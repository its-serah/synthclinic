"""
synthclinic.models.mri.vae
-----------------------
Convolutional Variational Autoencoder (VAE) for 2D MRI slices.

Architecture
------------
Encoder: 4× Conv2D blocks (stride 2) → flatten → μ, log σ²
Decoder: linear → 4× ConvTranspose2D blocks → tanh

The VAE serves as the *encoder* in the latent diffusion pipeline:
  1. VAE encodes real MRI slices x → latent z  (compression ratio ≈ 16×)
  2. DDPM is trained to denoise latent vectors z
  3. At generation time: DDPM samples ẑ → VAE decoder → synthetic MRI x̂

This mirrors the key idea of Rombach et al. (2022) "High-Resolution Image
Synthesis with Latent Diffusion Models", but at a scale appropriate for
brain MRI slices (64×64 or 128×128 inputs).

Reference
---------
Kingma, D.P. & Welling, M. (2014). "Auto-Encoding Variational Bayes."
  ICLR 2014. https://arxiv.org/abs/1312.6114
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    """Conv2D + GroupNorm + SiLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.SiLU(),
    )


def _deconv_block(in_c: int, out_c: int) -> nn.Sequential:
    """Bilinear upsample + Conv2D + GroupNorm + SiLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.SiLU(),
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    Convolutional encoder.

    Input  : (B, in_channels, H, W) — normalised MRI in [-1, 1]
    Output : μ, log σ² each of shape (B, latent_dim)

    Spatial path: H → H/2 → H/4 → H/8 → H/16 (4 stride-2 conv blocks)
    Final spatial size: (H/16, W/16)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        c = base_channels

        self.encoder = nn.Sequential(
            _conv_block(in_channels, c, stride=1),      # H×W
            _conv_block(c, c * 2, stride=2),            # H/2
            _conv_block(c * 2, c * 4, stride=2),        # H/4
            _conv_block(c * 4, c * 8, stride=2),        # H/8
            _conv_block(c * 8, c * 8, stride=2),        # H/16
        )

        spatial = img_size // 16
        flat_dim = c * 8 * spatial * spatial
        self._flat_dim = flat_dim
        self._spatial = spatial
        self._c8 = c * 8

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    """
    Convolutional decoder.

    Input  : (B, latent_dim)
    Output : (B, out_channels, H, W) in [-1, 1] via tanh
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        c = base_channels
        spatial = img_size // 16
        flat_dim = c * 8 * spatial * spatial

        self._spatial = spatial
        self._c8 = c * 8

        self.fc = nn.Linear(latent_dim, flat_dim)

        self.decoder = nn.Sequential(
            _deconv_block(c * 8, c * 8),   # H/8
            _deconv_block(c * 8, c * 4),   # H/4
            _deconv_block(c * 4, c * 2),   # H/2
            _deconv_block(c * 2, c),       # H
        )
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._c8, self._spatial, self._spatial)
        h = self.decoder(h)
        return torch.tanh(self.out_conv(h))


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    Full Variational Autoencoder combining encoder and decoder.

    Parameters
    ----------
    in_channels  : 1 for grayscale MRI
    base_channels: starting channel width (default 32)
    latent_dim   : dimension of latent code z
    img_size     : spatial size H = W (must be divisible by 16)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        assert img_size % 16 == 0, "img_size must be divisible by 16"

        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels, base_channels, latent_dim, img_size)
        self.decoder = VAEDecoder(in_channels, base_channels, latent_dim, img_size)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterisation trick: z = μ + ε·σ,  ε ~ N(0, I).

        At inference (eval mode), returns μ directly for deterministic decoding.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_hat  : reconstructed image in [-1, 1]
        mu     : latent mean
        logvar : latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ---------------------------------------------------------------------------
# VAE loss
# ---------------------------------------------------------------------------

def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE loss = Reconstruction loss + β · KL divergence.

    β = 1 → standard VAE.
    β > 1 → disentangled representations (β-VAE, Higgins et al., 2017).

    Returns
    -------
    total_loss, recon_loss, kl_loss
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl
