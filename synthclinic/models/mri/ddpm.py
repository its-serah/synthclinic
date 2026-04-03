"""
synthclinic.models.mri.ddpm
-----------------------
Denoising Diffusion Probabilistic Model (DDPM) operating in the VAE latent space.

The DDPM operates on 1D latent vectors (output of the VAE encoder),
not on raw pixel space.  This dramatically reduces compute compared to
pixel-space diffusion while retaining generation quality — the key insight
of Latent Diffusion Models (Rombach et al., 2022).

Pipeline
--------
Training:
  1. Encode MRI slice x → z via frozen VAE encoder
  2. Sample timestep t ~ Uniform(1, T)
  3. Add noise: z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
  4. Predict noise: ε̂ = UNet(z_t, t)
  5. Loss = MSE(ε, ε̂)

Sampling (DDIM deterministic, or DDPM stochastic):
  1. Sample z_T ~ N(0, I)
  2. Iteratively denoise z_T → z_{T-1} → … → z_0
  3. Decode z_0 → x̂ via VAE decoder

References
----------
Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
  NeurIPS 2020. https://arxiv.org/abs/2006.11239
Song, J. et al. (2021). "Denoising Diffusion Implicit Models."
  ICLR 2021. https://arxiv.org/abs/2010.02502
Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models."
  CVPR 2022. https://arxiv.org/abs/2112.10752
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding for timestep t, as in the original DDPM.

    Input  : (B,)  integer timesteps
    Output : (B, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


# ---------------------------------------------------------------------------
# UNet for latent DDPM  (operates on 1D latent vectors)
# ---------------------------------------------------------------------------

class LatentUNet(nn.Module):
    """
    Simple MLP-based "UNet" operating on 1D latent vectors z ∈ ℝ^latent_dim.

    For 1D latent codes (not 2D feature maps), a residual MLP is more
    appropriate than a convolutional UNet.  This provides:
    - Skip connections (residual)
    - Time conditioning at every block

    Architecture per block:
        z → Linear → LayerNorm → SiLU → [+ time_emb] → Linear → LayerNorm → SiLU → + residual

    Parameters
    ----------
    latent_dim  : dimension of z
    hidden_dim  : hidden layer width
    time_emb_dim: sinusoidal time embedding width
    depth       : number of residual blocks
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        time_emb_dim: int = 128,
        depth: int = 6,
    ) -> None:
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, hidden_dim),
        )

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden_dim) for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim) — noisy latent
        t : (B,) — integer timesteps

        Returns
        -------
        (B, latent_dim) — predicted noise ε̂
        """
        t_emb = self.time_embed(t)          # (B, hidden_dim)
        h = self.input_proj(z) + t_emb      # (B, hidden_dim)

        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """
    Linear beta schedule from Ho et al. (2020).

    β_t linearly interpolated from β_start to β_end over T steps.
    Pre-computes α_t, ᾱ_t (cumulative product) for fast forward diffusion.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("posterior_variance",
                      betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def register(self, name: str, val: torch.Tensor) -> None:
        setattr(self, name, val)

    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample z_t ~ q(z_t | z_0).

        z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)   # (B, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        z_t = sqrt_ab * z_0 + sqrt_1mab * noise
        return z_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: LatentUNet,
        z_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: sample z_{t-1} ~ p_θ(z_{t-1} | z_t).

        Uses the DDPM posterior mean formula.
        """
        t_tensor = torch.full((z_t.size(0),), t, device=z_t.device, dtype=torch.long)
        eps_hat = model(z_t, t_tensor)

        betas_t = self.betas[t]
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = (1.0 / self.alphas[t]).sqrt()

        # Predicted z_0
        pred_z0 = sqrt_recip_alpha * (z_t - betas_t / sqrt_1mab * eps_hat)
        pred_z0 = pred_z0.clamp(-4, 4)  # clip for stability

        if t == 0:
            return pred_z0

        # Posterior variance
        posterior_var = self.posterior_variance[t]
        noise = torch.randn_like(z_t)
        return pred_z0 + posterior_var.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: LatentUNet,
        n_samples: int,
        latent_dim: int,
        device: torch.device,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: sample z_0 ~ p_θ(z_0).

        Parameters
        ----------
        steps : optional DDIM-style stride (sample every `steps` timestep).
                ``None`` → full T steps (DDPM).

        Returns
        -------
        (n_samples, latent_dim) tensor of denoised latent vectors.
        """
        model.eval()
        z = torch.randn(n_samples, latent_dim, device=device)

        timesteps = list(range(self.T - 1, -1, -1))
        if steps is not None:
            stride = max(1, self.T // steps)
            timesteps = timesteps[::stride]

        for t in timesteps:
            z = self.p_sample(model, z, t)

        return z
