"""
synthclinic.models.ecg.timegan
--------------------------
TimeGAN — Time-series Generative Adversarial Networks.

Reference: Yoon, J., Jarrett, D., & van der Schaar, M. (2019).
  "Time-series Generative Adversarial Networks."
  Advances in Neural Information Processing Systems (NeurIPS), 32.
  https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

Architecture
------------
TimeGAN decomposes time-series generation into a latent space learned jointly
by four networks:

  Embedder   E : X → H          Maps real sequences to a latent representation.
  Recovery   R : H → X̂         Reconstructs sequences from latent space.
  Generator  G : Z → Ê         Generates latent sequences from Gaussian noise.
  Supervisor S : H_t → Ĥ_{t+1} Captures step-wise temporal dynamics.
  Discriminator D : H → [0,1]  Distinguishes real from generated latent seqs.

Training has three phases:
  Phase 1 — Autoencoder  : Train E + R with MSE reconstruction loss.
  Phase 2 — Supervisor   : Train S to predict the next latent step.
  Phase 3 — Joint        : Adversarial training of G + S vs D, with
                           E + R continuing to reconstruct.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class _GRUBlock(nn.Module):
    """Shared GRU backbone used by all TimeGAN sub-networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out  # (B, T, out_dim)


class Embedder(nn.Module):
    """
    E: X → H  Maps real data to a latent embedding.

    Input  : (B, T, input_dim)
    Output : (B, T, hidden_dim)  — values in (0, 1) via sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = _GRUBlock(input_dim, hidden_dim, num_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rnn(x)
        return self.proj(h)


class Recovery(nn.Module):
    """
    R: H → X̂  Reconstructs data from latent embedding.

    Input  : (B, T, hidden_dim)
    Output : (B, T, input_dim)  — sigmoid activation
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = _GRUBlock(hidden_dim, hidden_dim, num_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.rnn(h)
        return self.proj(out)


class Generator(nn.Module):
    """
    G: Z → Ê  Generates latent sequences from Gaussian noise.

    Input  : (B, T, noise_dim)
    Output : (B, T, hidden_dim)  — sigmoid activation
    """

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = _GRUBlock(noise_dim, hidden_dim, num_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.rnn(z)
        return self.proj(out)


class Supervisor(nn.Module):
    """
    S: H_t → Ĥ_{t+1}  Predicts the next latent step (temporal dynamics).

    Uses num_layers - 1 GRU layers to stay shallower than the main networks.

    Input  : (B, T, hidden_dim)
    Output : (B, T, hidden_dim)  — sigmoid activation
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        sup_layers = max(1, num_layers - 1)
        self.rnn = _GRUBlock(hidden_dim, hidden_dim, sup_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.rnn(h)
        return self.proj(out)


class Discriminator(nn.Module):
    """
    D: H → [0, 1]  Classifies real vs generated latent sequences.

    Uses a bidirectional GRU to capture both forward and backward dynamics.

    Input  : (B, T, hidden_dim)
    Output : (B, T, 1)  — raw logits (use BCEWithLogitsLoss)
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = _GRUBlock(hidden_dim, hidden_dim, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.rnn(h)
        return self.proj(out)  # (B, T, 1)


# ---------------------------------------------------------------------------
# TimeGAN — composed module
# ---------------------------------------------------------------------------

class TimeGANModel(nn.Module):
    """
    Composed TimeGAN model holding all sub-networks.

    Parameters
    ----------
    seq_len    : length of each time-series segment (T)
    input_dim  : feature dimension (1 for univariate ECG)
    hidden_dim : GRU hidden state size
    noise_dim  : noise vector dimension for Generator
    num_layers : number of GRU stacked layers
    """

    def __init__(
        self,
        seq_len: int = 256,
        input_dim: int = 1,
        hidden_dim: int = 24,
        noise_dim: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def random_noise(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample Gaussian noise Z ~ N(0, I) for the Generator."""
        return torch.randn(batch_size, self.seq_len, self.noise_dim, device=device)

    def generate_latent(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Generate latent sequences: Ê = G(Z)."""
        z = self.random_noise(batch_size, device)
        return self.generator(z)

    def generate_data(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """End-to-end generation: X̂ = R(G(Z))."""
        e_hat = self.generate_latent(batch_size, device)
        return self.recovery(e_hat)
