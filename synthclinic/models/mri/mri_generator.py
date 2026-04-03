"""
synthclinic.models.mri.mri_generator
---------------------------------
``MRIGenerator`` — end-to-end MRI synthesis using a two-stage pipeline:
  Stage 1: Train a VAE to compress 2D MRI slices into a compact latent space.
  Stage 2: Train a latent DDPM to model the latent distribution.

Generation:
  DDPM samples a latent z → VAE decoder produces a synthetic MRI slice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthclinic.base import BaseGenerator
from synthclinic.data.preprocessing.mri_preprocess import MRISliceDataset, denormalise
from synthclinic.models.mri.vae import VAE, vae_loss
from synthclinic.models.mri.ddpm import LatentUNet, LinearNoiseSchedule

logger = logging.getLogger(__name__)


class MRIGenerator(BaseGenerator):
    """
    Synthetic brain MRI slice generator via latent diffusion (VAE + DDPM).

    Parameters
    ----------
    config : dict
        Supported keys:

        Stage 1 — VAE
        ``img_size``         : input spatial size, must be divisible by 16 (default 64)
        ``vae_base_channels``: encoder/decoder channel width (default 32)
        ``latent_dim``       : VAE latent dimension (default 128)
        ``vae_beta``         : KL weight in VAE loss (default 1.0)
        ``vae_epochs``       : VAE training epochs (default 100)
        ``vae_lr``           : VAE Adam lr (default 1e-3)

        Stage 2 — DDPM
        ``diffusion_steps``  : T in the noise schedule (default 1000)
        ``ddpm_epochs``      : DDPM training epochs (default 200)
        ``ddpm_lr``          : DDPM Adam lr (default 3e-4)
        ``ddpm_hidden_dim``  : LatentUNet hidden width (default 512)
        ``ddpm_depth``       : LatentUNet residual block count (default 6)

        Shared
        ``batch_size``       : mini-batch size (default 32)
        ``log_every``        : log metrics every N epochs (default 20)
        ``sampling_steps``   : DDIM-style steps at inference (default 100)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        super().__init__(config)
        self._vae: Optional[VAE] = None
        self._ddpm_model: Optional[LatentUNet] = None
        self._schedule: Optional[LinearNoiseSchedule] = None

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def train(self, data: np.ndarray) -> Dict[str, float]:
        """
        Two-stage training on MRI slices.

        Parameters
        ----------
        data : np.ndarray, shape (N, 1, H, W), dtype float32, values in [0, 1]

        Returns
        -------
        dict of final losses from both stages
        """
        cfg = self.config
        img_size = cfg.get("img_size", 64)
        latent_dim = cfg.get("latent_dim", 128)
        batch_size = cfg.get("batch_size", 32)
        log_every = cfg.get("log_every", 20)

        dataset = MRISliceDataset(data, img_size=img_size, augment=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # ----------------------------------------------------------------
        # Stage 1: Train VAE
        # ----------------------------------------------------------------
        logger.info("Stage 1: VAE training …")
        vae_metrics = self._train_vae(loader, cfg, img_size, latent_dim, log_every)

        # ----------------------------------------------------------------
        # Stage 2: Encode all data with frozen VAE, then train DDPM
        # ----------------------------------------------------------------
        logger.info("Stage 2: Encoding latents with trained VAE …")
        latents = self._encode_dataset(loader)

        logger.info("Stage 2: DDPM training on %d latent vectors …", len(latents))
        ddpm_metrics = self._train_ddpm(latents, cfg, latent_dim, log_every)

        self.is_trained = True
        return {**vae_metrics, **ddpm_metrics}

    def generate(self, n_samples: int, sampling_steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Generate *n_samples* synthetic MRI slices.

        Returns
        -------
        np.ndarray, shape (n_samples, 1, H, W), dtype float32, values in [0, 1]
        """
        self._require_trained()
        cfg = self.config
        steps = sampling_steps or cfg.get("sampling_steps", 100)
        latent_dim = cfg.get("latent_dim", 128)

        self._vae.eval()
        self._ddpm_model.eval()

        batch_size = min(n_samples, 32)
        all_images = []
        generated = 0

        with torch.no_grad():
            while generated < n_samples:
                bs = min(batch_size, n_samples - generated)
                # Sample latents via DDPM
                z_0 = self._schedule.sample(
                    self._ddpm_model, bs, latent_dim, self.device, steps=steps
                )
                # Decode to image space
                x_hat = self._vae.decode(z_0)  # (bs, 1, H, W) in [-1, 1]
                x_hat = denormalise(x_hat)      # → [0, 1]
                all_images.append(x_hat.cpu().numpy())
                generated += bs

        self._vae.train()
        self._ddpm_model.train()
        return np.concatenate(all_images, axis=0)

    def save(self, path: str | Path) -> None:
        self._require_trained()
        out_dir = Path(path) / "mri_generator"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._vae.state_dict(), out_dir / "vae_weights.pt")
        torch.save(self._ddpm_model.state_dict(), out_dir / "ddpm_weights.pt")

        meta = {
            **self.config,
            "latent_dim": self.config.get("latent_dim", 128),
            "img_size": self.config.get("img_size", 64),
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("MRIGenerator saved to %s", out_dir)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path) / "mri_generator"
        if not in_dir.exists():
            in_dir = Path(path)

        with open(in_dir / "config.json") as f:
            meta = json.load(f)

        cfg = meta
        img_size = cfg.get("img_size", 64)
        latent_dim = cfg.get("latent_dim", 128)

        self._vae = VAE(
            in_channels=1,
            base_channels=cfg.get("vae_base_channels", 32),
            latent_dim=latent_dim,
            img_size=img_size,
        ).to(self.device)
        self._vae.load_state_dict(
            torch.load(in_dir / "vae_weights.pt", map_location=self.device)
        )
        self._vae.eval()

        self._ddpm_model = LatentUNet(
            latent_dim=latent_dim,
            hidden_dim=cfg.get("ddpm_hidden_dim", 512),
            depth=cfg.get("ddpm_depth", 6),
        ).to(self.device)
        self._ddpm_model.load_state_dict(
            torch.load(in_dir / "ddpm_weights.pt", map_location=self.device)
        )

        self._schedule = LinearNoiseSchedule(
            T=cfg.get("diffusion_steps", 1000),
            device=self.device,
        )

        self.is_trained = True
        logger.info("MRIGenerator loaded from %s", in_dir)

    # ------------------------------------------------------------------
    # Private training helpers
    # ------------------------------------------------------------------

    def _train_vae(
        self,
        loader: DataLoader,
        cfg: Dict,
        img_size: int,
        latent_dim: int,
        log_every: int,
    ) -> Dict[str, float]:
        self._vae = VAE(
            in_channels=1,
            base_channels=cfg.get("vae_base_channels", 32),
            latent_dim=latent_dim,
            img_size=img_size,
        ).to(self.device)

        opt = optim.Adam(self._vae.parameters(), lr=cfg.get("vae_lr", 1e-3))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.get("vae_epochs", 100)
        )
        beta = cfg.get("vae_beta", 1.0)
        epochs = cfg.get("vae_epochs", 100)

        for epoch in tqdm(range(epochs), desc="VAE"):
            epoch_loss = 0.0
            for x in loader:
                x = x.to(self.device)
                x_hat, mu, logvar = self._vae(x)
                loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._vae.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % log_every == 0:
                logger.info(
                    "[VAE %d/%d] loss=%.4f recon=%.4f kl=%.4f",
                    epoch + 1, epochs,
                    epoch_loss / len(loader), recon.item(), kl.item(),
                )

        return {"vae_loss": epoch_loss / len(loader)}

    def _encode_dataset(self, loader: DataLoader) -> torch.Tensor:
        """Encode all training slices into latent space using the frozen VAE."""
        self._vae.eval()
        latents = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                mu, _ = self._vae.encode(x)
                latents.append(mu.cpu())
        self._vae.train()
        return torch.cat(latents, dim=0)  # (N, latent_dim)

    def _train_ddpm(
        self,
        latents: torch.Tensor,
        cfg: Dict,
        latent_dim: int,
        log_every: int,
    ) -> Dict[str, float]:
        self._schedule = LinearNoiseSchedule(
            T=cfg.get("diffusion_steps", 1000),
            device=self.device,
        )
        self._ddpm_model = LatentUNet(
            latent_dim=latent_dim,
            hidden_dim=cfg.get("ddpm_hidden_dim", 512),
            depth=cfg.get("ddpm_depth", 6),
        ).to(self.device)

        opt = optim.Adam(self._ddpm_model.parameters(), lr=cfg.get("ddpm_lr", 3e-4))
        epochs = cfg.get("ddpm_epochs", 200)
        batch_size = cfg.get("batch_size", 32)

        # Build simple DataLoader over the encoded latents
        from torch.utils.data import TensorDataset
        lat_loader = DataLoader(
            TensorDataset(latents),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        for epoch in tqdm(range(epochs), desc="DDPM"):
            epoch_loss = 0.0
            for (z_0,) in lat_loader:
                z_0 = z_0.to(self.device)
                B = z_0.size(0)

                t = torch.randint(0, self._schedule.T, (B,), device=self.device)
                z_t, noise = self._schedule.q_sample(z_0, t)
                eps_hat = self._ddpm_model(z_t, t)

                loss = nn.functional.mse_loss(eps_hat, noise)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._ddpm_model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()

            if (epoch + 1) % log_every == 0:
                logger.info(
                    "[DDPM %d/%d] mse_loss=%.6f",
                    epoch + 1, epochs, epoch_loss / len(lat_loader),
                )

        return {"ddpm_loss": epoch_loss / len(lat_loader)}
