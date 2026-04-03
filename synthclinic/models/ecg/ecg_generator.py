"""
synthclinic.models.ecg.ecg_generator
---------------------------------
``ECGGenerator`` — BaseGenerator wrapper around TimeGAN with the full
three-phase training procedure from Yoon et al. (2019).

Training phases
---------------
1. **Autoencoder** (``ae_epochs``):  E + R minimise MSE reconstruction loss.
2. **Supervisor**  (``sup_epochs``): S minimises step-wise prediction loss on
   real latent sequences H produced by the frozen Embedder.
3. **Joint**       (``joint_epochs``): G + S are trained adversarially against D.
   E + R continue to be updated with reconstruction + supervised regularisation.
   Two discriminator updates per generator update (common GAN practice).

Loss coefficients (all tunable via config)
------------------------------------------
  γ  (gamma)  : weight on supervised loss in the generator objective
  η  (eta)    : weight on supervised loss in the embedder objective
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from synthclinic.base import BaseGenerator
from synthclinic.models.ecg.timegan import TimeGANModel

logger = logging.getLogger(__name__)


class ECGGenerator(BaseGenerator):
    """
    Synthetic ECG generator using TimeGAN.

    Parameters
    ----------
    config : dict
        Supported keys:

        ``seq_len``        : segment length T (default 256)
        ``input_dim``      : feature dim — 1 for univariate ECG (default 1)
        ``hidden_dim``     : GRU hidden size (default 24)
        ``noise_dim``      : Generator noise dimension (default 8)
        ``num_layers``     : GRU layers (default 3)
        ``ae_epochs``      : autoencoder pre-training epochs (default 200)
        ``sup_epochs``     : supervisor pre-training epochs (default 200)
        ``joint_epochs``   : joint adversarial training epochs (default 600)
        ``batch_size``     : mini-batch size (default 128)
        ``lr``             : learning rate for all optimisers (default 1e-3)
        ``gamma``          : supervised loss weight in G objective (default 1.0)
        ``eta``            : supervised loss weight in E objective (default 0.1)
        ``log_every``      : log metrics every N epochs (default 50)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        super().__init__(config)
        self._net: Optional[TimeGANModel] = None
        self._history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def train(self, data: np.ndarray) -> Dict[str, float]:
        """
        Train TimeGAN on ECG segments.

        Parameters
        ----------
        data : np.ndarray, shape (N, T) or (N, T, 1), dtype float32

        Returns
        -------
        dict of final-epoch losses
        """
        cfg = self.config
        data = self._prepare_input(data)  # (N, T, 1)

        seq_len = data.shape[1]
        input_dim = data.shape[2]

        self._net = TimeGANModel(
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=cfg.get("hidden_dim", 24),
            noise_dim=cfg.get("noise_dim", 8),
            num_layers=cfg.get("num_layers", 3),
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(data))
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 128),
            shuffle=True,
            drop_last=True,
        )

        lr = cfg.get("lr", 1e-3)
        opt_E = optim.Adam(
            list(self._net.embedder.parameters())
            + list(self._net.recovery.parameters()),
            lr=lr,
        )
        opt_S = optim.Adam(self._net.supervisor.parameters(), lr=lr)
        opt_G = optim.Adam(
            list(self._net.generator.parameters())
            + list(self._net.supervisor.parameters()),
            lr=lr,
        )
        opt_D = optim.Adam(self._net.discriminator.parameters(), lr=lr)

        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        gamma = cfg.get("gamma", 1.0)
        eta = cfg.get("eta", 0.1)
        log_every = cfg.get("log_every", 50)

        # ----------------------------------------------------------------
        # Phase 1: Autoencoder (Embedder + Recovery)
        # ----------------------------------------------------------------
        ae_epochs = cfg.get("ae_epochs", 200)
        logger.info("Phase 1: Autoencoder training (%d epochs) …", ae_epochs)
        for epoch in tqdm(range(ae_epochs), desc="AE"):
            for (X,) in loader:
                X = X.to(self.device)
                H = self._net.embedder(X)
                X_tilde = self._net.recovery(H)

                # Reconstruction loss
                l_recon = mse(X, X_tilde)
                # Optional: supervised loss on real latent sequences
                H_sup = self._net.supervisor(H)
                l_sup = mse(H[:, 1:, :], H_sup[:, :-1, :])

                loss = l_recon + 0.1 * l_sup
                opt_E.zero_grad()
                loss.backward()
                opt_E.step()

            if (epoch + 1) % log_every == 0:
                logger.info("[AE %d/%d] recon=%.4f", epoch + 1, ae_epochs, l_recon.item())

        # ----------------------------------------------------------------
        # Phase 2: Supervisor pre-training (Supervisor only)
        # ----------------------------------------------------------------
        sup_epochs = cfg.get("sup_epochs", 200)
        logger.info("Phase 2: Supervisor training (%d epochs) …", sup_epochs)
        for epoch in tqdm(range(sup_epochs), desc="Supervisor"):
            for (X,) in loader:
                X = X.to(self.device)
                with torch.no_grad():
                    H = self._net.embedder(X)
                H_sup = self._net.supervisor(H)
                l_sup = mse(H[:, 1:, :], H_sup[:, :-1, :])

                opt_S.zero_grad()
                l_sup.backward()
                opt_S.step()

            if (epoch + 1) % log_every == 0:
                logger.info("[Sup %d/%d] sup_loss=%.4f", epoch + 1, sup_epochs, l_sup.item())

        # ----------------------------------------------------------------
        # Phase 3: Joint adversarial training
        # ----------------------------------------------------------------
        joint_epochs = cfg.get("joint_epochs", 600)
        logger.info("Phase 3: Joint training (%d epochs) …", joint_epochs)

        final_metrics: Dict[str, float] = {}
        g_losses, d_losses = [], []

        for epoch in tqdm(range(joint_epochs), desc="Joint"):
            for (X,) in loader:
                X = X.to(self.device)
                B = X.size(0)

                # -------- Generator step --------
                E_hat = self._net.generate_latent(B, self.device)     # fake latent
                H = self._net.embedder(X)                              # real latent
                H_hat_sup = self._net.supervisor(E_hat)                # supervised fake

                # Adversarial: fool the discriminator
                Y_fake_e = self._net.discriminator(E_hat)
                l_adv_g_e = bce(Y_fake_e, torch.ones_like(Y_fake_e))

                Y_fake = self._net.discriminator(H_hat_sup)
                l_adv_g = bce(Y_fake, torch.ones_like(Y_fake))

                # Supervised loss
                l_sup_g = mse(H[:, 1:, :], H_hat_sup[:, :-1, :])

                # Moment matching (mean + variance per feature)
                X_hat = self._net.recovery(E_hat)
                l_mom = (
                    torch.mean(torch.abs(X.mean(0) - X_hat.mean(0)))
                    + torch.mean(torch.abs(X.var(0) - X_hat.var(0)))
                )

                l_G = (
                    l_adv_g
                    + l_adv_g_e
                    + gamma * l_sup_g
                    + 100 * l_mom
                )
                opt_G.zero_grad()
                l_G.backward()
                nn.utils.clip_grad_norm_(self._net.generator.parameters(), 1.0)
                opt_G.step()

                # -------- Embedder update --------
                H = self._net.embedder(X)
                X_tilde = self._net.recovery(H)
                H_sup = self._net.supervisor(H)
                l_recon = mse(X, X_tilde)
                l_sup_e = mse(H[:, 1:, :], H_sup[:, :-1, :])
                l_E = l_recon + eta * l_sup_e
                opt_E.zero_grad()
                l_E.backward()
                opt_E.step()

                # -------- Discriminator step (×2) --------
                for _ in range(2):
                    H = self._net.embedder(X).detach()
                    E_hat = self._net.generate_latent(B, self.device).detach()
                    H_hat_sup = self._net.supervisor(E_hat).detach()

                    Y_real = self._net.discriminator(H)
                    Y_fake = self._net.discriminator(H_hat_sup)
                    Y_fake_e = self._net.discriminator(E_hat)

                    l_D = (
                        bce(Y_real, torch.ones_like(Y_real))
                        + bce(Y_fake, torch.zeros_like(Y_fake))
                        + gamma * bce(Y_fake_e, torch.zeros_like(Y_fake_e))
                    )

                    # Only update if discriminator is not too strong
                    if l_D.item() > 0.15:
                        opt_D.zero_grad()
                        l_D.backward()
                        opt_D.step()

            g_losses.append(l_G.item())
            d_losses.append(l_D.item())

            if (epoch + 1) % log_every == 0:
                logger.info(
                    "[Joint %d/%d] G=%.4f  D=%.4f  recon=%.4f  sup=%.4f",
                    epoch + 1, joint_epochs,
                    l_G.item(), l_D.item(), l_recon.item(), l_sup_g.item(),
                )

        self._history = {"generator_loss": g_losses, "discriminator_loss": d_losses}
        final_metrics = {
            "generator_loss": g_losses[-1],
            "discriminator_loss": d_losses[-1],
            "recon_loss": l_recon.item(),
        }
        self.is_trained = True
        logger.info("ECGGenerator training complete. Final G=%.4f D=%.4f",
                    final_metrics["generator_loss"], final_metrics["discriminator_loss"])
        return final_metrics

    def generate(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate *n_samples* synthetic ECG segments.

        Returns
        -------
        np.ndarray, shape (n_samples, T, 1), values in (0, 1)
        """
        self._require_trained()
        self._net.eval()
        segments = []
        batch_size = min(n_samples, 256)
        generated = 0
        with torch.no_grad():
            while generated < n_samples:
                bs = min(batch_size, n_samples - generated)
                X_hat = self._net.generate_data(bs, self.device)
                segments.append(X_hat.cpu().numpy())
                generated += bs
        self._net.train()
        return np.concatenate(segments, axis=0)  # (N, T, 1)

    def save(self, path: str | Path) -> None:
        self._require_trained()
        out_dir = Path(path) / "ecg_generator"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), out_dir / "timegan_weights.pt")

        meta = {
            **self.config,
            "seq_len": self._net.seq_len,
            "input_dim": self._net.input_dim,
            "hidden_dim": self._net.hidden_dim,
            "noise_dim": self._net.noise_dim,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("ECGGenerator saved to %s", out_dir)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path) / "ecg_generator"
        if not in_dir.exists():
            in_dir = Path(path)

        with open(in_dir / "config.json") as f:
            meta = json.load(f)

        self._net = TimeGANModel(
            seq_len=meta["seq_len"],
            input_dim=meta["input_dim"],
            hidden_dim=meta["hidden_dim"],
            noise_dim=meta["noise_dim"],
            num_layers=meta.get("num_layers", 3),
        ).to(self.device)

        self._net.load_state_dict(
            torch.load(in_dir / "timegan_weights.pt", map_location=self.device)
        )
        self._net.eval()
        self.is_trained = True
        logger.info("ECGGenerator loaded from %s", in_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input(data: np.ndarray) -> np.ndarray:
        """Ensure data is shape (N, T, input_dim) float32."""
        data = data.astype(np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]  # (N, T) → (N, T, 1)
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
        return data
