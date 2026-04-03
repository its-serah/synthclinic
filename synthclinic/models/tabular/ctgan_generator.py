"""
synthclinic.models.tabular.ctgan_generator
---------------------------------------
Tabular synthetic data generator using CTGAN and TVAE from the SDV library.

CTGAN (Conditional Tabular GAN) — Xu et al., NeurIPS 2019.
TVAE (Tabular VAE) — same paper, VAE baseline.

Both models handle mixed data types (continuous + categorical) natively
via mode-specific normalisation and conditional vector sampling.

Reference: https://github.com/sdv-dev/CTGAN
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd

from synthclinic.base import BaseGenerator

logger = logging.getLogger(__name__)


class TabularGenerator(BaseGenerator):
    """
    Wraps SDV's CTGAN / TVAE with the GenMed ``BaseGenerator`` interface.

    Parameters
    ----------
    config : dict
        Supported keys:

        ``model``          : ``"ctgan"`` (default) or ``"tvae"``
        ``epochs``         : training epochs (default 300)
        ``batch_size``     : mini-batch size (default 500)
        ``generator_dim``  : CTGAN generator layer sizes (default [256, 256])
        ``discriminator_dim`` : CTGAN discriminator layer sizes (default [256, 256])
        ``embedding_dim``  : TVAE embedding dim (default 128)
        ``verbose``        : print per-epoch loss (default False)
        ``cuda``           : use GPU if available (default True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        super().__init__(config)
        self._model_type: str = config.get("model", "ctgan").lower()
        self._sdv_model = None
        self._columns: list = []

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit CTGAN or TVAE on *data*.

        Parameters
        ----------
        data : pd.DataFrame
            Clean tabular medical data — can contain a mix of continuous and
            categorical columns.  Missing values should be imputed beforehand
            (see ``TabularPreprocessor``).

        Returns
        -------
        dict with key ``"status"`` = ``"ok"`` (SDV does not expose epoch losses).
        """
        self._columns = list(data.columns)
        self._sdv_model = self._build_sdv_model(data)

        logger.info(
            "Training %s on %d rows × %d cols …",
            self._model_type.upper(), len(data), len(data.columns),
        )
        self._sdv_model.fit(data)
        self.is_trained = True
        logger.info("%s training complete.", self._model_type.upper())
        return {"status": "ok"}

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Sample *n_samples* synthetic rows.

        Returns
        -------
        pd.DataFrame with the same columns as the training data.
        """
        self._require_trained()
        logger.info("Generating %d synthetic tabular samples …", n_samples)
        synthetic = self._sdv_model.sample(num_rows=n_samples)
        return synthetic

    def save(self, path: str | Path) -> None:
        """
        Save the fitted SDV model to *path*.

        Creates a ``<path>/tabular_generator/`` directory containing:
        - ``model.pkl``  : the serialised SDV model
        - ``config.json``: generator config
        """
        self._require_trained()
        out_dir = Path(path) / "tabular_generator"
        out_dir.mkdir(parents=True, exist_ok=True)

        import pickle
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(self._sdv_model, f)

        meta = {**self.config, "model_type": self._model_type, "columns": self._columns}
        with open(out_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("TabularGenerator saved to %s", out_dir)

    def load(self, path: str | Path) -> None:
        """Restore a previously saved TabularGenerator."""
        in_dir = Path(path) / "tabular_generator"
        if not in_dir.exists():
            # Try path directly
            in_dir = Path(path)

        import pickle
        with open(in_dir / "model.pkl", "rb") as f:
            self._sdv_model = pickle.load(f)

        cfg_path = in_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                meta = json.load(f)
            self._model_type = meta.get("model_type", self._model_type)
            self._columns = meta.get("columns", [])

        self.is_trained = True
        logger.info("TabularGenerator loaded from %s", in_dir)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_sdv_model(self, data: pd.DataFrame):
        cfg = self.config
        use_cuda = cfg.get("cuda", True) and str(self.device) != "cpu"

        # SDV ≥ 1.9 requires metadata to be detected from the DataFrame
        try:
            from sdv.metadata import Metadata as _Metadata
            metadata = _Metadata.detect_from_dataframe(data)
        except (ImportError, AttributeError):
            from sdv.metadata import SingleTableMetadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)

        if self._model_type == "ctgan":
            from sdv.single_table import CTGANSynthesizer
            return CTGANSynthesizer(
                metadata=metadata,
                epochs=cfg.get("epochs", 300),
                batch_size=cfg.get("batch_size", 500),
                generator_dim=tuple(cfg.get("generator_dim", [256, 256])),
                discriminator_dim=tuple(cfg.get("discriminator_dim", [256, 256])),
                verbose=cfg.get("verbose", False),
                cuda=use_cuda,
            )

        elif self._model_type == "tvae":
            from sdv.single_table import TVAESynthesizer
            return TVAESynthesizer(
                metadata=metadata,
                epochs=cfg.get("epochs", 300),
                batch_size=cfg.get("batch_size", 500),
                embedding_dim=cfg.get("embedding_dim", 128),
                compress_dims=tuple(cfg.get("compress_dims", [128, 128])),
                decompress_dims=tuple(cfg.get("decompress_dims", [128, 128])),
                cuda=use_cuda,
            )

        else:
            raise ValueError(
                f"Unknown tabular model '{self._model_type}'. Choose 'ctgan' or 'tvae'."
            )
