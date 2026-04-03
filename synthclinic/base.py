"""
synthclinic.base
-----------
Abstract base class shared by all GenMed modality generators.

Every generator exposes the same four-method contract:
    train(data)        -> dict of training metrics
    generate(n)        -> synthetic data matching training format
    save(path)         -> persist weights + metadata
    load(path)         -> restore weights + metadata
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for all GenMed modality generators.

    Subclasses must implement: train, generate, save, load.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained: bool = False
        logger.info("%s initialised — device: %s", self.__class__.__name__, self.device)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, data: Any) -> Dict[str, float]:
        """
        Fit the generative model on *data*.

        Parameters
        ----------
        data:
            Modality-specific training data. Type varies by subclass
            (``np.ndarray``, ``pd.DataFrame``, ``list[str]``, …).

        Returns
        -------
        dict
            Training diagnostics — at minimum ``{"loss": float}``.
            Subclasses may return richer dictionaries (e.g. generator /
            discriminator losses, perplexity, ELBO, …).
        """

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> Any:
        """
        Draw *n_samples* synthetic samples from the trained model.

        Must call ``_require_trained()`` before generation.

        Returns
        -------
        Synthetic data in the same format / dtype as the training input.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist model weights and any configuration needed for reload.

        Parameters
        ----------
        path:
            Target directory (created if absent) or file path.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Restore model weights from a previously saved checkpoint.

        Sets ``self.is_trained = True`` on success.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        """Raise ``RuntimeError`` if the model has not been trained yet."""
        if not self.is_trained:
            raise RuntimeError(
                f"{self.__class__.__name__} must be trained before calling generate(). "
                "Call .train(data) first or restore a checkpoint with .load(path)."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(device={self.device}, status={status})"
