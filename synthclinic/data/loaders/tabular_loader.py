"""
synthclinic.data.loaders.tabular_loader
-----------------------------------
Loads open-access tabular medical datasets for the GenMed tabular generator.

Supported datasets
------------------
``heart_disease``
    UCI Heart Disease (Cleveland), 303 samples, 13 clinical features.
    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

``diabetes``
    PIMA Indian Diabetes Dataset, 768 samples, 8 features.
    Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

``combined``
    A reduced, harmonised view of both datasets aligned on shared columns.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote URLs (mirrored, no login required)
# ---------------------------------------------------------------------------
_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
_PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

_HEART_COLS: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
_PIMA_COLS: List[str] = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]


class TabularLoader:
    """
    Downloads and caches tabular medical datasets locally.

    Parameters
    ----------
    dataset:
        One of ``"heart_disease"``, ``"diabetes"``, or ``"combined"``.
    cache_dir:
        Local directory for cached CSVs.
    """

    def __init__(
        self,
        dataset: Literal["heart_disease", "diabetes", "combined"] = "heart_disease",
        cache_dir: str = "data/raw/tabular",
    ) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Return a clean ``pd.DataFrame`` ready for the generative model."""
        loaders = {
            "heart_disease": self._load_heart_disease,
            "diabetes": self._load_pima,
            "combined": self._load_combined,
        }
        if self.dataset not in loaders:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Choose from: {list(loaders.keys())}"
            )
        df = loaders[self.dataset]()
        logger.info(
            "Loaded %s dataset: %d rows × %d cols",
            self.dataset, len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_heart_disease(self) -> pd.DataFrame:
        cache = self.cache_dir / "heart_disease.csv"
        if not cache.exists():
            logger.info("Downloading UCI Heart Disease dataset …")
            raw = self._fetch(
                _HEART_URL,
                fallback="https://raw.githubusercontent.com/trane293/brats2017/master/"
                         "clev_heart_dis.csv",
            )
            df = pd.read_csv(
                io.StringIO(raw), header=None, names=_HEART_COLS, na_values="?"
            )
            df.to_csv(cache, index=False)
        else:
            df = pd.read_csv(cache)

        df = df.dropna().reset_index(drop=True)
        # Binarise target (0 = no disease, 1 = disease)
        df["target"] = (df["target"] > 0).astype(int)
        return df

    def _load_pima(self) -> pd.DataFrame:
        cache = self.cache_dir / "pima_diabetes.csv"
        if not cache.exists():
            logger.info("Downloading PIMA Diabetes dataset …")
            raw = self._fetch(_PIMA_URL)
            df = pd.read_csv(io.StringIO(raw), header=None, names=_PIMA_COLS)
            df.to_csv(cache, index=False)
        else:
            df = pd.read_csv(cache)
        return df.reset_index(drop=True)

    def _load_combined(self) -> pd.DataFrame:
        """
        Harmonise both datasets onto shared demographic and vital columns.
        Useful for testing multi-source tabular generation.
        """
        hd = self._load_heart_disease()[
            ["age", "sex", "trestbps", "chol", "thalach", "target"]
        ].rename(columns={
            "trestbps": "blood_pressure",
            "chol": "cholesterol",
            "thalach": "max_heart_rate",
            "target": "label",
        })
        hd["source"] = "heart_disease"

        dm = self._load_pima()[
            ["age", "blood_pressure", "outcome"]
        ].rename(columns={"outcome": "label"})
        dm["source"] = "diabetes"

        common = ["age", "blood_pressure", "label", "source"]
        combined = pd.concat([hd[common], dm[common]], ignore_index=True)
        return combined

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch(url: str, fallback: str | None = None, timeout: int = 30) -> str:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            if fallback:
                logger.warning("Primary URL failed (%s), trying fallback …", exc)
                r = requests.get(fallback, timeout=timeout)
                r.raise_for_status()
                return r.text
            raise
