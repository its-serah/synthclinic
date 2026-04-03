"""
synthclinic.data.loaders.clinical_loader
--------------------------------------
Loads the MTSamples medical transcription dataset (open-access, ~4 000 records).

MTSamples contains de-identified clinical transcriptions from 40+ medical
specialties.  No registration or credentialed access required.

Primary source : https://www.mtsamples.com/
CSV mirror     : https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
                 (or GitHub mirror used here)
HuggingFace    : Gaborandi/mtsamples  (fallback)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_MTSAMPLES_URLS = [
    # Primary GitHub mirror
    "https://raw.githubusercontent.com/benjamin-croker/mtsamples/main/mtsamples.csv",
    # Secondary
    "https://raw.githubusercontent.com/terrence-lau/mt-samples/main/mtsamples.csv",
]


class ClinicalNotesLoader:
    """
    Downloads and caches the MTSamples clinical transcription dataset.

    Parameters
    ----------
    cache_dir:
        Local directory for the cached CSV.
    specialties:
        Optional list of medical specialties to keep (case-insensitive).
        ``None`` → all specialties.
    min_length:
        Minimum character length of a transcription to include.
        Short / empty notes are discarded.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw/clinical",
        specialties: Optional[List[str]] = None,
        min_length: int = 200,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.specialties = [s.lower() for s in specialties] if specialties else None
        self.min_length = min_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Return a cleaned DataFrame with at least two columns:

        - ``transcription`` : full clinical note text
        - ``medical_specialty`` : speciality label
        """
        df = self._fetch()
        df = self._clean(df)
        logger.info(
            "Clinical notes: %d records | %d specialties",
            len(df),
            df["medical_specialty"].nunique(),
        )
        return df

    def texts(self) -> List[str]:
        """Convenience — return only the transcription strings as a list."""
        return self.load()["transcription"].tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self) -> pd.DataFrame:
        cache = self.cache_dir / "mtsamples.csv"
        if cache.exists():
            return pd.read_csv(cache)

        for url in _MTSAMPLES_URLS:
            try:
                logger.info("Downloading MTSamples from %s …", url)
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df.to_csv(cache, index=False)
                return df
            except Exception as exc:
                logger.warning("URL failed (%s): %s", url, exc)

        logger.warning("Direct download failed. Trying HuggingFace datasets …")
        return self._fetch_huggingface(cache)

    def _fetch_huggingface(self, cache: Path) -> pd.DataFrame:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("Gaborandi/mtsamples", split="train", trust_remote_code=True)
            df = ds.to_pandas()
            df.to_csv(cache, index=False)
            return df
        except Exception as exc:
            raise RuntimeError(
                "Could not download MTSamples via any method. "
                "Check your internet connection or place mtsamples.csv in "
                f"{self.cache_dir} manually."
            ) from exc

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Identify key columns flexibly
        text_col = next(
            (c for c in df.columns if "transcription" in c or c == "text"), None
        )
        spec_col = next(
            (c for c in df.columns if "specialty" in c), None
        )

        rename_map = {}
        if text_col and text_col != "transcription":
            rename_map[text_col] = "transcription"
        if spec_col and spec_col != "medical_specialty":
            rename_map[spec_col] = "medical_specialty"
        df = df.rename(columns=rename_map)

        # Guard: ensure required columns exist
        if "transcription" not in df.columns:
            raise ValueError(
                f"Could not find a transcription column in MTSamples. "
                f"Available columns: {list(df.columns)}"
            )
        if "medical_specialty" not in df.columns:
            df["medical_specialty"] = "unknown"

        # Filter empty / too-short notes
        df = df.dropna(subset=["transcription"])
        df = df[df["transcription"].str.strip().str.len() >= self.min_length]

        # Optional specialty filter
        if self.specialties:
            mask = df["medical_specialty"].str.lower().isin(self.specialties)
            df = df[mask]
            if df.empty:
                logger.warning(
                    "No notes found for specialties %s. Ignoring filter.",
                    self.specialties,
                )
                df = self._clean.__wrapped__(self, df) if hasattr(self._clean, "__wrapped__") else df

        return df[["transcription", "medical_specialty"]].reset_index(drop=True)
