"""
synthclinic.data.preprocessing.tabular_preprocess
----------------------------------------------
Preprocessing for tabular medical data fed into CTGAN / TVAE.

SDV's CTGAN handles its own internal transformation, but we expose an
explicit preprocessing step so the same clean DataFrame can be used for
both generative training and downstream ML evaluation (TSTR).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Prepares a medical DataFrame for CTGAN / TVAE training.

    Responsibilities
    ----------------
    * Identify continuous vs categorical columns automatically.
    * Impute missing values (median for continuous, mode for categorical).
    * Optionally standardise continuous columns (useful for evaluation / TSTR).
    * Expose metadata dict in SDV ``Metadata`` format.

    Parameters
    ----------
    continuous_cols:
        Explicit list of continuous column names.  ``None`` → auto-detect
        (columns with > ``cat_threshold`` unique values are treated as
        continuous).
    categorical_cols:
        Explicit list of categorical column names.
    cat_threshold:
        Max unique values for a column to be considered categorical when
        auto-detecting.
    scale:
        Whether to apply ``StandardScaler`` to continuous columns.
    """

    def __init__(
        self,
        continuous_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        cat_threshold: int = 15,
        scale: bool = False,
    ) -> None:
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.cat_threshold = cat_threshold
        self.scale = scale
        self._scaler: Optional[StandardScaler] = None
        self._cont_cols_fitted: List[str] = []
        self._cat_cols_fitted: List[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self._fit(df)
        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using previously fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first.")
        return self._transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse StandardScaler on continuous columns."""
        if self._scaler is None or not self._cont_cols_fitted:
            return df.copy()
        out = df.copy()
        present = [c for c in self._cont_cols_fitted if c in out.columns]
        out[present] = self._scaler.inverse_transform(out[present])
        return out

    @property
    def sdv_metadata(self) -> Dict:
        """
        Return a minimal SDV-compatible metadata dict.

        Usage::

            from sdv.metadata import SingleTableMetadata
            meta = SingleTableMetadata.load_from_dict(preprocessor.sdv_metadata)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first.")
        columns = {}
        for c in self._cont_cols_fitted:
            columns[c] = {"sdtype": "numerical"}
        for c in self._cat_cols_fitted:
            columns[c] = {"sdtype": "categorical"}
        return {"columns": columns, "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1"}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        if self.continuous_cols is not None:
            self._cont_cols_fitted = [c for c in self.continuous_cols if c in df.columns]
        else:
            self._cont_cols_fitted = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique() > self.cat_threshold
            ]

        if self.categorical_cols is not None:
            self._cat_cols_fitted = [c for c in self.categorical_cols if c in df.columns]
        else:
            self._cat_cols_fitted = [
                c for c in df.columns if c not in self._cont_cols_fitted
            ]

        if self.scale and self._cont_cols_fitted:
            self._scaler = StandardScaler()
            self._scaler.fit(df[self._cont_cols_fitted].fillna(
                df[self._cont_cols_fitted].median()
            ))

        self.is_fitted = True
        logger.info(
            "Tabular preprocessor — continuous: %s | categorical: %s",
            self._cont_cols_fitted, self._cat_cols_fitted,
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Impute continuous
        for c in self._cont_cols_fitted:
            if c in out.columns:
                out[c] = out[c].fillna(out[c].median())

        # Impute categorical
        for c in self._cat_cols_fitted:
            if c in out.columns:
                mode = out[c].mode()
                if not mode.empty:
                    out[c] = out[c].fillna(mode.iloc[0])

        # Optional scaling
        if self.scale and self._scaler and self._cont_cols_fitted:
            present = [c for c in self._cont_cols_fitted if c in out.columns]
            out[present] = self._scaler.transform(out[present])

        return out
