"""
synthclinic.evaluation.fidelity
----------------------------
Fidelity metrics: how statistically similar is the synthetic data to real data?

Metrics
-------
MMD (Maximum Mean Discrepancy)
    Kernel two-sample test.  MMD ≈ 0 → distributions are identical.
    Used for ECG (signal space) and tabular (feature space).

TSTR (Train on Synthetic, Test on Real)
    Train a downstream classifier on *synthetic* data, evaluate on *real* data.
    High TSTR accuracy → synthetic data is a useful substitute for real data.
    Reported as the ratio to TRTR (Train Real, Test Real) — the ideal baseline.

Feature-level KS + Chi-squared tests  (tabular)
    Per-column comparison of continuous (KS) and categorical (chi-squared)
    feature distributions.  Reported as mean p-value across features.

References
----------
Gretton, A. et al. (2012). "A Kernel Two-Sample Test." JMLR.
Yoon, J. et al. (2019). "Time-series Generative Adversarial Networks." NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy
# ---------------------------------------------------------------------------

def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """
    Unbiased MMD² with RBF kernel k(x, y) = exp(-γ||x-y||²).

    Parameters
    ----------
    X : (n, d) — real samples
    Y : (m, d) — synthetic samples
    gamma : RBF bandwidth.  ``None`` → median heuristic: γ = 1/(2·median(||xi-xj||²))

    Returns
    -------
    float — MMD² estimate.  Negative values (due to unbiased estimator) are
    clipped to 0.
    """
    X = X.reshape(len(X), -1).astype(np.float64)
    Y = Y.reshape(len(Y), -1).astype(np.float64)

    if gamma is None:
        # Median heuristic on a subsample
        sub = min(500, len(X))
        dists = np.sum((X[:sub, None] - X[None, :sub]) ** 2, axis=-1)
        median_sq = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_sq + 1e-8)

    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return float(max(0.0, mmd2))


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    sq_dists = (
        np.sum(A ** 2, axis=1, keepdims=True)
        + np.sum(B ** 2, axis=1)
        - 2 * A @ B.T
    )
    return np.exp(-gamma * np.clip(sq_dists, 0, None))


# ---------------------------------------------------------------------------
# Train on Synthetic, Test on Real (TSTR)
# ---------------------------------------------------------------------------

def tstr_score(
    real_X: np.ndarray,
    real_y: np.ndarray,
    synth_X: np.ndarray,
    synth_y: np.ndarray,
    classifier: str = "random_forest",
) -> Dict[str, float]:
    """
    Compute TSTR and TRTR scores for a downstream classification task.

    Parameters
    ----------
    real_X, real_y     : real train + test (split internally)
    synth_X, synth_y   : synthetic train data
    classifier         : ``"random_forest"`` or ``"logistic"``

    Returns
    -------
    dict with keys: ``"tstr"`` (F1), ``"trtr"`` (F1), ``"tstr_trtr_ratio"``
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    real_X = real_X.reshape(len(real_X), -1)
    synth_X = synth_X.reshape(len(synth_X), -1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        real_X, real_y, test_size=0.25, random_state=42, stratify=real_y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    synth_X_s = scaler.transform(synth_X)

    def _clf():
        if classifier == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42)
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # TRTR: train real, test real (upper bound)
    clf_trtr = _clf().fit(X_tr_s, y_tr)
    trtr = f1_score(y_te, clf_trtr.predict(X_te_s), average="macro", zero_division=0)

    # TSTR: train synthetic, test real
    clf_tstr = _clf().fit(synth_X_s, synth_y)
    tstr = f1_score(y_te, clf_tstr.predict(X_te_s), average="macro", zero_division=0)

    ratio = tstr / (trtr + 1e-8)
    logger.info("TSTR=%.4f  TRTR=%.4f  ratio=%.4f", tstr, trtr, ratio)
    return {"tstr": tstr, "trtr": trtr, "tstr_trtr_ratio": ratio}


# ---------------------------------------------------------------------------
# Per-feature distribution tests (tabular)
# ---------------------------------------------------------------------------

def feature_distribution_tests(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Per-feature KS test (continuous) and chi-squared test (categorical).

    Returns
    -------
    dict with:
        ``"mean_ks_statistic"`` : average KS statistic across continuous cols
        ``"mean_ks_pvalue"``    : average KS p-value  (> 0.05 = distributions match)
        ``"mean_chi2_pvalue"``  : average chi-squared p-value for categorical cols
        ``"n_continuous"``      : number of continuous columns tested
        ``"n_categorical"``     : number of categorical columns tested
    """
    if categorical_cols is None:
        categorical_cols = [
            c for c in real_df.columns
            if real_df[c].dtype == object or real_df[c].nunique() <= 15
        ]
    continuous_cols = [c for c in real_df.columns if c not in categorical_cols]

    ks_stats, ks_pvals, chi2_pvals = [], [], []

    for col in continuous_cols:
        if col not in synth_df.columns:
            continue
        stat, pval = ks_2samp(
            real_df[col].dropna().values,
            synth_df[col].dropna().values,
        )
        ks_stats.append(stat)
        ks_pvals.append(pval)

    for col in categorical_cols:
        if col not in synth_df.columns:
            continue
        cats = set(real_df[col].dropna()) | set(synth_df[col].dropna())
        real_counts = real_df[col].value_counts().reindex(cats, fill_value=0)
        synth_counts = synth_df[col].value_counts().reindex(cats, fill_value=0)
        try:
            _, pval, _, _ = chi2_contingency(
                np.stack([real_counts.values, synth_counts.values])
            )
            chi2_pvals.append(pval)
        except Exception:
            pass

    results = {
        "mean_ks_statistic": float(np.mean(ks_stats)) if ks_stats else float("nan"),
        "mean_ks_pvalue": float(np.mean(ks_pvals)) if ks_pvals else float("nan"),
        "mean_chi2_pvalue": float(np.mean(chi2_pvals)) if chi2_pvals else float("nan"),
        "n_continuous": len(ks_stats),
        "n_categorical": len(chi2_pvals),
    }
    logger.info("Feature tests: KS_stat=%.4f KS_p=%.4f chi2_p=%.4f",
                results["mean_ks_statistic"],
                results["mean_ks_pvalue"],
                results["mean_chi2_pvalue"])
    return results
