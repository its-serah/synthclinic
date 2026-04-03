"""
synthclinic.evaluation.privacy
---------------------------
Privacy metrics: can synthetic samples be traced back to real patients?

Metrics
-------
DCR (Distance to Closest Record)
    For each synthetic record, compute its nearest-neighbour distance to the
    real training set.  Low DCR → risk of memorisation / re-identification.
    Compare DCR(synthetic, train) vs DCR(holdout, train) — if similar,
    synthetic data is no more revealing than held-out real data.

NNDR (Nearest-Neighbour Distance Ratio)
    Ratio of the nearest- to second-nearest-neighbour distance for synthetic
    records.  NNDR close to 1 → the synthetic point sits "between" real
    points, not on top of them.

Membership Inference Attack (MIA) — simple attribute inference baseline
    Train a logistic regressor on (synthetic, real) data with label 1=real.
    If MIA accuracy ≈ 0.5, the synthetic data is indistinguishable.
    Accuracy > 0.7 is a warning sign.

References
----------
Jordon, J. et al. (2022). "Synthetic Data — what, why and how?" arXiv:2205.03257
Hayes, J. et al. (2019). "LOGAN: Membership Inference Attacks Against GANs." arXiv.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCR — Distance to Closest Record
# ---------------------------------------------------------------------------

def dcr_score(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    real_holdout: Optional[np.ndarray] = None,
    subsample: int = 2000,
) -> Dict[str, float]:
    """
    Compute DCR statistics for synthetic vs real data.

    Parameters
    ----------
    real_train  : (N, d)  training data the generator was fitted on
    synthetic   : (M, d)  generated samples
    real_holdout: (K, d)  held-out real data (never seen by the generator).
                  If provided, compares DCR_synth vs DCR_holdout.
    subsample   : cap N and M before computing to keep O(N·M) tractable.

    Returns
    -------
    dict with:
        ``"dcr_mean"``       : mean distance from synthetic to nearest real train record
        ``"dcr_5th_pct"``    : 5th percentile (worst-case proximity)
        ``"dcr_holdout_mean"``: DCR from holdout to train (if provided)
        ``"privacy_gap"``    : dcr_mean / dcr_holdout_mean  (> 1 is safer)
    """
    real_train = real_train.reshape(len(real_train), -1).astype(np.float64)
    synthetic = synthetic.reshape(len(synthetic), -1).astype(np.float64)

    # Subsample for efficiency
    if len(real_train) > subsample:
        idx = np.random.choice(len(real_train), subsample, replace=False)
        real_train = real_train[idx]
    if len(synthetic) > subsample:
        idx = np.random.choice(len(synthetic), subsample, replace=False)
        synthetic = synthetic[idx]

    scaler = StandardScaler()
    real_train_s = scaler.fit_transform(real_train)
    synthetic_s = scaler.transform(synthetic)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(real_train_s)

    synth_dists, _ = nn.kneighbors(synthetic_s)
    synth_dists = synth_dists.flatten()

    results: Dict[str, float] = {
        "dcr_mean": float(synth_dists.mean()),
        "dcr_5th_pct": float(np.percentile(synth_dists, 5)),
    }

    if real_holdout is not None:
        holdout_s = scaler.transform(
            real_holdout.reshape(len(real_holdout), -1).astype(np.float64)[:subsample]
        )
        hold_dists, _ = nn.kneighbors(holdout_s)
        dcr_holdout = float(hold_dists.mean())
        results["dcr_holdout_mean"] = dcr_holdout
        results["privacy_gap"] = float(results["dcr_mean"] / (dcr_holdout + 1e-8))

    logger.info(
        "DCR: mean=%.4f  5th%%=%.4f  gap=%.2fx",
        results["dcr_mean"],
        results["dcr_5th_pct"],
        results.get("privacy_gap", float("nan")),
    )
    return results


# ---------------------------------------------------------------------------
# NNDR — Nearest-Neighbour Distance Ratio
# ---------------------------------------------------------------------------

def nndr_score(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    subsample: int = 2000,
) -> Dict[str, float]:
    """
    Nearest-Neighbour Distance Ratio for synthetic records.

    NNDR_i = d(s_i, r_1) / d(s_i, r_2)  where r_1, r_2 are the two
    nearest real training records to synthetic point s_i.

    NNDR close to 1.0 → s_i sits between real points (diverse, not memorised).
    NNDR close to 0.0 → s_i is very close to one specific real point (risk).

    Returns
    -------
    dict with ``"nndr_mean"``, ``"nndr_median"``, ``"nndr_5th_pct"``
    """
    real_s = real_train.reshape(len(real_train), -1).astype(np.float64)
    synth_s = synthetic.reshape(len(synthetic), -1).astype(np.float64)

    if len(real_s) > subsample:
        real_s = real_s[np.random.choice(len(real_s), subsample, replace=False)]
    if len(synth_s) > subsample:
        synth_s = synth_s[np.random.choice(len(synth_s), subsample, replace=False)]

    scaler = StandardScaler()
    real_s = scaler.fit_transform(real_s)
    synth_s = scaler.transform(synth_s)

    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn.fit(real_s)
    dists, _ = nn.kneighbors(synth_s)  # (M, 2)

    nndr = dists[:, 0] / (dists[:, 1] + 1e-8)
    results = {
        "nndr_mean": float(nndr.mean()),
        "nndr_median": float(np.median(nndr)),
        "nndr_5th_pct": float(np.percentile(nndr, 5)),
    }
    logger.info("NNDR: mean=%.4f  median=%.4f", results["nndr_mean"], results["nndr_median"])
    return results


# ---------------------------------------------------------------------------
# Membership Inference Attack (simple baseline)
# ---------------------------------------------------------------------------

def membership_inference_attack(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    subsample: int = 1000,
) -> Dict[str, float]:
    """
    Simple MIA: train a classifier to distinguish real from synthetic data.

    If accuracy ≈ 0.5, synthetic data is indistinguishable from real.
    Accuracy > 0.7 suggests memorisation or poor diversity.

    Parameters
    ----------
    real_train : real training samples (label=1)
    synthetic  : generated samples (label=0)

    Returns
    -------
    dict with ``"mia_accuracy"``, ``"mia_advantage"`` (accuracy - 0.5)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = min(subsample, len(real_train), len(synthetic))
    real_sub = real_train[np.random.choice(len(real_train), n, replace=False)]
    synth_sub = synthetic[np.random.choice(len(synthetic), n, replace=False)]

    X = np.concatenate([real_sub, synth_sub]).reshape(2 * n, -1).astype(np.float64)
    y = np.array([1] * n + [0] * n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    results = {
        "mia_accuracy": float(acc),
        "mia_advantage": float(acc - 0.5),
    }
    logger.info("MIA: accuracy=%.4f  advantage=%.4f", acc, acc - 0.5)
    return results
