"""
synthclinic.data.preprocessing.ecg_preprocess
------------------------------------------
ECG preprocessing utilities: detrending, z-score normalisation,
sliding-window augmentation, and a PyTorch Dataset wrapper for TimeGAN.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def detrend_segments(signals: np.ndarray) -> np.ndarray:
    """
    Remove linear trend from each ECG segment (baseline wander removal).

    Parameters
    ----------
    signals : (N, T) float array

    Returns
    -------
    (N, T) float array — detrended
    """
    from scipy.signal import detrend
    return detrend(signals, axis=1).astype(np.float32)


def bandpass_filter(
    signals: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Removes baseline wander (< 0.5 Hz) and high-frequency noise (> 40 Hz).
    Standard clinical recommendation for diagnostic ECG.

    Parameters
    ----------
    signals : (N, T) float array
    fs      : sampling frequency in Hz (MIT-BIH = 360 Hz)
    lowcut  : high-pass cutoff (Hz)
    highcut : low-pass cutoff (Hz)
    """
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    b, a = butter(
        N=4,
        Wn=[lowcut / nyq, highcut / nyq],
        btype="bandpass",
    )
    filtered = filtfilt(b, a, signals, axis=1)
    return filtered.astype(np.float32)


def zscore_normalise(signals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-segment z-score normalisation.  Preferred over min-max when
    feeding into TimeGAN because it preserves amplitude relationships.
    """
    mu = signals.mean(axis=1, keepdims=True)
    sigma = signals.std(axis=1, keepdims=True) + eps
    return ((signals - mu) / sigma).astype(np.float32)


def sliding_window_augment(
    signals: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Re-segment a (N, T) array using overlapping windows of *window_size*.
    Increases dataset size without introducing inter-subject contamination.

    Parameters
    ----------
    signals     : (N, T) float array — original fixed-length segments
    window_size : desired output segment length
    stride      : step between windows (stride < window_size → overlap)

    Returns
    -------
    (M, window_size) float array  where M ≥ N
    """
    N, T = signals.shape
    windows = []
    for seg in signals:
        starts = range(0, T - window_size + 1, stride)
        windows.extend(seg[s: s + window_size] for s in starts)
    return np.stack(windows).astype(np.float32)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for TimeGAN training.

    Returns a (T, 1) tensor per item — TimeGAN expects
    shape (batch, seq_len, features).

    Parameters
    ----------
    signals : (N, T) float32 array
    augment : if True, apply bandpass filter + z-score normalisation
    """

    def __init__(self, signals: np.ndarray, augment: bool = True) -> None:
        if augment:
            signals = bandpass_filter(signals)
            signals = zscore_normalise(signals)
        # Shape: (N, T) → store as (N, T)
        self.data = torch.from_numpy(signals)  # (N, T)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # TimeGAN input shape: (seq_len, input_dim=1)
        return self.data[idx].unsqueeze(-1)  # (T, 1)
