"""
synthclinic.data.loaders.ecg_loader
--------------------------------
Loads the PhysioNet MIT-BIH Arrhythmia Database (mitdb).

Records are downloaded automatically via ``wfdb`` on first use and cached
locally.  The dataset is open-access: https://physionet.org/content/mitdb/

Each 30-minute recording (360 Hz, 2 leads) is segmented into fixed-length
windows. Lead MLII (index 0) is used by default as it provides the clearest
QRS morphology for generation quality assessment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# All 48 MIT-BIH record IDs
_ALL_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


class ECGLoader:
    """
    Loads MIT-BIH ECG signals as fixed-length, normalised numpy arrays.

    Parameters
    ----------
    data_dir:
        Local cache directory for downloaded ``.dat`` / ``.hea`` files.
    segment_length:
        Samples per segment. Default 256 ≈ 0.71 s at 360 Hz.
        Common thesis choices: 128, 256, 512, 1024.
    lead:
        Lead index to extract — 0 = MLII, 1 = V-lead (record-dependent).
    records:
        Subset of record IDs to use.  ``None`` → all 48 records.
    max_segments_per_record:
        Cap per record to avoid memory issues; set ``None`` for no cap.
    """

    _DB = "mitdb"

    def __init__(
        self,
        data_dir: str = "data/raw/ecg",
        segment_length: int = 256,
        lead: int = 0,
        records: Optional[List[str]] = None,
        max_segments_per_record: int = 300,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.segment_length = segment_length
        self.lead = lead
        self.records = records if records is not None else _ALL_RECORDS
        self.max_per_record = max_segments_per_record

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download (if needed) and segment all requested records.

        Returns
        -------
        signals : np.ndarray, shape ``(N, segment_length)``, dtype float32
            Each row is a min-max-normalised ECG segment in [-1, 1].
        record_ids : np.ndarray, shape ``(N,)``, dtype str
            Source record ID for each segment (useful for stratified splits).
        """
        import wfdb  # lazy import — only required for ECG modality

        all_signals, all_ids = [], []

        for rec_id in self.records:
            try:
                segs, ids = self._load_record(rec_id, wfdb)
                all_signals.append(segs)
                all_ids.append(ids)
                logger.debug("Record %s → %d segments", rec_id, len(segs))
            except Exception as exc:
                logger.warning("Skipping record %s: %s", rec_id, exc)

        if not all_signals:
            raise RuntimeError(
                "No ECG records could be loaded. "
                "Check that 'wfdb' is installed and the network is reachable."
            )

        X = np.concatenate(all_signals, axis=0).astype(np.float32)
        ids = np.concatenate(all_ids, axis=0)
        logger.info("ECG loader: %d segments from %d records", len(X), len(self.records))
        return X, ids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_record(
        self, rec_id: str, wfdb
    ) -> Tuple[np.ndarray, np.ndarray]:
        record_path = str(self.data_dir / rec_id)

        # Download if not cached
        if not (self.data_dir / f"{rec_id}.dat").exists():
            logger.info("Downloading MIT-BIH record %s …", rec_id)
            wfdb.dl_database(self._DB, str(self.data_dir), records=[rec_id])

        record = wfdb.rdrecord(record_path)
        signal: np.ndarray = record.p_signal[:, self.lead]  # (total_samples,)

        # Segment into non-overlapping windows
        n_full = len(signal) // self.segment_length
        if self.max_per_record:
            n_full = min(n_full, self.max_per_record)

        segments = np.stack(
            [signal[i * self.segment_length: (i + 1) * self.segment_length]
             for i in range(n_full)]
        )  # (n_full, segment_length)

        # Per-segment min-max normalisation to [-1, 1]
        mins = segments.min(axis=1, keepdims=True)
        maxs = segments.max(axis=1, keepdims=True)
        denom = np.where((maxs - mins) > 1e-8, maxs - mins, 1.0)
        segments = 2.0 * (segments - mins) / denom - 1.0

        ids = np.array([rec_id] * n_full)
        return segments, ids
