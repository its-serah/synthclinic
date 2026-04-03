#!/usr/bin/env python
"""
experiments/train_ecg.py
--------------------------
Train the ECG generator (TimeGAN) on MIT-BIH Arrhythmia Database.

Usage
-----
    python experiments/train_ecg.py --config configs/ecg.yaml
    python experiments/train_ecg.py --config configs/ecg.yaml --records 100 101 102
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from omegaconf import OmegaConf

from synthclinic.data.loaders.ecg_loader import ECGLoader
from synthclinic.data.preprocessing.ecg_preprocess import ECGDataset, bandpass_filter, zscore_normalise
from synthclinic.models.ecg.ecg_generator import ECGGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ecg")


def main(args):
    cfg = OmegaConf.load(args.config)

    # ── Data ──────────────────────────────────────────────────────────
    records = args.records or None
    loader = ECGLoader(
        data_dir=cfg.data.data_dir,
        segment_length=cfg.data.segment_length,
        lead=cfg.data.lead,
        records=records,
        max_segments_per_record=cfg.data.max_segments_per_record,
    )
    signals, _ = loader.load()
    logger.info("Loaded %d ECG segments, shape %s", len(signals), signals.shape)

    if cfg.preprocessing.augment:
        signals = bandpass_filter(
            signals,
            lowcut=cfg.preprocessing.bandpass_low,
            highcut=cfg.preprocessing.bandpass_high,
        )
        signals = zscore_normalise(signals)

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    generator = ECGGenerator(config=model_cfg)

    # ── Train ─────────────────────────────────────────────────────────
    metrics = generator.train(signals)
    logger.info("Training complete: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────
    generator.save(cfg.output.checkpoint_dir)
    logger.info("Checkpoint saved to %s", cfg.output.checkpoint_dir)

    # ── Generate ──────────────────────────────────────────────────────
    synthetic = generator.generate(cfg.generate.n_samples)
    out_path = Path(cfg.output.synthetic_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, synthetic)
    logger.info("Saved %d synthetic ECG segments to %s", len(synthetic), out_path)

    # ── Evaluate ──────────────────────────────────────────────────────
    if cfg.evaluation.run_morphology:
        from synthclinic.evaluation.clinical import ecg_morphology_metrics
        morph = ecg_morphology_metrics(synthetic)
        logger.info("ECG morphology: %s", morph)

    if cfg.evaluation.run_mmd:
        from synthclinic.evaluation.fidelity import mmd_rbf
        sub = min(500, len(signals), len(synthetic))
        mmd = mmd_rbf(
            signals[:sub].reshape(sub, -1),
            synthetic[:sub].reshape(sub, -1),
        )
        logger.info("MMD²=%.6f", mmd)

    if cfg.evaluation.run_privacy:
        from synthclinic.evaluation.privacy import dcr_score
        dcr = dcr_score(signals.reshape(len(signals), -1),
                        synthetic.reshape(len(synthetic), -1))
        logger.info("Privacy DCR: %s", dcr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenMed ECG generator")
    parser.add_argument("--config", default="configs/ecg.yaml")
    parser.add_argument("--records", nargs="+", default=None,
                        help="Restrict to specific MIT-BIH record IDs")
    main(parser.parse_args())
