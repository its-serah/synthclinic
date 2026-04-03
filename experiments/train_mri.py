#!/usr/bin/env python
"""
experiments/train_mri.py
--------------------------
Train the MRI generator (VAE + latent DDPM).

Usage
-----
    python experiments/train_mri.py --config configs/mri.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from omegaconf import OmegaConf

from synthclinic.data.loaders.mri_loader import MRILoader
from synthclinic.models.mri.mri_generator import MRIGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_mri")


def main(args):
    cfg = OmegaConf.load(args.config)

    # ── Data ──────────────────────────────────────────────────────────
    loader = MRILoader(
        data_dir=cfg.data.data_dir,
        slice_axis=cfg.data.slice_axis,
        target_size=(cfg.model.img_size, cfg.model.img_size),
        max_volumes=cfg.data.max_volumes,
        slice_range=tuple(cfg.data.slice_range),
    )
    slices = loader.load()
    logger.info("MRI data shape: %s", slices.shape)

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    generator = MRIGenerator(config=model_cfg)

    # ── Train ─────────────────────────────────────────────────────────
    metrics = generator.train(slices)
    logger.info("Training complete: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────
    generator.save(cfg.output.checkpoint_dir)
    logger.info("Checkpoint saved to %s", cfg.output.checkpoint_dir)

    # ── Generate ──────────────────────────────────────────────────────
    synthetic = generator.generate(
        n_samples=cfg.generate.n_samples,
        sampling_steps=cfg.generate.sampling_steps,
    )
    out_path = Path(cfg.output.synthetic_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, synthetic)
    logger.info("Saved %d synthetic MRI slices to %s", len(synthetic), out_path)

    # ── Evaluate ──────────────────────────────────────────────────────
    if cfg.evaluation.run_mmd:
        from synthclinic.evaluation.fidelity import mmd_rbf
        sub = min(200, len(slices), len(synthetic))
        mmd = mmd_rbf(
            slices[:sub].reshape(sub, -1),
            synthetic[:sub].reshape(sub, -1),
        )
        logger.info("MMD²=%.6f", mmd)

    if cfg.evaluation.run_privacy:
        from synthclinic.evaluation.privacy import dcr_score
        dcr = dcr_score(
            slices.reshape(len(slices), -1),
            synthetic.reshape(len(synthetic), -1),
        )
        logger.info("Privacy DCR: %s", dcr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenMed MRI generator")
    parser.add_argument("--config", default="configs/mri.yaml")
    main(parser.parse_args())
