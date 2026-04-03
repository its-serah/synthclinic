#!/usr/bin/env python
"""
experiments/train_tabular.py
------------------------------
Train the tabular generator (CTGAN or TVAE) on open medical datasets.

Usage
-----
    python experiments/train_tabular.py --config configs/tabular.yaml
    python experiments/train_tabular.py --config configs/tabular.yaml --model tvae
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from synthclinic.data.loaders.tabular_loader import TabularLoader
from synthclinic.data.preprocessing.tabular_preprocess import TabularPreprocessor
from synthclinic.models.tabular.ctgan_generator import TabularGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tabular")


def main(args):
    cfg = OmegaConf.load(args.config)
    if args.model:
        cfg.model.type = args.model

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading %s dataset …", cfg.data.dataset)
    loader = TabularLoader(
        dataset=cfg.data.dataset,
        cache_dir=cfg.data.cache_dir,
    )
    df = loader.load()

    preprocessor = TabularPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    logger.info("Clean dataset: %d rows × %d cols", len(df_clean), len(df_clean.columns))

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["model"] = model_cfg.pop("type", "ctgan")
    generator = TabularGenerator(config=model_cfg)

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Training %s …", model_cfg["model"].upper())
    metrics = generator.train(df_clean)
    logger.info("Training complete: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────
    ckpt = cfg.output.checkpoint_dir
    generator.save(ckpt)
    logger.info("Model saved to %s", ckpt)

    # ── Generate ──────────────────────────────────────────────────────
    n = cfg.generate.n_samples
    synthetic = generator.generate(n)
    logger.info("Generated %d synthetic rows", len(synthetic))

    out_path = Path(cfg.output.synthetic_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(out_path, index=False)
    logger.info("Synthetic data saved to %s", out_path)

    # ── Evaluate ──────────────────────────────────────────────────────
    if cfg.evaluation.run_feature_tests:
        from synthclinic.evaluation.fidelity import feature_distribution_tests
        results = feature_distribution_tests(df_clean, synthetic)
        logger.info("Feature tests: %s", results)

    if cfg.evaluation.run_tstr:
        target_col = "target" if "target" in df_clean.columns else df_clean.columns[-1]
        from synthclinic.evaluation.fidelity import tstr_score
        import numpy as np
        real_X = df_clean.drop(columns=[target_col]).values
        real_y = df_clean[target_col].values
        synth_X = synthetic.drop(columns=[target_col], errors="ignore").values
        synth_y = synthetic[target_col].values if target_col in synthetic.columns else real_y[:len(synth_X)]
        tstr = tstr_score(real_X, real_y, synth_X, synth_y)
        logger.info("TSTR: %s", tstr)

    if cfg.evaluation.run_privacy:
        from synthclinic.evaluation.privacy import dcr_score, membership_inference_attack
        import numpy as np
        real_arr = df_clean.select_dtypes("number").values
        synth_arr = synthetic.select_dtypes("number").values
        dcr = dcr_score(real_arr, synth_arr)
        mia = membership_inference_attack(real_arr, synth_arr)
        logger.info("Privacy — DCR: %s", dcr)
        logger.info("Privacy — MIA: %s", mia)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenMed tabular generator")
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--model", default=None, help="Override model type (ctgan|tvae)")
    main(parser.parse_args())
