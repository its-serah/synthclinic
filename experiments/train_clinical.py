#!/usr/bin/env python
"""
experiments/train_clinical.py
-------------------------------
Fine-tune GPT-2 (+ optional LoRA) on MTSamples clinical transcriptions.

Usage
-----
    python experiments/train_clinical.py --config configs/clinical.yaml
    python experiments/train_clinical.py --config configs/clinical.yaml --lora
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from synthclinic.data.loaders.clinical_loader import ClinicalNotesLoader
from synthclinic.models.clinical_notes.clinical_generator import ClinicalNotesGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_clinical")


def main(args):
    cfg = OmegaConf.load(args.config)
    if args.lora:
        cfg.model.use_lora = True

    # ── Data ──────────────────────────────────────────────────────────
    specialties = cfg.data.get("specialties", None)
    loader = ClinicalNotesLoader(
        cache_dir=cfg.data.cache_dir,
        specialties=list(specialties) if specialties else None,
        min_length=cfg.data.min_length,
    )
    df = loader.load()
    logger.info("Loaded %d clinical notes", len(df))

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    generator = ClinicalNotesGenerator(config=model_cfg)

    # ── Train ─────────────────────────────────────────────────────────
    metrics = generator.train(df)
    logger.info("Training complete: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────
    generator.save(cfg.output.checkpoint_dir)
    logger.info("Checkpoint saved to %s", cfg.output.checkpoint_dir)

    # ── Generate ──────────────────────────────────────────────────────
    gen_cfg = cfg.generate
    prompt = gen_cfg.get("prompt", None)
    notes = generator.generate(
        n_samples=gen_cfg.n_samples,
        prompt=prompt,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
    )
    logger.info("Generated %d clinical notes", len(notes))

    out_path = Path(cfg.output.synthetic_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n---\n\n".join(notes))
    logger.info("Saved to %s", out_path)

    # ── Evaluate ──────────────────────────────────────────────────────
    if cfg.evaluation.run_text_quality:
        from synthclinic.evaluation.clinical import text_quality_metrics
        reference = df["transcription"].sample(min(50, len(df)), random_state=42).tolist()
        qual = text_quality_metrics(notes, reference=reference)
        logger.info("Text quality: %s", qual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenMed clinical notes generator")
    parser.add_argument("--config", default="configs/clinical.yaml")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    main(parser.parse_args())
