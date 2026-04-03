#!/usr/bin/env python
"""
experiments/evaluate_all.py
-----------------------------
Load all four trained generators, run the full evaluation suite,
and generate a portfolio of VirtualPatient digital twins.

Requires all four modalities to have been trained first:
    python experiments/train_tabular.py
    python experiments/train_ecg.py
    python experiments/train_clinical.py
    python experiments/train_mri.py

Usage
-----
    python experiments/evaluate_all.py
    python experiments/evaluate_all.py --n_patients 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_all")

CHECKPOINTS = {
    "tabular": "checkpoints/tabular",
    "ecg": "checkpoints/ecg",
    "clinical": "checkpoints/clinical",
    "mri": "checkpoints/mri",
}


def load_generators():
    """Load all generators from their checkpoints (skip missing ones with a warning)."""
    generators = {}

    try:
        from synthclinic.models.tabular.ctgan_generator import TabularGenerator
        gen = TabularGenerator()
        gen.load(CHECKPOINTS["tabular"])
        generators["tabular"] = gen
        logger.info("✓ Tabular generator loaded")
    except Exception as e:
        logger.warning("Tabular generator not available: %s", e)

    try:
        from synthclinic.models.ecg.ecg_generator import ECGGenerator
        gen = ECGGenerator()
        gen.load(CHECKPOINTS["ecg"])
        generators["ecg"] = gen
        logger.info("✓ ECG generator loaded")
    except Exception as e:
        logger.warning("ECG generator not available: %s", e)

    try:
        from synthclinic.models.clinical_notes.clinical_generator import ClinicalNotesGenerator
        gen = ClinicalNotesGenerator()
        gen.load(CHECKPOINTS["clinical"])
        generators["clinical"] = gen
        logger.info("✓ Clinical notes generator loaded")
    except Exception as e:
        logger.warning("Clinical generator not available: %s", e)

    try:
        from synthclinic.models.mri.mri_generator import MRIGenerator
        gen = MRIGenerator()
        gen.load(CHECKPOINTS["mri"])
        generators["mri"] = gen
        logger.info("✓ MRI generator loaded")
    except Exception as e:
        logger.warning("MRI generator not available: %s", e)

    return generators


def run_cross_modal_evaluation(generators: dict) -> dict:
    """Run evaluation on loaded generators and return a results dict."""
    import numpy as np
    results = {}

    # Tabular
    if "tabular" in generators:
        from synthclinic.data.loaders.tabular_loader import TabularLoader
        from synthclinic.data.preprocessing.tabular_preprocess import TabularPreprocessor
        from synthclinic.evaluation.fidelity import feature_distribution_tests, tstr_score
        from synthclinic.evaluation.privacy import dcr_score

        loader = TabularLoader()
        preprocessor = TabularPreprocessor()
        real_df = preprocessor.fit_transform(loader.load())
        synth_df = generators["tabular"].generate(len(real_df))

        tab_results = {}
        tab_results.update(feature_distribution_tests(real_df, synth_df))
        target = "target" if "target" in real_df.columns else real_df.columns[-1]
        tab_results.update(tstr_score(
            real_df.drop(columns=[target]).values, real_df[target].values,
            synth_df.drop(columns=[target], errors="ignore").values,
            synth_df[target].values if target in synth_df.columns else real_df[target].values,
        ))
        tab_results.update(dcr_score(
            real_df.select_dtypes("number").values,
            synth_df.select_dtypes("number").values,
        ))
        results["tabular"] = tab_results
        logger.info("Tabular evaluation: %s", tab_results)

    # ECG
    if "ecg" in generators:
        from synthclinic.data.loaders.ecg_loader import ECGLoader
        from synthclinic.evaluation.fidelity import mmd_rbf
        from synthclinic.evaluation.clinical import ecg_morphology_metrics
        from synthclinic.evaluation.privacy import dcr_score

        real_signals, _ = ECGLoader(max_segments_per_record=100).load()
        synth_signals = generators["ecg"].generate(500)

        ecg_results = {}
        ecg_results["mmd2"] = mmd_rbf(
            real_signals[:300].reshape(300, -1),
            synth_signals[:300].reshape(300, -1),
        )
        ecg_results.update(ecg_morphology_metrics(synth_signals[:200]))
        ecg_results.update(dcr_score(
            real_signals.reshape(len(real_signals), -1),
            synth_signals.reshape(len(synth_signals), -1),
        ))
        results["ecg"] = ecg_results
        logger.info("ECG evaluation: %s", ecg_results)

    # Save results
    out_path = Path("data/processed/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation results saved to %s", out_path)

    return results


def generate_digital_twins(generators: dict, n_patients: int = 5) -> None:
    """Generate and save VirtualPatient digital twins."""
    from synthclinic.digital_twin.patient import DigitalTwinFactory

    factory = DigitalTwinFactory(
        tabular_generator=generators.get("tabular"),
        ecg_generator=generators.get("ecg"),
        clinical_generator=generators.get("clinical"),
        mri_generator=generators.get("mri"),
    )

    patients = factory.generate(n_patients)
    for patient in patients:
        save_path = Path("data/processed/patients") / patient.patient_id
        patient.save(save_path)
        patient.summary()

    logger.info("Generated and saved %d virtual patients", len(patients))


def main(args):
    generators = load_generators()
    if not generators:
        logger.error(
            "No trained generators found. Run the training scripts first:\n"
            "  python experiments/train_tabular.py\n"
            "  python experiments/train_ecg.py\n"
            "  python experiments/train_clinical.py\n"
            "  python experiments/train_mri.py"
        )
        return

    run_cross_modal_evaluation(generators)
    generate_digital_twins(generators, n_patients=args.n_patients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all GenMed generators and generate digital twins")
    parser.add_argument("--n_patients", type=int, default=5,
                        help="Number of virtual patients to generate")
    main(parser.parse_args())
