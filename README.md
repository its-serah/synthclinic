# SynthClinic: Generative AI for Medical Data Synthesis

A modular research framework for generating realistic **synthetic medical data** across four clinical modalities, without reproducing real patients.

> Developed as part of a Master's research thesis, inspired by the SynthClinic PhD project at the University of Technology of Troyes (UTT).

---

## Modalities

| Modality | Model | Open Dataset |
|---|---|---|
| **ECG signals** | TimeGAN (Yoon et al., 2019) | PhysioNet MIT-BIH Arrhythmia (auto-download via `wfdb`) |
| **MRI images** | VAE + DDPM (latent diffusion) | IXI Brain Dataset (manual download) |
| **Clinical notes** | GPT-2 fine-tuning + LoRA | MTSamples medical transcriptions (auto-download) |
| **Lab results** | CTGAN / TVAE (SDV) | UCI Heart Disease + PIMA Diabetes (auto-download) |

All datasets are **100% open-access** — no credentialed access required.

---

## Installation

```bash
git clone <repo>
cd synthclinic
pip install -e .
```

### MRI dataset (optional — required for MRI generator)

1. Download `IXI-T1.tar` from https://brain-development.org/ixi-dataset/
2. Extract to `data/raw/mri/IXI/`

Without the IXI dataset, `MRILoader` will generate synthetic Gaussian phantoms for pipeline testing.

---

## Quick Start

### Train a tabular generator

```bash
python experiments/train_tabular.py --config configs/tabular.yaml
```

### Train an ECG generator (TimeGAN)

```bash
python experiments/train_ecg.py --config configs/ecg.yaml
```

### Train a clinical notes generator (GPT-2 + LoRA)

```bash
python experiments/train_clinical.py --config configs/clinical.yaml
```

### Train an MRI generator (VAE + DDPM)

```bash
python experiments/train_mri.py --config configs/mri.yaml
```

### Run full evaluation

```bash
python experiments/evaluate_all.py
```

---

## Project Structure

```
synthclinic/
├── synthclinic/                    # Core library
│   ├── base.py                # BaseGenerator ABC
│   ├── data/
│   │   ├── loaders/           # Per-modality data loaders
│   │   └── preprocessing/     # Normalisation, windowing, tokenisation
│   ├── models/
│   │   ├── tabular/           # CTGAN / TVAE wrapper
│   │   ├── ecg/               # TimeGAN
│   │   ├── clinical_notes/    # GPT-2 + LoRA fine-tuner
│   │   └── mri/               # VAE + DDPM
│   ├── evaluation/
│   │   ├── fidelity.py        # FID, MMD, TSTR
│   │   ├── privacy.py         # DCR, membership inference
│   │   └── clinical.py        # Domain-specific metrics
│   └── digital_twin/
│       └── patient.py         # Multimodal virtual patient object
├── configs/                   # YAML experiment configs
├── experiments/               # Training and evaluation entry points
├── notebooks/                 # Analysis and visualisation
└── data/                      # Raw and processed datasets (gitignored)
```

---

## Evaluation Framework

### Fidelity
- **FID** (Fréchet Inception Distance) for MRI images
- **MMD** (Maximum Mean Discrepancy) for ECG and tabular
- **TSTR** (Train on Synthetic, Test on Real) — downstream utility

### Privacy
- **DCR** (Distance to Closest Record) — non-traceability
- **Membership Inference Attack** — worst-case privacy bound

### Clinical validity
- ECG: morphology metrics (R-peak detection rate, QRS duration distribution)
- Clinical notes: BLEU/ROUGE vs reference corpus, perplexity
- Tabular: per-feature statistical tests (KS, chi-squared)

---

## Digital Twin

The `Patient` class in `synthclinic/digital_twin/patient.py` composes outputs from all four generators into a single structured object representing a *virtual patient* with correlated multimodal data.

```python
from synthclinic.digital_twin.patient import VirtualPatient, DigitalTwinFactory

factory = DigitalTwinFactory(
    tabular_generator=tabular_gen,
    ecg_generator=ecg_gen,
    clinical_generator=clinical_gen,
    mri_generator=mri_gen,
)
patient = factory.generate()
patient.summary()
```

---

## References

- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). **Time-series Generative Adversarial Networks**. NeurIPS.
- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. NeurIPS.
- Kingma, D.P. & Welling, M. (2014). **Auto-Encoding Variational Bayes**. ICLR.
- Xu, L. et al. (2019). **Modeling Tabular Data using Conditional GAN**. NeurIPS.
- Hu, E.J. et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR.
