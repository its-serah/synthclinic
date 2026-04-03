"""
GenMed: Generative AI framework for synthetic medical data synthesis.

Modalities
----------
- Tabular (lab results)  : CTGAN / TVAE via SDV
- ECG signals            : TimeGAN
- Clinical notes         : GPT-2 + LoRA fine-tuning
- MRI images             : VAE + DDPM (latent diffusion)
"""

__version__ = "0.1.0"

from synthclinic.base import BaseGenerator

__all__ = ["BaseGenerator", "__version__"]
