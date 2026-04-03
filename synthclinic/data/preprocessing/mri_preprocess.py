"""
synthclinic.data.preprocessing.mri_preprocess
------------------------------------------
MRI slice preprocessing: augmentation transforms and PyTorch Dataset wrapper.

Input:  np.ndarray of shape (N, 1, H, W) with values in [0, 1]
Output: torch.Tensor of shape (1, H, W) per sample, values in [-1, 1]

Scaling to [-1, 1] is required for DDPM / VAE training because:
  - The VAE decoder uses tanh activation (output in [-1, 1])
  - The DDPM noise schedule expects data centred around 0
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def build_transforms(
    img_size: int = 64,
    augment: bool = True,
) -> transforms.Compose:
    """
    Build a torchvision transform pipeline for MRI slices.

    Parameters
    ----------
    img_size:
        Final spatial size (assumes square images).
    augment:
        If ``True``, apply random horizontal flip and small rotation.
        Disable for validation / test sets.

    Returns
    -------
    ``transforms.Compose`` instance.  Input: PIL Image or Tensor in [0, 1].
    Output: Tensor (1, H, W) in [-1, 1].
    """
    ops = []

    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]

    # Ensure correct size
    ops += [
        transforms.Resize((img_size, img_size), antialias=True),
        # Scale from [0, 1] → [-1, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    return transforms.Compose(ops)


class MRISliceDataset(Dataset):
    """
    PyTorch Dataset for MRI slices.

    Parameters
    ----------
    slices:
        np.ndarray of shape ``(N, 1, H, W)``, values in [0, 1].
    img_size:
        Resize target (square).  Should match the VAE / DDPM input size.
    augment:
        Apply random flips / rotations during training.
    """

    def __init__(
        self,
        slices: np.ndarray,
        img_size: int = 64,
        augment: bool = True,
    ) -> None:
        # Convert to float32 tensor in [0, 1]
        self._data = torch.from_numpy(slices.astype(np.float32))  # (N, 1, H, W)
        self._transform = build_transforms(img_size=img_size, augment=augment)
        logger.info("MRISliceDataset: %d slices, size %d×%d", len(slices), img_size, img_size)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._data[idx]  # (1, H, W) in [0, 1]
        return self._transform(img)  # (1, H, W) in [-1, 1]


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse the [-1, 1] normalisation back to [0, 1].

    Use before saving generated images or computing pixel-level metrics.
    """
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)
