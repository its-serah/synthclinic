"""
synthclinic.data.loaders.mri_loader
--------------------------------
Loads 2D axial MRI slices from NIfTI volumes (IXI Brain Dataset format).

IXI Dataset (open-access, no registration):
  https://brain-development.org/ixi-dataset/

Download IXI-T1.tar and extract to ``data/raw/mri/IXI/``.

If no NIfTI files are found, ``MRILoader.load()`` returns synthetic Gaussian
blob phantoms so the full pipeline can be tested without the dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_IXI_URL = "https://brain-development.org/ixi-dataset/"
_SETUP_MSG = (
    "\nTo use real MRI data:\n"
    f"  1. Download IXI-T1.tar from {_IXI_URL}\n"
    "  2. Extract to data/raw/mri/IXI/\n"
    "Continuing with synthetic Gaussian phantom data for now.\n"
)


class MRILoader:
    """
    Loads 2D axial slices from NIfTI brain MRI volumes.

    Parameters
    ----------
    data_dir:
        Directory containing ``.nii`` or ``.nii.gz`` files.
    slice_axis:
        Axis to slice along: 0 = sagittal, 1 = coronal, 2 = axial (default).
    target_size:
        (H, W) to resize each slice to.  64×64 is recommended for training
        speed; use 128×128 for higher-quality generation.
    max_volumes:
        Maximum number of NIfTI volumes to load.  ``None`` = all found.
    slice_range:
        Fraction (start, end) of depth slices to keep per volume.
        Default (0.25, 0.75) discards blank skull-base / crown slices.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/mri",
        slice_axis: int = 2,
        target_size: Tuple[int, int] = (64, 64),
        max_volumes: Optional[int] = 20,
        slice_range: Tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.slice_axis = slice_axis
        self.target_size = target_size
        self.max_volumes = max_volumes
        self.slice_range = slice_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> np.ndarray:
        """
        Return MRI slices as a float32 array.

        Returns
        -------
        np.ndarray, shape ``(N, 1, H, W)``, dtype float32, values in [0, 1]
        """
        nii_files = sorted(
            list(self.data_dir.rglob("*.nii.gz"))
            + list(self.data_dir.rglob("*.nii"))
        )

        if not nii_files:
            logger.warning(_SETUP_MSG)
            return self.synthetic_phantoms(n=200, size=self.target_size)

        if self.max_volumes is not None:
            nii_files = nii_files[: self.max_volumes]

        slices_list = []
        for path in nii_files:
            try:
                vol_slices = self._extract_slices(path)
                slices_list.append(vol_slices)
                logger.debug("%s → %d slices", path.name, len(vol_slices))
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        if not slices_list:
            logger.warning("All volumes failed to load. Using phantoms.")
            return self.synthetic_phantoms(n=200, size=self.target_size)

        data = np.concatenate(slices_list, axis=0).astype(np.float32)
        logger.info("MRI loader: %d slices from %d volumes", len(data), len(nii_files))
        return data

    # ------------------------------------------------------------------
    # Synthetic demo data
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic_phantoms(
        n: int = 200,
        size: Tuple[int, int] = (64, 64),
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate Gaussian blob phantoms to test the pipeline without real MRI.

        Returns
        -------
        np.ndarray, shape ``(n, 1, H, W)``, dtype float32, values in [0, 1]
        """
        rng = np.random.default_rng(seed)
        H, W = size
        slices = []
        for _ in range(n):
            img = np.zeros((H, W), dtype=np.float32)
            # Elliptical brain outline
            cy, cx = H // 2, W // 2
            ry, rx = int(H * 0.42), int(W * 0.38)
            y_grid, x_grid = np.ogrid[:H, :W]
            brain_mask = ((y_grid - cy) / ry) ** 2 + ((x_grid - cx) / rx) ** 2 <= 1.0
            img[brain_mask] = rng.uniform(0.05, 0.15)

            # Tissue blobs
            for _ in range(rng.integers(4, 9)):
                bcy = rng.integers(int(cy - ry * 0.7), int(cy + ry * 0.7))
                bcx = rng.integers(int(cx - rx * 0.7), int(cx + rx * 0.7))
                sigma = rng.uniform(3, 10)
                amplitude = rng.uniform(0.3, 0.9)
                blob = amplitude * np.exp(
                    -((y_grid - bcy) ** 2 + (x_grid - bcx) ** 2) / (2 * sigma ** 2)
                )
                img += blob.astype(np.float32) * brain_mask

            img = np.clip(img, 0, 1)
            slices.append(img[np.newaxis])  # (1, H, W)

        return np.stack(slices)  # (n, 1, H, W)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_slices(self, path: Path) -> np.ndarray:
        import nibabel as nib  # type: ignore
        from PIL import Image  # type: ignore

        vol_img = nib.load(str(path))
        vol: np.ndarray = vol_img.get_fdata(dtype=np.float32)

        if vol.ndim == 4:
            vol = vol[..., 0]  # first time point

        n_depth = vol.shape[self.slice_axis]
        start = int(self.slice_range[0] * n_depth)
        end = int(self.slice_range[1] * n_depth)

        H, W = self.target_size
        slices = []
        for idx in range(start, end):
            if self.slice_axis == 0:
                sl = vol[idx, :, :]
            elif self.slice_axis == 1:
                sl = vol[:, idx, :]
            else:
                sl = vol[:, :, idx]

            # Resize
            arr = Image.fromarray(sl).resize((W, H), Image.BILINEAR)
            sl_r = np.array(arr, dtype=np.float32)

            # Normalise to [0, 1]
            mn, mx = sl_r.min(), sl_r.max()
            if mx - mn > 1e-8:
                sl_r = (sl_r - mn) / (mx - mn)
            else:
                sl_r = np.zeros_like(sl_r)

            slices.append(sl_r[np.newaxis])  # (1, H, W)

        return np.stack(slices)  # (S, 1, H, W)
