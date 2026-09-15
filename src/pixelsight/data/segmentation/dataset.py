"""
PixelSight Segmentation Dataset

Optimized PyTorch dataset for Sentinel-2 segmentation patches.

Image:
    (4, 128, 128) float32

Label:
    (128, 128) uint8

Classes:
    0   = Tree
    1   = Shrubland
    2   = Grassland
    3   = Cropland
    4   = Built-up
    5   = Bare
    6   = Water
    255 = Ignore

Training augmentation:
    - Random horizontal flip
    - Random vertical flip
    - Random 90-degree rotation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES = 7

IGNORE_INDEX = 255

CLASS_NAMES = (
    "Tree",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare",
    "Water",
)


# ============================================================
# DATASET
# ============================================================

class PixelSightSegmentationDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        expected_bands: int = 4,
        patch_size: int = 128,
        augment: bool = False,
        validate_samples: bool = False,
    ):
        super().__init__()

        self.root = Path(root)

        self.image_dir = self.root / "images"

        self.label_dir = self.root / "labels"

        self.expected_bands = expected_bands

        self.patch_size = patch_size

        self.augment = augment

        # ----------------------------------------------------
        # Validation behavior
        # ----------------------------------------------------
        #
        # Full validation of every .npy file is useful during
        # dataset debugging, but doing it repeatedly during
        # training adds unnecessary CPU/I/O overhead.
        #
        # Default is False for speed.
        # ----------------------------------------------------

        self.validate_samples = validate_samples

        # ----------------------------------------------------
        # Directory checks
        # ----------------------------------------------------

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}"
            )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.label_dir.exists():

            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}"
            )

        # ----------------------------------------------------
        # Image files
        # ----------------------------------------------------

        self.image_files = sorted(
            self.image_dir.glob("*.npy")
        )

        if not self.image_files:

            raise RuntimeError(
                f"No .npy image patches found in "
                f"{self.image_dir}"
            )

        # ----------------------------------------------------
        # Label files
        # ----------------------------------------------------

        self.label_files = {
            path.name: path
            for path in self.label_dir.glob("*.npy")
        }

        # ----------------------------------------------------
        # Missing labels
        # ----------------------------------------------------

        missing_labels = [
            path.name
            for path in self.image_files
            if path.name not in self.label_files
        ]

        if missing_labels:

            raise RuntimeError(
                f"Missing labels for "
                f"{len(missing_labels)} image patches. "
                f"Examples: {missing_labels[:5]}"
            )

        # ----------------------------------------------------
        # Extra labels
        # ----------------------------------------------------

        extra_labels = [
            name
            for name in self.label_files
            if not (self.image_dir / name).exists()
        ]

        if extra_labels:

            raise RuntimeError(
                f"Found {len(extra_labels)} label patches "
                f"without corresponding images. "
                f"Examples: {extra_labels[:5]}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self) -> int:

        return len(self.image_files)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index: int):

        image_path = self.image_files[index]

        label_path = self.label_files[
            image_path.name
        ]

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image = np.load(
            image_path
        )

        label = np.load(
            label_path
        )

        # ----------------------------------------------------
        # Optional validation
        # ----------------------------------------------------

        if self.validate_samples:

            self._validate_image(
                image,
                image_path,
            )

            self._validate_label(
                label,
                label_path,
            )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            image, label = self._augment(
                image,
                label,
            )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.astype(
                np.float32,
                copy=False,
            )
        )

        label = torch.from_numpy(
            label.astype(
                np.int64,
                copy=False,
            )
        )

        return image, label

    # ========================================================
    # AUGMENTATION
    # ========================================================

    def _augment(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ):

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if np.random.random() < 0.5:

            image = np.flip(
                image,
                axis=2,
            ).copy()

            label = np.flip(
                label,
                axis=1,
            ).copy()

        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if np.random.random() < 0.5:

            image = np.flip(
                image,
                axis=1,
            ).copy()

            label = np.flip(
                label,
                axis=0,
            ).copy()

        # ----------------------------------------------------
        # Random 90-degree rotation
        # ----------------------------------------------------

        k = np.random.randint(
            0,
            4,
        )

        if k != 0:

            image = np.rot90(
                image,
                k=k,
                axes=(1, 2),
            ).copy()

            label = np.rot90(
                label,
                k=k,
                axes=(0, 1),
            ).copy()

        return image, label

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image: np.ndarray,
        path: Path,
    ) -> None:

        expected_shape = (
            self.expected_bands,
            self.patch_size,
            self.patch_size,
        )

        if image.shape != expected_shape:

            raise ValueError(
                f"Invalid image shape in {path}: "
                f"expected {expected_shape}, "
                f"got {image.shape}"
            )

        if image.dtype != np.float32:

            raise ValueError(
                f"Invalid image dtype in {path}: "
                f"expected float32, "
                f"got {image.dtype}"
            )

        if not np.isfinite(image).all():

            raise ValueError(
                f"Non-finite values found in image: "
                f"{path}"
            )

        image_min = float(image.min())

        image_max = float(image.max())

        if (
            image_min < 0.0
            or image_max > 1.0
        ):

            raise ValueError(
                f"Image values outside [0, 1] in {path}: "
                f"min={image_min:.6f}, "
                f"max={image_max:.6f}"
            )

    # ========================================================
    # LABEL VALIDATION
    # ========================================================

    def _validate_label(
        self,
        label: np.ndarray,
        path: Path,
    ) -> None:

        expected_shape = (
            self.patch_size,
            self.patch_size,
        )

        if label.shape != expected_shape:

            raise ValueError(
                f"Invalid label shape in {path}: "
                f"expected {expected_shape}, "
                f"got {label.shape}"
            )

        if label.dtype != np.uint8:

            raise ValueError(
                f"Invalid label dtype in {path}: "
                f"expected uint8, "
                f"got {label.dtype}"
            )

        valid_values = np.unique(label)

        allowed_values = set(
            range(NUM_CLASSES)
        )

        allowed_values.add(
            IGNORE_INDEX
        )

        invalid_values = [
            int(value)
            for value in valid_values
            if int(value) not in allowed_values
        ]

        if invalid_values:

            raise ValueError(
                f"Invalid class IDs in {path}: "
                f"{invalid_values}"
            )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    def class_distribution(self) -> dict[int, int]:

        counts = {
            class_id: 0
            for class_id in range(NUM_CLASSES)
        }

        for label_path in self.label_files.values():

            label = np.load(
                label_path
            )

            self._validate_label(
                label,
                label_path,
            )

            for class_id in range(NUM_CLASSES):

                counts[class_id] += int(
                    np.sum(
                        label == class_id
                    )
                )

        return counts


# ============================================================
# DATASET FACTORY
# ============================================================

def create_dataset(
    split: str,
    base_dir: str | Path = "dataset/segmentation/patches",
    augment: bool = False,
    validate_samples: bool = False,
) -> PixelSightSegmentationDataset:

    root = Path(base_dir) / split

    return PixelSightSegmentationDataset(
        root=root,
        expected_bands=4,
        patch_size=128,
        augment=augment,
        validate_samples=validate_samples,
    )