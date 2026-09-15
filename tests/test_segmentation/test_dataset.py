"""
Tests for PixelSight segmentation dataset.
"""

from pathlib import Path

import torch

from pixelsight.data.segmentation.dataset import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    PixelSightSegmentationDataset,
    create_dataset,
)


DATASET_ROOT = Path("dataset/segmentation/patches")


def test_train_dataset_exists():
    root = DATASET_ROOT / "train"

    assert root.exists()
    assert (root / "images").exists()
    assert (root / "labels").exists()


def test_train_dataset_loads():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    assert len(dataset) == 7225


def test_sample_shapes_and_types():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert image.shape == (4, 128, 128)
    assert label.shape == (128, 128)

    assert image.dtype == torch.float32
    assert label.dtype == torch.int64


def test_image_range():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    image, _ = dataset[0]

    assert torch.isfinite(image).all()
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_label_values():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    _, label = dataset[0]

    unique_values = torch.unique(label).tolist()

    allowed_values = set(range(NUM_CLASSES))
    allowed_values.add(IGNORE_INDEX)

    assert set(unique_values).issubset(allowed_values)


def test_all_class_names_exist():
    assert NUM_CLASSES == 7
    assert len(CLASS_NAMES) == 7

    expected_names = (
        "Tree",
        "Shrubland",
        "Grassland",
        "Cropland",
        "Built-up",
        "Bare",
        "Water",
    )

    assert CLASS_NAMES == expected_names


def test_image_label_pairs_match():
    train_root = DATASET_ROOT / "train"

    image_names = {
        path.name
        for path in (train_root / "images").glob("*.npy")
    }

    label_names = {
        path.name
        for path in (train_root / "labels").glob("*.npy")
    }

    assert image_names == label_names


def test_dataset_factory():
    dataset = create_dataset("train")

    assert len(dataset) == 7225


def test_validation_dataset_loads():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "validation"
    )

    assert len(dataset) == 7225


def test_test_dataset_loads():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "test"
    )

    assert len(dataset) == 7149


def test_all_splits_have_expected_size():
    train_dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    validation_dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "validation"
    )

    test_dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "test"
    )

    assert len(train_dataset) == 7225
    assert len(validation_dataset) == 7225
    assert len(test_dataset) == 7149


def test_class_distribution():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    counts = dataset.class_distribution()

    assert set(counts.keys()) == set(range(NUM_CLASSES))

    for class_id in range(NUM_CLASSES):
        assert counts[class_id] >= 0


def test_class_distribution_contains_pixels():
    dataset = PixelSightSegmentationDataset(
        DATASET_ROOT / "train"
    )

    counts = dataset.class_distribution()

    # Every class should occur somewhere in the training scene.
    for class_id in range(NUM_CLASSES):
        assert counts[class_id] > 0