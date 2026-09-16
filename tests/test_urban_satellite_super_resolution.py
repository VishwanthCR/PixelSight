from pathlib import Path

import numpy as np
import torch

from urban_satellite_super_resolution.data.synthetic import generate_sample
from urban_satellite_super_resolution.data.raster import assert_aligned
from urban_satellite_super_resolution.evaluation.metrics import image_metrics, segmentation_metrics
from urban_satellite_super_resolution.inference.uncertainty import mc_predict
from urban_satellite_super_resolution.models import MultiTaskUrbanSR
from urban_satellite_super_resolution.utils.io import write_geotiff


def test_model_forward_and_uncertainty():
    model = MultiTaskUrbanSR(features=8, blocks=1)
    outputs = mc_predict(model, torch.rand(1, 4, 16, 16), passes=3)
    assert outputs["super_resolved"].shape == (1, 4, 64, 64)
    assert outputs["urban_probabilities"].shape == (1, 5, 64, 64)
    assert outputs["uncertainty_std"].shape == (1, 4, 64, 64)


def test_synthetic_alignment_and_georeferencing(tmp_path: Path):
    paths = generate_sample(tmp_path / "sample", width=32, height=32)
    assert_aligned(paths["hr"], paths["labels"])
    output = write_geotiff(tmp_path / "out.tif", np.zeros((4, 32, 32), dtype=np.float32), paths["lr"])
    with __import__("rasterio").open(output) as src:
        assert src.width == 32 and src.height == 32
        assert abs(src.res[0] - 2.5) < 1e-6


def test_metrics_are_finite():
    target = np.ones((4, 8, 8), dtype=np.float32)
    metrics = image_metrics(target, target)
    assert metrics["mae"] == 0.0
    segmentation = segmentation_metrics(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))
    assert segmentation["accuracy"] == 1.0
