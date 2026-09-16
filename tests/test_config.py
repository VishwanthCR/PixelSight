"""Tests for framework configuration, artifact manifests, and model protocols."""

from pathlib import Path

import pytest
import torch

from urban_satellite_super_resolution.config import FrameworkConfig, load_config
from urban_satellite_super_resolution.inference.outputs import create_manifest
from urban_satellite_super_resolution.models.base import SuperResolutionModel
from urban_satellite_super_resolution.models.ldsr_adapter import LDSRS2Adapter


def test_default_framework_config():
    config = FrameworkConfig()
    assert config.project.name == "urban_satellite_super_resolution"
    assert config.input.scale_factor == 4
    assert config.preprocessing.tile_size == 128
    assert config.model.backend == "ldsr_s2"
    assert config.model.in_channels == 4
    assert config.uncertainty.passes == 20
    assert config.urban.task == "semantic_segmentation"
    assert config.training.batch_size == 8
    assert config.loss.charbonnier_weight == 1.0


def test_load_canonical_config():
    config_path = Path("configs/urban_satellite_super_resolution.yaml")
    assert config_path.is_file()

    config = load_config(config_path)
    assert config.project.name == "urban_satellite_super_resolution"
    assert config.project.seed == 42
    assert config.input.bands == ["B02", "B03", "B04", "B08"]
    assert config.input.expected_resolution_m == 10
    assert config.preprocessing.tile_size == 128
    assert config.preprocessing.stride == 96
    assert config.model.backend == "ldsr_s2"
    assert config.uncertainty.method == "mc_dropout"
    assert config.training.learning_rate == 0.0002
    assert config.evaluation.require_independent_reference is True


def test_load_nonexistent_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("configs/nonexistent_config.yaml"))


def test_artifact_manifest_creation(tmp_path: Path):
    manifest = create_manifest(
        job_id="test_job_123",
        model_name="LDSR-S2",
        input_path="data/test/lr/scene.tif",
        preprocessing=[{"operation": "normalize_reflectance"}],
        artifacts={"super_resolved": "super_resolution/sr.tif"},
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["job_id"] == "test_job_123"
    assert manifest_dict["model"]["name"] == "LDSR-S2"
    assert manifest_dict["status_flags"]["proxy_uncertainty"] is True
    assert manifest_dict["status_flags"]["independent_reference_available"] is False

    saved_path = manifest.save(tmp_path / "manifest.json")
    assert saved_path.is_file()
    assert '"test_job_123"' in saved_path.read_text(encoding="utf-8")


def test_ldsr_adapter_protocol():
    adapter = LDSRS2Adapter()
    assert isinstance(adapter, SuperResolutionModel)
    assert adapter.name == "LDSR-S2"
    assert adapter.scale == 4
