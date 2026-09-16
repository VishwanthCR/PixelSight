"""Canonical configuration loading, validation, and schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "urban_satellite_super_resolution"
    version: str = "0.1.0"
    seed: int = 42
    output_root: str = "results/framework"


@dataclass
class InputConfig:
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    expected_resolution_m: float = 10.0
    scale_factor: int = 4
    target_spacing_m: float = 2.5
    normalization: str = "training_statistics"


@dataclass
class PreprocessingConfig:
    tile_size: int = 128
    stride: int = 96
    overlap: int = 32
    resampling_image: str = "bilinear"
    resampling_label: str = "nearest"
    reject_cloud_fraction: float = 0.10
    reject_nodata_fraction: float = 0.10
    require_crs: bool = True
    require_alignment: bool = True


@dataclass
class ModelConfig:
    backend: str = "ldsr_s2"
    fallback_backend: str = "edsr_multitask"
    in_channels: int = 4
    out_channels: int = 4
    scale: int = 4
    sampling_steps: int = 100
    features: int = 48
    blocks: int = 6
    dropout: float = 0.15
    urban_classes: int = 5


@dataclass
class UncertaintyConfig:
    enabled: bool = True
    method: str = "mc_dropout"
    passes: int = 20
    thresholds: list[float] = field(default_factory=lambda: [0.33, 0.66])
    calibration_reference_required: bool = True


@dataclass
class UrbanConfig:
    enabled: bool = True
    task: str = "semantic_segmentation"
    classes: list[str] = field(
        default_factory=lambda: ["building", "road", "impervious", "vegetation", "water"]
    )
    instance_counting: bool = False
    checkpoint: str = "checkpoints/segmentation/unet_worldcover_best.pth"


@dataclass
class ChangeConfig:
    enabled: bool = True
    require_same_crs: bool = True
    require_same_grid: bool = True
    threshold: str = "learned_or_validation_calibrated"


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    mixed_precision: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints/framework"
    checkpoint: str = "checkpoints/urban_satellite_super_resolution.pt"


@dataclass
class LossConfig:
    charbonnier_weight: float = 1.0
    ssim_weight: float = 0.1
    sam_weight: float = 0.05
    segmentation_dice_weight: float = 1.0
    segmentation_focal_weight: float = 1.0
    uncertainty_nll_weight: float = 0.1
    sr_weight: float = 1.0
    segmentation_weight: float = 1.0
    uncertainty_weight: float = 0.1
    ignore_index: int = 255


@dataclass
class EvaluationConfig:
    lpips_optional: bool = True
    metrics: list[str] = field(
        default_factory=lambda: [
            "psnr",
            "ssim",
            "mae",
            "rmse",
            "sam",
            "iou",
            "f1",
            "precision",
            "recall",
            "ece",
        ]
    )
    require_independent_reference: bool = True
    geographic_grouping: list[str] = field(
        default_factory=lambda: ["scene", "region", "city"]
    )


@dataclass
class FrameworkConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    urban: UrbanConfig = field(default_factory=UrbanConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        result = dataclasses.asdict(self)
        result.pop("raw", None)
        return result


def _populate(target_cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return target_cls()
    fields = getattr(target_cls, "__dataclass_fields__", {})
    kwargs = {key: value for key, value in data.items() if key in fields}
    return target_cls(**kwargs)


def load_config(path: str | Path) -> FrameworkConfig:
    """Load and validate canonical framework configuration from YAML."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    # Backward compatibility with prototype configs
    input_data = raw.get("input") or {}
    if not input_data and "data" in raw:
        input_data = {
            "bands": raw["data"].get("input_bands", ["B02", "B03", "B04", "B08"]),
            "scale_factor": raw["data"].get("scale", 4),
            "normalization": raw["data"].get("normalization", "training_statistics"),
        }

    preprocessing_data = raw.get("preprocessing") or {}
    if not preprocessing_data and "data" in raw:
        preprocessing_data = {
            "tile_size": raw["data"].get("tile_size", 128),
            "stride": raw["data"].get("stride", 96),
        }

    uncertainty_data = raw.get("uncertainty") or {}
    if not uncertainty_data and "inference" in raw:
        uncertainty_data = {
            "passes": raw["inference"].get("mc_dropout_passes", 20),
            "thresholds": raw["inference"].get("confidence_thresholds", [0.05, 0.15]),
        }

    return FrameworkConfig(
        project=_populate(ProjectConfig, raw.get("project")),
        input=_populate(InputConfig, input_data),
        preprocessing=_populate(PreprocessingConfig, preprocessing_data),
        model=_populate(ModelConfig, raw.get("model")),
        uncertainty=_populate(UncertaintyConfig, uncertainty_data),
        urban=_populate(UrbanConfig, raw.get("urban")),
        change=_populate(ChangeConfig, raw.get("change")),
        training=_populate(TrainingConfig, raw.get("training")),
        loss=_populate(LossConfig, raw.get("loss")),
        evaluation=_populate(EvaluationConfig, raw.get("evaluation")),
        raw=raw,
    )
