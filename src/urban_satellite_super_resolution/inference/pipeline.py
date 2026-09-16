"""Scene inference with georeferenced uncertainty and urban outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..data.raster import read_reflectance
from ..models import MultiTaskUrbanSR
from .uncertainty import mc_predict
from ..utils.io import write_geotiff, write_report

URBAN_CLASSES = ("building", "road", "impervious", "vegetation", "water")


def _count_class_regions(probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    from scipy import ndimage

    counts = {}
    for index, name in enumerate(URBAN_CLASSES):
        mask = probabilities[index] >= threshold
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())[1:]
        counts[name] = int((sizes >= 2).sum())
    return counts


def _display(data: np.ndarray) -> np.ndarray:
    rgb = data[[2, 1, 0]] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0)
    output = np.zeros_like(rgb)
    for index, band in enumerate(rgb):
        low, high = np.percentile(band[np.isfinite(band)], [2, 98]) if np.isfinite(band).any() else (0, 1)
        output[index] = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
    return np.moveaxis(output, 0, -1)


def _write_quicklook(directory: Path, input_data: np.ndarray, sr: np.ndarray, uncertainty: np.ndarray, probabilities: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(16, 15))
    axes[0, 0].imshow(_display(input_data)); axes[0, 0].set_title("Observed Sentinel-2 input (10 m)")
    axes[0, 1].imshow(_display(sr)); axes[0, 1].set_title("Inferred SR estimate (~2.5 m)")
    axes[0, 2].imshow(uncertainty.mean(0), cmap="magma"); axes[0, 2].set_title("Predictive uncertainty")
    for index, name in enumerate(URBAN_CLASSES):
        row, column = divmod(index + 3, 3)
        axes[row, column].imshow(probabilities[index], cmap="viridis", vmin=0, vmax=1)
        axes[row, column].set_title(f"{name} probability")
    for axis in axes.ravel(): axis.axis("off")
    figure.tight_layout(); figure.savefig(directory / "quicklook.png", dpi=160); plt.close(figure)


def infer_scene(input_path: str | Path, output_dir: str | Path, *, checkpoint: str | Path | None = None, passes: int = 20, device: str | None = None, change_input: str | Path | None = None) -> dict[str, Any]:
    input_path, output_dir = Path(input_path), Path(output_dir)
    data, info = read_reflectance(input_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskUrbanSR(in_channels=data.shape[0], out_channels=data.shape[0], urban_classes=len(URBAN_CLASSES))
    model.to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
    tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    outputs = mc_predict(model, tensor, passes=passes)
    sr = outputs["super_resolved"][0].cpu().numpy()
    uncertainty = outputs["uncertainty_std"][0].cpu().numpy()
    probabilities = outputs["urban_probabilities"][0].cpu().numpy()
    confidence = outputs["confidence_classes"][0, 0].cpu().numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_geotiff(output_dir / "super_resolved.tif", sr, input_path)
    write_geotiff(output_dir / "uncertainty_std.tif", uncertainty, input_path)
    write_geotiff(output_dir / "confidence_classes.tif", confidence, input_path, categorical=True)
    for index, name in enumerate(URBAN_CLASSES):
        write_geotiff(output_dir / f"{name}_probability.tif", probabilities[index], input_path)
    if change_input:
        second, _ = read_reflectance(change_input)
        if second.shape != data.shape:
            raise ValueError("Two-date change inputs must have matching band and spatial shapes")
        change = np.clip(1.0 - np.exp(-8.0 * np.mean(np.abs(data - second), axis=0)), 0.0, 1.0)
        write_geotiff(output_dir / "urban_change_probability.tif", change, input_path)
    sr_input = torch.nn.functional.interpolate(outputs["super_resolved"], size=data.shape[-2:], mode="area")
    sr_outputs = mc_predict(model, sr_input, passes=passes)
    sr_probabilities = sr_outputs["urban_probabilities"][0].cpu().numpy()
    input_object_counts = _count_class_regions(probabilities)
    sr_object_counts = _count_class_regions(sr_probabilities)
    object_comparison = {
        name: {
            "input_object_count": input_object_counts[name],
            "sr_object_count": sr_object_counts[name],
            "object_count_change": sr_object_counts[name] - input_object_counts[name],
        }
        for name in URBAN_CLASSES
    }
    _write_quicklook(output_dir, data, sr, uncertainty, probabilities)
    report = {
        "model": "MultiTaskUrbanSR baseline",
        "input": str(input_path), "input_resolution_m": 10.0, "target_pixel_spacing_m": 2.5,
        "inference": {"device": device, "mc_dropout_passes": passes},
        "outputs": {"super_resolved": "super_resolved.tif", "uncertainty": "uncertainty_std.tif", "confidence": "confidence_classes.tif", "quicklook": "quicklook.png", "urban_probability_maps": [f"{name}_probability.tif" for name in URBAN_CLASSES], "urban_change_probability": "urban_change_probability.tif" if change_input else None},
        "confidence_summary": {"high": int((confidence == 0).sum()), "medium": int((confidence == 1).sum()), "low": int((confidence == 2).sum())},
        "object_count_comparison": {
            "input_source": str(input_path),
            "sr_source": "super_resolved.tif",
            "classes": object_comparison,
            "note": "Both views use the same urban probability head. The SR view is resolution-matched before classification; differences are model-output differences, not observed real-world change.",
        },
        "scientific_caveats": ["The enhanced image is a model-derived estimate, never observed 2.5 m ground truth.", "Fine-scale details and urban probabilities are inferred and should be interpreted with the uncertainty raster.", "Validation requires geographically separated paired data and suitable reference labels."],
    }
    report_path = write_report(output_dir / "report.json", report)
    markdown = "# Urban Satellite Super-Resolution Report\n\n"
    markdown += "**Scientific status:** The enhanced image is a model-derived estimate, not observed 2.5 m ground truth.\n\n"
    markdown += f"- Input: `{input_path}`\n- Input spacing: `{report['input_resolution_m']} m`\n- Target spacing: `{report['target_pixel_spacing_m']} m`\n- MC-dropout passes: `{passes}`\n- Device: `{device}`\n\n"
    markdown += "## Confidence summary\n\n"
    for level, count in report["confidence_summary"].items():
        markdown += f"- {level}: {count} pixels\n"
    markdown += "\n## Caveats\n\n" + "\n".join(f"- {item}" for item in report["scientific_caveats"]) + "\n"
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {"report": report_path, "markdown_report": output_dir / "report.md", **report}
