from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


CLASS_NAMES = (
    "tree",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare",
    "water",
)
DISPLAY_NAMES = {
    "tree": "Trees",
    "shrubland": "Shrubland",
    "grassland": "Grassland",
    "cropland": "Cropland",
    "built_up": "Houses / built-up regions",
    "bare": "Bare land",
    "water": "Water",
}
TREE_CLASS = 0
BUILT_UP_CLASS = 4
PATCH_SIZE = 128
NUM_CLASSES = len(CLASS_NAMES)


def _checkpoint_state(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if isinstance(state, dict):
            return state
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Segmentation checkpoint does not contain a model state dictionary.")


def _predict_tiles(image: np.ndarray, model: object, torch: object, device: str) -> np.ndarray:
    height, width = image.shape[1:]
    prediction = np.zeros((height, width), dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for top in range(0, height, PATCH_SIZE):
            for left in range(0, width, PATCH_SIZE):
                bottom = min(top + PATCH_SIZE, height)
                right = min(left + PATCH_SIZE, width)
                tile = image[:, top:bottom, left:right]
                padded = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                padded[:, : tile.shape[1], : tile.shape[2]] = tile
                tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
                logits = model(tensor)
                tile_prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                prediction[top:bottom, left:right] = tile_prediction[: bottom - top, : right - left]
    return prediction


def _count_regions(mask: np.ndarray, minimum_pixels: int = 2) -> tuple[int, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel())[1:]
    accepted = sizes >= minimum_pixels
    return int(accepted.sum()), int(mask.sum())


def _save_planning_map(prediction: np.ndarray, output_path: Path) -> None:
    palette = np.array(
        [
            [40, 180, 90],
            [120, 170, 80],
            [170, 210, 100],
            [220, 190, 70],
            [210, 90, 55],
            [150, 135, 110],
            [50, 125, 210],
        ],
        dtype=np.uint8,
    )
    output = palette[np.clip(prediction, 0, NUM_CLASSES - 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path, format="PNG")


def _save_classification(prediction: np.ndarray, reference_path: Path, output_path: Path) -> None:
    import rasterio

    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(prediction.astype(np.uint8), 1)


def compare_urban_analysis(input_analysis: dict, sr_analysis: dict) -> dict:
    """Compare object classifications from native input and SR output."""
    input_classes = input_analysis.get("object_counts", {})
    sr_classes = sr_analysis.get("object_counts", {})
    comparison = {}
    for class_name in sorted(set(input_classes) | set(sr_classes)):
        before = input_classes.get(class_name, {})
        after = sr_classes.get(class_name, {})
        before_count = int(before.get("object_count", 0))
        after_count = int(after.get("object_count", 0))
        before_area = float(before.get("area_percent", 0.0))
        after_area = float(after.get("area_percent", 0.0))
        comparison[class_name] = {
            "label": after.get("label", before.get("label", class_name)),
            "input_object_count": before_count,
            "sr_object_count": after_count,
            "object_count_change": after_count - before_count,
            "input_area_percent": before_area,
            "sr_area_percent": after_area,
            "area_change_percent_points": after_area - before_area,
        }
    return {
        "input_source": "preprocessing/normalized.tif",
        "sr_source": "super_resolution/sr.tif",
        "classes": comparison,
        "interpretation": (
            "This comparison shows how the segmentation model classifies the native input versus the generated SR estimate. "
            "Changes indicate model-output differences and must not be interpreted as observed real-world change."
        ),
    }


def run_urban_analysis(
    input_path: Path,
    output_map: Path,
    *,
    classified_output: Path | None = None,
    checkpoint_path: Path,
    device: str,
) -> dict:
    import rasterio
    import torch
    from pixelsight.models.segmentation.unet import create_unet

    with rasterio.open(input_path) as source:
        image = source.read().astype(np.float32)

    if image.shape[0] != 4:
        raise ValueError(f"Urban analysis requires 4 bands, received {image.shape[0]}.")
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.nanmax(image)) > 1.5:
        image /= 10000.0
    image = np.clip(image, 0.0, 1.0)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_unet(in_channels=4, num_classes=NUM_CLASSES, base_channels=32)
    model.load_state_dict(_checkpoint_state(checkpoint))
    model.to(device)
    prediction = _predict_tiles(image, model, torch, device)
    _save_planning_map(prediction, output_map)
    if classified_output is not None:
        _save_classification(prediction, input_path, classified_output)

    tree_clusters, tree_pixels = _count_regions(prediction == TREE_CLASS)
    built_clusters, built_pixels = _count_regions(prediction == BUILT_UP_CLASS)
    total_pixels = prediction.size
    object_counts = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        clusters, pixels = _count_regions(prediction == class_id)
        object_counts[class_name] = {
            "label": DISPLAY_NAMES[class_name],
            "object_count": clusters,
            "pixel_count": pixels,
            "area_percent": float(pixels / total_pixels * 100.0),
        }
    other_object_count = sum(
        value["object_count"] for name, value in object_counts.items() if name not in {"tree", "built_up"}
    )
    interpretation = (
        f"The super-resolved scene contains an estimated {tree_clusters} connected tree region(s) "
        f"and {built_clusters} connected built-up region(s), reported here as house-area estimates. "
        f"The remaining classified land-cover regions contribute {other_object_count} other connected object region(s). "
        f"Built-up coverage is {built_pixels / total_pixels * 100.0:.1f}% and tree coverage is "
        f"{tree_pixels / total_pixels * 100.0:.1f}% of the analyzed raster."
    )
    return {
        "status": "estimated",
        "source": "super_resolved.tif",
        "map": "analysis/urban_planning_map.png",
        "classified_raster": "analysis/urban_classes.tif" if classified_output is not None else None,
        "class_names": list(CLASS_NAMES),
        "object_counts": object_counts,
        "trees": tree_clusters,
        "houses": built_clusters,
        "other_objects": other_object_count,
        "interpretation": interpretation,
        "class_pixel_counts": {
            name: int((prediction == index).sum()) for index, name in enumerate(CLASS_NAMES)
        },
        "tree_clusters": tree_clusters,
        "tree_pixels": tree_pixels,
        "estimated_building_clusters": built_clusters,
        "built_up_pixels": built_pixels,
        "coverage": {
            "tree_percent": float(tree_pixels / total_pixels * 100.0),
            "built_up_percent": float(built_pixels / total_pixels * 100.0),
        },
        "limitations": [
            "Counts are connected classified regions in the super-resolved estimate, not surveyed individual trees or houses.",
            "The segmentation checkpoint was trained on land-cover classes. Houses are therefore reported as connected built-up regions, not guaranteed individual building footprints.",
            "Roads and impervious surfaces are not separate classes in the current trained checkpoint; they are included in the land-cover classification and other-object total where applicable.",
            "RGB PNG/JPEG uploads use a luminance-derived NIR proxy, so planning estimates are less reliable than true Sentinel-2 B08 input.",
        ],
    }
