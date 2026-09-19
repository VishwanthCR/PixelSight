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


def run_urban_pipeline(
    native_path: Path,
    sr_path: Path,
    job_dir: Path,
    *,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> dict:
    """
    Dedicated urban analysis pipeline executing segmentation across native and SR rasters,
    producing built-up masks, difference rasters, previews, and neutral comparative statistics.
    """
    import json
    import rasterio

    ckpt = checkpoint_path or (
        Path(__file__).resolve().parents[3] / "checkpoints" / "segmentation" / "unet_worldcover_best.pth"
    )
    urban_dir = job_dir / "application" / "urban"
    previews_dir = urban_dir / "previews"
    urban_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    sr_seg_tif = urban_dir / "segmentation_sr.tif"
    sr_seg_png = previews_dir / "segmentation_sr.png"
    native_seg_tif = urban_dir / "segmentation_native.tif"
    native_seg_png = previews_dir / "segmentation_native.png"
    builtup_tif = urban_dir / "builtup.tif"
    builtup_png = previews_dir / "builtup_preview.png"
    diff_tif = urban_dir / "urban_difference.tif"
    diff_png = previews_dir / "urban_difference.png"

    # 1. Run segmentation on SR and Native
    sr_analysis = run_urban_analysis(
        sr_path,
        sr_seg_png,
        classified_output=sr_seg_tif,
        checkpoint_path=ckpt,
        device=device,
    )
    native_analysis = run_urban_analysis(
        native_path,
        native_seg_png,
        classified_output=native_seg_tif,
        checkpoint_path=ckpt,
        device=device,
    )

    # 2. Extract predictions for masks and differences
    with rasterio.open(sr_seg_tif) as src:
        sr_pred = src.read(1)
        sr_profile = src.profile.copy()

    with rasterio.open(native_seg_tif) as src:
        native_pred = src.read(1)

    # 3. Built-up mask (class 4)
    builtup_mask = (sr_pred == BUILT_UP_CLASS).astype(np.uint8)
    sr_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(builtup_tif, "w", **sr_profile) as dst:
        dst.write(builtup_mask, 1)
        dst.set_band_description(1, "Built-up mask (1=built-up, 0=other)")

    # Built-up preview
    builtup_rgb = np.zeros((*builtup_mask.shape, 3), dtype=np.uint8)
    builtup_rgb[builtup_mask == 1] = [220, 70, 50]  # Red for built-up
    builtup_rgb[builtup_mask == 0] = [30, 40, 50]   # Dark slate for background
    Image.fromarray(builtup_rgb).save(builtup_png, format="PNG")

    # 4. Difference map
    sr_h, sr_w = sr_pred.shape
    native_img = Image.fromarray(native_pred)
    native_resampled = np.asarray(native_img.resize((sr_w, sr_h), Image.NEAREST), dtype=np.uint8)
    diff_mask = (sr_pred != native_resampled).astype(np.uint8)

    with rasterio.open(diff_tif, "w", **sr_profile) as dst:
        dst.write(diff_mask, 1)
        dst.set_band_description(1, "Segmentation difference (1=disagreement, 0=agreement)")

    diff_rgb = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
    diff_rgb[diff_mask == 1] = [240, 200, 60]  # Yellow for reclassified
    diff_rgb[diff_mask == 0] = [40, 45, 55]    # Dark for consistent
    Image.fromarray(diff_rgb).save(diff_png, format="PNG")

    # 5. Indicators & Summary
    total_pixels = max(int(sr_pred.size), 1)
    veg_mask = np.isin(sr_pred, [0, 1, 2, 3])
    water_mask = (sr_pred == 6)
    bare_mask = (sr_pred == 5)

    veg_pixels = int(veg_mask.sum())
    water_pixels = int(water_mask.sum())
    bare_pixels = int(bare_mask.sum())
    built_pixels = int(builtup_mask.sum())

    comparison = compare_urban_analysis(native_analysis, sr_analysis)

    result = {
        "application": "urban",
        "scientific_status": "Model-output comparison",
        "indicators": {
            "built_up_area_pixels": built_pixels,
            "built_up_fraction": float(built_pixels / total_pixels),
            "vegetation_area_pixels": veg_pixels,
            "vegetation_fraction": float(veg_pixels / total_pixels),
            "water_area_pixels": water_pixels,
            "water_fraction": float(water_pixels / total_pixels),
            "bare_area_pixels": bare_pixels,
            "bare_fraction": float(bare_pixels / total_pixels),
            "connected_built_up_regions": sr_analysis.get("houses", 0),
            "connected_tree_regions": sr_analysis.get("trees", 0),
        },
        "class_distribution": {
            CLASS_NAMES[cid]: {
                "label": DISPLAY_NAMES[CLASS_NAMES[cid]],
                "pixel_count": int((sr_pred == cid).sum()),
                "percent": float((sr_pred == cid).sum() / total_pixels * 100.0),
            }
            for cid in range(NUM_CLASSES)
        },
        "comparison": comparison,
        "interpretations": [
            f"Built-up area covers {built_pixels / total_pixels * 100.0:.1f}% across {sr_analysis.get('houses', 0)} connected cluster(s).",
            f"Vegetation canopy (trees/shrub/grass/crop) covers {veg_pixels / total_pixels * 100.0:.1f}% of the scene.",
            "Downstream segmentation results reflect model inference patterns; PixelSight research establishes that LDSR does not outperform bicubic baselines on standardized land-cover metrics.",
        ],
        "limitations": [
            "Segmentation is performed by a UNet trained on ESA WorldCover land-cover classes, not cadastral building surveys.",
            "Output differences between native and SR representations indicate neural model sensitivity, NOT observed physical land-use transformation.",
        ],
        "outputs": {
            "segmentation_native_geotiff": "application/urban/segmentation_native.tif",
            "segmentation_sr_geotiff": "application/urban/segmentation_sr.tif",
            "builtup_mask_geotiff": "application/urban/builtup.tif",
            "urban_difference_geotiff": "application/urban/urban_difference.tif",
            "segmentation_native_preview": "application/urban/previews/segmentation_native.png",
            "segmentation_sr_preview": "application/urban/previews/segmentation_sr.png",
            "builtup_preview": "application/urban/previews/builtup_preview.png",
            "urban_difference_preview": "application/urban/previews/urban_difference.png",
            "metrics": "application/urban/metrics.json",
        },
    }

    metrics_path = urban_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

