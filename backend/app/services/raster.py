from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from backend.app.schemas import InspectionResponse


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")
MODEL_SIZE = 128
SCALE = 4


def _band_name(src: rasterio.DatasetReader, index: int) -> str | None:
    description = src.descriptions[index - 1]
    if description:
        return description.strip().upper()

    tags = src.tags(index)
    for key in ("band_name", "band", "name", "BANDNAME", "DESCRIPTION"):
        value = tags.get(key)
        if value:
            return value.strip().upper()
    return None


def inspect_raster(path: Path, filename: str | None = None) -> InspectionResponse:
    display_name = filename or path.name
    try:
        with rasterio.open(path) as src:
            names = [_band_name(src, index) for index in range(1, src.count + 1)]
            errors: list[str] = []
            warnings: list[str] = []

            if src.count != len(REQUIRED_BANDS):
                errors.append(
                    f"Expected exactly 4 bands ({'/'.join(REQUIRED_BANDS)}), found {src.count}."
                )
            elif tuple(names) != REQUIRED_BANDS:
                errors.append(
                    "Band descriptions must explicitly identify B02, B03, B04, B08 in that order."
                )

            if src.width < MODEL_SIZE or src.height < MODEL_SIZE:
                warnings.append(
                    f"Image is smaller than the {MODEL_SIZE}x{MODEL_SIZE} model tile; it will be padded during analysis."
                )
            if src.crs is None:
                errors.append("A CRS is required for geospatial processing.")
            if abs(src.transform.a) == 0 or abs(src.transform.e) == 0:
                errors.append("The raster transform is invalid.")

            resolution = [float(abs(src.res[0])), float(abs(src.res[1]))]
            if not np.allclose(resolution, [10.0, 10.0], atol=0.01):
                warnings.append("Input resolution is not 10m; preprocessing may be required.")

            dtype = src.dtypes[0] if src.count else None
            requires_preprocessing = dtype != "float32"
            if dtype == "float32":
                sample = src.read(masked=True)
                if sample.size and (float(sample.min()) < 0 or float(sample.max()) > 1):
                    requires_preprocessing = True

            return InspectionResponse(
                valid=not errors,
                compatible=not errors,
                filename=display_name,
                format=src.driver,
                width=src.width,
                height=src.height,
                bands=src.count,
                band_names=names,
                dtype=dtype,
                crs=str(src.crs) if src.crs else None,
                transform=list(src.transform),
                resolution=resolution,
                bounds=[float(value) for value in src.bounds],
                nodata=[float(value) if value is not None else None for value in src.nodatavals],
                band_metadata=[src.tags(index) for index in range(1, src.count + 1)],
                requires_preprocessing=requires_preprocessing,
                estimated_output_dimensions=[src.width * SCALE, src.height * SCALE],
                errors=errors,
                warnings=warnings,
            )
    except (rasterio.errors.RasterioIOError, ValueError) as exc:
        return InspectionResponse(
            valid=False,
            compatible=False,
            filename=display_name,
            errors=[f"Unable to inspect raster: {exc}"],
        )


def preprocess_raster(source: Path, destination: Path) -> tuple[InspectionResponse, list[dict[str, Any]]]:
    inspection = inspect_raster(source)
    if not inspection.compatible:
        raise ValueError("Input is not compatible with the LDSR-S2 Sentinel-2 adapter.")

    with rasterio.open(source) as src:
        raw = src.read()
        profile = src.profile.copy()
        descriptions = src.descriptions
        max_value = float(np.nanmax(raw))
        image = raw.astype(np.float32)
        operations: list[dict[str, Any]] = []

        if max_value > 1.5:
            image /= 10000.0
            operations.append({"operation": "normalize_reflectance", "scale": 10000})

        finite = np.isfinite(image)
        if not finite.all():
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            operations.append({"operation": "replace_non_finite_values", "replacement": 0.0})

        clipped = np.clip(image, 0.0, 1.0)
        if not np.array_equal(image, clipped):
            operations.append({"operation": "clip_reflectance", "range": [0.0, 1.0]})
        image = clipped

        profile.update(dtype="float32", count=4, compress="deflate", tiled=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(image)
            for index, description in enumerate(descriptions, start=1):
                dst.set_band_description(index, description or REQUIRED_BANDS[index - 1])

    if inspection.dtype != "float32":
        operations.append({"operation": "convert_dtype", "source": inspection.dtype, "target": "float32"})
    return inspect_raster(destination), operations
