"""
Sentinel-2 preprocessing utilities for PixelSight Plan 2.

Plan 2 uses Sentinel-2 Level-2A:
B02, B03, B04, B08 at native 10 m resolution.

Raw Sentinel-2 imagery must remain outside Git.
"""

from pathlib import Path

import numpy as np


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")


def validate_bands(bands):
    """Validate that all required multispectral bands are present."""
    missing = [
        band for band in REQUIRED_BANDS
        if band not in bands
    ]

    if missing:
        raise ValueError(
            f"Missing required bands: {missing}"
        )


def normalize_reflectance(
    image,
    scale_factor=10000.0
):
    """
    Convert Sentinel-2 integer surface-reflectance values
    to approximately [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Sentinel-2 image.
    scale_factor : float
        Reflectance scaling factor.

    Returns
    -------
    np.ndarray
        Float32 reflectance image clipped to [0, 1].
    """

    image = image.astype(np.float32)

    image = image / scale_factor

    return np.clip(
        image,
        0.0,
        1.0
    )


def create_valid_mask(
    image,
    nodata_value=0
):
    """
    Create a valid-pixel mask.

    A pixel is considered valid when all four spectral
    channels are non-nodata.
    """

    return np.all(
        image != nodata_value,
        axis=-1
    )


def extract_patches(
    image,
    patch_size=128,
    stride=128
):
    """
    Extract spatially aligned multispectral patches.

    Parameters
    ----------
    image : np.ndarray
        H x W x C image.
    patch_size : int
        Spatial patch size.
    stride : int
        Distance between patch origins.

    Returns
    -------
    np.ndarray
        N x patch_size x patch_size x C patches.
    """

    height, width, channels = image.shape

    patches = []

    for y in range(
        0,
        height - patch_size + 1,
        stride
    ):
        for x in range(
            0,
            width - patch_size + 1,
            stride
        ):
            patch = image[
                y:y + patch_size,
                x:x + patch_size,
                :
            ]

            patches.append(patch)

    if not patches:
        return np.empty(
            (
                0,
                patch_size,
                patch_size,
                channels
            ),
            dtype=np.float32
        )

    return np.stack(
        patches
    ).astype(np.float32)