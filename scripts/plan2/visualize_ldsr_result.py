from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

INPUT_TIF = Path(
    "dataset/plan2/geotiff/"
    "train_test_512_10m.tif"
)

LDSR_100_TIF = Path(
    "dataset/plan2/ldsr_s2_test/"
    "train_test_512_ldsr_s2_2p5m.tif"
)

LDSR_10_TIF = Path(
    "dataset/plan2/ldsr_s2_test/"
    "train_test_512_ldsr_s2_2p5m_10steps.tif"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIGURES_DIR = Path(
    "results/plan2/figures"
)

QUALITATIVE_DIR = Path(
    "results/plan2/qualitative"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_tif(path):
    """
    Read a 4-band GeoTIFF as float32.

    Returns:
        data: C,H,W
        profile
    """

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with rasterio.open(path) as src:

        data = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

    if not np.isfinite(data).all():

        raise RuntimeError(
            f"NaN or infinite values found in:\n{path}"
        )

    return data, profile


def percentile_normalize(image, low=2, high=98):
    """
    Normalize an image for visualization only.

    Does NOT modify the scientific data.
    """

    image = image.astype(
        np.float32
    )

    lo = np.percentile(
        image,
        low
    )

    hi = np.percentile(
        image,
        high
    )

    if hi <= lo:

        return np.zeros_like(
            image
        )

    result = (
        image - lo
    ) / (
        hi - lo
    )

    return np.clip(
        result,
        0,
        1
    )


def make_rgb(data):
    """
    Sentinel-2 RGB:
        B04 = Red
        B03 = Green
        B02 = Blue

    Input:
        C,H,W

    Output:
        H,W,3
    """

    rgb = np.stack(
        [
            data[2],
            data[1],
            data[0],
        ],
        axis=-1
    )

    return percentile_normalize(
        rgb
    )


def make_false_color(data):
    """
    Sentinel-2 false color:
        B08 = Red
        B04 = Green
        B03 = Blue
    """

    false_color = np.stack(
        [
            data[3],
            data[2],
            data[1],
        ],
        axis=-1
    )

    return percentile_normalize(
        false_color
    )


def save_image(
    image,
    path,
    title=None,
    figsize=(10, 8),
    cmap=None
):

    plt.figure(
        figsize=figsize
    )

    if image.ndim == 3:

        plt.imshow(
            image
        )

    else:

        plt.imshow(
            image,
            cmap=cmap
        )

        plt.colorbar(
            fraction=0.046,
            pad=0.04
        )

    if title is not None:

        plt.title(
            title,
            fontsize=14
        )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PixelSight — LDSR-S2 "
        "10 STEP vs 100 STEP VISUALIZATION"
    )
    print("=" * 70)

    print()

    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    print(
        "Checking input files..."
    )

    for path in [
        INPUT_TIF,
        LDSR_100_TIF,
        LDSR_10_TIF,
    ]:

        print(
            f"  {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n\n"
                "This full-scene 10-vs-100-step comparison requires both "
                "GeoTIFF outputs. They are not recreated automatically.\n\n"
                "For the completed 100-step patch outputs already in this "
                "project, run:\n"
                "  python scripts/plan2/"
                "generate_ldsr_sample_comparisons.py"
            )

    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 2. READ DATA
    # ========================================================

    print(
        "Reading original 10m image..."
    )

    input_data, input_profile = read_tif(
        INPUT_TIF
    )

    print(
        f"  Shape: {input_data.shape}"
    )

    print()

    print(
        "Reading 100-step LDSR-S2 output..."
    )

    ldsr_100, sr100_profile = read_tif(
        LDSR_100_TIF
    )

    print(
        f"  Shape: {ldsr_100.shape}"
    )

    print()

    print(
        "Reading 10-step LDSR-S2 output..."
    )

    ldsr_10, sr10_profile = read_tif(
        LDSR_10_TIF
    )

    print(
        f"  Shape: {ldsr_10.shape}"
    )

    print()

    # ========================================================
    # 3. VALIDATE SHAPES
    # ========================================================

    if input_data.shape != (
        4,
        512,
        512,
    ):

        raise RuntimeError(
            "Unexpected input shape: "
            f"{input_data.shape}"
        )

    if ldsr_100.shape != (
        4,
        2048,
        2048,
    ):

        raise RuntimeError(
            "Unexpected 100-step output shape: "
            f"{ldsr_100.shape}"
        )

    if ldsr_10.shape != (
        4,
        2048,
        2048,
    ):

        raise RuntimeError(
            "Unexpected 10-step output shape: "
            f"{ldsr_10.shape}"
        )

    # ========================================================
    # 4. PRINT METADATA
    # ========================================================

    print("=" * 70)
    print("IMAGE INFORMATION")
    print("=" * 70)

    print()

    print("Original:")
    print(
        "  Resolution:",
        input_profile["transform"].a,
        "m"
    )

    print(
        "  Size:",
        input_data.shape[2],
        "x",
        input_data.shape[1]
    )

    print(
        "  CRS:",
        input_profile["crs"]
    )

    print()

    print("LDSR-S2 100 steps:")
    print(
        "  Resolution:",
        sr100_profile["transform"].a,
        "m"
    )

    print(
        "  Size:",
        ldsr_100.shape[2],
        "x",
        ldsr_100.shape[1]
    )

    print()

    print("LDSR-S2 10 steps:")
    print(
        "  Resolution:",
        sr10_profile["transform"].a,
        "m"
    )

    print(
        "  Size:",
        ldsr_10.shape[2],
        "x",
        ldsr_10.shape[1]
    )

    print()

    # ========================================================
    # 5. CREATE DISPLAY IMAGES
    # ========================================================

    print(
        "Creating RGB images..."
    )

    input_rgb = make_rgb(
        input_data
    )

    rgb_100 = make_rgb(
        ldsr_100
    )

    rgb_10 = make_rgb(
        ldsr_10
    )

    print(
        "Creating false-color images..."
    )

    input_false = make_false_color(
        input_data
    )

    false_100 = make_false_color(
        ldsr_100
    )

    false_10 = make_false_color(
        ldsr_10
    )

    # ========================================================
    # 6. RGB COMPARISON
    # ========================================================

    print(
        "Creating RGB comparison..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(
        input_rgb
    )

    axes[0].set_title(
        "Original Sentinel-2 — 10m",
        fontsize=14
    )

    axes[1].imshow(
        rgb_100
    )

    axes[1].set_title(
        "LDSR-S2 — 100 steps — 2.5m",
        fontsize=14
    )

    axes[2].imshow(
        rgb_10
    )

    axes[2].set_title(
        "LDSR-S2 — 10 steps — 2.5m",
        fontsize=14
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    rgb_path = (
        FIGURES_DIR
        / "input_vs_ldsr_100_vs_10_rgb.png"
    )

    plt.savefig(
        rgb_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        rgb_path
    )

    # ========================================================
    # 7. FALSE COLOR COMPARISON
    # ========================================================

    print(
        "Creating false-color comparison..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(
        input_false
    )

    axes[0].set_title(
        "Original Sentinel-2 — 10m",
        fontsize=14
    )

    axes[1].imshow(
        false_100
    )

    axes[1].set_title(
        "LDSR-S2 — 100 steps",
        fontsize=14
    )

    axes[2].imshow(
        false_10
    )

    axes[2].set_title(
        "LDSR-S2 — 10 steps",
        fontsize=14
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    false_path = (
        FIGURES_DIR
        / "input_vs_ldsr_100_vs_10_false_color.png"
    )

    plt.savefig(
        false_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        false_path
    )

    # ========================================================
    # 8. 100 vs 10 RGB ONLY
    # ========================================================

    print(
        "Creating direct 100-vs-10 RGB comparison..."
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )

    axes[0].imshow(
        rgb_100
    )

    axes[0].set_title(
        "LDSR-S2 — 100 steps",
        fontsize=15
    )

    axes[1].imshow(
        rgb_10
    )

    axes[1].set_title(
        "LDSR-S2 — 10 steps",
        fontsize=15
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    path = (
        QUALITATIVE_DIR
        / "ldsr_100_vs_10_rgb.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 9. FALSE COLOR 100 VS 10
    # ========================================================

    print(
        "Creating direct false-color comparison..."
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )

    axes[0].imshow(
        false_100
    )

    axes[0].set_title(
        "LDSR-S2 — 100 steps",
        fontsize=15
    )

    axes[1].imshow(
        false_10
    )

    axes[1].set_title(
        "LDSR-S2 — 10 steps",
        fontsize=15
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    path = (
        QUALITATIVE_DIR
        / "ldsr_100_vs_10_false_color.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 10. INDIVIDUAL BAND COMPARISON
    # ========================================================

    print(
        "Creating individual band comparison..."
    )

    band_names = [
        "B02",
        "B03",
        "B04",
        "B08",
    ]

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 18)
    )

    for band_index, band_name in enumerate(
        band_names
    ):

        # --------------------------------------------
        # Normalize each band independently
        # --------------------------------------------

        original = percentile_normalize(
            input_data[band_index]
        )

        image_100 = percentile_normalize(
            ldsr_100[band_index]
        )

        image_10 = percentile_normalize(
            ldsr_10[band_index]
        )

        axes[
            band_index,
            0
        ].imshow(
            original
        )

        axes[
            band_index,
            0
        ].set_title(
            f"{band_name} — Original 10m"
        )

        axes[
            band_index,
            1
        ].imshow(
            image_100
        )

        axes[
            band_index,
            1
        ].set_title(
            f"{band_name} — 100 steps"
        )

        axes[
            band_index,
            2
        ].imshow(
            image_10
        )

        axes[
            band_index,
            2
        ].set_title(
            f"{band_name} — 10 steps"
        )

        for column in range(3):

            axes[
                band_index,
                column
            ].axis(
                "off"
            )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "band_comparison_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 11. 100 VS 10 DIFFERENCE
    # ========================================================

    print(
        "Calculating 100-step vs 10-step difference..."
    )

    difference = (
        np.abs(
            ldsr_100
            - ldsr_10
        )
    )

    mean_difference = (
        difference.mean()
    )

    max_difference = (
        difference.max()
    )

    rmse_difference = np.sqrt(
        np.mean(
            (
                ldsr_100
                - ldsr_10
            ) ** 2
        )
    )

    print(
        f"Mean absolute difference: "
        f"{mean_difference:.8f}"
    )

    print(
        f"RMSE difference: "
        f"{rmse_difference:.8f}"
    )

    print(
        f"Maximum difference: "
        f"{max_difference:.8f}"
    )

    print()

    # --------------------------------------------------------
    # RGB difference
    # --------------------------------------------------------

    rgb_difference = np.mean(
        np.abs(
            ldsr_100
            - ldsr_10
        ),
        axis=0
    )

    rgb_difference_display = (
        percentile_normalize(
            rgb_difference,
            low=1,
            high=99
        )
    )

    save_image(
        rgb_difference_display,
        QUALITATIVE_DIR
        / "ldsr_100_vs_10_difference.png",
        title=(
            "Absolute difference — "
            "LDSR-S2 100 vs 10 steps"
        ),
        cmap="gray"
    )

    # ========================================================
    # 12. DIFFERENCE PER BAND
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    for i, band_name in enumerate(
        band_names
    ):

        diff = np.abs(
            ldsr_100[i]
            - ldsr_10[i]
        )

        display = percentile_normalize(
            diff,
            low=1,
            high=99
        )

        axes[i].imshow(
            display,
            cmap="gray"
        )

        axes[i].set_title(
            band_name
        )

        axes[i].axis(
            "off"
        )

    plt.suptitle(
        "LDSR-S2 100-step vs 10-step absolute difference",
        fontsize=15
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "band_difference_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 13. SPECTRAL STATISTICS
    # ========================================================

    print(
        "Calculating spectral statistics..."
    )

    input_mean = []
    mean_100 = []
    mean_10 = []

    input_std = []
    std_100 = []
    std_10 = []

    for i in range(4):

        input_mean.append(
            float(
                input_data[i].mean()
            )
        )

        mean_100.append(
            float(
                ldsr_100[i].mean()
            )
        )

        mean_10.append(
            float(
                ldsr_10[i].mean()
            )
        )

        input_std.append(
            float(
                input_data[i].std()
            )
        )

        std_100.append(
            float(
                ldsr_100[i].std()
            )
        )

        std_10.append(
            float(
                ldsr_10[i].std()
            )
        )

    # ========================================================
    # 14. MEAN COMPARISON
    # ========================================================

    x = np.arange(
        len(band_names)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width,
        input_mean,
        width,
        label="Original 10m"
    )

    ax.bar(
        x,
        mean_100,
        width,
        label="LDSR-S2 100 steps"
    )

    ax.bar(
        x + width,
        mean_10,
        width,
        label="LDSR-S2 10 steps"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        band_names
    )

    ax.set_ylabel(
        "Mean reflectance"
    )

    ax.set_title(
        "Mean spectral reflectance"
    )

    ax.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "spectral_mean_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 15. STD COMPARISON
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width,
        input_std,
        width,
        label="Original 10m"
    )

    ax.bar(
        x,
        std_100,
        width,
        label="LDSR-S2 100 steps"
    )

    ax.bar(
        x + width,
        std_10,
        width,
        label="LDSR-S2 10 steps"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        band_names
    )

    ax.set_ylabel(
        "Standard deviation"
    )

    ax.set_title(
        "Spectral standard deviation"
    )

    ax.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "spectral_std_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 16. SPATIAL DETAIL COMPARISON
    # ========================================================

    print(
        "Calculating spatial detail..."
    )

    def gradient_strength(image):

        gx = np.diff(
            image,
            axis=1
        )

        gy = np.diff(
            image,
            axis=0
        )

        return float(
            np.mean(
                np.sqrt(
                    gx[:-1] ** 2
                    +
                    gy[:, :-1] ** 2
                )
            )
        )

    spatial_100 = []
    spatial_10 = []

    for i in range(4):

        spatial_100.append(
            gradient_strength(
                ldsr_100[i]
            )
        )

        spatial_10.append(
            gradient_strength(
                ldsr_10[i]
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        spatial_100,
        width,
        label="100 steps"
    )

    ax.bar(
        x + width / 2,
        spatial_10,
        width,
        label="10 steps"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        band_names
    )

    ax.set_ylabel(
        "Mean gradient magnitude"
    )

    ax.set_title(
        "Spatial detail comparison"
    )

    ax.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "spatial_detail_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 17. ZOOMED RGB
    # ========================================================

    print(
        "Creating RGB zoom..."
    )

    # Central region of image.
    # Same geographic area, displayed at different
    # native pixel dimensions.

    crop_start = 768
    crop_end = 1280

    rgb_100_zoom = rgb_100[
        crop_start:crop_end,
        crop_start:crop_end
    ]

    # For the original 10m image, approximately
    # corresponding central region.

    input_start = 192
    input_end = 320

    input_rgb_zoom = input_rgb[
        input_start:input_end,
        input_start:input_end
    ]

    rgb_10_zoom = rgb_10[
        crop_start:crop_end,
        crop_start:crop_end
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(
        input_rgb_zoom
    )

    axes[0].set_title(
        "Original 10m"
    )

    axes[1].imshow(
        rgb_100_zoom
    )

    axes[1].set_title(
        "LDSR-S2 100 steps — 2.5m"
    )

    axes[2].imshow(
        rgb_10_zoom
    )

    axes[2].set_title(
        "LDSR-S2 10 steps — 2.5m"
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    path = (
        QUALITATIVE_DIR
        / "rgb_zoom_100_vs_10.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )

    # ========================================================
    # 18. FINAL CSV
    # ========================================================

    csv_path = (
        FIGURES_DIR
        / "ldsr_100_vs_10_statistics.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "band,"
            "input_mean,"
            "ldsr_100_mean,"
            "ldsr_10_mean,"
            "input_std,"
            "ldsr_100_std,"
            "ldsr_10_std,"
            "ldsr_100_spatial_detail,"
            "ldsr_10_spatial_detail\n"
        )

        for i, band_name in enumerate(
            band_names
        ):

            f.write(
                f"{band_name},"
                f"{input_mean[i]:.10f},"
                f"{mean_100[i]:.10f},"
                f"{mean_10[i]:.10f},"
                f"{input_std[i]:.10f},"
                f"{std_100[i]:.10f},"
                f"{std_10[i]:.10f},"
                f"{spatial_100[i]:.10f},"
                f"{spatial_10[i]:.10f}\n"
            )

        f.write("\n")

        f.write(
            "overall_mean_absolute_difference,"
            f"{mean_difference:.10f}\n"
        )

        f.write(
            "overall_rmse_difference,"
            f"{rmse_difference:.10f}\n"
        )

        f.write(
            "overall_maximum_difference,"
            f"{max_difference:.10f}\n"
        )

    print(
        "Saved:",
        csv_path
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        "Figures:"
    )

    print(
        f"  {FIGURES_DIR}"
    )

    print()

    print(
        "Qualitative images:"
    )

    print(
        f"  {QUALITATIVE_DIR}"
    )

    print()

    print(
        "100-step vs 10-step:"
    )

    print(
        f"  Mean absolute difference: "
        f"{mean_difference:.8f}"
    )

    print(
        f"  RMSE: "
        f"{rmse_difference:.8f}"
    )

    print(
        f"  Maximum difference: "
        f"{max_difference:.8f}"
    )

    print()

    print(
        "The visualization normalization is "
        "for display only."
    )

    print(
        "The underlying GeoTIFF reflectance values "
        "are not modified."
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
