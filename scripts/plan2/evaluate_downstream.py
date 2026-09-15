# ============================================================
# PixelSight Plan 2
# Downstream Geospatial Evaluation
#
# Comparison:
#
#   Baseline:
#       Sentinel-2 10 m
#           ↓
#       Bicubic 4x upsampling
#           ↓
#       ~2.5 m
#
#   Proposed:
#       Sentinel-2 10 m
#           ↓
#       LDSR-S2 4x
#           ↓
#       ~2.5 m
#
# Labels:
#       Sentinel-2 SCL classes
#
# Metrics:
#       IoU
#       F1
#
# IMPORTANT:
# SCL is a proxy label source. It is NOT genuine 2.5 m
# ground-truth segmentation.
# ============================================================


# ============================================================
# IMPORTANT: Use a non-GUI Matplotlib backend.
# This prevents Tkinter "main thread is not in main loop"
# errors on Windows.
# ============================================================

import matplotlib

matplotlib.use("Agg")


# ============================================================
# Imports
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    jaccard_score,
)

import matplotlib.pyplot as plt


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# INPUT FILES
# ============================================================

INPUT_10M = (
    ROOT
    / "dataset"
    / "plan2"
    / "geotiff"
    / "train_test_512_10m.tif"
)


SR_FILE = (
    ROOT
    / "dataset"
    / "plan2"
    / "geotiff"
    / "sr.tif"
)


SCL_FILE = (
    ROOT
    / "dataset"
    / "plan2"
    / "sentinel2"
    / "train"
    / "T43PHM_20260303T050649_SCL_20m.jp2"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "plan2"
    / "downstream"
)


METRICS_FILE = (
    OUTPUT_DIR
    / "downstream_metrics.csv"
)


BASELINE_MAP = (
    OUTPUT_DIR
    / "baseline_classification.png"
)


SR_MAP = (
    OUTPUT_DIR
    / "ldsr_classification.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42


# Maximum training pixels.
# This keeps the SIH experiment reasonably fast.
MAX_TRAIN_SAMPLES = 50000


# Sentinel-2 SCL classes:
#
# 2 = Dark Area Pixels
# 4 = Vegetation
# 5 = Bare Soils
# 6 = Water
#
# Other classes are excluded from this proxy experiment.

VALID_CLASSES = [
    2,
    4,
    5,
    6,
]


CLASS_NAMES = {
    2: "Dark Area Pixels",
    4: "Vegetation",
    5: "Bare Soil",
    6: "Water",
}


# ============================================================
# FILE CHECK
# ============================================================

def check_files():

    print()
    print("Checking required files...")

    required_files = [
        INPUT_10M,
        SR_FILE,
        SCL_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired file not found:\n"
                f"{path}"
            )

        print(
            f"Found: {path}"
        )

    print(
        "All required files found."
    )


# ============================================================
# LOAD 10 m IMAGE
# ============================================================

def load_10m_image():

    print()
    print(
        "Loading 10 m Sentinel-2 image..."
    )

    with rasterio.open(
        INPUT_10M
    ) as src:

        image = src.read()

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        bounds = src.bounds

        resolution = src.res

    # Rasterio gives:
    #
    # C x H x W
    #
    # Convert to:
    #
    # H x W x C

    image = np.moveaxis(
        image,
        0,
        -1,
    )

    print(
        f"10 m image shape: "
        f"{image.shape}"
    )

    print(
        f"Resolution: "
        f"{resolution}"
    )

    print(
        f"CRS: "
        f"{crs}"
    )

    print(
        f"Bounds: "
        f"{bounds}"
    )

    return (
        image,
        profile,
        transform,
        crs,
        bounds,
    )


# ============================================================
# LOAD LDSR-S2 IMAGE
# ============================================================

def load_sr_image():

    print()
    print(
        "Loading LDSR-S2 image..."
    )

    with rasterio.open(
        SR_FILE
    ) as src:

        sr = src.read()

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        bounds = src.bounds

        resolution = src.res

    sr = np.moveaxis(
        sr,
        0,
        -1,
    )

    print(
        f"LDSR-S2 shape: "
        f"{sr.shape}"
    )

    print(
        f"Resolution: "
        f"{resolution}"
    )

    print(
        f"CRS: "
        f"{crs}"
    )

    print(
        f"Bounds: "
        f"{bounds}"
    )

    return (
        sr,
        profile,
        transform,
        crs,
        bounds,
    )


# ============================================================
# NORMALIZE REFLECTANCE
# ============================================================

def normalize_reflectance(
    image
):

    image = image.astype(
        np.float32
    )

    # Sentinel-2 reflectance is commonly
    # represented using a 10000 scale.
    #
    # If values already look like [0,1],
    # do not divide again.

    if image.max() > 2.0:

        image = (
            image / 10000.0
        )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


# ============================================================
# BICUBIC BASELINE
# ============================================================

def upsample_baseline(
    image
):

    print()
    print(
        "Creating bicubic baseline..."
    )

    with rasterio.open(
        INPUT_10M
    ) as src:

        baseline = src.read(
            out_shape=(
                src.count,
                src.height * 4,
                src.width * 4,
            ),
            resampling=Resampling.cubic,
        )

    baseline = np.moveaxis(
        baseline,
        0,
        -1,
    )

    print(
        f"Bicubic baseline shape: "
        f"{baseline.shape}"
    )

    baseline = (
        normalize_reflectance(
            baseline
        )
    )

    return baseline


# ============================================================
# LOAD SCL LABELS
# ============================================================

def load_scl_labels(
    target_shape,
    target_transform,
    target_crs,
):

    print()
    print(
        "Loading SCL proxy labels..."
    )

    height, width = (
        target_shape
    )

    labels = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    with rasterio.open(
        SCL_FILE
    ) as src:

        # SCL is categorical data.
        #
        # Therefore nearest-neighbour
        # resampling must be used.

        reproject(
            source=rasterio.band(
                src,
                1,
            ),
            destination=labels,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

    print(
        "SCL labels reprojected to:"
    )

    print(
        labels.shape
    )

    unique, counts = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    print()
    print(
        "SCL class distribution:"
    )

    for cls, count in zip(
        unique,
        counts,
    ):

        name = CLASS_NAMES.get(
            int(cls),
            "Excluded / Other",
        )

        print(
            f"  Class {int(cls)} "
            f"({name}): "
            f"{int(count):,} pixels"
        )

    return labels


# ============================================================
# CREATE VALID MASK
# ============================================================

def create_valid_mask(
    labels,
    evaluation_classes=None,
):

    if evaluation_classes is None:

        evaluation_classes = (
            VALID_CLASSES
        )

    return np.isin(
        labels,
        evaluation_classes,
    )


# ============================================================
# DETECT CLASSES ACTUALLY PRESENT
# ============================================================

def get_present_classes(
    labels
):

    present = []

    print()
    print(
        "Checking evaluation classes..."
    )

    for cls in VALID_CLASSES:

        count = int(
            np.sum(
                labels == cls
            )
        )

        if count > 0:

            present.append(
                cls
            )

            print(
                f"  Class {cls} "
                f"({CLASS_NAMES[cls]}): "
                f"AVAILABLE "
                f"({count:,} pixels)"
            )

        else:

            print(
                f"  Class {cls} "
                f"({CLASS_NAMES[cls]}): "
                f"ABSENT — excluded"
            )

    if len(present) == 0:

        raise RuntimeError(
            "None of the configured SCL "
            "classes are present."
        )

    print()
    print(
        "Classes used for evaluation:"
    )

    print(
        present
    )

    return present


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(
    image
):

    image = image.astype(
        np.float32
    )

    output = np.zeros_like(
        image
    )

    for band in range(
        image.shape[-1]
    ):

        values = (
            image[:, :, band]
        )

        low = np.percentile(
            values,
            2,
        )

        high = np.percentile(
            values,
            98,
        )

        if high > low:

            output[
                :,
                :,
                band
            ] = (
                values - low
            ) / (
                high - low
            )

        else:

            output[
                :,
                :,
                band
            ] = 0.0

    return np.clip(
        output,
        0.0,
        1.0,
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(
    labels,
    valid_mask,
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    valid_indices = (
        np.flatnonzero(
            valid_mask.ravel()
        )
    )

    if len(valid_indices) < 100:

        raise RuntimeError(
            "Not enough valid pixels "
            "for evaluation."
        )

    rng.shuffle(
        valid_indices
    )

    # 20% training
    # 80% testing

    split = int(
        len(valid_indices)
        * 0.20
    )

    train_indices = (
        valid_indices[:split]
    )

    test_indices = (
        valid_indices[split:]
    )

    # Limit training size
    # for speed.

    if (
        len(train_indices)
        > MAX_TRAIN_SAMPLES
    ):

        train_indices = (
            rng.choice(
                train_indices,
                size=MAX_TRAIN_SAMPLES,
                replace=False,
            )
        )

    print()
    print(
        f"Training pixels: "
        f"{len(train_indices):,}"
    )

    print(
        f"Testing pixels : "
        f"{len(test_indices):,}"
    )

    return (
        train_indices,
        test_indices,
    )


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(
    image,
    labels,
    train_indices,
    test_indices,
    evaluation_classes,
    name,
):

    print()
    print(
        "=" * 60
    )

    print(
        f"Evaluating: {name}"
    )

    print(
        "=" * 60
    )

    height, width, bands = (
        image.shape
    )

    features = image.reshape(
        -1,
        bands,
    )

    label_vector = (
        labels.ravel()
    )

    X_train = (
        features[
            train_indices
        ]
    )

    y_train = (
        label_vector[
            train_indices
        ]
    )

    X_test = (
        features[
            test_indices
        ]
    )

    y_test = (
        label_vector[
            test_indices
        ]
    )

    print(
        f"Training Random Forest "
        f"on {len(X_train):,} pixels..."
    )

    classifier = (
        RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print(
        "Predicting test pixels..."
    )

    y_pred = classifier.predict(
        X_test
    )

    # --------------------------------------------------------
    # Macro IoU
    # --------------------------------------------------------

    iou = jaccard_score(
        y_test,
        y_pred,
        labels=evaluation_classes,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        labels=evaluation_classes,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Per-class IoU
    # --------------------------------------------------------

    per_class_iou = (
        jaccard_score(
            y_test,
            y_pred,
            labels=evaluation_classes,
            average=None,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Per-class F1
    # --------------------------------------------------------

    per_class_f1 = (
        f1_score(
            y_test,
            y_pred,
            labels=evaluation_classes,
            average=None,
            zero_division=0,
        )
    )

    print()
    print(
        f"{name}"
    )

    print(
        f"Macro IoU: "
        f"{iou:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{f1:.4f}"
    )

    print()
    print(
        "Per-class metrics:"
    )

    for (
        cls,
        cls_iou,
        cls_f1,
    ) in zip(
        evaluation_classes,
        per_class_iou,
        per_class_f1,
    ):

        print(
            f"  Class {cls} "
            f"({CLASS_NAMES[cls]}): "
            f"IoU={cls_iou:.4f}, "
            f"F1={cls_f1:.4f}"
        )

    result = {
        "Method": name,
        "IoU": float(iou),
        "F1": float(f1),
    }

    # Add per-class metrics dynamically.
    #
    # This means absent classes will not
    # appear as artificial zero values.

    for (
        cls,
        cls_iou,
        cls_f1,
    ) in zip(
        evaluation_classes,
        per_class_iou,
        per_class_f1,
    ):

        result[
            f"Class_{cls}_IoU"
        ] = float(cls_iou)

        result[
            f"Class_{cls}_F1"
        ] = float(cls_f1)

    return result


# ============================================================
# CREATE CLASSIFICATION MAP
# ============================================================

def create_classification_map(
    image,
    labels,
    evaluation_classes,
    output_path,
    title,
):

    print()
    print(
        "Creating classification map:"
    )

    print(
        f"  {title}"
    )

    height, width, bands = (
        image.shape
    )

    valid_mask = (
        create_valid_mask(
            labels,
            evaluation_classes,
        )
    )

    features = image.reshape(
        -1,
        bands,
    )

    valid_indices = (
        np.flatnonzero(
            valid_mask.ravel()
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_size = min(
        30000,
        len(valid_indices),
    )

    train_indices = (
        rng.choice(
            valid_indices,
            size=sample_size,
            replace=False,
        )
    )

    classifier = (
        RandomForestClassifier(
            n_estimators=60,
            max_depth=16,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
    )

    classifier.fit(
        features[
            train_indices
        ],
        labels.ravel()[
            train_indices
        ],
    )

    predictions = np.zeros(
        height * width,
        dtype=np.uint8,
    )

    predictions[
        valid_indices
    ] = classifier.predict(
        features[
            valid_indices
        ]
    )

    predictions = (
        predictions.reshape(
            height,
            width,
        )
    )

    # --------------------------------------------------------
    # Plot without opening a GUI.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 10)
    )

    ax = fig.add_subplot(
        111
    )

    ax.imshow(
        predictions,
        interpolation="nearest",
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved classification map:"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "PixelSight Plan 2"
    )

    print(
        "Downstream Geospatial Evaluation"
    )

    print(
        "=" * 60
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "SCL labels are being used as "
        "proxy downstream labels."
    )

    print(
        "This is NOT a genuine 2.5 m "
        "ground-truth benchmark."
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load 10 m Sentinel-2
    # --------------------------------------------------------

    (
        image_10m,
        _,
        transform_10m,
        crs_10m,
        bounds_10m,
    ) = load_10m_image()

    # --------------------------------------------------------
    # Load LDSR-S2
    # --------------------------------------------------------

    (
        sr_image,
        _,
        sr_transform,
        sr_crs,
        sr_bounds,
    ) = load_sr_image()

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    expected_height = (
        image_10m.shape[0]
        * 4
    )

    expected_width = (
        image_10m.shape[1]
        * 4
    )

    if (
        sr_image.shape[0]
        != expected_height
    ):

        raise RuntimeError(
            "LDSR height does not match "
            "expected 4x scale.\n"
            f"Expected: {expected_height}\n"
            f"Actual: {sr_image.shape[0]}"
        )

    if (
        sr_image.shape[1]
        != expected_width
    ):

        raise RuntimeError(
            "LDSR width does not match "
            "expected 4x scale.\n"
            f"Expected: {expected_width}\n"
            f"Actual: {sr_image.shape[1]}"
        )

    # --------------------------------------------------------
    # CRS check
    # --------------------------------------------------------

    if crs_10m != sr_crs:

        raise RuntimeError(
            "CRS mismatch between "
            "10 m input and LDSR output.\n"
            f"10 m CRS: {crs_10m}\n"
            f"SR CRS: {sr_crs}"
        )

    # --------------------------------------------------------
    # Spatial bounds
    # --------------------------------------------------------

    print()
    print(
        "Spatial alignment check:"
    )

    print(
        f"10 m bounds: "
        f"{bounds_10m}"
    )

    print(
        f"SR bounds : "
        f"{sr_bounds}"
    )

    # --------------------------------------------------------
    # Create bicubic baseline
    # --------------------------------------------------------

    baseline = (
        upsample_baseline(
            image_10m
        )
    )

    # --------------------------------------------------------
    # Normalize LDSR image
    # --------------------------------------------------------

    sr_image = (
        normalize_reflectance(
            sr_image
        )
    )

    # --------------------------------------------------------
    # Normalize features
    # --------------------------------------------------------

    print()
    print(
        "Normalizing baseline features..."
    )

    baseline_features = (
        normalize_features(
            baseline
        )
    )

    print(
        "Normalizing LDSR-S2 features..."
    )

    sr_features = (
        normalize_features(
            sr_image
        )
    )

    # --------------------------------------------------------
    # Load SCL
    # --------------------------------------------------------

    labels = (
        load_scl_labels(
            target_shape=(
                sr_features.shape[:2]
            ),
            target_transform=(
                sr_transform
            ),
            target_crs=(
                sr_crs
            ),
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Automatically detect which requested
    # classes actually exist.
    # --------------------------------------------------------

    evaluation_classes = (
        get_present_classes(
            labels
        )
    )

    # --------------------------------------------------------
    # Valid evaluation mask
    # --------------------------------------------------------

    valid_mask = (
        create_valid_mask(
            labels,
            evaluation_classes,
        )
    )

    valid_pixel_count = int(
        valid_mask.sum()
    )

    print()
    print(
        f"Valid evaluation pixels: "
        f"{valid_pixel_count:,}"
    )

    # --------------------------------------------------------
    # Same train/test pixels for both methods
    # --------------------------------------------------------

    (
        train_indices,
        test_indices,
    ) = create_train_test_split(
        labels,
        valid_mask,
    )

    # --------------------------------------------------------
    # Evaluate baseline
    # --------------------------------------------------------

    baseline_result = (
        train_and_evaluate(
            baseline_features,
            labels,
            train_indices,
            test_indices,
            evaluation_classes,
            "Bicubic Baseline",
        )
    )

    # --------------------------------------------------------
    # Evaluate LDSR-S2
    # --------------------------------------------------------

    sr_result = (
        train_and_evaluate(
            sr_features,
            labels,
            train_indices,
            test_indices,
            evaluation_classes,
            "LDSR-S2",
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            baseline_result,
            sr_result,
        ]
    )

    results.to_csv(
        METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 60
    )

    print()

    print(
        results[
            [
                "Method",
                "IoU",
                "F1",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    baseline_iou = (
        baseline_result["IoU"]
    )

    sr_iou = (
        sr_result["IoU"]
    )

    baseline_f1 = (
        baseline_result["F1"]
    )

    sr_f1 = (
        sr_result["F1"]
    )

    iou_change = (
        sr_iou
        - baseline_iou
    )

    f1_change = (
        sr_f1
        - baseline_f1
    )

    print()

    print(
        f"IoU change: "
        f"{iou_change:+.4f}"
    )

    print(
        f"F1 change : "
        f"{f1_change:+.4f}"
    )

    # Percentage improvement.
    #
    # Only calculate if baseline > 0.

    if baseline_iou > 0:

        iou_percentage = (
            iou_change
            / baseline_iou
        ) * 100.0

        print(
            f"IoU relative change: "
            f"{iou_percentage:+.2f}%"
        )

    if baseline_f1 > 0:

        f1_percentage = (
            f1_change
            / baseline_f1
        ) * 100.0

        print(
            f"F1 relative change: "
            f"{f1_percentage:+.2f}%"
        )

    # --------------------------------------------------------
    # Classification maps
    # --------------------------------------------------------

    create_classification_map(
        baseline_features,
        labels,
        evaluation_classes,
        BASELINE_MAP,
        "PixelSight — Bicubic Baseline Classification",
    )

    create_classification_map(
        sr_features,
        labels,
        evaluation_classes,
        SR_MAP,
        "PixelSight — LDSR-S2 Classification",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "DOWNSTREAM EVALUATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print()
    print(
        "Metrics:"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print()
    print(
        "Baseline classification:"
    )

    print(
        f"  {BASELINE_MAP}"
    )

    print()
    print(
        "LDSR-S2 classification:"
    )

    print(
        f"  {SR_MAP}"
    )

    print()
    print(
        "Classes evaluated:"
    )

    for cls in evaluation_classes:

        print(
            f"  {cls} - "
            f"{CLASS_NAMES[cls]}"
        )

    print()
    print(
        "Note:"
    )

    print(
        "The reported IoU/F1 values are "
        "proxy downstream metrics based "
        "on Sentinel-2 SCL labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()