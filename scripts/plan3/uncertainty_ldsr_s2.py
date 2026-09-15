from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import opensr_model


# ============================================================
# PixelSight Plan 3
# LDSR-S2 Stochastic Uncertainty Estimation
# ============================================================

INPUT_PATH = Path(
    "dataset/plan2/patches/train/patch_00000.npz"
)

OUTPUT_DIR = Path("results/plan3")

N_VARIATIONS = 5
SAMPLING_STEPS = 100
SCALE = 4


# ============================================================
# Model
# ============================================================

def load_model(device):
    print("\n=== Loading LDSR-S2 ===")

    config_path = (
        Path(opensr_model.__file__).parent
        / "configs"
        / "config_10m.yaml"
    )

    checkpoint_path = (
        Path.cwd()
        / "opensr-ldsrs2_v1_0_0.ckpt"
    )

    print(f"Local config     : {config_path}")
    print(f"Local checkpoint : {checkpoint_path}")

    if not config_path.exists():
        raise FileNotFoundError(
            f"LDSR-S2 config not found: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"LDSR-S2 checkpoint not found: {checkpoint_path}"
        )

    config = OmegaConf.load(config_path)

    print("Creating LDSR-S2 model...")

    model = opensr_model.SRLatentDiffusion(
        config,
        device=device,
    )

    print("Loading pretrained checkpoint...")

    model.load_pretrained(str(checkpoint_path))

    model.eval()

    print("LDSR-S2 model loaded successfully.")

    return model


# ============================================================
# Random seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Single stochastic inference
# ============================================================

@torch.no_grad()
def run_single(model, image, seed, device):
    set_seed(seed)

    # HWC -> BCHW
    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)

    start = time.perf_counter()

    sr = model.forward(
        tensor,
        sampling_steps=SAMPLING_STEPS,
    )

    elapsed = time.perf_counter() - start

    # BCHW -> HWC
    sr = (
        sr.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
        .astype(np.float32)
    )

    return sr, elapsed


# ============================================================
# Save uncertainty heatmap
# ============================================================

def save_heatmap(uncertainty_map, path):
    plt.figure(figsize=(8, 8))

    plt.imshow(uncertainty_map)

    plt.colorbar(
        label="Mean pixel-wise uncertainty"
    )

    plt.title(
        "PixelSight Plan 3 - LDSR-S2 Uncertainty"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("PixelSight Plan 3")
    print("LDSR-S2 Stochastic Uncertainty Estimation")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device          : {device}")

    if device.type == "cuda":
        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Input patch     : {INPUT_PATH}")
    print("Patch size      : 128 × 128")
    print(f"Scale           : {SCALE}×")
    print(
        f"Sampling steps  : {SAMPLING_STEPS}"
    )
    print(f"N variations    : {N_VARIATIONS}")

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input patch not found: {INPUT_PATH}"
        )

    data = np.load(INPUT_PATH)

    print("\nInput fields:", data.files)

    image = data["image"].astype(np.float32)
    mask = data["mask"]

    print(
        f"Input image shape: {image.shape}"
    )

    print(
        f"Input range: "
        f"{image.min():.6f} → "
        f"{image.max():.6f}"
    )

    print(
        f"Valid pixels: "
        f"{int(mask.sum())} / {mask.size}"
    )

    if image.shape != (128, 128, 4):
        raise ValueError(
            f"Expected (128,128,4), got {image.shape}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Stochastic variations
    # --------------------------------------------------------

    samples = []
    timings = []

    print("\n=== Running stochastic variations ===")

    for i in range(N_VARIATIONS):

        seed = i

        print(
            f"\nVariation {i + 1}/{N_VARIATIONS}"
            f"  seed={seed}"
        )

        sr, elapsed = run_single(
            model=model,
            image=image,
            seed=seed,
            device=device,
        )

        print(
            f"Runtime : {elapsed:.3f} seconds"
        )

        print(
            f"Shape   : {sr.shape}"
        )

        print(
            f"Range   : "
            f"{sr.min():.6f} → "
            f"{sr.max():.6f}"
        )

        print(
            f"Mean    : {sr.mean():.6f}"
        )

        print(
            f"Std     : {sr.std():.6f}"
        )

        sample_path = (
            OUTPUT_DIR
            / f"sample_seed_{seed:03d}.npy"
        )

        np.save(sample_path, sr)

        print(
            f"Saved   : {sample_path}"
        )

        samples.append(sr)
        timings.append(elapsed)

    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    samples = np.stack(
        samples,
        axis=0,
    )

    print("\n=== Stacked samples ===")

    print(
        f"Samples shape: {samples.shape}"
    )

    # --------------------------------------------------------
    # Pixel-wise mean
    # --------------------------------------------------------

    mean_sr = np.mean(
        samples,
        axis=0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Pixel-wise standard deviation
    # --------------------------------------------------------

    std_sr = np.std(
        samples,
        axis=0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Average channels -> single uncertainty heatmap
    # --------------------------------------------------------

    uncertainty_map = np.mean(
        std_sr,
        axis=2,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    mean_path = OUTPUT_DIR / "mean_sr.npy"
    std_path = OUTPUT_DIR / "std_sr.npy"
    uncertainty_path = (
        OUTPUT_DIR / "uncertainty_map.npy"
    )
    heatmap_path = (
        OUTPUT_DIR / "uncertainty_heatmap.png"
    )

    np.save(mean_path, mean_sr)
    np.save(std_path, std_sr)
    np.save(uncertainty_path, uncertainty_map)

    save_heatmap(
        uncertainty_map,
        heatmap_path,
    )

    print("\n=== P3 outputs ===")

    print(
        f"Mean SR       : {mean_path}"
    )

    print(
        f"Std SR        : {std_path}"
    )

    print(
        f"Uncertainty   : {uncertainty_path}"
    )

    print(
        f"Heatmap       : {heatmap_path}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n=== Uncertainty statistics ===")

    print(
        f"Mean uncertainty : "
        f"{uncertainty_map.mean():.8f}"
    )

    print(
        f"Std uncertainty  : "
        f"{uncertainty_map.std():.8f}"
    )

    print(
        f"Minimum          : "
        f"{uncertainty_map.min():.8f}"
    )

    print(
        f"Maximum          : "
        f"{uncertainty_map.max():.8f}"
    )

    print(
        f"Median           : "
        f"{np.median(uncertainty_map):.8f}"
    )

    # --------------------------------------------------------
    # Check stochasticity
    # --------------------------------------------------------

    print("\n=== Stochasticity check ===")

    max_pairwise_difference = 0.0

    for i in range(N_VARIATIONS):
        for j in range(i + 1, N_VARIATIONS):

            difference = np.max(
                np.abs(
                    samples[i] - samples[j]
                )
            )

            print(
                f"seed {i} vs seed {j}: "
                f"max abs difference = "
                f"{difference:.10f}"
            )

            max_pairwise_difference = max(
                max_pairwise_difference,
                float(difference),
            )

    if max_pairwise_difference == 0.0:
        print(
            "\nWARNING: All stochastic outputs "
            "are identical."
        )
    else:
        print(
            "\nSUCCESS: LDSR-S2 outputs vary "
            "across random seeds."
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    timing_path = (
        OUTPUT_DIR / "timing_summary.csv"
    )

    with open(
        timing_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "seed",
                "time_seconds",
            ]
        )

        for seed, elapsed in zip(
            range(N_VARIATIONS),
            timings,
        ):
            writer.writerow(
                [
                    seed,
                    f"{elapsed:.6f}",
                ]
            )

        writer.writerow(
            [
                "mean",
                f"{np.mean(timings):.6f}",
            ]
        )

        writer.writerow(
            [
                "total",
                f"{np.sum(timings):.6f}",
            ]
        )

    print(
        f"\nTiming CSV: {timing_path}"
    )

    print("\n" + "=" * 65)
    print("PLAN 3 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()