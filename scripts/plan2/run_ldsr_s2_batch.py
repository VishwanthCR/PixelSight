from io import StringIO
from pathlib import Path
import argparse
import csv
import time

import numpy as np
import requests
import torch
from omegaconf import OmegaConf

import opensr_model


CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/"
    "refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)


def load_model(device):
    print("\n=== Loading LDSR-S2 ===")

    print("Downloading official configuration...")

    response = requests.get(
        CONFIG_URL,
        timeout=30,
    )
    response.raise_for_status()

    config = OmegaConf.load(
        StringIO(response.text)
    )

    print("Configuration loaded.")

    print("Creating LDSR-S2 model...")

    model = opensr_model.SRLatentDiffusion(
        config,
        device=device,
    )

    print("Model created.")

    print(
        f"Loading checkpoint: "
        f"{config.ckpt_version}"
    )

    model.load_pretrained(
        config.ckpt_version
    )

    print("Checkpoint loaded.")

    if model.training:
        raise RuntimeError(
            "LDSR-S2 is unexpectedly in training mode."
        )

    print("Model is in evaluation mode.")

    return model


def process_patch(
    model,
    patch_file,
    output_file,
    device,
    sampling_steps,
):
    with np.load(patch_file) as data:
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(bool)

    if image.shape != (128, 128, 4):
        raise RuntimeError(
            f"Unexpected input shape: {image.shape}"
        )

    tensor = torch.from_numpy(
        image
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=torch.float32,
    )

    start = time.perf_counter()

    with torch.inference_mode():

        sr = model.forward(
            tensor,
            sampling_steps=sampling_steps,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if tuple(sr.shape) != (1, 4, 512, 512):
        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(sr.shape)}"
        )

    sr_numpy = (
        sr.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Save immediately so completed patches can be resumed.
    np.savez_compressed(
        output_file,
        sr=sr_numpy,
        mask=mask,
    )

    return elapsed, sr_numpy


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Maximum number of patches to process.",
    )

    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=== PixelSight LDSR-S2 Batch Inference ===")
    print(f"Split: {args.split}")
    print(f"Device: {device}")

    if device == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    input_dir = Path(
        f"dataset/plan2/patches/{args.split}"
    )

    output_dir = Path(
        f"dataset/plan2/ldsr_s2/{args.split}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_files = sorted(
        input_dir.glob("patch_*.npz")
    )

    print(
        f"Available patches: "
        f"{len(patch_files)}"
    )

    if args.max_patches is not None:
        patch_files = patch_files[
            :args.max_patches
        ]

    print(
        f"Patches selected: "
        f"{len(patch_files)}"
    )

    model = load_model(device)

    log_file = output_dir / "inference_log.csv"

    log_exists = log_file.exists()

    with open(
        log_file,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        if not log_exists:
            writer.writerow([
                "patch",
                "status",
                "time_seconds",
                "sr_min",
                "sr_max",
                "sr_mean",
                "sr_std",
                "error",
            ])

        completed = 0
        skipped = 0
        failed = 0
        total_time = 0.0

        for index, patch_file in enumerate(
            patch_files,
            start=1,
        ):

            patch_name = patch_file.stem

            output_file = (
                output_dir /
                f"{patch_name}_sr.npz"
            )

            # Resume support.
            if output_file.exists():
                skipped += 1

                print(
                    f"[{index}/{len(patch_files)}] "
                    f"{patch_name} — SKIP "
                    f"(already exists)"
                )

                continue

            print(
                f"\n[{index}/{len(patch_files)}] "
                f"Processing {patch_name}..."
            )

            try:

                elapsed, sr = process_patch(
                    model=model,
                    patch_file=patch_file,
                    output_file=output_file,
                    device=device,
                    sampling_steps=args.sampling_steps,
                )

                completed += 1
                total_time += elapsed

                writer.writerow([
                    patch_name,
                    "success",
                    f"{elapsed:.3f}",
                    f"{sr.min():.8f}",
                    f"{sr.max():.8f}",
                    f"{sr.mean():.8f}",
                    f"{sr.std():.8f}",
                    "",
                ])

                f.flush()

                print(
                    f"SUCCESS — "
                    f"{elapsed:.2f} sec"
                )

                print(
                    f"SR range: "
                    f"{sr.min():.6f} -> "
                    f"{sr.max():.6f}"
                )

            except Exception as e:

                failed += 1

                writer.writerow([
                    patch_name,
                    "failed",
                    "",
                    "",
                    "",
                    "",
                    "",
                    str(e),
                ])

                f.flush()

                print(
                    f"FAILED: {e}"
                )

    print("\n=== Batch Complete ===")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    if completed > 0:
        print(
            f"Average inference time: "
            f"{total_time / completed:.2f} sec/patch"
        )

        print(
            f"Estimated time for "
            f"selected patches: "
            f"{total_time / 3600:.2f} hours"
        )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()