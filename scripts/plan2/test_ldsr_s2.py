from io import StringIO
from pathlib import Path

import numpy as np
import requests
import torch
from omegaconf import OmegaConf

import opensr_model


PATCH = Path(
    "dataset/plan2/patches/train/patch_00000.npz"
)

OUTPUT = Path(
    "dataset/plan2/ldsr_test"
)

CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/"
    "refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)


def main():
    print("=== PixelSight LDSR-S2 Test ===")

    # ---------------------------------------------------------
    # 1. Device
    # ---------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # 2. Download/load official LDSR-S2 config
    # ---------------------------------------------------------

    print("Downloading official LDSR-S2 configuration...")

    response = requests.get(
        CONFIG_URL,
        timeout=30,
    )

    response.raise_for_status()

    config = OmegaConf.load(
        StringIO(response.text)
    )

    print("Configuration loaded.")

    # ---------------------------------------------------------
    # 3. Create official pretrained model
    # ---------------------------------------------------------

    print("Creating LDSR-S2 model...")

    model = opensr_model.SRLatentDiffusion(
        config,
        device=device,
    )

    print("Model created.")

    # ---------------------------------------------------------
    # 4. Load official pretrained checkpoint
    # ---------------------------------------------------------

    print(
        f"Loading pretrained checkpoint: "
        f"{config.ckpt_version}"
    )

    model.load_pretrained(
        config.ckpt_version
    )

    print("Pretrained checkpoint loaded.")

    if model.training:
        raise RuntimeError(
            "LDSR-S2 model is unexpectedly "
            "in training mode."
        )

    print("Model is in evaluation mode.")

    # ---------------------------------------------------------
    # 5. Load PixelSight 128x128 native patch
    # ---------------------------------------------------------

    if not PATCH.exists():
        raise FileNotFoundError(
            f"Patch not found: {PATCH}"
        )

    with np.load(PATCH) as data:
        image = data["image"].astype(
            np.float32
        )

        mask = data["mask"].astype(bool)

    print(f"Input patch: {PATCH}")
    print(f"Input shape: {image.shape}")
    print(
        f"Input range: "
        f"{image.min():.6f} -> "
        f"{image.max():.6f}"
    )
    print(
        f"Valid pixels: "
        f"{mask.mean() * 100:.2f}%"
    )

    if image.shape != (128, 128, 4):
        raise RuntimeError(
            f"Expected (128,128,4), "
            f"got {image.shape}"
        )

    # ---------------------------------------------------------
    # 6. Convert HWC -> BCHW
    # ---------------------------------------------------------

    tensor = torch.from_numpy(
        image
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=torch.float32,
    )

    print(
        f"Model input: "
        f"{tuple(tensor.shape)}"
    )

    # ---------------------------------------------------------
    # 7. Run LDSR-S2
    # ---------------------------------------------------------

    print(
        "Running LDSR-S2 "
        "(100 sampling steps)..."
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        sr = model.forward(
            tensor,
            sampling_steps=100,
        )

    # ---------------------------------------------------------
    # 8. Inspect output
    # ---------------------------------------------------------

    print(
        f"Output shape: "
        f"{tuple(sr.shape)}"
    )

    print(
        f"Output range: "
        f"{sr.min().item():.6f} -> "
        f"{sr.max().item():.6f}"
    )

    if device == "cuda":

        memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory: "
            f"{memory_gb:.2f} GB"
        )

    expected_shape = (
        1,
        4,
        512,
        512,
    )

    if tuple(sr.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SR shape: "
            f"{tuple(sr.shape)}"
        )

    # ---------------------------------------------------------
    # 9. Save result
    # ---------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sr_numpy = (
        sr.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    np.save(
        OUTPUT / "patch_00000_sr.npy",
        sr_numpy,
    )

    np.save(
        OUTPUT / "patch_00000_mask.npy",
        mask,
    )

    print(
        f"Saved SR result to: "
        f"{OUTPUT / 'patch_00000_sr.npy'}"
    )

    print(
        "\nLDSR-S2 single-patch test PASSED."
    )


if __name__ == "__main__":
    main()