from io import StringIO
from pathlib import Path
import os

import requests
import torch
from omegaconf import OmegaConf
import opensr_model
import opensr_utils


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT = Path(
    "dataset/plan2/geotiff/train_B02_B03_B04_B08_10m.tif"
)

OUTPUT_DIR = Path("dataset/plan2/ldsr_s2_geotiff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_URL = (
    "https://raw.githubusercontent.com/ESAOpenSR/"
    "opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Checks
# ---------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(f"Input not found: {INPUT}")

print("=" * 70)
print("PixelSight — LDSR-S2 GeoTIFF Inference")
print("=" * 70)

print(f"Input : {INPUT}")
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

print()


# ---------------------------------------------------------
# Load official LDSR-S2 configuration
# ---------------------------------------------------------

print("Downloading official LDSR-S2 configuration...")

response = requests.get(CONFIG_URL, timeout=30)
response.raise_for_status()

config = OmegaConf.load(StringIO(response.text))

print("Configuration loaded.")


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

print("Creating LDSR-S2 model...")

model = opensr_model.SRLatentDiffusion(
    config,
    device=DEVICE,
)

print("Model created.")


# ---------------------------------------------------------
# Load pretrained checkpoint
# ---------------------------------------------------------

print("Loading pretrained LDSR-S2 checkpoint...")

model.load_pretrained(config.ckpt_version)

model.eval()

print("Pretrained checkpoint loaded.")
print("Model ready for inference.")
print()


# ---------------------------------------------------------
# OpenSR Utils large-file pipeline
# ---------------------------------------------------------

print("Creating large-file processing job...")
print()
print("Parameters:")
print("  Window       : 128 x 128 LR pixels")
print("  Scale        : 4x")
print("  Overlap      : 12 LR pixels")
print("  Border trim  : 2 LR pixels")
print()

sr_job = opensr_utils.large_file_processing(
    root=str(INPUT),
    model=model,
    window_size=(128, 128),
    factor=4,
    overlap=12,
    eliminate_border_px=2,
    device=DEVICE,
    gpus=0,
)


# ---------------------------------------------------------
# Start inference
# ---------------------------------------------------------

print("Starting LDSR-S2 super-resolution...")
print("WARNING: This processes the complete 10980 x 10980 scene.")
print()

result = sr_job.start_super_resolution()

print()
print("=" * 70)
print("LDSR-S2 processing finished.")
print("=" * 70)

print("Result:")
print(result)