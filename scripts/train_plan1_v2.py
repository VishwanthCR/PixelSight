import sys
from pathlib import Path

# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src/
sys.path.insert(0, str(ROOT / "src"))


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pixelsight.models.swinir_v2 import SwinIRResidual


# ============================================================
# Configuration
# ============================================================

# -----------------------------
# Plan 1 geographic split
# -----------------------------
#
# Region 1 = TRAIN
# Region 2 = VALIDATION
# Region 3 = FINAL TEST
#
# Region 3 is deliberately NOT loaded by this script.

TRAIN_LR = (
    ROOT
    / "dataset"
    / "region1"
    / "lr_patches.npy"
)

TRAIN_HR = (
    ROOT
    / "dataset"
    / "region1"
    / "hr_patches.npy"
)

VAL_LR = (
    ROOT
    / "dataset"
    / "region2"
    / "lr_patches.npy"
)

VAL_HR = (
    ROOT
    / "dataset"
    / "region2"
    / "hr_patches.npy"
)


# -----------------------------
# Checkpoint
# -----------------------------

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "swinir_v2_best.pth"
)


# -----------------------------
# Training parameters
# -----------------------------

EPOCHS = 50

BATCH_SIZE = 16

LEARNING_RATE = 2e-4

NUM_WORKERS = 0


# -----------------------------
# Device
# -----------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(lr_path, hr_path):
    """
    Load LR/HR numpy arrays and convert them from:

        LR: (N, H, W, C)
        HR: (N, H, W, C)

    to PyTorch format:

        LR: (N, C, H, W)
        HR: (N, C, H, W)
    """

    lr = np.load(lr_path).astype(
        np.float32
    )

    hr = np.load(hr_path).astype(
        np.float32
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )

    if len(lr) != len(hr):
        raise ValueError(
            "LR and HR datasets contain "
            "different numbers of patches."
        )

    # --------------------------------------------------------
    # Validate expected dimensions
    # --------------------------------------------------------

    if lr.ndim != 4:
        raise ValueError(
            f"Expected LR to have 4 dimensions, "
            f"got {lr.ndim}."
        )

    if hr.ndim != 4:
        raise ValueError(
            f"Expected HR to have 4 dimensions, "
            f"got {hr.ndim}."
        )

    if lr.shape[1:] != (32, 32, 4):
        raise ValueError(
            f"Unexpected LR shape: {lr.shape}"
        )

    if hr.shape[1:] != (64, 64, 4):
        raise ValueError(
            f"Unexpected HR shape: {hr.shape}"
        )

    # --------------------------------------------------------
    # Convert NHWC -> NCHW
    # --------------------------------------------------------

    lr = torch.from_numpy(
        lr
    ).permute(
        0,
        3,
        1,
        2
    )

    hr = torch.from_numpy(
        hr
    ).permute(
        0,
        3,
        1,
        2
    )

    return lr, hr


# ============================================================
# Count empty patches
# ============================================================

def count_empty_patches(hr):
    """
    Count patches whose HR target is completely zero.

    A patch is considered empty when the absolute sum
    across all channels and pixels is below 1e-8.
    """

    empty_mask = torch.sum(
        torch.abs(hr),
        dim=(1, 2, 3)
    ) < 1e-8

    empty_count = int(
        empty_mask.sum().item()
    )

    content_count = (
        len(hr)
        - empty_count
    )

    return empty_count, content_count


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion
):
    """
    Evaluate the model on Region 2.

    Region 3 is never used here.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for lr, hr in loader:

            lr = lr.to(
                DEVICE,
                non_blocking=True
            )

            hr = hr.to(
                DEVICE,
                non_blocking=True
            )

            prediction = model(lr)

            loss = criterion(
                prediction,
                hr
            )

            batch_size = lr.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# Main training function
# ============================================================

def main():

    print("=" * 60)
    print("PixelSight — SwinIR v2 Training")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # ========================================================
    # Load Region 1 training data
    # ========================================================

    print("=" * 60)
    print("TRAINING DATA — REGION 1")
    print("=" * 60)

    train_lr, train_hr = load_dataset(
        TRAIN_LR,
        TRAIN_HR
    )

    print()

    print(
        "Training patches:",
        len(train_lr)
    )

    # --------------------------------------------------------
    # Count empty/content patches
    # --------------------------------------------------------

    empty_count, content_count = (
        count_empty_patches(train_hr)
    )

    print(
        "Empty training patches:",
        empty_count
    )

    print(
        "Content training patches:",
        content_count
    )

    print()

    # ========================================================
    # Load Region 2 validation data
    # ========================================================

    print("=" * 60)
    print("VALIDATION DATA — REGION 2")
    print("=" * 60)

    val_lr, val_hr = load_dataset(
        VAL_LR,
        VAL_HR
    )

    print()

    print(
        "Validation patches:",
        len(val_lr)
    )

    val_empty_count, val_content_count = (
        count_empty_patches(val_hr)
    )

    print(
        "Empty validation patches:",
        val_empty_count
    )

    print(
        "Content validation patches:",
        val_content_count
    )

    print()

    # ========================================================
    # Explicit Region 3 protection
    # ========================================================

    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(
        "Train      : Region 1"
    )

    print(
        "Validation : Region 2"
    )

    print(
        "Final test : Region 3"
    )

    print(
        "Region 3 loaded by this script: NO"
    )

    print()

    # ========================================================
    # Create datasets
    # ========================================================

    train_dataset = TensorDataset(
        train_lr,
        train_hr
    )

    val_dataset = TensorDataset(
        val_lr,
        val_hr
    )

    # ========================================================
    # Create data loaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # Create SwinIR v2 model
    # ========================================================

    print("=" * 60)
    print("MODEL")
    print("=" * 60)

    model = SwinIRResidual(
        in_channels=4,
        out_channels=4,
        scale=2,
        embed_dim=60,
        depths=6,
        num_heads=6,
        window_size=8
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model:",
        "SwinIRResidual"
    )

    print(
        "Parameters:",
        parameter_count
    )

    print()

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.L1Loss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # Checkpoint directory
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Best model tracking
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    # ========================================================
    # Training
    # ========================================================

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Training mode
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        total_samples = 0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for lr, hr in train_loader:

            lr = lr.to(
                DEVICE,
                non_blocking=True
            )

            hr = hr.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            prediction = model(lr)

            # ------------------------------------------------
            # L1 loss
            # ------------------------------------------------

            loss = criterion(
                prediction,
                hr
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update parameters
            # ------------------------------------------------

            optimizer.step()

            batch_size = lr.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        # ----------------------------------------------------
        # Epoch training loss
        # ----------------------------------------------------

        if total_samples == 0:
            raise RuntimeError(
                "Training dataset is empty."
            )

        train_loss = (
            running_loss
            / total_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        # ----------------------------------------------------
        # Print epoch results
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d} | "
            f"Train L1: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss,

                "config": {
                    "model":
                        "swinir_residual",

                    "in_channels":
                        4,

                    "out_channels":
                        4,

                    "scale":
                        2,

                    "embed_dim":
                        60,

                    "depths":
                        6,

                    "num_heads":
                        6,

                    "window_size":
                        8,

                    "reconstruction":
                        "bicubic_residual",

                    "epochs":
                        EPOCHS,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "train_region":
                        "region1",

                    "validation_region":
                        "region2",

                    "test_region":
                        "region3",

                    "train_content_only":
                        False
                }
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print(
                f"  ✓ New best model saved "
                f"(epoch {epoch})"
            )

    # ========================================================
    # Final summary
    # ========================================================

    print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation L1:",
        best_val_loss
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print()

    print(
        "Region 3 was not used during training "
        "or validation."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()