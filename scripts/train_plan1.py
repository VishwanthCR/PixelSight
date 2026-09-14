import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from pixelsight.models.swinir import SwinIR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

TRAIN_LR = Path("dataset/region1/lr_patches.npy")
TRAIN_HR = Path("dataset/region1/hr_patches.npy")

VAL_LR = Path("dataset/region2/lr_patches.npy")
VAL_HR = Path("dataset/region2/hr_patches.npy")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================
# Dataset
# ============================================================

class PixelSightDataset(Dataset):

    def __init__(self, lr_path, hr_path, content_only=False):

        self.lr = np.load(lr_path)
        self.hr = np.load(hr_path)

        assert len(self.lr) == len(self.hr)

        if content_only:
            # Keep patches containing actual HR information.
            mask = np.any(self.hr > 1e-8, axis=(1, 2, 3))

            self.lr = self.lr[mask]
            self.hr = self.hr[mask]

        print(f"Loaded {len(self.lr)} patches")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, index):

        lr = torch.from_numpy(self.lr[index]).permute(2, 0, 1).float()
        hr = torch.from_numpy(self.hr[index]).permute(2, 0, 1).float()

        return lr, hr


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("PixelSight SwinIR Training")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("\nLoading training dataset...")

train_dataset = PixelSightDataset(
    TRAIN_LR,
    TRAIN_HR,
    content_only=True
)

print("\nLoading validation dataset...")

val_dataset = PixelSightDataset(
    VAL_LR,
    VAL_HR,
    content_only=False
)


# ============================================================
# DataLoaders
# ============================================================

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


# ============================================================
# Model
# ============================================================

model = SwinIR(
    in_channels=4,
    out_channels=4,
    scale=2,
    embed_dim=60,
    depths=6,
    num_heads=6,
    window_size=8
).to(DEVICE)


print("\nModel parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# Loss / Optimizer
# ============================================================

criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# Mixed Precision
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for lr, hr in train_loader:

        lr = lr.to(DEVICE, non_blocking=True)
        hr = hr.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            prediction = model(lr)

            loss = criterion(prediction, hr)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * lr.size(0)

    train_loss /= len(train_dataset)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(DEVICE, non_blocking=True)
            hr = hr.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                prediction = model(lr)

                loss = criterion(prediction, hr)

            val_loss += loss.item() * lr.size(0)

    val_loss /= len(val_dataset)


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            CHECKPOINT_DIR / "swinir_best.pth"
        )

        best_marker = "  <-- BEST"

    else:
        best_marker = ""


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"Train L1: {train_loss:.6f} "
        f"Val L1: {val_loss:.6f} "
        f"LR: {current_lr:.2e}"
        f"{best_marker}"
    )


print("\n" + "=" * 60)
print("Training complete")
print("Best validation loss:", best_val_loss)
print("Best model:", CHECKPOINT_DIR / "swinir_best.pth")
print("=" * 60)