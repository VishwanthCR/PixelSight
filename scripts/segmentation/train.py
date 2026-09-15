"""
PixelSight - Stable Optimized Semantic Segmentation Training

Features:
    - Spatial augmentation for training
    - Weighted Cross Entropy
    - Dice loss
    - AMP mixed precision
    - Batch size 32
    - Multi-worker DataLoader
    - Persistent workers
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Best validation checkpoint

Classes:
    0 = Tree
    1 = Shrubland
    2 = Grassland
    3 = Cropland
    4 = Built-up
    5 = Bare
    6 = Water
    255 = Ignore
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from pixelsight.data.segmentation.dataset import create_dataset
from pixelsight.models.segmentation.unet import (
    create_unet,
    count_parameters,
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Tree",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare",
    "Water",
]

NUM_CLASSES = 7
IGNORE_INDEX = 255


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train stable optimized "
            "PixelSight U-Net"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML configuration",
    )

    return parser.parse_args()


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    dataset,
    num_classes: int,
):

    print("\nCalculating class weights...")

    distribution = dataset.class_distribution()

    counts = np.array(
        [
            distribution[class_id]
            for class_id in range(num_classes)
        ],
        dtype=np.float64,
    )

    counts = np.maximum(
        counts,
        1.0,
    )

    frequencies = (
        counts
        / counts.sum()
    )

    inverse_frequency = (
        1.0
        / frequencies
    )

    weights = (
        inverse_frequency
        / inverse_frequency.mean()
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("\nClass weights:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_id} "
            f"{class_name:<12}: "
            f"{weights[class_id].item():.6f}"
        )

    return weights


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ):

        super().__init__()

        self.num_classes = num_classes

        self.ignore_index = ignore_index

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        valid_mask = (
            targets
            != self.ignore_index
        )

        safe_targets = targets.clone()

        safe_targets[
            ~valid_mask
        ] = 0

        target_one_hot = F.one_hot(
            safe_targets,
            num_classes=self.num_classes,
        )

        target_one_hot = (
            target_one_hot
            .permute(
                0,
                3,
                1,
                2,
            )
            .float()
        )

        valid_mask = valid_mask.unsqueeze(1)

        probabilities = (
            probabilities
            * valid_mask
        )

        target_one_hot = (
            target_one_hot
            * valid_mask
        )

        intersection = (
            probabilities
            * target_one_hot
        ).sum(
            dim=(0, 2, 3)
        )

        probability_sum = probabilities.sum(
            dim=(0, 2, 3)
        )

        target_sum = target_one_hot.sum(
            dim=(0, 2, 3)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probability_sum
            + target_sum
            + self.smooth
        )

        present_classes = (
            target_sum > 0
        )

        if present_classes.any():

            dice = dice[
                present_classes
            ]

            return 1.0 - dice.mean()

        return logits.sum() * 0.0


# ============================================================
# COMBINED LOSS
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        ignore_index=255,
        ce_weight=0.7,
        dice_weight=0.3,
    ):

        super().__init__()

        self.ce_weight = ce_weight

        self.dice_weight = dice_weight

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

        self.dice = DiceLoss(
            num_classes=NUM_CLASSES,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits,
        targets,
    ):

        ce_loss = self.cross_entropy(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        return (
            self.ce_weight * ce_loss
            +
            self.dice_weight * dice_loss
        )


# ============================================================
# DATA LOADERS
# ============================================================

def create_loaders(
    train_dataset,
    validation_dataset,
    batch_size,
    num_workers,
    pin_memory,
):

    use_pin_memory = (
        pin_memory
        and torch.cuda.is_available()
    )

    persistent_workers = (
        num_workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    training=True,
    print_frequency=50,
    use_amp=False,
):

    if training:

        model.train()

        phase = "Train"

    else:

        model.eval()

        phase = "Validation"

    running_loss = 0.0

    total_samples = 0

    for batch_index, (
        images,
        labels,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels,
                )

            if training:

                if scaler is not None:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    optimizer.step()

        actual_batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.detach().item()
            * actual_batch_size
        )

        total_samples += (
            actual_batch_size
        )

        if (
            (batch_index + 1)
            % print_frequency
            == 0
            or
            (batch_index + 1)
            == len(loader)
        ):

            print(
                f"  {phase:<10} "
                f"Batch "
                f"{batch_index + 1:>4}"
                f"/{len(loader)} "
                f"Loss: "
                f"{loss.detach().item():.6f}"
            )

    if total_samples == 0:

        return 0.0

    return (
        running_loss
        / total_samples
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PixelSight Segmentation"
    )

    print(
        "Stable Optimized U-Net Training"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    args = parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed = config.get(
        "seed",
        42,
    )

    set_seed(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU Memory: "
            f"{gpu_memory:.2f} GB"
        )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    print(
        "AMP mixed precision: "
        f"{'ENABLED' if use_amp else 'DISABLED'}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data_config = config["data"]

    base_dir = Path(
        data_config.get(
            "base_dir",
            "dataset/segmentation/patches",
        )
    )

    train_split = data_config.get(
        "train_split",
        "train",
    )

    validation_split = data_config.get(
        "validation_split",
        "validation",
    )

    print(
        "\nLoading datasets..."
    )

    train_dataset = create_dataset(
        split=train_split,
        base_dir=base_dir,
        augment=True,
        validate_samples=False,
    )

    validation_dataset = create_dataset(
        split=validation_split,
        base_dir=base_dir,
        augment=False,
        validate_samples=False,
    )

    print(
        f"Train samples      : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_dataset):,}"
    )

    print(
        "Training augmentation: ENABLED"
    )

    print(
        "Validation augmentation: DISABLED"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    training_config = config[
        "training"
    ]

    batch_size = training_config.get(
        "batch_size",
        32,
    )

    num_workers = training_config.get(
        "num_workers",
        4,
    )

    pin_memory = training_config.get(
        "pin_memory",
        True,
    )

    learning_rate = training_config.get(
        "learning_rate",
        5e-5,
    )

    weight_decay = training_config.get(
        "weight_decay",
        1e-4,
    )

    optimizer_name = training_config.get(
        "optimizer",
        "adamw",
    ).lower()

    epochs = training_config.get(
        "epochs",
        50,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
    ) = create_loaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        "\nDataLoader:"
    )

    print(
        f"  Batch size        : "
        f"{batch_size}"
    )

    print(
        f"  Workers           : "
        f"{num_workers}"
    )

    print(
        f"  Pin memory        : "
        f"{pin_memory}"
    )

    print(
        f"  Persistent workers: "
        f"{num_workers > 0}"
    )

    print(
        f"  Train batches     : "
        f"{len(train_loader)}"
    )

    print(
        f"  Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset,
        NUM_CLASSES,
    )

    class_weights = class_weights.to(
        device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_config = config[
        "model"
    ]

    in_channels = model_config.get(
        "input_channels",
        4,
    )

    num_classes = model_config.get(
        "num_classes",
        NUM_CLASSES,
    )

    base_channels = model_config.get(
        "base_channels",
        32,
    )

    model = create_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )

    model = model.to(device)

    print(
        "\nModel:"
    )

    print(
        f"  Parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_config = training_config.get(
        "loss",
        {},
    )

    ce_weight = loss_config.get(
        "ce_weight",
        1.0,
    )

    dice_weight = loss_config.get(
        "dice_weight",
        0.0,
    )

    criterion = CombinedLoss(
        class_weights=class_weights,
        ignore_index=IGNORE_INDEX,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )

    print(
        "\nLoss:"
    )

    print(
        "  Weighted Cross Entropy + Dice"
    )

    print(
        f"  CE weight   : "
        f"{ce_weight}"
    )

    print(
        f"  Dice weight : "
        f"{dice_weight}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    #
    # If validation loss does not improve for 3 epochs,
    # reduce LR by factor 0.5.
    #
    # Example:
    #
    # 5e-5
    # ↓
    # 2.5e-5
    # ↓
    # 1.25e-5
    #
    # This helps stabilize training when validation
    # improvement stalls.
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    print(
        "\nLearning-rate scheduler:"
    )

    print(
        "  ReduceLROnPlateau"
    )

    print(
        "  Factor   : 0.5"
    )

    print(
        "  Patience : 3 epochs"
    )

    print(
        "  Min LR   : 1e-6"
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    if use_amp:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping_patience = training_config.get(
        "early_stopping_patience",
        8,
    )

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_config = config.get(
        "checkpoint",
        {},
    )

    checkpoint_directory = Path(
        checkpoint_config.get(
            "directory",
            "checkpoints/segmentation",
        )
    )

    checkpoint_name = checkpoint_config.get(
        "best_model",
        "unet_worldcover_aug_dice_best.pth",
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_directory
        / checkpoint_name
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_config = config.get(
        "logging",
        {}
    )

    print_frequency = logging_config.get(
        "print_frequency",
        50,
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print(
        "\nTraining:"
    )

    print(
        f"  Epochs             : "
        f"{epochs}"
    )

    print(
        f"  Batch size         : "
        f"{batch_size}"
    )

    print(
        f"  Learning rate      : "
        f"{learning_rate}"
    )

    print(
        f"  Weight decay       : "
        f"{weight_decay}"
    )

    print(
        f"  Optimizer          : "
        f"{optimizer_name}"
    )

    print(
        f"  Early stop patience: "
        f"{early_stopping_patience}"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    total_start_time = time.time()

    print(
        "\n" + "-" * 70
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start_time = time.time()

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            training=True,
            print_frequency=print_frequency,
            use_amp=use_amp,
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            training=False,
            print_frequency=print_frequency,
            use_amp=use_amp,
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # TIMING
        # ====================================================

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        print(
            f"\nTrain loss      : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Learning rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch time      : "
            f"{epoch_time / 60:.2f} minutes"
        )

        # ====================================================
        # GPU MEMORY
        # ====================================================

        if device.type == "cuda":

            peak_memory = (
                torch.cuda
                .max_memory_allocated()
                / (1024 ** 3)
            )

            print(
                f"Peak GPU memory : "
                f"{peak_memory:.2f} GB"
            )

            torch.cuda.reset_peak_memory_stats()

        # ====================================================
        # BEST MODEL
        # ====================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "best_validation_loss":
                    best_validation_loss,

                "config":
                    config,

                "class_names":
                    CLASS_NAMES,

                "num_classes":
                    num_classes,

                "in_channels":
                    in_channels,

                "base_channels":
                    base_channels,

                "seed":
                    seed,

                "augmentation":
                    True,

                "loss":
                    "weighted_cross_entropy_plus_dice",

                "ce_weight":
                    ce_weight,

                "dice_weight":
                    dice_weight,

                "amp":
                    use_amp,

                "batch_size":
                    batch_size,

                "num_workers":
                    num_workers,

                "learning_rate":
                    learning_rate,
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                f"  NEW BEST MODEL SAVED"
            )

            print(
                f"  Path: "
                f"{checkpoint_path}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{early_stopping_patience})"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No validation improvement "
                f"for "
                f"{early_stopping_patience} "
                f"epochs."
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_time = (
        time.time()
        - total_start_time
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STABLE OPTIMIZED TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch           : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Total training time  : "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Checkpoint           : "
        f"{checkpoint_path}"
    )

    print(
        f"AMP                  : "
        f"{'Enabled' if use_amp else 'Disabled'}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()