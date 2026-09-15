"""
PixelSight Segmentation U-Net

Lightweight U-Net baseline for 4-band Sentinel-2 imagery.

Input:
    [B, 4, H, W]

Output:
    [B, 7, H, W]

Classes:
    0 = Tree
    1 = Shrubland
    2 = Grassland
    3 = Cropland
    4 = Built-up
    5 = Bare
    6 = Water

Ignore label:
    255

The ignore label is handled by the loss/evaluation code,
not by the model itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn


NUM_CLASSES = 7
INPUT_BANDS = 4


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Max-pooling followed by DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling followed by concatenation with the encoder skip
    connection and DoubleConv.

    Bilinear interpolation is used instead of transposed
    convolution to keep the baseline lightweight.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        # Handle possible spatial-size differences caused by
        # odd input dimensions.
        diff_height = skip.size(2) - x.size(2)
        diff_width = skip.size(3) - x.size(3)

        if diff_height != 0 or diff_width != 0:
            x = nn.functional.pad(
                x,
                [
                    diff_width // 2,
                    diff_width - diff_width // 2,
                    diff_height // 2,
                    diff_height - diff_height // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for PixelSight multispectral segmentation.

    Default architecture:

        4
        ↓
        32
        ↓
        64
        ↓
        128
        ↓
        256
        ↓
        512 bottleneck
        ↓
        256
        ↓
        128
        ↓
        64
        ↓
        32
        ↓
        7 classes
    """

    def __init__(
        self,
        in_channels: int = INPUT_BANDS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be greater than zero."
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be greater than zero."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            base_channels,
        )

        self.enc2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.enc3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.enc4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        # Bottleneck
        self.bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        # Decoder
        self.up4 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.up3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.up1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        # Final classifier
        self.classifier = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Tensor with shape [B, in_channels, H, W].

        Returns:
            Tensor with shape [B, num_classes, H, W].
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B,C,H,W], got {x.shape}"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.size(1)}"
            )

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.classifier(d1)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def create_unet(
    in_channels: int = INPUT_BANDS,
    num_classes: int = NUM_CLASSES,
    base_channels: int = 32,
) -> UNet:
    """
    Convenience factory for creating the PixelSight U-Net.
    """

    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )


if __name__ == "__main__":
    model = create_unet()

    print("=" * 60)
    print("PixelSight Segmentation U-Net")
    print("=" * 60)

    print(f"Input bands    : {model.in_channels}")
    print(f"Output classes : {model.num_classes}")
    print(f"Base channels  : {model.base_channels}")
    print(f"Parameters     : {count_parameters(model):,}")

    dummy_input = torch.randn(
        2,
        model.in_channels,
        128,
        128,
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape     : {tuple(dummy_input.shape)}")
    print(f"Output shape    : {tuple(output.shape)}")

    print("\nModel test: PASS")