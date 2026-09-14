import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelsight.models.swinir import RSTB


class SwinIRResidual(nn.Module):
    """
    Lightweight 4-channel SwinIR for 2x multispectral
    super-resolution using bicubic residual learning.

    The model predicts a residual correction to the bicubic
    upsampled image:

        SR = Bicubic(LR) + Residual(LR)

    The final residual layer is initialized to zero so that
    the model starts exactly from the bicubic baseline.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        scale=2,
        embed_dim=60,
        depths=6,
        num_heads=6,
        window_size=8
    ):
        super().__init__()

        self.scale = scale
        self.window_size = window_size

        # ----------------------------------------------------
        # Shallow feature extraction
        # ----------------------------------------------------

        self.conv_first = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Deep feature extraction
        # ----------------------------------------------------

        self.body = RSTB(
            dim=embed_dim,
            depth=depths,
            window_size=window_size,
            num_heads=num_heads
        )

        self.conv_after_body = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual upsampling
        # ----------------------------------------------------

        self.conv_before_upsample = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim * scale * scale,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(scale),

            nn.Conv2d(
                embed_dim,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

        # ----------------------------------------------------
        # Zero-initialize the final residual layer
        #
        # This makes the initial residual exactly zero:
        #
        #     residual = 0
        #
        # Therefore the initial model is exactly:
        #
        #     SR = Bicubic(LR)
        #
        # This is especially useful for empty patches because:
        #
        #     zero LR -> zero bicubic -> zero residual -> zero SR
        # ----------------------------------------------------

        nn.init.zeros_(self.upsample[-1].weight)
        nn.init.zeros_(self.upsample[-1].bias)

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # Shallow features
        # ----------------------------------------------------

        shallow = self.conv_first(x)

        B, C, H, W = shallow.shape

        features = shallow.flatten(
            2
        ).transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # Swin Transformer body
        # ----------------------------------------------------

        features = self.body(
            features,
            H,
            W
        )

        # Convert sequence back to image features

        features = features.transpose(
            1,
            2
        ).view(
            B,
            -1,
            H,
            W
        )

        features = self.conv_after_body(
            features
        )

        # Local residual connection

        features = features + shallow

        # ----------------------------------------------------
        # Generate learned residual
        # ----------------------------------------------------

        features = self.conv_before_upsample(
            features
        )

        residual = self.upsample(
            features
        )

        # ----------------------------------------------------
        # Final reconstruction
        # ----------------------------------------------------

        output = bicubic + residual

        return output