import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Window utilities
# ============================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, -1)
    )

    return x


# ============================================================
# Window Attention
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=6
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        relative_size = (2 * window_size - 1) ** 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size, num_heads)
        )

        coords = torch.arange(window_size)

        coords = torch.stack(
            torch.meshgrid(
                coords,
                coords,
                indexing="ij"
            )
        )

        coords_flatten = torch.flatten(
            coords,
            1
        )

        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )

        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1

        relative_coords[:, :, 0] *= (
            2 * window_size - 1
        )

        relative_position_index = (
            relative_coords.sum(-1)
        )

        self.register_buffer(
            "relative_position_index",
            relative_position_index
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

    def forward(self, x):

        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                self.head_dim
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attention = q @ k.transpose(-2, -1)

        relative_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
            .view(
                self.window_size ** 2,
                self.window_size ** 2,
                -1
            )
            .permute(2, 0, 1)
            .contiguous()
        )

        attention = attention + relative_bias.unsqueeze(0)

        attention = F.softmax(
            attention,
            dim=-1
        )

        x = (
            attention @ v
        ).transpose(
            1, 2
        ).reshape(
            B_,
            N,
            C
        )

        return self.proj(x)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        dim,
        expansion=4
    ):
        super().__init__()

        hidden = dim * expansion

        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# Swin Transformer Block
# ============================================================

class SwinBlock(nn.Module):

    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=6
    ):
        super().__init__()

        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim)

    def forward(self, x, H, W):

        B, L, C = x.shape

        shortcut = x

        x = self.norm1(x)

        x = x.view(
            B,
            H,
            W,
            C
        )

        windows = window_partition(
            x,
            self.window_size
        )

        attention_windows = self.attn(
            windows
        )

        x = window_reverse(
            attention_windows,
            self.window_size,
            H,
            W
        )

        x = x.view(
            B,
            H * W,
            C
        )

        x = shortcut + x

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# Residual Swin Transformer Block
# ============================================================

class RSTB(nn.Module):

    def __init__(
        self,
        dim,
        depth=6,
        window_size=8,
        num_heads=6
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim,
                window_size,
                num_heads
            )
            for _ in range(depth)
        ])

        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, H, W):

        residual = x

        for block in self.blocks:
            x = block(x, H, W)

        B, L, C = x.shape

        x = x.transpose(
            1, 2
        ).view(
            B,
            C,
            H,
            W
        )

        x = self.conv(x)

        x = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        return residual + x


# ============================================================
# Lightweight 4-channel SwinIR
# ============================================================

class SwinIR(nn.Module):

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

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        # Deep feature extraction
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

        # Upsampling
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

    def forward(self, x):

        B, C, H, W = x.shape

        # Feature extraction
        shallow = self.conv_first(x)

        features = shallow.flatten(
            2
        ).transpose(
            1,
            2
        )

        # Swin blocks
        features = self.body(
            features,
            H,
            W
        )

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

        features = features + shallow

        # Upsample
        features = self.conv_before_upsample(
            features
        )

        output = self.upsample(
            features
)

        output = torch.sigmoid(output)

        return output