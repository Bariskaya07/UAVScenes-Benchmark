"""
PPX (Parallel Pooling Mixer) Encoder for CMNeXt
Paper: Delivering Arbitrary-Modal Semantic Segmentation (Section 3.3)

PPX is a lightweight encoder for auxiliary modalities (HAG, Depth, etc.)
that replaces heavy transformer backbones with efficient pooling operations.

PPX Block formulation:
1. f_hat = DW-Conv7x7(f)
2. f_hat = Sum(Pool_kxk(f_hat)) + f_hat,  k in {3, 7, 11}
3. w = Sigmoid(Conv1x1(f_hat)); f_w = w * f + f
4. output = FFN(f_w) + SE(f_w)

Output channels: [64, 128, 320, 512] (matches MiT-B2)
Output strides: [4, 8, 16, 32]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Channel attention mechanism that adaptively recalibrates
    channel-wise feature responses.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class FFN(nn.Module):
    """Feed-Forward Network with depthwise separable convolutions.

    Expands channels, applies depthwise conv, then projects back.
    """
    def __init__(self, dim, expansion_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = dim * expansion_ratio

        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PPXBlock(nn.Module):
    """Parallel Pooling Mixer Block.

    Core building block of PPX encoder. Uses parallel multi-scale pooling
    instead of self-attention for efficient feature extraction.

    Args:
        dim: Input/output channel dimension
        kernel_size: Kernel size for depthwise conv (default: 7)
        pool_sizes: List of pooling kernel sizes (default: [3, 7, 11])
        expansion_ratio: FFN expansion ratio (default: 4)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(self, dim, kernel_size=7, pool_sizes=[3, 7, 11],
                 expansion_ratio=4, dropout=0.0):
        super().__init__()

        # Step 1: Depthwise Conv 7x7
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size,
                                  padding=kernel_size // 2, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)

        # Step 2: Parallel Pooling (multi-scale)
        self.pools = nn.ModuleList([
            nn.AvgPool2d(k, stride=1, padding=k // 2) for k in pool_sizes
        ])

        # Step 3: Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Step 4: FFN + SE
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = FFN(dim, expansion_ratio, dropout)
        self.se = SEBlock(dim)

    def forward(self, x):
        identity = x

        # Step 1: Depthwise conv
        x = self.norm1(self.dw_conv(x))

        # Step 2: Parallel pooling - aggregate multi-scale context
        pooled = sum(pool(x) for pool in self.pools)
        x = pooled + x

        # Step 3: Gating - adaptive feature selection
        w = self.gate(x)
        x = w * identity + identity  # Gated residual

        # Step 4: FFN + SE with residual
        x = self.norm2(x)
        x = x + self.ffn(x) + self.se(x)

        return x


class PPXStage(nn.Module):
    """PPX Stage: Downsampling + N x PPXBlock.

    Each stage reduces spatial resolution and increases channels.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_blocks: Number of PPX blocks in this stage
        stride: Downsampling stride (4 for stage 1, 2 for others)
    """
    def __init__(self, in_channels, out_channels, num_blocks, stride=2):
        super().__init__()

        # Patch Embedding / Downsampling
        # Stage 1: stride=4 (Conv 7x7, stride=4)
        # Stage 2-4: stride=2 (Conv 3x3, stride=2)
        if stride == 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels)
            )

        # PPX Blocks (maintain spatial resolution within stage)
        self.blocks = nn.Sequential(*[
            PPXBlock(out_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPXEncoder(nn.Module):
    """PPX Encoder: 4-stage encoder matching MiT-B2 output structure.

    Designed to be a lightweight alternative to transformer backbones
    for auxiliary modalities in multi-modal segmentation.

    Output channels: [64, 128, 320, 512] (same as MiT-B2)
    Output strides: [4, 8, 16, 32]

    For 768x768 input:
    - Stage 1: 768/4 = 192x192
    - Stage 2: 768/8 = 96x96
    - Stage 3: 768/16 = 48x48
    - Stage 4: 768/32 = 24x24

    Args:
        in_chans: Number of input channels (default: 3)
        embed_dims: Channel dimensions for each stage (default: [64, 128, 320, 512])
        num_blocks: Number of PPX blocks per stage (default: [3, 4, 6, 3])
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(self, in_chans=3, embed_dims=[64, 128, 320, 512],
                 num_blocks=[3, 4, 6, 3], dropout=0.0):
        super().__init__()

        self.embed_dims = embed_dims

        # Stage 1: Input -> 64 channels, stride 4
        self.stage1 = PPXStage(in_chans, embed_dims[0], num_blocks[0], stride=4)

        # Stage 2: 64 -> 128 channels, stride 2 (total stride 8)
        self.stage2 = PPXStage(embed_dims[0], embed_dims[1], num_blocks[1], stride=2)

        # Stage 3: 128 -> 320 channels, stride 2 (total stride 16)
        self.stage3 = PPXStage(embed_dims[1], embed_dims[2], num_blocks[2], stride=2)

        # Stage 4: 320 -> 512 channels, stride 2 (total stride 32)
        self.stage4 = PPXStage(embed_dims[2], embed_dims[3], num_blocks[3], stride=2)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"PPXEncoder initialized (random init) - dims: {embed_dims}, blocks: {num_blocks}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of multi-scale features:
            - [B, 64, H/4, W/4]
            - [B, 128, H/8, W/8]
            - [B, 320, H/16, W/16]
            - [B, 512, H/32, W/32]
        """
        outputs = []

        x = self.stage1(x)
        outputs.append(x)  # H/4

        x = self.stage2(x)
        outputs.append(x)  # H/8

        x = self.stage3(x)
        outputs.append(x)  # H/16

        x = self.stage4(x)
        outputs.append(x)  # H/32

        return outputs


# Factory functions
def ppx_encoder_b2(in_chans=3, **kwargs):
    """PPX Encoder matching MiT-B2 output structure.

    This is the recommended encoder for auxiliary modalities in CMNeXt.
    """
    return PPXEncoder(
        in_chans=in_chans,
        embed_dims=[64, 128, 320, 512],
        num_blocks=[3, 4, 6, 3],
        **kwargs
    )


def ppx_encoder_small(in_chans=3, **kwargs):
    """Smaller PPX Encoder for lighter models."""
    return PPXEncoder(
        in_chans=in_chans,
        embed_dims=[32, 64, 160, 256],
        num_blocks=[2, 2, 4, 2],
        **kwargs
    )


def ppx_encoder_large(in_chans=3, **kwargs):
    """Larger PPX Encoder for heavier models."""
    return PPXEncoder(
        in_chans=in_chans,
        embed_dims=[64, 128, 320, 512],
        num_blocks=[3, 6, 12, 3],
        **kwargs
    )


if __name__ == '__main__':
    # Test PPX Encoder output shapes
    print("Testing PPXEncoder...")

    model = ppx_encoder_b2(in_chans=3)
    x = torch.randn(1, 3, 768, 768)

    outputs = model(x)

    expected_shapes = [
        (1, 64, 192, 192),   # Stage 1: 768/4
        (1, 128, 96, 96),    # Stage 2: 768/8
        (1, 320, 48, 48),    # Stage 3: 768/16
        (1, 512, 24, 24),    # Stage 4: 768/32
    ]

    print("\nOutput shapes:")
    for i, (out, expected) in enumerate(zip(outputs, expected_shapes)):
        status = "OK" if out.shape == expected else "FAIL"
        print(f"  Stage {i+1}: {tuple(out.shape)} (expected {expected}) [{status}]")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
