"""
CMNeXt: Multi-Modal Semantic Segmentation with Hub2Fuse Mechanism
Paper: Delivering Arbitrary-Modal Semantic Segmentation

Hub2Fuse paradigm:
1. Hub modality (RGB) as the central representation
2. Auxiliary modalities (HAG, Depth, etc.) fused via cross-attention
3. Self-rectification module for robust fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .backbones.mit import (
    MixVisionTransformer, mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
)
from .ppx import PPXEncoder, ppx_encoder_b2


class ConvModule(nn.Module):
    """Convolution + BatchNorm + ReLU module."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FeatureRectifyModule(nn.Module):
    """Self-Rectification Module for multi-modal feature alignment.

    This module rectifies auxiliary modality features using the hub modality
    as guidance, improving cross-modal alignment.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, hub_feat, aux_feat):
        """
        Args:
            hub_feat: Hub modality features (RGB)
            aux_feat: Auxiliary modality features (HAG, etc.)
        Returns:
            Rectified auxiliary features
        """
        # Channel attention from hub
        channel_weight = self.channel_attention(hub_feat)
        # Spatial attention from hub
        spatial_weight = self.spatial_attention(hub_feat)

        # Apply attention to auxiliary features
        rectified = aux_feat * channel_weight * spatial_weight
        return rectified


class CrossModalFusion(nn.Module):
    """Cross-Modal Fusion module with Spatial Reduction Attention (SRA).

    Uses SegFormer-style spatial reduction to avoid memory explosion.
    Instead of full N×N attention, reduces K/V to (N/sr_ratio²) tokens.

    Memory savings:
    - Full attention: 256×256 = 65,536 tokens → 4.3B elements
    - SRA (sr_ratio=8): 32×32 = 1,024 tokens → 67M elements (64x savings!)
    """
    def __init__(self, dim, num_heads=4, sr_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_hub = nn.LayerNorm(dim)
        self.norm_aux = nn.LayerNorm(dim)

        # Spatial Reduction for K/V (critical for memory!)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, hub_feat, aux_feat):
        """
        Args:
            hub_feat: [B, C, H, W] Hub modality features
            aux_feat: [B, C, H, W] Auxiliary modality features
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = hub_feat.shape

        # Query from hub (full resolution)
        hub_flat = hub_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        hub_norm = self.norm_hub(hub_flat)
        q = self.q_proj(hub_norm).reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key/Value from auxiliary (SPATIALLY REDUCED to save memory!)
        if self.sr_ratio > 1:
            # Reduce spatial resolution: [B, C, H, W] -> [B, C, H/sr, W/sr]
            aux_reduced = self.sr(aux_feat)
            _, _, h_r, w_r = aux_reduced.shape
            aux_flat = aux_reduced.flatten(2).transpose(1, 2)  # [B, (H/sr)*(W/sr), C]
            aux_flat = self.sr_norm(aux_flat)
        else:
            aux_flat = aux_feat.flatten(2).transpose(1, 2)

        aux_norm = self.norm_aux(aux_flat)

        # K/V from reduced auxiliary
        kv = self.kv_proj(aux_norm).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, N_reduced, head_dim]

        # Attention: Q (full) × K (reduced) -> memory efficient!
        # [B, heads, HW, head_dim] × [B, heads, head_dim, N_reduced] = [B, heads, HW, N_reduced]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Aggregate: [B, heads, HW, N_reduced] × [B, heads, N_reduced, head_dim]
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.out_proj(out)

        # Reshape back to spatial
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Residual connection
        return hub_feat + out


class Hub2FuseBlock(nn.Module):
    """Hub2Fuse Block: Core fusion mechanism of CMNeXt.

    Combines:
    1. Feature Rectification
    2. Cross-Modal Fusion (with Spatial Reduction)
    3. Feature Enhancement

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        sr_ratio: Spatial reduction ratio for memory efficiency
                  - Stage 1 (256×256): sr_ratio=8 → 32×32
                  - Stage 2 (128×128): sr_ratio=4 → 32×32
                  - Stage 3 (64×64): sr_ratio=2 → 32×32
                  - Stage 4 (32×32): sr_ratio=1 → no reduction
    """
    def __init__(self, dim, num_heads=4, sr_ratio=8):
        super().__init__()
        self.rectify = FeatureRectifyModule(dim)
        self.fusion = CrossModalFusion(dim, num_heads, sr_ratio=sr_ratio)
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, hub_feat, aux_feat):
        """
        Args:
            hub_feat: Hub modality features (RGB)
            aux_feat: Auxiliary modality features (HAG)
        Returns:
            Fused and enhanced features
        """
        # Step 1: Rectify auxiliary features using hub guidance
        rectified_aux = self.rectify(hub_feat, aux_feat)

        # Step 2: Cross-modal fusion
        fused = self.fusion(hub_feat, rectified_aux)

        # Step 3: Feature enhancement
        enhanced = self.enhance(fused)

        return enhanced


class SegFormerHead(nn.Module):
    """SegFormer-style MLP decoder head.

    Aggregates multi-scale features and produces final segmentation.
    """
    def __init__(self, in_channels=[64, 128, 320, 512], embed_dim=256, num_classes=19, dropout_ratio=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Linear layers for each scale
        self.linear_c4 = ConvModule(in_channels[3], embed_dim, 1)
        self.linear_c3 = ConvModule(in_channels[2], embed_dim, 1)
        self.linear_c2 = ConvModule(in_channels[1], embed_dim, 1)
        self.linear_c1 = ConvModule(in_channels[0], embed_dim, 1)

        # Fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Final classifier
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: List of multi-scale features [c1, c2, c3, c4]
        Returns:
            Segmentation logits [B, num_classes, H/4, W/4]
        """
        c1, c2, c3, c4 = features

        # Get target size (1/4 of input)
        n, _, h, w = c1.shape

        # Process each scale
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1)

        # Concatenate and fuse
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        fused = self.linear_fuse(_c)

        # Final prediction
        out = self.dropout(fused)
        out = self.linear_pred(out)

        return out


class CMNeXt(nn.Module):
    """CMNeXt: Multi-Modal Semantic Segmentation Network.

    Architecture (Paper-compliant):
    - Hub backbone: MiT (Mix Vision Transformer) for RGB (ImageNet pretrained)
    - Auxiliary encoder: PPX (Parallel Pooling Mixer) for HAG (random init)
    - Hub2Fuse modules at each scale for cross-modal fusion
    - SegFormer-style decoder head

    Note: Following the paper, auxiliary modalities use PPX blocks instead of
    transformer backbones. PPX is randomly initialized (no pretrained weights).

    Args:
        backbone: Backbone type ('mit_b0', 'mit_b1', 'mit_b2', etc.)
        num_classes: Number of segmentation classes
        modals: List of input modalities ['img', 'hag']
        pretrained: Path to pretrained backbone weights (only for RGB branch)
    """
    def __init__(self, backbone='mit_b2', num_classes=19, modals=['img', 'hag'],
                 pretrained=None, embed_dim=256, aux_in_chans=3):
        super().__init__()
        self.modals = modals
        self.num_classes = num_classes
        self.aux_in_chans = aux_in_chans

        # Backbone factory
        backbone_factory = {
            'mit_b0': mit_b0,
            'mit_b1': mit_b1,
            'mit_b2': mit_b2,
            'mit_b3': mit_b3,
            'mit_b4': mit_b4,
            'mit_b5': mit_b5,
        }

        # Get backbone dimensions
        if backbone in ['mit_b0']:
            in_channels = [32, 64, 160, 256]
        else:
            in_channels = [64, 128, 320, 512]

        # Hub backbone (RGB) - ImageNet pretrained
        self.hub_backbone = backbone_factory[backbone]()

        # Auxiliary encoder (HAG) - PPX with random init (paper-compliant)
        # aux_in_chans: 1 for UAVScenes HAG, 2 for DELIVER LiDAR, 3 for backward compat
        if len(modals) > 1:
            self.aux_encoder = ppx_encoder_b2(in_chans=aux_in_chans)
            print(f"[CMNeXt] Auxiliary encoder input channels: {aux_in_chans}")

            # Spatial reduction ratios for memory-efficient attention
            # Stage 1 (256×256): sr=8 → 32×32, Stage 2 (128×128): sr=4 → 32×32
            # Stage 3 (64×64): sr=2 → 32×32, Stage 4 (32×32): sr=1 → no reduction
            sr_ratios = [8, 4, 2, 1]

            # Hub2Fuse modules for each scale (with SRA for memory efficiency)
            self.fuse_blocks = nn.ModuleList([
                Hub2FuseBlock(dim, num_heads=max(1, dim // 64), sr_ratio=sr)
                for dim, sr in zip(in_channels, sr_ratios)
            ])

        # Decoder head
        self.decode_head = SegFormerHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        # Load pretrained weights
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained_path):
        """Load pretrained backbone weights (only for RGB hub backbone).

        Note: Following the paper, auxiliary encoder (PPX) remains randomly
        initialized. ImageNet pretrained weights are only for the hub modality.
        """
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Load hub backbone only (RGB branch)
            hub_msg = self.hub_backbone.load_state_dict(state_dict, strict=False)
            print(f"Hub backbone (RGB) loaded with ImageNet pretrained: {hub_msg}")

            # PPX encoder for auxiliary modality stays randomly initialized
            if hasattr(self, 'aux_encoder'):
                print("Auxiliary encoder (PPX) initialized randomly (paper-compliant)")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def forward(self, inputs):
        """
        Args:
            inputs: List of input tensors [rgb, hag] or single tensor for RGB-only

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        if isinstance(inputs, (list, tuple)):
            rgb = inputs[0]  # Hub modality
            aux = inputs[1] if len(inputs) > 1 else None  # Auxiliary modality
        else:
            rgb = inputs
            aux = None

        # Get original input size for final upsampling
        input_size = rgb.shape[2:]

        # Extract hub features
        hub_features = self.hub_backbone(rgb)  # List of [c1, c2, c3, c4]

        # Multi-modal fusion
        if aux is not None and hasattr(self, 'aux_encoder'):
            # Extract auxiliary features using PPX encoder
            aux_features = self.aux_encoder(aux)

            # Fuse at each scale
            fused_features = []
            for i, (hub_f, aux_f, fuse_block) in enumerate(zip(hub_features, aux_features, self.fuse_blocks)):
                fused = fuse_block(hub_f, aux_f)
                fused_features.append(fused)
        else:
            fused_features = hub_features

        # Decode
        logits = self.decode_head(fused_features)

        # Upsample to original size
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits


def cmnext_b0(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b0', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=256)


def cmnext_b1(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b1', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=256)


def cmnext_b2(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b2', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=768)


def cmnext_b3(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b3', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=768)


def cmnext_b4(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b4', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=768)


def cmnext_b5(num_classes=19, modals=['img', 'hag'], pretrained=None):
    return CMNeXt(backbone='mit_b5', num_classes=num_classes, modals=modals,
                  pretrained=pretrained, embed_dim=768)
