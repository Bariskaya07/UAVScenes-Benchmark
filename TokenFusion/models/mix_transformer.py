"""
Mix Vision Transformer (MiT) with TokenFusion for Multi-Modal Segmentation

This implementation is adapted from TokenFusion paper for UAVScenes RGB+HAG segmentation.
Key features:
- Parallel processing for 2 modalities (RGB + HAG)
- PredictorLG for token importance scoring
- TokenExchange for swapping uninformative tokens
- Modality-specific LayerNorm

Reference:
    Wang et al. "Multimodal Token Fusion for Vision Transformers" CVPR 2022
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .modules import ModuleParallel, LayerNormParallel, num_parallel, TokenExchange


class DWConv(nn.Module):
    """Depthwise Convolution for positional encoding in MLP."""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP with Depthwise Convolution for spatial mixing."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(act_layer())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = [self.dwconv(x[0], H, W), self.dwconv(x[1], H, W)]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention with Spatial Reduction and Token Exchange."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)

        self.exchange = TokenExchange()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]

        if self.sr_ratio > 1:
            x = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            x = self.sr(x)
            x = [x_.reshape(B, C, -1).permute(0, 2, 1) for x_ in x]
            x = self.norm(x)
            kv = self.kv(x)
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        else:
            kv = self.kv(x)
            kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]

        k, v = [kv[0][0], kv[1][0]], [kv[0][1], kv[1][1]]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)

        # Apply token exchange if mask is provided
        if mask is not None:
            x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(x, mask)]
            x = self.exchange(x, mask, mask_threshold=0.02)

        return x


class PredictorLG(nn.Module):
    """
    Token Importance Predictor (from DynamicViT).

    Predicts importance score for each token using a small MLP.
    Scores are used to determine which tokens to exchange between modalities.
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 2),
                nn.LogSoftmax(dim=-1)
            ) for _ in range(num_parallel)
        ])

    def forward(self, x):
        """
        Predict importance scores for tokens.

        Args:
            x: List of [B, N, C] tensors (one per modality)

        Returns:
            List of [B, N, 2] score tensors (LogSoftmax outputs)
        """
        return [self.score_nets[i](x[i]) for i in range(num_parallel)]


class Block(nn.Module):
    """Transformer Block with Token Fusion support."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormParallel, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio
        )
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        f = self.drop_path(self.attn(self.norm1(x), H, W, mask))
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
        f = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding for hierarchical structure."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = ModuleParallel(nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        ))
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """
    Mix Vision Transformer with TokenFusion for Multi-Modal Segmentation.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of classes (not used, kept for compatibility)
        embed_dims: Embedding dimensions for each stage
        num_heads: Number of attention heads for each stage
        mlp_ratios: MLP expansion ratios
        qkv_bias: Add bias to QKV projections
        qk_scale: Override default QK scale
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        depths: Number of blocks in each stage
        sr_ratios: Spatial reduction ratios for attention
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=LayerNormParallel,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # Patch embeddings for each stage
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4,
            in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # Token importance predictors for each stage
        self.score_predictor = nn.ModuleList([
            PredictorLG(embed_dims[i]) for i in range(len(depths))
        ])

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stage 1
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
            ) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage 2
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
            ) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage 3
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
            ) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage 4
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
            ) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        """Reset drop path rate for fine-tuning."""
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.module.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.module.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.module.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.module.drop_prob = dpr[cur + i]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def forward_features(self, x):
        """
        Forward pass through encoder.

        Args:
            x: List of [B, C, H, W] tensors for each modality

        Returns:
            Tuple of:
                - List of multi-scale features for each modality
                - List of token importance masks
        """
        B = x[0].shape[0]
        outs0, outs1 = [], []
        masks = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            score = self.score_predictor[0](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]
            masks.append(mask)
            x = blk(x, H, W, mask)
        x = self.norm1(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            score = self.score_predictor[1](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]
            masks.append(mask)
            x = blk(x, H, W, mask)
        x = self.norm2(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            score = self.score_predictor[2](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]
            masks.append(mask)
            x = blk(x, H, W, mask)
        x = self.norm3(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            score = self.score_predictor[3](x)
            mask = [F.softmax(score_.reshape(B, -1, 2), dim=2)[:, :, 0] for score_ in score]
            masks.append(mask)
            x = blk(x, H, W, mask)
        x = self.norm4(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        return [outs0, outs1], masks

    def forward(self, x):
        x, masks = self.forward_features(x)
        return x, masks


# Model variants following SegFormer configurations

class mit_b0(MixVisionTransformer):
    """MiT-B0: Tiny model."""
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )


class mit_b1(MixVisionTransformer):
    """MiT-B1: Small model."""
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )


class mit_b2(MixVisionTransformer):
    """
    MiT-B2: Base model (used for fair comparison with CMNeXt).

    Configuration:
        - embed_dims: [64, 128, 320, 512]
        - depths: [3, 4, 6, 3]
        - Total blocks: 16
    """
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )


class mit_b3(MixVisionTransformer):
    """MiT-B3: Large model."""
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )


class mit_b4(MixVisionTransformer):
    """MiT-B4: Extra large model."""
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )


class mit_b5(MixVisionTransformer):
    """MiT-B5: Huge model."""
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )
