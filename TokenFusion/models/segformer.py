"""
WeTr Model (SegFormer with TokenFusion) for Semantic Segmentation

This module implements the WeTr model that combines:
- MixVisionTransformer encoder with TokenFusion
- SegFormer-style MLP decoder

Reference:
    Wang et al. "Multimodal Token Fusion for Vision Transformers" CVPR 2022
    Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation" NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer
from .modules import num_parallel


class MLP(nn.Module):
    """
    Linear Embedding layer for SegFormer decoder.

    Projects feature maps to a unified embedding dimension.
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    """
    SegFormer MLP Decoder.

    Takes multi-scale features from encoder and produces segmentation output.
    All features are projected to same embedding dimension, upsampled,
    concatenated, and fused.
    """

    def __init__(self, feature_strides=None, in_channels=None, embedding_dim=256, num_classes=19):
        super().__init__()

        if in_channels is None:
            in_channels = [64, 128, 320, 512]
        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_strides = feature_strides

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP projections for each scale
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # Fusion and prediction
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through decoder.

        Args:
            x: List of 4 feature maps [c1, c2, c3, c4] from encoder stages

        Returns:
            Segmentation logits at 1/4 resolution
        """
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        # Project and upsample each scale to c1 resolution
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Concatenate and fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class WeTr(nn.Module):
    """
    WeTr: TokenFusion SegFormer for Multi-Modal Semantic Segmentation.

    Combines MixVisionTransformer encoder with TokenFusion mechanism
    and SegFormer-style MLP decoder.

    Args:
        backbone: Name of MiT backbone ('mit_b0', 'mit_b1', 'mit_b2', etc.)
        num_classes: Number of segmentation classes (19 for UAVScenes)
        embedding_dim: Decoder embedding dimension
        pretrained: Path to pretrained weights or bool to load default
    """

    def __init__(self, backbone='mit_b2', num_classes=19, embedding_dim=256, pretrained=None):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = num_parallel

        # Initialize encoder
        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims

        # Load pretrained weights if provided
        if pretrained:
            self._load_pretrained(pretrained)

        # Initialize decoder (shared for both modalities)
        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes
        )

        # Learnable weights for ensemble
        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def _load_pretrained(self, pretrained):
        """Load pretrained weights for encoder."""
        if isinstance(pretrained, str):
            print(f"Loading pretrained weights from: {pretrained}")
            state_dict = torch.load(pretrained, map_location='cpu')

            # Remove classification head if present
            if 'head.weight' in state_dict:
                state_dict.pop('head.weight')
            if 'head.bias' in state_dict:
                state_dict.pop('head.bias')

            # Expand state dict for parallel modalities
            state_dict = expand_state_dict(
                self.encoder.state_dict(),
                state_dict,
                self.num_parallel
            )

            # Load weights
            missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            print("Pretrained weights loaded successfully!")

    def get_param_groups(self):
        """
        Get parameter groups for optimizer.

        Returns:
            List of 3 parameter groups:
            - encoder non-norm params
            - encoder norm params
            - decoder params
        """
        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: List of 2 tensors [rgb, hag], each [B, 3, H, W]

        Returns:
            Tuple of:
                - List of 3 outputs [rgb_pred, hag_pred, ensemble_pred]
                - List of token importance masks
        """
        # Encoder forward (returns multi-scale features for each modality)
        x, masks = self.encoder(x)

        # Decoder forward for each modality
        x = [self.decoder(x[0]), self.decoder(x[1])]

        # Learned ensemble of modality predictions
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(self.num_parallel):
            ens = ens + alpha_soft[l] * x[l].detach()
        x.append(ens)

        return x, masks


def expand_state_dict(model_dict, state_dict, num_parallel):
    """
    Expand single-modal pretrained weights to parallel modality model.

    For TokenFusion, we need to expand weights for:
    - LayerNormParallel (ln_0, ln_1, ...)
    - Other shared modules

    Args:
        model_dict: Target model state dict
        state_dict: Pretrained state dict (single modality)
        num_parallel: Number of parallel modalities

    Returns:
        Expanded state dict
    """
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()

    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')

        # Direct match
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]

        # LayerNormParallel expansion (ln_0, ln_1, etc.)
        for i in range(num_parallel):
            ln = f'.ln_{i}'
            replace = ln in model_dict_key_re
            model_dict_key_re_clean = model_dict_key_re.replace(ln, '')

            if replace and model_dict_key_re_clean in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re_clean]

    return model_dict


def build_model(backbone='mit_b2', num_classes=19, embedding_dim=256, pretrained=None):
    """
    Build WeTr model.

    Args:
        backbone: MiT backbone variant
        num_classes: Number of classes (19 for UAVScenes)
        embedding_dim: Decoder embedding dimension
        pretrained: Path to pretrained weights

    Returns:
        WeTr model instance
    """
    model = WeTr(
        backbone=backbone,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model
    model = WeTr('mit_b2', num_classes=19, embedding_dim=256, pretrained=None)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward pass
    rgb = torch.rand(2, 3, 768, 768)
    hag = torch.rand(2, 3, 768, 768)
    x = [rgb, hag]

    outputs, masks = model(x)
    print(f"RGB output shape: {outputs[0].shape}")
    print(f"HAG output shape: {outputs[1].shape}")
    print(f"Ensemble output shape: {outputs[2].shape}")
    print(f"Number of mask layers: {len(masks)}")
