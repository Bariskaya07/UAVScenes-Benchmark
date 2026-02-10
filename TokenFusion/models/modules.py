"""
TokenFusion Core Modules

This module contains the core components for multimodal token fusion:
- TokenExchange: Swaps uninformative tokens between modalities
- ModuleParallel: Wrapper for running same module on parallel modality inputs
- LayerNormParallel: Individual LayerNorm for each modality
"""

import torch
import torch.nn as nn


# Number of parallel modalities (RGB + HAG = 2)
num_parallel = 2


class TokenExchange(nn.Module):
    """
    Token Exchange Module for Multimodal Fusion.

    Dynamically exchanges uninformative tokens between two modalities.
    Tokens with importance score below threshold are replaced with
    corresponding tokens from the other modality.

    Forward:
        x: List of [B, N, C] tensors (one per modality)
        mask: List of [B, N, 1] importance score tensors
        mask_threshold: Threshold for token exchange (default: 0.02)

    Returns:
        List of fused [B, N, C] tensors
    """

    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        """
        Exchange tokens between modalities based on importance scores.

        Args:
            x: List of 2 tensors, each [B, N, C]
            mask: List of 2 tensors, each [B, N, 1] - importance scores from PredictorLG
            mask_threshold: Score threshold below which tokens are exchanged

        Returns:
            List of 2 tensors with exchanged tokens
        """
        # Initialize output tensors
        x0 = torch.zeros_like(x[0])
        x1 = torch.zeros_like(x[1])

        # For modality 0: keep tokens with score >= threshold, replace others with modality 1
        x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        x0[mask[0] < mask_threshold] = x[1][mask[0] < mask_threshold]

        # For modality 1: keep tokens with score >= threshold, replace others with modality 0
        x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]

        return [x0, x1]


class ModuleParallel(nn.Module):
    """
    Module Parallel Wrapper.

    Wraps a single module to process multiple modality inputs in parallel.
    Uses shared weights for all modalities.

    Args:
        module: The module to wrap (e.g., nn.Linear, nn.Conv2d)

    Example:
        >>> linear = ModuleParallel(nn.Linear(256, 512))
        >>> x_parallel = [x_rgb, x_hag]  # List of tensors
        >>> out = linear(x_parallel)  # Returns list of outputs
    """

    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        """
        Apply module to each input in parallel.

        Args:
            x_parallel: List of tensors (one per modality)

        Returns:
            List of outputs from applying module to each input
        """
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    """
    Layer Normalization with separate statistics per modality.

    Unlike ModuleParallel(nn.LayerNorm), this maintains separate LayerNorm
    instances for each modality. This is important because different
    modalities (e.g., RGB vs HAG) have different statistical distributions.

    Args:
        num_features: Number of features (embedding dimension)

    Example:
        >>> ln = LayerNormParallel(256)
        >>> x_parallel = [x_rgb, x_hag]
        >>> out = ln(x_parallel)  # Each normalized with its own statistics
    """

    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        # Create separate LayerNorm for each modality
        for i in range(num_parallel):
            setattr(self, f'ln_{i}', nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        """
        Apply modality-specific LayerNorm to each input.

        Args:
            x_parallel: List of tensors (one per modality)

        Returns:
            List of normalized tensors
        """
        return [getattr(self, f'ln_{i}')(x) for i, x in enumerate(x_parallel)]


class DropPathParallel(nn.Module):
    """
    Drop Path (Stochastic Depth) for parallel modalities.

    Applies the same drop path pattern to all modalities.

    Args:
        drop_prob: Probability of dropping a path
    """

    def __init__(self, drop_prob=0.0):
        super(DropPathParallel, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x_parallel):
        if self.drop_prob == 0.0 or not self.training:
            return x_parallel

        keep_prob = 1 - self.drop_prob
        # Shape: (B, 1, 1) for broadcasting
        shape = (x_parallel[0].shape[0],) + (1,) * (x_parallel[0].ndim - 1)
        # Generate same random tensor for all modalities (consistent drop)
        random_tensor = keep_prob + torch.rand(shape, dtype=x_parallel[0].dtype, device=x_parallel[0].device)
        random_tensor.floor_()  # binarize

        outputs = []
        for x in x_parallel:
            output = x.div(keep_prob) * random_tensor
            outputs.append(output)

        return outputs


class Conv2dParallel(nn.Module):
    """
    Parallel Conv2d for patch embedding.

    Wraps nn.Conv2d to process multiple modality inputs.

    Args:
        Same as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2dParallel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x_parallel):
        return [self.conv(x) for x in x_parallel]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.

    This is the same as DropPath from timm but works with single tensors.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop Path for single tensors."""

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
