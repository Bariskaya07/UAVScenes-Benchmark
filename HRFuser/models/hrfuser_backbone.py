"""
HRFuser Backbone - Standalone PyTorch Implementation

Adapted from:
- HRFuser: https://github.com/timbroed/HRFuser
- HRFormer: https://arxiv.org/abs/2110.09408
- HRNet: https://arxiv.org/abs/1904.04514

All mmcv/mmdet dependencies have been removed for standalone usage.
This implements HRFuserHRFormerBased backbone for multi-modal fusion
(RGB + auxiliary modality like HAG/LiDAR).

Output: List of 4 multi-resolution feature maps [C1, C2, C3, C4]
with channels matching the config (e.g. [18, 36, 72, 144] for Tiny).
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from torch.nn.functional import pad, softmax, dropout, linear


# ---------------------------------------------------------------------------
# Utility functions (replacing mmcv equivalents)
# ---------------------------------------------------------------------------

def nchw_to_nlc(x):
    """Convert [N, C, H, W] -> [N, L, C]."""
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] -> [N, C, H, W]."""
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nlc2nchw2nlc(module, x, hw_shape, contiguous=False):
    """Convert NLC->NCHW, apply module, convert back NCHW->NLC."""
    H, W = hw_shape
    B, L, C = x.shape
    x = x.transpose(1, 2).reshape(B, C, H, W)
    if contiguous:
        x = x.contiguous()
    x = module(x)
    x = x.flatten(2).transpose(1, 2)
    return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


def _make_norm(norm_type, num_features, **kwargs):
    """Create normalization layer."""
    if norm_type == 'BN' or norm_type == 'SyncBN':
        return nn.BatchNorm2d(num_features, **kwargs)
    elif norm_type == 'LN':
        eps = kwargs.get('eps', 1e-6)
        return nn.LayerNorm(num_features, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# ---------------------------------------------------------------------------
# ResNet building blocks (Bottleneck)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 with_cp=False, norm_cfg=None, conv_cfg=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.with_cp = with_cp

    @property
    def norm1(self):
        return self.bn1

    @property
    def norm2(self):
        return self.bn2

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 with_cp=False, norm_cfg=None, conv_cfg=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.with_cp = with_cp

    @property
    def norm1(self):
        return self.bn1

    @property
    def norm2(self):
        return self.bn2

    @property
    def norm3(self):
        return self.bn3

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# HRNet: HRModule (multi-branch fusion)
# ---------------------------------------------------------------------------

class HRModule(nn.Module):
    """High-Resolution Module for HRNet. Multi-branch with fusion."""

    def __init__(self, num_branches, blocks, num_blocks, in_channels,
                 num_channels, multiscale_output=True, with_cp=False,
                 conv_cfg=None, norm_cfg=None, **kwargs):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        pass  # Validation removed for simplicity

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels[branch_index],
                         num_channels[branch_index] * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion))

        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index],
                           stride, downsample=downsample, with_cp=self.with_cp))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index],
                               with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels[j], in_channels[i], 1, bias=False),
                        nn.BatchNorm2d(in_channels[i]),
                        nn.Upsample(scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels[j], in_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(in_channels[i])))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels[j], in_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(in_channels[j]),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


# ---------------------------------------------------------------------------
# HRFormer building blocks
# ---------------------------------------------------------------------------

class WindowMSA(nn.Module):
    """Window based multi-head self-attention module."""

    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True,
                 qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.,
                 with_rpe=True, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.Wh, self.Ww = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        self.with_rpe = with_rpe
        if self.with_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.Wh - 1) * (2 * self.Ww - 1), num_heads))
            coords_h = torch.arange(self.Wh)
            coords_w = torch.arange(self.Ww)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.Wh - 1
            relative_coords[:, :, 1] += self.Ww - 1
            relative_coords[:, :, 0] *= 2 * self.Ww - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.Wh * self.Ww, self.Wh * self.Ww, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class LocalWindowSelfAttention(nn.Module):
    """Local-window Self Attention (LSA) module."""

    def __init__(self, embed_dims, num_heads, window_size,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0.,
                 proj_drop_rate=0., with_rpe=True, with_pad_mask=False, **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMSA(
            embed_dims=embed_dims, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate, with_rpe=with_rpe)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        Wh, Ww = self.window_size

        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, Wh * Ww, C)

        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = pad(pad_mask, [0, 0, pad_w // 2, pad_w - pad_w // 2,
                                      pad_h // 2, pad_h - pad_h // 2], value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, 1)
            pad_mask = pad_mask.permute(0, 1, 3, 2, 4, 5)
            pad_mask = pad_mask.reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, pad_mask, **kwargs)
        else:
            out = self.attn(x, **kwargs)

        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H + pad_h, W + pad_w, C)
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)


class CrossFFN(nn.Module):
    """FFN with Depthwise Conv of HRFormer."""

    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 act_cfg=None, dw_act_cfg=None, norm_cfg=None, **kwargs):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                     stride=1, groups=hidden_channels, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x, H, W):
        return nlc2nchw2nlc(self.layers, x, (H, W))


class HRFormerBlock(nn.Module):
    """High-Resolution Block for HRFormer."""
    expansion = 1

    def __init__(self, in_channels, out_channels, num_heads, window_size=7,
                 mlp_ratio=4, drop_path=0.0, act_cfg=None,
                 norm_cfg=None, transformer_norm_cfg=None,
                 with_cp=False, with_rpe=True, with_pad_mask=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.attn = LocalWindowSelfAttention(
            in_channels, num_heads=num_heads, window_size=window_size,
            with_rpe=with_rpe, with_pad_mask=with_pad_mask)

        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6)
        self.ffn = CrossFFN(
            in_channels=in_channels,
            hidden_channels=int(in_channels * mlp_ratio),
            out_channels=out_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _inner_forward(self, x):
        B, C, H, W = x.size()
        x = nchw_to_nlc(x)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        x = nlc_to_nchw(x, (H, W))
        return x

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._inner_forward, x)
        else:
            out = self._inner_forward(x)
        return out


# ---------------------------------------------------------------------------
# HRFormer Module (replacing HRFomerModule from original)
# ---------------------------------------------------------------------------

class HRFormerModule(HRModule):
    """High-Resolution Module for HRFormer (with transformer blocks)."""

    def __init__(self, num_branches, block, num_blocks, in_channels,
                 num_channels, num_heads, num_window_sizes, num_mlp_ratios,
                 multiscale_output=True, drop_paths=None,
                 with_rpe=True, with_pad_mask=False,
                 conv_cfg=None, norm_cfg=None, transformer_norm_cfg=None,
                 with_cp=False):
        self.transformer_norm_cfg = transformer_norm_cfg
        self.drop_paths = drop_paths or [0.0]
        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.with_rpe = with_rpe
        self.with_pad_mask = with_pad_mask
        super().__init__(num_branches, block, num_blocks, in_channels,
                         num_channels, multiscale_output, with_cp,
                         conv_cfg, norm_cfg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        assert stride == 1 and self.in_channels[branch_index] == num_channels[branch_index]
        layers = []
        layers.append(block(
            self.in_channels[branch_index], num_channels[branch_index],
            num_heads=self.num_heads[branch_index],
            window_size=self.num_window_sizes[branch_index],
            mlp_ratio=self.num_mlp_ratios[branch_index],
            drop_path=self.drop_paths[0],
            with_rpe=self.with_rpe, with_pad_mask=self.with_pad_mask,
            with_cp=self.with_cp))
        self.in_channels[branch_index] = self.in_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(
                self.in_channels[branch_index], num_channels[branch_index],
                num_heads=self.num_heads[branch_index],
                window_size=self.num_window_sizes[branch_index],
                mlp_ratio=self.num_mlp_ratios[branch_index],
                drop_path=self.drop_paths[min(i, len(self.drop_paths) - 1)],
                with_rpe=self.with_rpe, with_pad_mask=self.with_pad_mask,
                with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Build fuse layers for HRFormer (depthwise separable convolutions)."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.in_channels
        fuse_layers = []
        for i in range(num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, bias=False),
                        nn.BatchNorm2d(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            sub_modules = [
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3,
                                         stride=2, padding=1, groups=num_inchannels[j], bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)]
                        else:
                            sub_modules = [
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3,
                                         stride=2, padding=1, groups=num_inchannels[j], bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.ReLU(False)]
                        conv3x3s.append(nn.Sequential(*sub_modules))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward with bilinear interpolation for upsampling (HRFormer style)."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


# ---------------------------------------------------------------------------
# HRFuser: Cross-Attention Fusion components
# ---------------------------------------------------------------------------

class WindowMCA(nn.Module):
    """Window based multi-head cross-attention (W-MCA) module."""

    def __init__(self, embed_dim, num_heads, window_size, qkv_bias=True,
                 qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.,
                 kdim=None, vdim=None, with_rpe=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.Wh, self.Ww = window_size
        self.num_heads = num_heads
        head_embed_dim = embed_dim // num_heads
        self.scale = qk_scale or head_embed_dim ** -0.5

        self.with_rpe = with_rpe
        if self.with_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.Wh - 1) * (2 * self.Ww - 1), num_heads))
            coords_h = torch.arange(self.Wh)
            coords_w = torch.arange(self.Ww)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.Wh - 1
            relative_coords[:, :, 1] += self.Ww - 1
            relative_coords[:, :, 0] *= 2 * self.Ww - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=qkv_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        B, N, C = query.shape
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.Wh * self.Ww, self.Wh * self.Ww, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class MultiWindowCrossAttention(nn.Module):
    """Multi-window Cross Attention (MWCA) module."""

    def __init__(self, window_size=7, with_pad_mask=False, **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMCA(window_size=self.window_size, **kwargs)

    def forward(self, x, y, H, W, **kwargs):
        assert x.shape == y.shape
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        Wh, Ww = self.window_size

        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        y = pad(y, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)
        y = y.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        y = y.permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)

        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = pad(pad_mask, [0, 0, pad_w // 2, pad_w - pad_w // 2,
                                      pad_h // 2, pad_h - pad_h // 2], value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, 1)
            pad_mask = pad_mask.permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, y, y, pad_mask, **kwargs)
        else:
            out = self.attn(x, y, y, **kwargs)

        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H + pad_h, W + pad_w, C)
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)


class HRFuserFusionBlock(nn.Module):
    """Cross-modality fusion block using Multi-Window Cross-Attention."""
    expansion = 1

    def __init__(self, in_channels, out_channels, num_heads, window_size=7,
                 mlp_ratio=4, drop_path=0.0, act_cfg=None,
                 norm_cfg=None, transformer_norm_cfg=None, with_cp=False,
                 num_fused_modalities=1, proj_drop_rate=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.num_fused_modalities = num_fused_modalities

        norm1_list = []
        norm2_list = []
        attn_list = []
        for i in range(self.num_fused_modalities):
            norm1_list.append(nn.LayerNorm(in_channels, eps=1e-6))
            norm2_list.append(nn.LayerNorm(out_channels, eps=1e-6))
            attn_list.append(MultiWindowCrossAttention(
                embed_dim=in_channels, num_heads=self.num_heads,
                window_size=self.window_size,
                proj_drop_rate=proj_drop_rate))
        self.norm1 = nn.ModuleList(norm1_list)
        self.norm2 = nn.ModuleList(norm2_list)
        self.attn = nn.ModuleList(attn_list)

        self.norm3 = nn.LayerNorm(out_channels, eps=1e-6)
        self.ffn = CrossFFN(
            in_channels=in_channels,
            hidden_channels=int(in_channels * self.mlp_ratio),
            out_channels=out_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _inner_forward(self, x, y):
        B, C, H, W = x.size()
        x = nchw_to_nlc(x)
        x_tmp = torch.empty_like(x).copy_(x)
        for i in range(self.num_fused_modalities):
            z = y[i]
            z = nchw_to_nlc(z)
            x = x + z + self.drop_path(self.attn[i](self.norm1[i](x_tmp), self.norm2[i](z), H, W))
        x = x + self.drop_path(self.ffn(self.norm3(x), H, W))
        x = nlc_to_nchw(x, (H, W))
        return x

    def forward(self, x, y):
        if self.with_cp and x.requires_grad:
            raise Exception('with_cp not supported with CA Fusion module')
        else:
            out = self._inner_forward(x, y)
        return out


# ---------------------------------------------------------------------------
# HRNet base class (standalone, no mmcv)
# ---------------------------------------------------------------------------

class HRNet(nn.Module):
    """HRNet backbone (standalone PyTorch)."""

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, extra, in_channels=3, conv_cfg=None,
                 norm_cfg=None, norm_eval=False, with_cp=False,
                 zero_init_residual=False, multiscale_output=True,
                 pretrained=None, init_cfg=None):
        super().__init__()
        self.pretrained = pretrained
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg or dict(type='BN')
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # Stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # Stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # Stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=multiscale_output)

    @property
    def norm1(self):
        return self.bn1

    @property
    def norm2(self):
        return self.bn2

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre_layer[-1]
                    out_ch = num_channels_cur_layer[i] if j == i - num_branches_pre else in_ch
                    conv_downsamples.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, with_cp=self.with_cp))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, with_cp=self.with_cp))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            hr_modules.append(HRModule(
                num_branches, block, num_blocks, in_channels, num_channels,
                reset_multiscale_output, with_cp=self.with_cp))
        return nn.Sequential(*hr_modules), in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()


# ---------------------------------------------------------------------------
# HRFormer (extends HRNet with transformer blocks)
# ---------------------------------------------------------------------------

class HRFormer(HRNet):
    """HRFormer backbone."""

    blocks_dict = {'BOTTLENECK': Bottleneck, 'HRFORMERBLOCK': HRFormerBlock, 'HRFORMER': HRFormerBlock}

    def __init__(self, extra, in_channels=3, conv_cfg=None,
                 norm_cfg=None, transformer_norm_cfg=None,
                 norm_eval=False, with_cp=False, multiscale_output=True,
                 drop_path_rate=0., zero_init_residual=False,
                 pretrained=None, init_cfg=None):

        # Stochastic depth
        depths = [
            extra[stage]['num_blocks'][0] * extra[stage]['num_modules']
            for stage in ['stage2', 'stage3', 'stage4']
        ]
        depth_s2, depth_s3, _ = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]

        self.transformer_norm_cfg = transformer_norm_cfg or dict(type='LN', eps=1e-6)
        self.with_rpe = extra.get('with_rpe', True)
        self.with_pad_mask = extra.get('with_pad_mask', False)

        super().__init__(
            extra=extra, in_channels=in_channels, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, norm_eval=norm_eval, with_cp=with_cp,
            zero_init_residual=zero_init_residual, multiscale_output=multiscale_output,
            pretrained=pretrained, init_cfg=init_cfg)

    def _make_stage(self, layer_config, num_inchannels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        # For Bottleneck blocks (stage1), use parent HRNet's _make_stage
        if block in (BasicBlock, Bottleneck):
            return super()._make_stage(layer_config, num_inchannels, multiscale_output)

        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['window_sizes']
        num_mlp_ratios = layer_config['mlp_ratios']
        drop_path_rates = layer_config.get('drop_path_rates', [0.0])

        modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(HRFormerModule(
                num_branches, block, num_blocks, num_inchannels, num_channels,
                num_heads, num_window_sizes, num_mlp_ratios,
                reset_multiscale_output,
                drop_paths=drop_path_rates[num_blocks[0] * i:num_blocks[0] * (i + 1)],
                with_rpe=self.with_rpe, with_pad_mask=self.with_pad_mask,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                transformer_norm_cfg=self.transformer_norm_cfg,
                with_cp=self.with_cp))
            num_inchannels = modules[-1].in_channels

        return nn.Sequential(*modules), num_inchannels


# ---------------------------------------------------------------------------
# HRFuserHRFormerBased: Main multi-modal backbone
# ---------------------------------------------------------------------------

class HRFuserHRFormerBased(HRFormer):
    """HRFuser backbone for multi-modal fusion.

    Takes RGB image and a list of auxiliary modality images,
    fuses them through Multi-Window Cross-Attention at each stage,
    and outputs multi-resolution features.
    """

    blocks_dict = {
        'BOTTLENECK': Bottleneck,
        'HRFORMER': HRFormerBlock,
        'CA': HRFuserFusionBlock,
        'MWCA': HRFuserFusionBlock,
    }

    def __init__(self, extra, in_channels=3, conv_cfg=None,
                 norm_cfg=None, transformer_norm_cfg=None,
                 norm_eval=False, with_cp=False, drop_path_rate=0.,
                 zero_init_residual=False, multiscale_output=True,
                 pretrained=None, init_cfg=None,
                 num_fused_modalities=1, mod_in_channels=None):
        super().__init__(
            extra=extra, in_channels=in_channels, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, transformer_norm_cfg=transformer_norm_cfg,
            norm_eval=norm_eval, with_cp=with_cp,
            drop_path_rate=drop_path_rate,
            zero_init_residual=zero_init_residual,
            multiscale_output=multiscale_output,
            pretrained=pretrained, init_cfg=init_cfg)

        if mod_in_channels is None:
            mod_in_channels = [3] * num_fused_modalities

        cfg = self.extra
        self.num_fused_modalities = num_fused_modalities
        self.pre_neck_fusion = True if cfg.get('LidarStageD') else False

        # Copy drop_path_rates to lidar stages
        self.extra['LidarStageB']['drop_path_rates'] = self.extra['stage2']['drop_path_rates']
        self.extra['LidarStageC']['drop_path_rates'] = self.extra['stage3']['drop_path_rates']
        if self.pre_neck_fusion:
            self.extra['LidarStageD']['drop_path_rates'] = self.extra['stage4']['drop_path_rates']

        # Build modality initial conv layers
        conv_a = []
        norm_a = []
        conv_b = []
        norm_b = []
        for i in range(self.num_fused_modalities):
            conv_a.append(nn.Conv2d(mod_in_channels[i], 64, 3, stride=2, padding=1, bias=False))
            norm_a.append(nn.BatchNorm2d(64))
            conv_b.append(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False))
            norm_b.append(nn.BatchNorm2d(64))
        self.conv_a = nn.ModuleList(conv_a)
        self.norm_a = nn.ModuleList(norm_a)
        self.conv_b = nn.ModuleList(conv_b)
        self.norm_b = nn.ModuleList(norm_b)

        # Stage A (modality initial bottleneck)
        self.stage_a_cfg = cfg['LidarStageA']
        num_channels_a = self.stage_a_cfg['num_channels'][0]
        block_a = self.blocks_dict[self.stage_a_cfg['block']]
        num_blocks_a = self.stage_a_cfg['num_blocks'][0]
        stage_a_out_channels = [[num_channels_a * block_a.expansion] for _ in range(self.num_fused_modalities)]
        modalities = []
        for i in range(self.num_fused_modalities):
            modalities.append(self._make_layer(block_a, 64, num_channels_a, num_blocks_a))
        self.layer_a = nn.ModuleList(modalities)

        # ModFusion A
        self.fusion_a_cfg = cfg['ModFusionA']
        num_channels_fa = self.fusion_a_cfg['num_channels']
        block_fa = self.blocks_dict[self.fusion_a_cfg['block']]
        num_channels_fa = [ch * block_fa.expansion for ch in num_channels_fa]
        self.transition_a = self._make_mod_transition_layer(stage_a_out_channels, num_channels_fa)
        self.fusion_a = self._make_multimodal_fusion(block_fa, self.fusion_a_cfg, num_channels_fa)

        # Stage B
        self.stage_b_cfg = cfg['LidarStageB']
        num_channels_b = self.stage_b_cfg['num_channels']
        block_b = self.blocks_dict[self.stage_b_cfg['block']]
        num_channels_b = [ch * block_b.expansion for ch in num_channels_b]
        self.stage_b, pre_stage_channels_b = self._make_mod_stage(self.stage_b_cfg, num_channels_b)

        # ModFusion B
        self.fusion_b_cfg = cfg['ModFusionB']
        num_channels_fb = self.fusion_b_cfg['num_channels']
        block_fb = self.blocks_dict[self.fusion_b_cfg['block']]
        self.transition_b = self._make_mod_transition_layer(pre_stage_channels_b, num_channels_fb)
        self.fusion_b = self._make_multimodal_fusion(block_fb, self.fusion_b_cfg, num_channels_fb)

        # Stage C
        self.stage_c_cfg = cfg['LidarStageC']
        num_channels_c = self.stage_c_cfg['num_channels']
        block_c = self.blocks_dict[self.stage_c_cfg['block']]
        num_channels_c = [ch * block_c.expansion for ch in num_channels_c]
        self.stage_c, pre_stage_channels_c = self._make_mod_stage(self.stage_c_cfg, num_channels_c)

        # ModFusion C
        self.fusion_c_cfg = cfg['ModFusionC']
        num_channels_fc = self.fusion_c_cfg['num_channels']
        block_fc = self.blocks_dict[self.fusion_c_cfg['block']]
        self.transition_c = self._make_mod_transition_layer(pre_stage_channels_c, num_channels_fc)
        self.fusion_c = self._make_multimodal_fusion(block_fc, self.fusion_c_cfg, num_channels_fc)

        # Optional Stage D
        if self.pre_neck_fusion:
            self.stage_d_cfg = cfg['LidarStageD']
            num_channels_d = self.stage_d_cfg['num_channels']
            block_d = self.blocks_dict[self.stage_d_cfg['block']]
            num_channels_d = [ch * block_d.expansion for ch in num_channels_d]
            self.stage_d, pre_stage_channels_d = self._make_mod_stage(self.stage_d_cfg, num_channels_d)

            self.fusion_d_cfg = cfg['ModFusionD']
            num_channels_fd = self.fusion_d_cfg['num_channels']
            block_fd = self.blocks_dict[self.fusion_d_cfg['block']]
            self.transition_d = self._make_mod_transition_layer(pre_stage_channels_d, num_channels_fd)
            self.fusion_d = self._make_multimodal_fusion(block_fd, self.fusion_d_cfg, num_channels_fd)

    def _make_mod_stage(self, layer_config, in_channels):
        pre_stage_channels = []
        modalities = []
        for _ in range(self.num_fused_modalities):
            tmp_stage, tmp_channels = self._make_stage(layer_config, list(in_channels))
            modalities.append(tmp_stage)
            pre_stage_channels.append(tmp_channels)
        return nn.ModuleList(modalities), pre_stage_channels

    def _make_mod_transition_layer(self, pre_stage_channels, num_channels):
        modalities = []
        for num_mod in range(self.num_fused_modalities):
            modalities.append(self._make_transition_layer(
                pre_stage_channels[num_mod], num_channels))
        return nn.ModuleList(modalities)

    def _make_multimodal_fusion(self, block, layer_config, num_inchannels):
        num_branches = layer_config['num_branches']
        num_channels = layer_config['num_channels']
        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['window_sizes']
        num_mlp_ratios = layer_config['mlp_ratios']
        drop_path = layer_config['drop_path']
        proj_drop_rate = layer_config.get('proj_drop_rate', 0.0)

        pre_branches = []
        for branch_index in range(num_branches):
            pre_branches.append(block(
                num_inchannels[branch_index], num_channels[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_path,
                num_fused_modalities=self.num_fused_modalities,
                proj_drop_rate=proj_drop_rate))
        return nn.ModuleList(pre_branches)

    def forward(self, x, x_mod):
        """Forward function.

        Args:
            x: RGB image tensor [B, 3, H, W]
            x_mod: list of auxiliary modality tensors, each [B, C, H, W]

        Returns:
            List of 4 multi-resolution feature maps
        """
        if not self.num_fused_modalities == len(x_mod):
            raise ValueError(f'num_fused_modalities ({self.num_fused_modalities}) '
                           f'!= len(x_mod) ({len(x_mod)})')

        # RGB stem
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Modality stems + Stage A
        n_list = []
        for k in range(self.num_fused_modalities):
            x_mod[k] = self.conv_a[k](x_mod[k])
            x_mod[k] = self.norm_a[k](x_mod[k])
            x_mod[k] = self.relu(x_mod[k])
            x_mod[k] = self.conv_b[k](x_mod[k])
            x_mod[k] = self.norm_b[k](x_mod[k])
            x_mod[k] = self.relu(x_mod[k])
            x_mod[k] = self.layer_a[k](x_mod[k])
            n_list.append([x_mod[k]])

        # Stage 2 & B
        x_list = []
        m_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_tmp = self.transition1[i](x)
            else:
                x_tmp = x
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_a[k][i] is not None:
                    m_tmp.append(self.transition_a[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            m_list.append(m_tmp)
            x_list.append(self.fusion_a[i](x_tmp, m_tmp))
        y_list = self.stage2(x_list)
        for k in range(self.num_fused_modalities):
            n_list[k] = self.stage_b[k]([m_list[0][k]])

        # Stage 3 & C
        x_list = []
        m_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_tmp = self.transition2[i](y_list[-1])
            else:
                x_tmp = y_list[i]
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_b[k][i] is not None:
                    m_tmp.append(self.transition_b[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            m_list.append(m_tmp)
            x_list.append(self.fusion_b[i](x_tmp, m_tmp))
        y_list = self.stage3(x_list)
        for k in range(self.num_fused_modalities):
            n_list[k] = self.stage_c[k]([m_list[0][k]])

        # Pre Stage 4 fusion
        x_list = []
        if self.pre_neck_fusion:
            m_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_tmp = self.transition3[i](y_list[-1])
            else:
                x_tmp = y_list[i]
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_c[k][i] is not None:
                    m_tmp.append(self.transition_c[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            if self.pre_neck_fusion:
                m_list.append(m_tmp)
            x_list.append(self.fusion_c[i](x_tmp, m_tmp))
        y_list = self.stage4(x_list)

        # Optional: Stage D + pre-neck fusion
        if self.pre_neck_fusion:
            for k in range(self.num_fused_modalities):
                n_list[k] = self.stage_d[k]([m_list[0][k]])
            x_list = []
            for i in range(self.stage4_cfg['num_branches']):
                x_tmp = y_list[i]
                m_tmp = []
                for k in range(self.num_fused_modalities):
                    if self.transition_d[k][i] is not None:
                        m_tmp.append(self.transition_d[k][i](n_list[k][0]))
                    else:
                        m_tmp.append(n_list[k][0])
                x_list.append(self.fusion_d[i](x_tmp, m_tmp))
            for i in range(len(x_list)):
                y_list[i] = self.relu(x_list[i])

        return y_list


# ---------------------------------------------------------------------------
# Config helper: HRFuser-T (Tiny)
# ---------------------------------------------------------------------------

def get_hrfuser_tiny_config():
    """Returns the HRFuser-T configuration dictionary."""
    return dict(
        LidarStageA=dict(
            num_modules=1, num_branches=1, block='BOTTLENECK',
            num_blocks=(2,), num_channels=(64,)),
        ModFusionA=dict(
            block='MWCA', num_branches=2,
            window_sizes=(7, 7), num_heads=(1, 2), mlp_ratios=(4, 4),
            num_channels=(18, 36), drop_path=0.0, proj_drop_rate=0.1),
        LidarStageB=dict(
            num_modules=1, num_branches=1, block='HRFORMER',
            window_sizes=(7,), num_heads=(1,), mlp_ratios=(4,),
            num_blocks=(2,), num_channels=(18,)),
        ModFusionB=dict(
            block='MWCA', num_branches=3,
            window_sizes=(7, 7, 7), num_heads=(1, 2, 4), mlp_ratios=(4, 4, 4),
            num_channels=(18, 36, 72), drop_path=0.0, proj_drop_rate=0.1),
        LidarStageC=dict(
            num_modules=3, num_branches=1, block='HRFORMER',
            window_sizes=(7,), num_heads=(1,), mlp_ratios=(4,),
            num_blocks=(2,), num_channels=(18,)),
        ModFusionC=dict(
            block='MWCA', num_branches=4,
            window_sizes=(7, 7, 7, 7), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4),
            num_channels=(18, 36, 72, 144), drop_path=0.0, proj_drop_rate=0.1),
        LidarStageD=None,
        stage1=dict(
            num_modules=1, num_branches=1, block='BOTTLENECK',
            num_blocks=(2,), num_channels=(64,)),
        stage2=dict(
            num_modules=1, num_branches=2, block='HRFORMER',
            window_sizes=(7, 7), num_heads=(1, 2), mlp_ratios=(4, 4),
            num_blocks=(2, 2), num_channels=(18, 36)),
        stage3=dict(
            num_modules=3, num_branches=3, block='HRFORMER',
            window_sizes=(7, 7, 7), num_heads=(1, 2, 4), mlp_ratios=(4, 4, 4),
            num_blocks=(2, 2, 2), num_channels=(18, 36, 72)),
        stage4=dict(
            num_modules=2, num_branches=4, block='HRFORMER',
            window_sizes=(7, 7, 7, 7), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4),
            num_blocks=(2, 2, 2, 2), num_channels=(18, 36, 72, 144)),
    )
