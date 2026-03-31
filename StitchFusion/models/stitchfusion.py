import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .encoders.mix_transformer import StitchFusionBackbone, load_mit_pretrained


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = norm_layer(c2)
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims, embed_dim=256, num_classes=19, norm_layer=nn.BatchNorm2d):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f'linear_c{i + 1}', MLP(dim, embed_dim))
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, norm_layer=norm_layer)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]
        for i, feature in enumerate(features[1:]):
            cf = getattr(self, f'linear_c{i + 2}')(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        return self.linear_pred(self.dropout(seg))


class StitchFusionSegmentation(nn.Module):
    def __init__(
        self,
        backbone='mit_b2',
        num_classes=19,
        moa_type='obMoA',
        moa_r=8,
        pretrained=None,
        embed_dim=768,
        activation_checkpoint=False,
        criterion=None,
        freeze_backbone=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        variant = backbone.split('_')[-1].upper() if '_' in backbone else backbone.replace('mit', '').upper()
        if not variant.startswith('B'):
            variant = 'B2'

        self.backbone = StitchFusionBackbone(
            model_name=variant,
            modals=('img', 'hag'),
            moa_type=moa_type,
            moa_r=moa_r,
            use_checkpoint=activation_checkpoint,
            norm_layer=norm_layer,
        )
        self.decode_head = SegFormerHead(
            self.backbone.channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            norm_layer=norm_layer,
        )
        self.criterion = criterion

        self.decode_head.apply(self._init_weights)
        if pretrained:
            self.init_pretrained(pretrained)
        if freeze_backbone:
            self.freeze_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(m.groups, 1)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained):
        return load_mit_pretrained(self.backbone, pretrained)

    def freeze_backbone(self):
        self.backbone.freeze_shared_backbone()

    def encode_decode(self, rgb, modal_x):
        ori_size = rgb.shape[2:]
        features = self.backbone([rgb, modal_x])
        out = self.decode_head(features)
        return F.interpolate(out, size=ori_size, mode='bilinear', align_corners=False)

    def forward(self, rgb, modal_x, label=None):
        out = self.encode_decode(rgb, modal_x)
        if label is not None and self.criterion is not None:
            return self.criterion(out, label.long())
        return out
