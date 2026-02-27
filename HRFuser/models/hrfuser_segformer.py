"""
HRFuser + SegFormer Head for Semantic Segmentation

Combines HRFuserHRFormerBased backbone with SegFormer MLP decode head.
Adapted for RGB + HAG (Height Above Ground) multi-modal semantic segmentation
on UAVScenes dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrfuser_backbone import HRFuserHRFormerBased, get_hrfuser_tiny_config


class MLP(nn.Module):
    """Linear Embedding for SegFormer decode head."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.

    Takes multi-scale features and produces per-pixel class predictions.
    """

    def __init__(self, feature_strides, in_channels, embedding_dim=256, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class HRFuserSegFormer(nn.Module):
    """HRFuser backbone + SegFormer decode head for semantic segmentation.

    This model takes RGB and HAG (Height Above Ground) inputs,
    extracts multi-scale features via HRFuser's multi-modal fusion backbone,
    and produces semantic segmentation predictions via SegFormer's MLP head.

    Args:
        num_classes (int): Number of semantic classes. Default: 19.
        embedding_dim (int): Embedding dimension for SegFormer head. Default: 256.
        backbone_config (dict): HRFuser backbone configuration. Default: HRFuser-T.
        drop_path_rate (float): Drop path rate for backbone. Default: 0.0.
        num_fused_modalities (int): Number of auxiliary modalities. Default: 1.
        mod_in_channels (list): Input channels for each auxiliary modality. Default: [3].
        pretrained (str): Path to pretrained backbone weights. Default: None.
    """

    def __init__(self, num_classes=19, embedding_dim=256,
                 backbone_config=None, drop_path_rate=0.0,
                 num_fused_modalities=1, mod_in_channels=None,
                 pretrained=None):
        super().__init__()
        self.num_classes = num_classes

        if backbone_config is None:
            backbone_config = get_hrfuser_tiny_config()

        if mod_in_channels is None:
            mod_in_channels = [3] * num_fused_modalities

        # Determine output channels from config
        stage4_channels = backbone_config['stage4']['num_channels']
        # HRFormerBlock has expansion=1, so output channels = num_channels
        self.in_channels = list(stage4_channels)
        self.feature_strides = [4, 8, 16, 32]

        # Build backbone
        self.backbone = HRFuserHRFormerBased(
            extra=backbone_config,
            in_channels=3,
            norm_cfg=dict(type='BN'),
            transformer_norm_cfg=dict(type='LN', eps=1e-6),
            norm_eval=False,
            drop_path_rate=drop_path_rate,
            num_fused_modalities=num_fused_modalities,
            mod_in_channels=mod_in_channels,
            pretrained=pretrained)

        # Build decoder
        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for decoder."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, pretrained_path):
        """Load pretrained HRFormer weights (RGB stream only).

        Args:
            pretrained_path: Path to HRFormer pretrained checkpoint.
        """
        if pretrained_path is None:
            return

        # PyTorch>=2.6 defaults to weights_only=True; HRFormer checkpoints may
        # include non-tensor metadata (e.g., yacs CfgNode), so disable it here.
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter out detection head keys and load backbone weights
        backbone_state = {}
        for k, v in state_dict.items():
            # Remove 'backbone.' prefix if present
            new_k = k.replace('backbone.', '')
            backbone_state[new_k] = v

        # Load with strict=False to allow missing modality-specific keys
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)

        loaded = len(state_dict) - len(unexpected)
        print(f"Loaded pretrained weights: {loaded} params loaded, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

    def get_param_groups(self):
        """Get parameter groups for optimizer (different lr for backbone vs decoder)."""
        backbone_params = []
        backbone_norm_params = []
        decoder_params = []

        for name, param in self.backbone.named_parameters():
            if 'norm' in name or 'bn' in name:
                backbone_norm_params.append(param)
            else:
                backbone_params.append(param)

        for param in self.decoder.parameters():
            decoder_params.append(param)

        return [backbone_params, backbone_norm_params, decoder_params]

    def forward(self, rgb, hag):
        """Forward pass.

        Args:
            rgb: RGB image tensor [B, 3, H, W]
            hag: HAG (Height Above Ground) tensor [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H/4, W/4]
        """
        # Extract multi-scale features with multi-modal fusion
        features = self.backbone(rgb, [hag])

        # Decode features to segmentation map
        out = self.decoder(features)

        return out


if __name__ == '__main__':
    # Quick shape test
    model = HRFuserSegFormer(num_classes=19, embedding_dim=256)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Test forward pass
    rgb = torch.randn(1, 3, 256, 256)
    hag = torch.randn(1, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        out = model(rgb, hag)
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input HAG shape: {hag.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [1, 19, 64, 64]")
