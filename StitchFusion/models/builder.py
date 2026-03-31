import torch.nn as nn

from .stitchfusion import StitchFusionSegmentation


class EncoderDecoder(StitchFusionSegmentation):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__(
            backbone=cfg.backbone,
            num_classes=cfg.num_classes,
            moa_type=getattr(cfg, 'moa_type', 'obMoA'),
            moa_r=getattr(cfg, 'moa_r', 8),
            pretrained=getattr(cfg, 'pretrained_model', None),
            embed_dim=getattr(cfg, 'decoder_embed_dim', 768),
            activation_checkpoint=getattr(cfg, 'activation_checkpoint', False),
            criterion=criterion,
            freeze_backbone=getattr(cfg, 'freeze_backbone', False),
            norm_layer=norm_layer,
        )
