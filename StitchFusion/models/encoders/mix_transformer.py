import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import DropPath, trunc_normal_


def _unwrap_checkpoint(checkpoint_data):
    if 'state_dict' in checkpoint_data:
        checkpoint_data = checkpoint_data['state_dict']
    if 'model' in checkpoint_data:
        checkpoint_data = checkpoint_data['model']
    return checkpoint_data


def load_mit_pretrained(model, pretrained):
    checkpoint_data = torch.load(pretrained, map_location='cpu')
    checkpoint_data = _unwrap_checkpoint(checkpoint_data)
    state_dict = {}
    for key, value in checkpoint_data.items():
        if key.startswith('patch_embed') or key.startswith('block') or key.startswith('norm'):
            state_dict[key] = value
    return model.load_state_dict(state_dict, strict=False)


def _bn_checkpoint_safe(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.training and m.track_running_stats:
                return False
    return True


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        padding = kernel // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding=padding, groups=dim)
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomPWConv(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = norm_layer(dim)
        nn.init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)


class BiDirectionalAdapter(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()
        self.adapter_down = nn.Linear(dim, r)
        self.adapter_mid = nn.Linear(r, r)
        self.adapter_up = nn.Linear(r, dim)
        self.dropout = nn.Dropout(0.1)

        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x = self.adapter_down(x)
        x = F.gelu(self.adapter_mid(x))
        x = self.dropout(x)
        return self.adapter_up(x)


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        inner_dim = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, inner_dim, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(inner_dim, channel, bias=False),
            nn.Sigmoid(),
        )
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2, norm_layer=norm_layer)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2, norm_layer=norm_layer)
        self.fc2 = nn.Linear(c2, c1)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        x = self.pwconv2(x + x1 + x2 + x3, H, W)
        return self.fc2(F.gelu(x))


class FeatureCross(nn.Module):
    def __init__(self, channels, num_modals, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channels = channels
        self.num_modals = num_modals
        self.linear_fusion_layers = nn.ModuleList(
            [nn.Linear(channel * self.num_modals, channel) for channel in self.channels]
        )
        self.mix_ffn = nn.ModuleList(
            [MixFFN(channel, channel, norm_layer=norm_layer) for channel in self.channels]
        )
        self.channel_attns = nn.ModuleList([ChannelAttentionBlock(channel) for channel in self.channels])

    def forward(self, x, layer_idx):
        B, _, H, W = x[0].shape
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.linear_fusion_layers[layer_idx](x)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W) + self.channel_attns[layer_idx](x_sum, H, W)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)


class PlainBlock(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, inputs, H, W):
        outputs = [x.clone() for x in inputs]
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.attn(self.norm1(outputs[i]), H, W))
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.mlp(self.norm2(outputs[i]), H, W))
        return outputs


class SharedAdapterBlock(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.0, moa_r=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))
        self.adap_t = BiDirectionalAdapter(dim, r=moa_r)
        self.adap2_t = BiDirectionalAdapter(dim, r=moa_r)

    def forward(self, inputs, H, W):
        outputs = [x.clone() for x in inputs]
        outputs_before_att = [x.clone() for x in outputs]
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.attn(self.norm1(outputs[i]), H, W))
        for i in range(len(outputs)):
            x_ori = outputs_before_att[i]
            for j in range(len(outputs)):
                if i != j:
                    outputs[j] = outputs[j] + self.drop_path(self.adap_t(self.norm1(x_ori)))

        outputs_before_mlp = [x.clone() for x in outputs]
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.mlp(self.norm2(outputs[i]), H, W))
        for i in range(len(outputs)):
            x_ori = outputs_before_mlp[i]
            for j in range(len(outputs)):
                if i != j:
                    outputs[j] = outputs[j] + self.drop_path(self.adap2_t(self.norm2(x_ori)))
        return outputs


class PairwiseAdapterBlock(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.0, num_modalities=2, moa_r=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

        for i in range(num_modalities):
            for j in range(num_modalities):
                if i < j:
                    setattr(self, f'adap_t_att{i + 1}{j + 1}', BiDirectionalAdapter(dim, r=moa_r))
                    setattr(self, f'adap_t_mlp{i + 1}{j + 1}', BiDirectionalAdapter(dim, r=moa_r))

    def forward(self, inputs, H, W):
        outputs = [x.clone() for x in inputs]
        outputs_before_att = [x.clone() for x in outputs]
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.attn(self.norm1(outputs[i]), H, W))
        for i in range(len(outputs)):
            x_ori = outputs_before_att[i]
            for j in range(len(outputs)):
                if i == j:
                    continue
                if i < j:
                    adapter = getattr(self, f'adap_t_att{i + 1}{j + 1}')
                else:
                    adapter = getattr(self, f'adap_t_att{j + 1}{i + 1}')
                outputs[j] = outputs[j] + self.drop_path(adapter(self.norm1(x_ori)))

        outputs_before_mlp = [x.clone() for x in outputs]
        for i in range(len(outputs)):
            outputs[i] = outputs[i] + self.drop_path(self.mlp(self.norm2(outputs[i]), H, W))
        for i in range(len(outputs)):
            x_ori = outputs_before_mlp[i]
            for j in range(len(outputs)):
                if i == j:
                    continue
                if i < j:
                    adapter = getattr(self, f'adap_t_mlp{i + 1}{j + 1}')
                else:
                    adapter = getattr(self, f'adap_t_mlp{j + 1}{i + 1}')
                outputs[j] = outputs[j] + self.drop_path(adapter(self.norm2(x_ori)))
        return outputs


MIT_SETTINGS = {
    'B0': ([32, 64, 160, 256], [2, 2, 2, 2]),
    'B1': ([64, 128, 320, 512], [2, 2, 2, 2]),
    'B2': ([64, 128, 320, 512], [3, 4, 6, 3]),
    'B3': ([64, 128, 320, 512], [3, 4, 18, 3]),
    'B4': ([64, 128, 320, 512], [3, 8, 27, 3]),
    'B5': ([64, 128, 320, 512], [3, 6, 40, 3]),
}


class StitchFusionBackbone(nn.Module):
    def __init__(
        self,
        model_name='B2',
        modals=('img', 'hag'),
        moa_type='obMoA',
        moa_r=8,
        use_checkpoint=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if model_name not in MIT_SETTINGS:
            raise ValueError(f'Unsupported MiT variant: {model_name}')
        if len(modals) != 2:
            raise ValueError(f'StitchFusion benchmark expects exactly 2 modalities, got {len(modals)}')

        embed_dims, depths = MIT_SETTINGS[model_name]
        self.channels = embed_dims
        self.num_stages = 4
        self.total_modalities = len(modals)
        self.use_checkpoint = bool(use_checkpoint)
        self.moa_type = str(moa_type).lower()
        self.moa_r = int(moa_r)

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        cur = 0

        for i in range(self.num_stages):
            patch_embed = PatchEmbed(
                3 if i == 0 else embed_dims[i - 1],
                embed_dims[i],
                7 if i == 0 else 3,
                4 if i == 0 else 2,
                7 // 2 if i == 0 else 3 // 2,
            )
            block = nn.ModuleList(
                [
                    self._make_block(
                        embed_dims[i],
                        num_heads[i],
                        sr_ratios[i],
                        dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
            cur += depths[i]

        self.feature_cross = FeatureCross(self.channels, self.total_modalities, norm_layer=norm_layer)

    def _make_block(self, dim, head, sr_ratio, dpr):
        if self.moa_type == 'obmoa':
            return PairwiseAdapterBlock(
                dim,
                head,
                sr_ratio=sr_ratio,
                dpr=dpr,
                num_modalities=self.total_modalities,
                moa_r=self.moa_r,
            )
        if self.moa_type in {'shared', 'sharedmoa'}:
            return SharedAdapterBlock(dim, head, sr_ratio=sr_ratio, dpr=dpr, moa_r=self.moa_r)
        if self.moa_type in {'none', 'plain'}:
            return PlainBlock(dim, head, sr_ratio=sr_ratio, dpr=dpr)
        raise ValueError(f'Unsupported moa_type={self.moa_type!r}')

    def _should_checkpoint_module(self, module, *inputs):
        return (
            self.use_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and _bn_checkpoint_safe(module)
            and any(torch.is_tensor(t) and t.requires_grad for t in inputs)
        )

    def _run_block(self, block, inputs, H, W):
        if not self._should_checkpoint_module(block, *inputs):
            return block(inputs, H, W)

        def custom_forward(*tensor_inputs):
            return tuple(block(list(tensor_inputs), H, W))

        return list(checkpoint(custom_forward, *inputs, use_reentrant=False))

    def _run_feature_cross(self, inputs, layer_idx):
        if not self._should_checkpoint_module(self.feature_cross, *inputs):
            return self.feature_cross(inputs, layer_idx)

        def custom_forward(*tensor_inputs):
            return self.feature_cross(list(tensor_inputs), layer_idx)

        return checkpoint(custom_forward, *inputs, use_reentrant=False)

    def freeze_shared_backbone(self):
        for name, param in self.named_parameters():
            keep_trainable = name.startswith('feature_cross') or 'adap' in name
            param.requires_grad = keep_trainable

    def forward(self, x):
        x_in = [tensor for tensor in x]
        outs = []

        for stage_idx in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{stage_idx + 1}')
            stage_tokens = []
            H = W = None
            for tensor in x_in:
                token, H, W = patch_embed(tensor)
                stage_tokens.append(token)

            blocks = getattr(self, f'block{stage_idx + 1}')
            for block in blocks:
                stage_tokens = self._run_block(block, stage_tokens, H, W)

            norm = getattr(self, f'norm{stage_idx + 1}')
            stage_features = []
            for token in stage_tokens:
                feature = norm(token).reshape(token.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
                stage_features.append(feature)

            fused = self._run_feature_cross(stage_features, stage_idx)
            outs.append(fused)
            x_in = stage_features

        return outs
