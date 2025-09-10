import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- ConvNeXt3D ----------
class LayerNormChannelsLast(nn.Module):
    """
    LayerNorm that expects channels-last tensors (B, D, H, W, C).
    For 3D, we permute between channels-first <-> channels-last as needed.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # to (B, D, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)  # back to (B, C, D, H, W)
        return x

class StochasticDepth(nn.Module):
    """DropPath for residual branch."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * (rand < keep_prob)

class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt block adapted to 3D:
      - Depthwise 3x3x3 conv
      - LayerNorm (channels-last)
      - 2x pointwise (1x1x1) convs with GELU
      - Residual with Stochastic Depth
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.ln = LayerNormChannelsLast(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # layer scale per-channel
        x = x * self.gamma.view(1, -1, 1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXt3D(nn.Module):
    """
    ConvNeXt3D-Tiny-ish:
      dims: [64, 128, 256, 512]
      depths per stage: [3, 3, 9, 3] (feel free to reduce if memory tight)
    """
    def __init__(self, num_classes, in_channels=1,
                 dims=(64, 128, 256, 512), depths=(3, 3, 9, 3),
                 drop_path_rate=0.1):
        super().__init__()

        # Stem (downsample by 4 overall via two stride-2 convs)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
        )

        # Stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        # Stages
        self.stages = nn.ModuleList()
        in_dim = dims[0]
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(ConvNeXtBlock3D(in_dim, drop_path=dpr[cur]))
                cur += 1
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            if i < 3:
                # Downsample between stages
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(in_dim),  # tiny overhead, but OK
                        nn.Conv3d(in_dim, dims[i+1], kernel_size=2, stride=2),
                    )
                )
                in_dim = dims[i+1]

        self.head_norm = LayerNormChannelsLast(in_dim)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Linear(in_dim, num_classes)

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        # Iterate over (stage, downsample) pairs
        # stages were appended as: stage0, down0, stage1, down1, stage2, down2, stage3
        for i, block in enumerate(self.stages):
            x = block(x)
        x = self.head_norm(x)
        x = self.pool(x).flatten(1)
        return self.head(x)  # logits

#################################################################################################################
import torch
import torch.nn as nn
import timm


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import torch
import torch.nn as nn
import timm


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    def __init__(self, pretrained = True, num_classes=14, in_chans=1, seg_out_ch=1, segmentation = True):
        super().__init__()
        # Backbone (EfficientNetV2)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=14  # remove classification head
        )  # timm backbone or ConvNeXt
        self.in_chans = in_chans
        self.num_classes = num_classes

        # Convert single-channel input to 3 channels if backbone expects 3
        if in_chans != 3:
            self.input_conv = nn.Conv2d(in_chans, 3, kernel_size=1)
        else:
            self.input_conv = nn.Identity()

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_linear = nn.Linear(self.backbone.num_features, num_classes)

        # Segmentation decoder (example)
        self.seg_decoder = nn.Sequential(
            nn.Conv3d(self.backbone.num_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, seg_out_ch, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, C, D, H, W] -> 5D tensor (depth included)
        """
        B, C, D, H, W = x.shape

        # Flatten depth into batch for backbone
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, D, C, H, W] -> [B*D, C, H, W]
        x = x.view(B * D, C, H, W)

        # Convert input channels if needed
        x = self.input_conv(x)  # -> [B*D, 3, H, W] for backbone

        # Backbone feature extraction
        feats = self.backbone(x)  # assume feats: [B*D, feat_dim, H', W']

        # -------------------
        # Classification branch
        # -------------------
        cls_feat = self.cls_pool(feats)         # [B*D, feat_dim, 1, 1]
        cls_feat = torch.flatten(cls_feat, 1)   # [B*D, feat_dim]
        cls_out = self.cls_linear(cls_feat)     # [B*D, num_classes]
        cls_out = cls_out.view(B, D, -1).mean(dim=1)  # [B, num_classes]

        # -------------------
        # Segmentation branch
        # -------------------
        seg_feat = feats.unsqueeze(2)           # [B*D, feat_dim, 1, H', W']
        seg_out = self.seg_decoder(seg_feat)    # [B*D, 1, D_out, H_out, W_out]
        seg_out = seg_out.view(B, D, seg_out.shape[1],
                               seg_out.shape[2], seg_out.shape[3], seg_out.shape[4])
        seg_out = seg_out.mean(dim=1)           # aggregate along depth -> [B, 1, D_out, H_out, W_out]

        return cls_out, seg_out

