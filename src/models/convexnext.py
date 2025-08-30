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
