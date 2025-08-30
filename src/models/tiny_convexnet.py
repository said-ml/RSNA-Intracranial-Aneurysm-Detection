import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Channels-last LayerNorm ----------
class LayerNormChannelsLast(nn.Module):
    """LayerNorm for channels-last 3D tensors (B, D, H, W, C)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # -> (B, D, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)  # -> (B, C, D, H, W)
        return x

# ---------- Stochastic Depth ----------
class StochasticDepth(nn.Module):
    """DropPath for residual branch."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, device=x.device, dtype=x.dtype)
        return x / keep_prob * (rand < keep_prob)

# ---------- ConvNeXt 3D Block ----------
class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
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
        x = x * self.gamma.view(1, -1, 1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x

# ---------- ConvNeXt3D ----------
class ConvNeXt3D(nn.Module):
    def __init__(self, num_classes, in_channels=1, dims=(64, 128, 256, 512),
                 depths=(3, 3, 9, 3), drop_path_rate=0.1):
        super().__init__()

        # Stem: downsample x4
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1)
        )

        # Stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        # Build stages
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
                        LayerNormChannelsLast(in_dim),
                        nn.Conv3d(in_dim, dims[i+1], kernel_size=2, stride=2)
                    )
                )
                in_dim = dims[i+1]

        self.head_norm = LayerNormChannelsLast(in_dim)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Linear(in_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.stages:
            x = block(x)
        x = self.head_norm(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

