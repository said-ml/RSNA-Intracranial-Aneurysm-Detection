import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- DenseNet3D ----------
class _DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm3d(inter_channels)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer3D(channels, growth_rate, bn_size, drop_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)

class _Transition3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.norm(x), inplace=True))
        return self.pool(x)

class DenseNet3D(nn.Module):
    """
    DenseNet-BC style 3D network.
    Args:
        growth_rate: channels added per layer.
        block_config: list with number of layers per dense block.
        init_channels: initial stem channels.
        bn_size: bottleneck size multiplier.
        drop_rate: dropout inside dense layers.
        num_classes: output logits size.
    """
    def __init__(self, num_classes, growth_rate=24, block_config=(4, 8, 12, 8),
                 init_channels=32, bn_size=4, drop_rate=0.0, in_channels=1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True)
        )

        channels = init_channels
        blocks = []
        for i, num_layers in enumerate(block_config):
            db = _DenseBlock3D(num_layers, channels, growth_rate, bn_size, drop_rate)
            channels = db.out_channels
            blocks.append(db)
            if i != len(block_config) - 1:
                out_channels = channels // 2
                blocks.append(_Transition3D(channels, out_channels))
                channels = out_channels
        self.features = nn.Sequential(*blocks)

        self.norm_final = nn.BatchNorm3d(channels)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(channels, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.norm_final(x), inplace=True)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)  # logits
        return x
