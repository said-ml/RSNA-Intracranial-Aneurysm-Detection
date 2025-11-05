import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """Residual block with proper shortcut projection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            identity = self.projection(x)
        out += identity
        out = self.relu(out)
        return out


class RSNA3DModel(nn.Module):
    """
    3D Residual Network for RSNA aneurysm detection.
    Flexible for num_classes (main head) and location prediction (13 locations)
    """

    def __init__(self, num_classes=1, num_locations=13, dropout_main=0.5, dropout_fc=0.3):
        super().__init__()
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        # Residual blocks
        self.res1 = ResidualBlock3D(32, 32, stride=1)
        self.pool1 = nn.MaxPool3d(2)
        self.res2 = ResidualBlock3D(32, 64, stride=1)
        self.pool2 = nn.MaxPool3d(2)
        self.res3 = ResidualBlock3D(64, 128, stride=1)
        self.pool3 = nn.MaxPool3d(2)
        self.res4 = ResidualBlock3D(128, 256, stride=1)

        # Global pool + FC
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout_main = nn.Dropout(dropout_main)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_fc = nn.Dropout(dropout_fc)

        # Heads
        self.main_head = nn.Linear(128, num_classes)
        self.location_head = nn.Linear(128, num_locations)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)
        x = self.res4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_main(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)

        main_pred = self.main_head(x)
        location_pred = self.location_head(x)
        return torch.cat([location_pred, main_pred], dim=1)
        #return main_pred, location_pred
