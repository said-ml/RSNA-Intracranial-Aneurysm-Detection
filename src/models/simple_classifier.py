
import torch
import torch.nn as nn

# --- Simple 3D classifier model ---
class RSNAClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=14):
        super().__init__()
        # Very lightweight 3D CNN backbone
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
            #nn.BatchNorm3d(32),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            #nn.BatchNorm3d(64),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            #nn.BatchNorm3d(128),
            nn.InstanceNorm3d(128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x

##############################################################################################
##############################################################################################





# Our model architecture (must match training)
class Simple3DCNN(nn.Module):
    def __init__(self, num_labels=14):
        super(Simple3DCNN, self).__init__()

        # Same architecture as in training
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

    def __init__(self, num_classes=1, num_locations=13, dropout_main=0.0, dropout_fc=0.0):
        super().__init__()
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=1, padding=1, bias=False),  # change 1 -> 2
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
        #self.pool3 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
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
        #print(f"x shape = {x.shape}")
        x = self.initial(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)  # now safe
        x = self.res4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_main(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)

        main_pred = self.main_head(x)
        main_pred = torch.sigmoid(main_pred)
        location_pred = self.location_head(x)
        location_pred = torch.sigmoid(location_pred)
        return torch.cat([location_pred, main_pred], dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock3D(nn.Module):
    """Residual block with optional projection for channel/stride mismatch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection is not None:
            identity = self.projection(x)
        out = out + identity
        out = self.relu(out)
        return out


class RSNA3DModel2(nn.Module):
    """
    Hybrid 3D CNN + Transformer model.

    Parameters
    ----------
    num_classes : int
        Number of main output classes (e.g. 1 for binary aneurysm present).
    num_locations : int
        Number of location outputs (13 in your setup).
    cnn_channels : int
        Base number of channels used in the CNN (will be scaled in blocks).
    embed_dim : int
        Transformer token embedding dimension.
    transformer_depth : int
        Number of Transformer encoder layers.
    transformer_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to FC layers.
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_locations: int = 13,
        cnn_channels: int = 32,
        embed_dim: int = 256,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- CNN encoder ---
        # initial conv expects 2 input channels (you said images have 2 channels)
        self.initial = nn.Sequential(
            nn.Conv3d(2, cnn_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.ReLU(inplace=True),
        )

        # residual stages (we'll downsample spatial H/W but keep depth D mostly intact)
        self.res1 = ResidualBlock3D(cnn_channels, cnn_channels, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.res2 = ResidualBlock3D(cnn_channels, cnn_channels * 2, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.res3 = ResidualBlock3D(cnn_channels * 2, cnn_channels * 4, stride=1)
        # final pooling: downsample H,W again but not depth
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # reduce channels to a nice number for transformer
        self.feature_proj_conv = nn.Conv3d(cnn_channels * 4, embed_dim, kernel_size=1)

        # small CNN pooled feature for optional concat with transformer CLS
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # --- Transformer encoder ---
        # We'll flatten spatial dims (D',H',W') -> N tokens and feed to Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=transformer_heads, dim_feedforward=embed_dim * 4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        # Learnable CLS token + positional embeddings (max size decided dynamically at forward; here we prepare none)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = None  # will init on first forward to match token count

        # Final fusion head - optionally combine transformer CLS and pooled CNN
        fusion_dim = embed_dim + embed_dim  # cls + pooled proj
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final FC and heads
        self.fc = nn.Linear(embed_dim, embed_dim // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.main_head = nn.Linear(embed_dim // 2, num_classes)
        self.location_head = nn.Linear(embed_dim // 2, num_locations)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # small initialization routine
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=2, D, H, W)
        returns: (B, num_locations + num_classes)
        """
        B = x.shape[0]

        # --- CNN encoder forward ---
        x = self.initial(x)             # (B, C1, D, H, W)
        x = self.res1(x)
        x = self.pool1(x)               # downsample H,W
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)               # final downsample H,W

        # x is (B, C_feat, Dp, Hp, Wp)
        # project channels -> embed_dim
        feat = self.feature_proj_conv(x)  # (B, E, Dp, Hp, Wp)

        # pooled cnn feature (for fusion)
        pooled = self.global_pool(feat)   # (B, E, 1,1,1)
        pooled = pooled.view(B, -1)       # (B, E)

        # --- Tokens for transformer ---
        B, E, Dp, Hp, Wp = feat.shape
        N = Dp * Hp * Wp
        tokens = feat.flatten(2)          # (B, E, N)
        tokens = tokens.permute(0, 2, 1)  # (B, N, E)

        # initialize pos_embed lazily to match N
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N + 1):
            # create positional embeddings (1, N+1, E)
            device = tokens.device
            self.pos_embed = nn.Parameter(torch.randn(1, N + 1, E, device=device) * 0.02)

        # project tokens if embed_dim mismatched (here already E==embed_dim)
        # prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        tokens_with_cls = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+N, E)

        tokens_with_cls = tokens_with_cls + self.pos_embed  # broadcast pos emb

        # Transformer expects (B, S, E) with batch_first=True
        t_out = self.transformer(tokens_with_cls)  # (B, 1+N, E)

        # take CLS token
        cls = t_out[:, 0, :]  # (B, E)

        # fuse with pooled cnn feature
        fused = torch.cat([cls, pooled], dim=1)  # (B, 2E)
        fused = self.fusion_proj(fused)          # (B, E)

        # final MLP -> heads
        x = self.fc(fused)
        x = self.relu(x)
        x = self.dropout(x)

        main_logits = self.main_head(x)
        loc_logits = self.location_head(x)

        # apply sigmoid for outputs between 0..1 (keeps previous behavior)
        main_pred = torch.sigmoid(main_logits)
        loc_pred = torch.sigmoid(loc_logits)

        out = torch.cat([loc_pred, main_pred], dim=1)
        return out
