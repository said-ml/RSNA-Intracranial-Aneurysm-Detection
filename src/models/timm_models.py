import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RSNAHybridModel(nn.Module):
    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super(RSNAHybridModel, self).__init__()

        # ----------------------------
        # Encoder (ResNet50 backbone)
        # ----------------------------
        self.encoder = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            in_chans=3  # CT slices as 3 channels (RGB or stacked slices)
        )
        encoder_channels = self.encoder.num_features  # 2048

        # ----------------------------
        # Classification head
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(encoder_channels, num_classes)
        )

        # ----------------------------
        # Segmentation head (UNet-like decoder)
        # ----------------------------
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, 512, kernel_size=2, stride=2),  # upsample
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),  # 1 channel = aneurysm mask
        )

    def forward(self, x):
        # ----------------------------
        # Encoder backbone
        # ----------------------------
        feats = self.encoder.forward_features(x)  # [B, 2048, H/32, W/32]

        # ----------------------------
        # Classification
        # ----------------------------
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # [B, 2048]
        logits = self.classifier(pooled)  # [B, num_classes]
        probs = torch.sigmoid(logits)

        # ----------------------------
        # Segmentation
        # ----------------------------
        seg_mask = self.seg_head(feats)  # [B, 1, H/4, W/4]
        seg_mask = F.interpolate(seg_mask, size=x.shape[2:], mode="bilinear", align_corners=False)
        seg_mask = torch.sigmoid(seg_mask)

        return logits, seg_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


class ResNet3D_MultiTask(nn.Module):
    def __init__(self, num_classes=14,
                 segmentation = True):
        super().__init__()

        self.segmentation = segmentation
        # Backbone: 3D ResNet18
        self.encoder = r3d_18(pretrained=True)

        # Modify first conv to accept single-channel input
        self.encoder.stem[0] = nn.Conv3d(
            1, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        # Classification head (global pooled features → linear)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

        # Segmentation decoder (UNet-like upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        cls_out = self.classifier(feats)

        seg_out = self.decoder(feats)
        # Force seg_out to match seg_true size (D,H,W)
        seg_out = F.interpolate(
            seg_out,
            size=x.shape[2:],   # (D,H,W) of input
            mode="trilinear",
            align_corners=False
        )


        if  self.segmentation:
            return cls_out, seg_out
        else:
            return cls_out


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


import torch
import torch.nn as nn
import timm

class SliceBased3DModel(nn.Module):
    def __init__(self, num_classes=2, backbone_name="resnet18", pretrained=True, segmentation=True):
        super().__init__()
        self.segmentation = segmentation

        # Backbone with 1 input channel
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=1  # <--- fixes your 1-channel grayscale input
        )

        encoder_channels = self.encoder.feature_info.channels()  # list of channels per stage
        last_channels = encoder_channels[-1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channels, num_classes)
        )

        # Segmentation decoder (example)
        if self.segmentation:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(last_channels, 64, kernel_size=4, stride=4),
                nn.Conv2d(64, 1, kernel_size=1)
            )

    def forward(self, x):
        """
        x: [B, 1, D, H, W]
        """
        B, C, D, H, W = x.shape

        # Flatten depth into batch for 2D backbone
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # [B*D, 1, H, W]

        # Encoder
        feats = self.encoder(x)
        last_feat = feats[-1]  # [B*D, C_last, H_enc, W_enc]

        # Classification head
        cls_out = self.classifier(last_feat)  # [B*D, num_classes]
        cls_out = cls_out.view(B, D, -1).mean(dim=1)  # combine depth

        # Segmentation head
        if self.segmentation:
            seg_out = self.decoder(last_feat)  # [B*D, 1, H_dec, W_dec]
            seg_out = F.interpolate(seg_out, size=(D, H, W), mode='trilinear', align_corners=False)
            seg_out = seg_out.view(B, 1, D, H, W)

        else:
            seg_out = None

        return cls_out, seg_out


