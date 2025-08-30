import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video im
ssification outputs (arteries + aneurysm present)
            segmentation: Whether to include segmentation head
            seg_channels: Number of output channels for segmentation mask
            dropout: Dropout before final classifier
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        self.segmentation = segmentation
        self.num_classes = num_classes

        # 3D ResNet backbone
        self.backbone = r3d_18(weights=None if not pretrained else "KINETICS400_V1")
        # Modify first conv to accept 1 channel
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        # Remove final FC layer
        self.backbone.fc = nn.Identity()

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

        # Optional segmentation head
        if segmentation:
            # Upsample features progressively to match input size
            self.seg_head = nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, seg_channels, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor [B, 1, D, H, W]

        Returns:
            classification: [B, num_classes]
            segmentation: [B, seg_channels, D, H, W] (if segmentation=True)
        """
        # Pass through backbone manually to keep feature map for segmentation
        features = self.backbone.stem(x)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)  # [B, 512, D/8, H/32, W/32] approx

        # Classification: global average pooling
        cls_features = F.adaptive_avg_pool3d(features, 1).flatten(1)
        cls_out = self.classifier(self.dropout(cls_features))

        # Segmentation: upsample to input spatial size
        seg_out = None
        if self.segmentation:
            seg_out = self.seg_head(features)
            # Optional: interpolate to match exact input size
            seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='trilinear', align_corners=False)

        # Clamp logits to avoid extreme values
        cls_out = torch.clamp(cls_out, -20, 20)

        return cls_out, seg_out


if __name__ == '__main__':
    # quick test
    model = Aneurysm3DNet(num_classes=14, segmentation=False)
    x = torch.randn(2, 1, 32, 256, 256)  # batch of 2
    cls_out, seg_out = model(x)
    print("Classification output shape:", cls_out.shape)
    print("Segmentation output:", seg_out)
