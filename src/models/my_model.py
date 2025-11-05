import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSegUnet3D(nn.Module):
    def __init__(self,
                 in_channels=2,
                 num_segments=1,
                 num_pseudo_masks=1,#13
                 num_labels=14):
        super(MultiSegUnet3D, self).__init__()

        self.in_channels = in_channels
        self.num_segments = num_segments
        self.num_pseudo_masks = num_pseudo_masks
        self.num_labels = num_labels

        # ---------------- Encoder ---------------- #
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # keep depth

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # downsample depth and spatial

        # ---------------- Bottleneck ---------------- #
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        # ---------------- Decoder ---------------- #
        # First upsample restores spatial and depth equally
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Second upsample restores only spatial dims (keep depth)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # ---------------- Classification Head ---------------- #
        self.clf_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, num_labels)
        )

        # ---------------- Segmentation Heads ---------------- #
        self.weak_seg_head = nn.Conv3d(32, num_segments, kernel_size=1)
        self.pseudo_seg_head = nn.Conv3d(32, num_pseudo_masks, kernel_size=1)

    def forward(self, x, target_mask=None):
        """
        x: [B, C, D, H, W]
        target_mask: Optional tensor for seg heads to align output depth
        """
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u1 = self.up1(b)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        # ---------------- Classification Head ---------------- #
        clf_out = torch.sigmoid(self.clf_head(d2))

        # ---------------- Segmentation Heads ---------------- #
        weak_seg_out = self.weak_seg_head(d2)
        pseudo_seg_out = self.pseudo_seg_head(d2)

        # If target_mask is provided, interpolate to its size
        #if target_mask is not None:
            #target_size = target_mask.shape[2:]  # D, H, W
            #weak_seg_out = F.interpolate(weak_seg_out, size=target_size, mode='trilinear', align_corners=False)
            #pseudo_seg_out = F.interpolate(pseudo_seg_out, size=target_size, mode='trilinear', align_corners=False)

        weak_seg_out = torch.sigmoid(weak_seg_out)
        pseudo_seg_out = torch.sigmoid(pseudo_seg_out)

        #return clf_out, weak_seg_out, pseudo_seg_out
        return clf_out






