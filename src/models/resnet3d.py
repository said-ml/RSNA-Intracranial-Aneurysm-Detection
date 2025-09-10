import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from typing import Optional, Tuple


class Aneurysm3DNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        segmentation: bool = not False,
        seg_channels: int = 1,
        dropout: float = 0.3,
        pretrained: bool = False
    ):
        """
        3D ResNet-based network for classification and optional segmentation

        Args:
            num_classes: Number of classification outputs (arteries + aneurysm present)
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
            seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='trilinear', align_corners=False)

        # Clamp logits to avoid extreme values
        cls_out = torch.clamp(cls_out, -20, 20)

        return cls_out, seg_out

################################################################modelMedicalRecalnet#############
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = [
    'resnet10', 'resnet18', 'resnet34', 'resnet50',
    'resnet101', 'resnet152', 'resnet200'
]

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False
    )


def downsample_basic_block(x, planes, stride):
    """Downsample using avg pooling + zero padding"""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        device=x.device, dtype=x.dtype
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1, in_channels=1, segmentation= False):
        super().__init__()
        self.inplanes = 64
        self.segmentation = segmentation

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.segmentation:
            #self.seg_head = nn.Conv3d(512 * block.expansion, num_classes, kernel_size=1)
            self.seg_head = nn.Sequential(  # this for resnet18
                nn.ConvTranspose3d(512, 256, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(256, 128, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(128, 64, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 1, 1)
            )
            self.seg_head2 = nn.Sequential(  # this for resnet50
                nn.ConvTranspose3d(
                    2048, 32, kernel_size=2, stride=2
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=1, bias=False)
            )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_orig = x  # save original input
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classification
        cls_out = self.avgpool(x)
        cls_out = torch.flatten(cls_out, 1)
        cls_out = self.fc(cls_out)

        # segmentation
        seg_out = None
        if self.segmentation:
            seg_out = self.seg_head(x)
            # interpolate to match ORIGINAL input size
            seg_out = F.interpolate(
                seg_out,
                size=x_orig.shape[2:],  # <-- key: use original input shape
                mode='trilinear',
                align_corners=False
            )
            return cls_out, seg_out
        else:
            return cls_out


# Factory functions
def resnet10(**kwargs):  return ResNet3D(BasicBlock, [1,1,1,1], **kwargs)
def resnet18(**kwargs):  return ResNet3D(BasicBlock, [2,2,2,2], **kwargs)
def resnet34(**kwargs):  return ResNet3D(BasicBlock, [3,4,6,3], **kwargs)
def resnet50(**kwargs):  return ResNet3D(Bottleneck, [3,4,6,3], **kwargs)
def resnet101(**kwargs): return ResNet3D(Bottleneck, [3,4,23,3], **kwargs)
def resnet152(**kwargs): return ResNet3D(Bottleneck, [3,8,36,3], **kwargs)
def resnet200(**kwargs): return ResNet3D(Bottleneck, [3,24,36,3], **kwargs)



if __name__ == '__main__':
    # Example: ResNet-18 with classification only
    model = resnet18(num_classes=14)

    # Example: ResNet-34 with classification + segmentation
    model_seg = resnet34(num_classes=14, segmentation=True)

    x = torch.randn(2, 1, 32, 256, 256)
    out = model(x)  # classification only
    #cls_out, seg_out = model_seg(x)  # classification + segmentation

    print("Classification only:", out.shape)
    #print("Classification + Segmentation:", cls_out.shape, seg_out.shape);exit()

    # Quick test
    #model = Aneurysm3DNet(num_classes=14, segmentation=False)
    #model = ResNet3D(num_classes=14, segmentation=False)

    #x = torch.randn(2, 1, 32, 256, 256)  # batch of 2
    #cls_out, seg_out = model_seg(x)
    #print("Classification output shape:", cls_out.shape)
    #print("Segmentation output:", seg_out)
