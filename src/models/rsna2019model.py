import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RSNA2019Model(nn.Module):
    """
    2D CNN backbone (EfficientNet/ResNet) + BiLSTM sequence model.
    Adapted for multi-label intracranial hemorrhage classification.
    """

    def __init__(self, backbone_name="efficientnet_b0", num_classes=13, hidden_size=256):
        super().__init__()
        # Backbone (any timm 2D model)
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # remove final FC

        # Sequence model across slices
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        x: (B, S, C, H, W)
        B = batch size, S = number of slices per study, C = channels
        """
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)  # merge batch + slices

        # CNN features
        feats = self.backbone(x)  # (B*S, in_features)
        feats = feats.view(b, s, -1)  # (B, S, in_features)

        # BiLSTM over slice sequence
        seq_out, _ = self.lstm(feats)  # (B, S, 2*hidden_size)

        # Slice-level predictions
        out = self.fc(self.dropout(seq_out))  # (B, S, num_classes)

        return out  # slice-level logits (can be pooled to exam-level)
