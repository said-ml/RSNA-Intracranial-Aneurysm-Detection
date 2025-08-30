import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers.models.fsmt.modeling_fsmt import SinusoidalPositionalEmbedding



class CFG:
    def __init__(self):
        self.backbone = 'resnet18'      # timm backbone
        self.windows = [0, 1, 2]        # number of channels/windows
        self.target = [i for i in range(14)]# ['label1','label2']  # number of output classes
        self.rnn_num_layers = 2
        self.rnn_dropout = 0.2
        self.head_dropout = 0.2
        self.loss = 'bce'
        self.offline_inference = False
        self.positive_weight = 2.0
        self.lr = 1e-3
        self.epochs = 30

cfg = CFG()

class Net(nn.Module):
    """
    Modified Daragh model to fit old training pipeline.
    Accepts input x of shape (B, 1, D, H, W)
    and outputs logits of shape (B, num_classes)
    """
    def __init__(self, cfg=cfg, num_classes = len(cfg.target)):
        super(Net, self).__init__()
        self.cfg = cfg
        pretrained = not False#cfg.offline_inference

        # Backbone
        #num_classes = len(cfg.target)
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=pretrained,
            in_chans=1,  # single channel input
            num_classes=num_classes #len(cfg.target)
        )
        hidden_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Positional embeddings (optional, can be removed if not needed)
        self.posemb = SinusoidalPositionalEmbedding(202, 16, 201)
        self.poschemb = SinusoidalPositionalEmbedding(4, 16, 3)

        # RNN (optional, can be reduced to 1D if you prefer)
        rnn_dim = hidden_size + self.posemb.embedding_dim + self.poschemb.embedding_dim
        self.rnn = nn.LSTM(
            rnn_dim,
            rnn_dim,
            batch_first=True,
            num_layers=cfg.rnn_num_layers,
            dropout=cfg.rnn_dropout,
            bidirectional=True
        )

        self.head = nn.Linear(rnn_dim * 2, len(cfg.target))
        self.dropout = nn.Dropout(cfg.head_dropout)

    def forward(self, x):
        """
        x: (B, 1, D, H, W)
        returns logits: (B, num_classes)
        """
        B, C, D, H, W = x.shape

        # Flatten slices: (B*D, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        feats = self.backbone(x_reshaped)  # (B*D, hidden)
        feats = feats.view(B, D, -1)  # (B, D, hidden)

        # -------- Positional Embeddings --------
        # slice positions → (B, D)
        slice_pos = torch.arange(D, device=x.device).unsqueeze(0).expand(B, D)
        posemb = self.posemb(slice_pos)  # (B, D, posemb_dim)

        # channel embeddings (dummy indices for 3 chans) → (B, D, 3)
        chlmat = torch.zeros(B, D, 3, device=x.device).long()
        poschemb = self.poschemb(chlmat)  # (B, D, poschemb_dim)

        # Ensure all are 3D tensors (B, D, feature_dim)
        if poschemb.dim() == 4:
            poschemb = poschemb.mean(2)  # reduce channel dim

        # Concatenate along feature dim
        embs = torch.cat([feats, posemb, poschemb], dim=-1)  # (B, D, rnn_dim)

        # -------- RNN --------
        logits_rnn, _ = self.rnn(embs)  # (B, D, 2*rnn_dim)
        logits_rnn = self.dropout(logits_rnn)

        # Pool across sequence length (D)
        logits = self.head(logits_rnn).mean(dim=1)  # (B, num_classes)

        return logits


