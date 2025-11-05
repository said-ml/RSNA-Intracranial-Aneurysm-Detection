# rsna_hybrid_model_stable.py
import torch
import torch.nn as nn
import timm
from transformers import DebertaV2Model, DebertaV2Config
from typing import Optional


# ---------- Pooling Layers ----------
class MeanPooling(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class GemPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        x = last_hidden_state.clamp(min=self.eps).pow(self.p)
        sum_embeddings = torch.sum(x * mask, dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.pow(1.0 / self.p)


# ---------- Stable Hybrid Model ----------
class RSNAHybridModel(nn.Module):
    """
    AMP-safe hybrid model for RSNA aneurysm detection.

    Output:
        concatenated tensor [location_pred(13), main_pred(1)] -> shape (B, 14)
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_locations: int = 13,
        encoder_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        transformer_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 768,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        pool: str = "gem",
        dropout_main: float = 0.3,
        gc: bool = True,
    ):
        super().__init__()

        # ----- Vision encoder -----
        self.image_encoder = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, in_chans=2
        )
        # Optional grad checkpointing inside vision encoder (safe-ish)
        if gc and hasattr(self.image_encoder, "set_grad_checkpointing"):
            try:
                self.image_encoder.set_grad_checkpointing(True)
            except Exception:
                pass

        n_features = getattr(self.image_encoder, "num_features", None)
        if n_features is None:
            raise RuntimeError("The chosen timm encoder does not expose 'num_features' attribute.")

        # projection to transformer embed dim
        self.proj = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # ----- Transformer (DeBERTa) -----
        config = DebertaV2Config.from_pretrained(transformer_name)
        # adjust to wanted dims (we intentionally keep vocabulary small because we're using inputs_embeds)
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.intermediate_size = hidden_size * 4
        config.vocab_size = 3
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1
        config.hidden_act = "gelu"

        self.transformer = DebertaV2Model(config)

        # IMPORTANT: do not enable gradient_checkpointing() inside DeBERTa here.
        # It interacts poorly with AMP + GradScaler in some setups and can cause
        # saved-tensor-free issues during backward. Keep it disabled unless you test carefully.
        # if gc:
        #     self.transformer.gradient_checkpointing_enable()

        # CLS token
        scale = hidden_size ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, hidden_size))

        # pooling
        self.pool = GemPooling() if pool == "gem" else MeanPooling()

        # fusion head (pool + cls)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_main),
        )

        # heads
        self.main_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Sigmoid())
        self.location_head = nn.Sequential(nn.Linear(hidden_size, num_locations), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """
        x: (B, 2, H, W) -> vision encoder -> transformer
        returns: (B, num_locations + num_classes)
        """
        # Stage 1: Vision encoder -> embedding
        # The timm encoder should return (B, n_features) since num_classes=0
        f = self.image_encoder(x)           # (B, n_features)
        f = self.proj(f).unsqueeze(1)       # (B, 1, hidden_size)

        # Build transformer inputs: prepend CLS token
        b, t, c = f.shape  # t should be 1 here (we treat the image as a single token)
        cls_emb = self.cls_embedding.repeat(b, 1, 1)  # (B,1,hidden_size)
        inputs = torch.cat([cls_emb, f], dim=1)       # (B, 1 + t, hidden_size)
        attn_mask = torch.ones((b, 1 + t), device=x.device)

        # --- Transformer forward: run with autocast disabled for safety inside transformer ---
        # Mixed precision at Trainer level is OK; we disable autocast here for the transformer subgraph
        # to avoid mismatches or saved-tensor issues with some HuggingFace models.
        with torch.cuda.amp.autocast(enabled=False):
            out = self.transformer(inputs_embeds=inputs, attention_mask=attn_mask)

        h = out.last_hidden_state  # (B, 1+t, hidden_size)
        pooled = self.pool(h, attn_mask)  # (B, hidden_size)
        cls_tok = h[:, 0, :]               # (B, hidden_size)

        # fuse
        z = torch.cat([pooled, cls_tok], dim=-1)  # (B, 2*hidden_size)
        z = self.fc1(z)                            # (B, hidden_size)

        # heads
        main_pred = self.main_head(z)        # (B, num_classes)
        loc_pred = self.location_head(z)     # (B, num_locations)

        return torch.cat([loc_pred, main_pred], dim=1)  # (B, num_locations + num_classes)


if __name__ == "__main__":
    # quick smoke test (small random input)
    model = RSNAHybridModel(num_classes=1, num_locations=13, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # convnext variant expects (B, 2, H, W) -> choose a small H/W for smoke test
    x = torch.randn(2, 2, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    print("out.shape:", out.shape)  # expect (2, 14)
