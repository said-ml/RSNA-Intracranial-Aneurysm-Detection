
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch import amp
from sklearn.model_selection import StratifiedShuffleSplit
import os

# local imports
from src.metric import mean_weighted_colwise_auc
from src.configurations import *
from  src.utils import*
from models.model1 import Robust3DCNN as Simple3DCNN
from models.daragh_model import Net as Simple3DCNN
from models.convexnext import ConvNeXt3D as Simple3DCNN
from models.rsna2019model import RSNA2019Model as Simple3DCNN
from models.densenet3d import DenseNet3D as Simple3DCNN
#from models.tiny_convexnet import ConvNeXt3D as Simple3DCNN
from src.losses import SmoothBCEWithLogitsLoss, AsymmetricLoss, FocalLoss
from src.dataset import AneurysmDataset


# =====================
# Training & Evaluation
# =====================
# =====================
# Training & Evaluation
# =====================
def compute_pos_weight(train_df: pd.DataFrame, label_cols: list, eps: float = 1.0) -> torch.Tensor:
    """pos_weight = (neg + eps) / (pos + eps) per column to counter class imbalance."""
    total = float(len(train_df))
    pos = train_df[label_cols].sum(axis=0).astype(float)
    neg = total - pos
    w = (neg + eps) / (pos + eps)
    return torch.tensor(w.values, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def evaluate_model(model: nn.Module, processor: DICOMProcessor, val_df: pd.DataFrame, series_dir: str,
                   batch_size: int = 1):
    """Validation with workers=0 to avoid CUDA+fork issues. Returns (MW-ColAUC, per-column AUC dict, mean BCE loss)."""
    model.eval()
    ds = AneurysmDataset(val_df, series_dir, processor)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS_VAL,
        pin_memory=False, persistent_workers=False
    )

    preds, trues = [], []
    val_loss = 0.0
    # Use pos_weight for consistency with training
    pw = compute_pos_weight(val_df, LABEL_COLS, eps=1.0).clone()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    for vols, labels in tqdm(dl, desc="Val", leave=False):
        vols = vols.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with amp.autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(vols)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
        val_loss += float(loss.item()) * vols.size(0)
        preds.append(probs.cpu().numpy())
        trues.append(labels.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)
    y_pred_df = pd.DataFrame(y_pred, columns=LABEL_COLS)
    y_true_df = pd.DataFrame(y_true, columns=LABEL_COLS)

    final, aucs = mean_weighted_colwise_auc(y_true_df, y_pred_df)
    val_loss = val_loss / max(len(ds), 1)
    return final, aucs, val_loss

def train_model(
    train_df: pd.DataFrame,
    series_dir: str,
    processor: DICOMProcessor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    aneurysm_present_boost: float = ANEURYSM_PRESENT_BOOST,
    patience: int = PATIENCE,
    save_path: str = "/kaggle/working/model_weights.pth",
    warm_start_path: str = None,
    monitor: str = "auc",  # "auc" (maximize MW-ColAUC) or "loss" (minimize val_loss)
) -> nn.Module:
    """Train 3D CNN with AMP, tqdm, early stopping. The best checkpoint is selected by `monitor`."""
    assert monitor in {"auc", "loss"}, "monitor must be 'auc' or 'loss'"

    # Stratified split by AP (simple hold-out)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    ap = train_df[AP_COL].values
    train_idx, val_idx = next(sss.split(train_df, ap))

    tr_df = train_df.iloc[train_idx].copy()
    va_df = train_df.iloc[val_idx].copy()

    # Optional subsampling for runtime practicality
    if TRAIN_MAX_SERIES is not None and len(tr_df) > TRAIN_MAX_SERIES:
        tr_df = tr_df.sample(TRAIN_MAX_SERIES, random_state=42)
    if VAL_MAX_SERIES is not None and len(va_df) > VAL_MAX_SERIES:
        va_df = va_df.sample(VAL_MAX_SERIES, random_state=42)

    # Datasets & loaders (create BEFORE touching CUDA)
    ds_tr = AneurysmDataset(tr_df, series_dir, processor)
    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=2 if NUM_WORKERS_TRAIN > 0 else None
    )

    # Now it is safe to create CUDA model
    model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)

    # Warm start if provided (load on CPU to avoid early CUDA use)
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            import torch
            state = torch.load(warm_start_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print(f"Warm-started from {warm_start_path}")
        except Exception as e:
            print(f"[Warm start warn] {e}")

    # Loss with pos_weight
    pos_weight = compute_pos_weight(tr_df, LABEL_COLS, eps=1.0).clone()
    if aneurysm_present_boost != 1.0:
        pos_weight[-1] = pos_weight[-1] * float(aneurysm_present_boost)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #criterion = SmoothBCEWithLogitsLoss(smoothing=.05, pos_weight=pos_weight)
    criterion = AsymmetricLoss()
    #criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=("max" if monitor == "auc" else "min"),
        patience=1
    )
    import torch

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #total_steps = len( dl_tr) * epochs  # e.g., 500 batches * 30 epochs

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
       # optimizer,
        #max_lr=1e-3,
       ##pct_start=0.3,
        #anneal_strategy='cos',
       # div_factor=25.0,  # initial LR = max_lr/div_factor
        #final_div_factor=1e4
    #)
    scaler = amp.GradScaler(enabled=USE_AMP)

    # Initialize best score
    best_score = -float('inf') if monitor == "auc" else float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_samples = 0

        pbar = tqdm(dl_tr, desc=f"Train {epoch}/{epochs}", leave=False)
        for vols, labels in pbar:
            vols = vols.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', enabled=USE_AMP):
                logits = model(vols)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * vols.size(0)
            n_samples += vols.size(0)
            pbar.set_postfix(loss=f"{running/max(n_samples,1):.4f}")

        train_loss = running / max(n_samples, 1)

        # Validation (workers=0 to avoid CUDA+fork)
        try:
            final_auc, per_col, val_loss = evaluate_model(model, processor, va_df, series_dir, batch_size=1)
            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | MW-ColAUC={final_auc:.4f}")
        except Exception as e:
            print(f"[Eval warning] {e}")
            final_auc, val_loss = -1.0, train_loss + 1.0

        # Choose monitored score
        score = final_auc if monitor == "auc" else val_loss

        # Step scheduler with monitored score
        scheduler.step(score)

        # Early stopping with monitored score
        is_better = (score > best_score) if monitor == "auc" else (score < best_score)
        if is_better:
            best_score = score
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Load best and save to /kaggle/working
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} (selected by monitor='{monitor}', best_score={best_score:.6f})")

    # Final validation summary
    try:
        final_auc, per_col, val_loss = evaluate_model(model, processor, va_df, series_dir, batch_size=1)
        print(f"[Final Val] val_loss={val_loss:.4f} | MW-ColAUC={final_auc:.4f}")
    except Exception as e:
        print(f"[Eval warning] {e}")

    model.eval()
    return model

# =======================================
# Global Initialization, Training & Save
# =======================================
print("Initializing processor (memory-only cache)...")
processor = DICOMProcessor(
    target_size=TARGET_SIZE,
    target_spacing_mm=TARGET_SPACING_MM,
    cta_window=CTA_WINDOW,
    mri_z_clip=MRI_Z_CLIP,
    lru_capacity=LRU_CAPACITY,
)

model = None  # model will be created inside train_model

if DO_TRAIN:
    try:
        full_df = pd.read_csv(TRAIN_CSV_PATH)
        print(f"Training on up to {TRAIN_MAX_SERIES} train series, batch={BATCH_SIZE}, epochs={EPOCHS}, patience={PATIENCE} ...")
        model = train_model(
            train_df=full_df,
            series_dir=SERIES_DIR,
            processor=processor,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            aneurysm_present_boost=ANEURYSM_PRESENT_BOOST,
            patience=PATIENCE,
            save_path="/model_weights.pth",
            warm_start_path="C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints/model_weights.pth"\
            if os.path.exists("C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints//model_weights.pth") else None,
            monitor="auc",  # save the epoch with the best validation MW-ColAUC
        )
    except Exception as e:
        print(f"[Train warning] {e}")
        # Fallback: create a model and save weights so that an artifact exists
        model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)
        torch.save(model.state_dict(), "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints/model_weights.pth")
        print("Saved a randomly initialized model to C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints/model_weights.pth")
else:
    # If training is disabled, still create and save a model skeleton
    model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)
    torch.save(model.state_dict(), "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints/model_weights.pth")
    print("Training disabled. Saved untrained model to /checkpoints/model_weights.pth")

print("Training notebook completed. Best epoch weights are saved at C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/checkpoints//model_weights.pth")