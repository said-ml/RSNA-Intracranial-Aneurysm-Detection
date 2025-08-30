import torch
import pandas as pd
from torch.utils.data import DataLoader
import pandas as pd
LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]
df = pd.read_csv(r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv")
labels = df[LABEL_COLS].values.astype(float)
print(f'labels shape={labels}');exit()
##########################################################################################################################################################
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# ----------------------------
# Temperature scaling module
# ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)


# ----------------------------
# Safe checkpoint loading
# ----------------------------

import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

def _load_model_for_fold(model_class, ckpt_path, device, num_classes):
    model = model_class(num_classes=num_classes).to(device)
    model.eval()
    ckpt = torch.load(ckpt_path,
                      map_location=device,
                      weights_only=True)

    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False) reported:")
        print(f"Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    return model


# ----------------------------
# Multi-fold inference with on-the-fly temperature
# ----------------------------
def multi_fold_inference(model_class, ckpt_paths, dataloader, device, num_classes=14, clamp_val=20.0):
    all_fold_probs = []
    series_uids = []

    for fold_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[Fold {fold_idx}] Loading: {ckpt_path}")
        model = _load_model_for_fold(model_class, ckpt_path, device, num_classes)

        fold_logits = []
        fold_labels = []  # only if you have labels for validation, optional
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                images, uids = batch
                images = images.to(device, non_blocking=True)

                out = model(images)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                ########################--------added code-------############
               # print("Logits stats:", logits.mean().item(), logits.std().item(), logits.min().item(),
                      #
                #logits.max().item());
                #exit()
                ##############################################################
                logits = torch.clamp(logits, min=-clamp_val, max=clamp_val)
                fold_logits.append(logits.cpu())

                if fold_idx == 0:
                    series_uids.extend(uids)

        fold_logits = torch.cat(fold_logits, dim=0)
        print(f"[Fold {fold_idx}] Collected logits shape: {tuple(fold_logits.shape)}")

        # Automatic temperature scaling (simple heuristic)
        logits_np = fold_logits.numpy()
        mean_abs = abs(logits_np).mean()
        T = max(0.5, min(5.0, mean_abs / 2))  # clamp between 0.5 and 5.0
        print(f"[Fold {fold_idx}] Automatic Temperature T = {T:.3f}")

        calibrated_probs = torch.sigmoid(fold_logits / T)
        all_fold_probs.append(calibrated_probs)

    # Ensemble: average probabilities across folds
    stacked_probs = torch.stack(all_fold_probs, dim=0)
    avg_probs = stacked_probs.mean(dim=0)
    return avg_probs, series_uids


# ----------------------------
# Submission helper
# ----------------------------
LABEL_C2OLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]


def create_submission_df(probs_tensor, series_uids):
    probs_np = probs_tensor.detach().cpu().numpy()
    df = pd.DataFrame(probs_np, columns=LABEL_COLS)

    location_probs = probs_tensor[:, :-1]
    present_probs = 1 - torch.prod(1 - location_probs, dim=1)
    df["Aneurysm Present"] = present_probs.detach().cpu().numpy()
    df.insert(0, "SeriesInstanceUID", series_uids)
    return df


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.dataset.aneurysm_dataset import AneurysmDataset
    from src.models.resnet3d import Aneurysm3DNet

    npz_dir = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    test_dataset = AneurysmDataset(npz_dir=npz_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=not False, num_workers=4)

    ckpt_paths = [
        "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch1.pt",
        "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch2.pt",
    ]

    probs_tensor, series_uids = multi_fold_inference(Aneurysm3DNet, ckpt_paths, test_loader, device)
    submission_df = create_submission_df(probs_tensor, series_uids)
    submission_df.to_csv("submission.csv", index=False)
    print(f'shape of DataFrame = {submission_df.shape}')
    print(submission_df.head(20))
    print(f'it is Done');exit()








##############################################################################################################################################################
# ----------------------------
# Safe checkpoint loading
# ----------------------------
def _load_model_for_fold(model_class, ckpt_path, device, num_classes):
    model = model_class(num_classes=num_classes).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)

    # Prefer explicit key
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Handle DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Strict load: if it fails, show keys and raise so we don't silently use random weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False) reported:")
        print(f"       Missing keys    ({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
        print(f"       Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")
        # Try a strict load if no classifier mismatch; otherwise keep non-strict but warn
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[WARN] strict=True failed: {e}")
            print("[WARN] Proceeding with strict=False weights. Ensure your model definition matches the checkpoint exactly.")

    return model


# ----------------------------
# Multi-fold inference
# ----------------------------
def multi_fold_inferencegjh(model_class, ckpt_paths, dataloader, device, num_classes=14, clamp_val=20.0):
    """
    Returns:
        probs:  Tensor [N, num_classes] with probabilities in [0,1]
        series_uids: list[str] length N
    """
    all_fold_logits = []
    series_uids = []

    for fold_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[Fold {fold_idx}] Loading: {ckpt_path}")
        model = _load_model_for_fold(model_class, ckpt_path, device, num_classes)

        fold_logits = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                # Expect dataset to return (image_tensor, uid_string)
                images, uids = batch
                images = images.to(device, non_blocking=True)

                out = model(images)
                # Unpack if model returns (cls_logits, seg_logits)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                # Sanity checks
                if logits.ndim != 2 or logits.size(1) != num_classes:
                    raise RuntimeError(
                        f"[Fold {fold_idx}] Unexpected logits shape {tuple(logits.shape)}; "
                        f"expected [B, {num_classes}]."
                    )

                # Clamp to avoid saturation
                logits = torch.clamp(logits, min=-clamp_val, max=clamp_val)
                fold_logits.append(logits.cpu())

                # Collect UIDs only once (first fold)
                if fold_idx == 0:
                    series_uids.extend(uids)

                if step == 0:
                    with torch.no_grad():
                        _min, _max = logits.min().item(), logits.max().item()
                        print(f"[Fold {fold_idx}] First batch logits range after clamp: [{_min:.3f}, {_max:.3f}]")

        fold_logits = torch.cat(fold_logits, dim=0)  # [N, num_classes]
        print(f"[Fold {fold_idx}] Collected logits shape: {tuple(fold_logits.shape)}")
        all_fold_logits.append(fold_logits)

    # Stack folds and average logits: [F, N, C] -> [N, C]
    stacked = torch.stack(all_fold_logits, dim=0)
    avg_logits = stacked.mean(dim=0)

    # Final probabilities
    probs = torch.sigmoid(avg_logits)

    # Final sanity checks
    if len(series_uids) != probs.size(0):
        raise RuntimeError(
            f"UID count {len(series_uids)} != predictions {probs.size(0)}. "
            "Check dataloader order or UID collection."
        )

    with torch.no_grad():
        pmin, pmax = probs.min().item(), probs.max().item()
        print(f"[Ensemble] Probabilities range: [{pmin:.4f}, {pmax:.4f}]")
        print("[Ensemble] Sample probs[0]:", probs[0].tolist())

    return probs, series_uids


def multi_fold_inference(model_class, ckpt_paths, dataloader, device, num_classes=14, clamp_val=20.0, temperature=1.0):
    all_fold_logits = []
    series_uids = []

    for fold_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[Fold {fold_idx}] Loading: {ckpt_path}")
        model = _load_model_for_fold(model_class, ckpt_path, device, num_classes)

        fold_logits = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                images, uids = batch
                images = images.to(device, non_blocking=True)
                out = model(images)
                logits = out[0] if isinstance(out, (tuple,list)) else out
                # Clamp extreme logits
                logits = torch.clamp(logits, min=-clamp_val, max=clamp_val)
                fold_logits.append(logits.cpu())

                if fold_idx == 0:
                    series_uids.extend(uids)

                if step == 0:
                    _min, _max = logits.min().item(), logits.max().item()
                    print(f"[Fold {fold_idx}] First batch logits range after clamp: [{_min:.3f}, {_max:.3f}]")

        fold_logits = torch.cat(fold_logits, dim=0)
        print(f"[Fold {fold_idx}] Collected logits shape: {tuple(fold_logits.shape)}")
        all_fold_logits.append(fold_logits)

    stacked = torch.stack(all_fold_logits, dim=0)


    # After averaging logits
    avg_logits = stacked.mean(dim=0)

    # Temperature scaling
    T = 10.0
    scaled_logits = avg_logits / T

    # Probabilities
    probs = torch.sigmoid(scaled_logits)
    if len(series_uids) != probs.size(0):
        raise RuntimeError(f"UID count {len(series_uids)} != predictions {probs.size(0)}")

    print(f"[Ensemble] Probabilities range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
    print("[Ensemble] Sample probs[2]:", probs[2].tolist())
    return probs, series_uids ,avg_logits  # also return logits if needed

# ----------------------------
# Submission helper
# ----------------------------
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

def create_submission_dfjk(probs_tensor, series_uids):
    probs_np = probs_tensor.detach().cpu().numpy()
    df = pd.DataFrame(probs_np, columns=LABEL_COLS)

    location_probs = probs_tensor[:, :-1]

    # Union rule (probability of at least one aneurysm)
    present_probs = 1 - torch.prod(1 - location_probs, dim=1)

    df["Aneurysm Present"] = present_probs.detach().cpu().numpy()
    df.insert(0, "SeriesInstanceUID", series_uids)

    return df

def create_submission_df(probs_tensor, series_uids):
    probs_np = probs_tensor.detach().cpu().numpy()
    df = pd.DataFrame(probs_np, columns=LABEL_COLS)

    location_probs = probs_tensor[:, :-1]
    present_probs = 1 - torch.prod(1 - location_probs, dim=1)
    df["Aneurysm Present"] = present_probs.detach().cpu().numpy()
    df.insert(0, "SeriesInstanceUID", series_uids)
    return df

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.dataset.aneurysm_dataset import AneurysmDataset
    from src.models.resnet3d import Aneurysm3DNet

    # Load test dataset
    npz_dir = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    test_dataset = AneurysmDataset(npz_dir=npz_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Checkpoints
    ckpt_paths = ["C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch1.pt"]

    # --- Optional: Collect validation logits to fit temperature ---
    # Example: you must implement val_loader = DataLoader(validation_dataset, ...)
    # val_logits, val_labels = collect_validation_logits(model_class=Aneurysm3DNet, ckpt_paths=ckpt_paths, val_loader=val_loader, device=device)
    # T = fit_temperature(val_logits, val_labels, device=device)

    T = 5.0  # quick hack: set manually if you don’t have validation

    preds_tensor, series_uids, _ = multi_fold_inference(Aneurysm3DNet, ckpt_paths, test_loader, device, temperature=T)
    submission_df = create_submission_df(preds_tensor, series_uids)

    submission_df.to_csv("submission_calibrated.csv", index=False)
    print(submission_df.head(20));exit()



# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.dataset.aneurysm_dataset import AneurysmDataset  # your dataset class
    from src.models.resnet3d import Aneurysm3DNet #as ResNet3D18  # your model class

    npz_dir: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    # 1. Load test dataset
    test_dataset = AneurysmDataset(npz_dir=npz_dir, mode="test")  # implement split="test" in your dataset
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    #######################################################################################################################
    # 2. Define your checkpoint paths (all folds)
    ckpt_paths = [
        "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch1.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch2.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch3.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch4.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch5.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch6.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch7.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch8.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch9.pt",
        #"C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch10.pt",
    ]

    #######################################################################################################################
    model = Aneurysm3DNet(num_classes=14).to(device)
    ckpt = torch.load(ckpt_paths[0], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    sample, _ = next(iter(test_loader))
    sample = sample.to(device)
    out = model(sample)
    print(out)


    preds_tensor, series_uids = multi_fold_inference(Aneurysm3DNet, ckpt_paths, test_loader, device)
    submission_df = create_submission_df(preds_tensor, series_uids)

    # Save CSV
    submission_df.to_csv("submission.csv", index=False)
    print(submission_df.head(20));exit()
# Example usage
# -------------------------------
if __name__ == "__main__":
    from src.dataset.aneurysm_dataset import AneurysmDataset  # your dataset class
    from src.models.resnet3d import Aneurysm3DNet as ResNet3D18  # your model class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_dir: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    # 1. Load test dataset
    test_dataset = AneurysmDataset(npz_dir=npz_dir, mode="test")  # implement split="test" in your dataset
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 2. Define your checkpoint paths (all folds)
    ckpt_paths = [
        "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch1.pt",
        # "src/training/checkpoints/checkpoint_epoch2.pt",
        # "src/training/checkpoints/checkpoint_epoch2.pt",
        # add all folds you trained
    ]

    preds, uids = multi_fold_inference(
    lambda: Aneurysm3DNet(ResNet3D18(backbone_out_features=512), num_classes=15),
    ckpt_paths, test_loader, device)

    submission_df = create_submission(preds, uids)
    #display(submission_df.head())
    submission_df.to_csv("submission.csv", index=False);exit()

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

import numpy as np

# ----------------------------
# Define your model (backbone + classifier)
# ----------------------------
class Aneurysm3DNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = ResNet3D18(pretrained=True)
        self.classifier = nn.Linear(512, num_classes)
        # optional: segmentation head, etc.

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        # If backbone returns tuple (features, aux), just pick features
        if isinstance(features, tuple):
            features = features[0]

        out = self.classifier(features)

        if return_features:
            return out, features  # only if you need features
        else:
            return out
# ----------------------------
# Multi-fold inference function
# ----------------------------
def multi_fold_inference(model_class, ckpt_paths, test_loader, device):
    all_preds = []

    for ckpt_path in ckpt_paths:
        # load model
        model = Aneurysm3DNet(num_classes=14).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                images, uids = batch
                images = images.to(device)

                out = model(images)
                if isinstance(out, tuple):  # handle tuple output
                    out = out[0]
                fold_preds.append(out.cpu())

        all_preds.append(torch.cat(fold_preds, dim=0))

    # Average across folds
    avg_preds = torch.stack(all_preds, dim=0).mean(dim=0)
    return avg_preds


# ----------------------------
# Create submission DataFrame
# ----------------------------
def create_submission(preds, series_uids):
    """
    preds: numpy array (N_samples, 15)
    series_uids: list of SeriesInstanceUIDs
    """
    columns = [
        "SeriesInstanceUID",
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present"
    ]
    submission_df = pd.DataFrame(np.column_stack([series_uids, preds]), columns=columns)
    return submission_df

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    from src.dataset.aneurysm_dataset import AneurysmDataset  # your dataset class
    from src.models.resnet3d import Aneurysm3DNet as ResNet3D18  # your model class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_dir: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    # 1. Load test dataset
    test_dataset = AneurysmDataset(npz_dir=npz_dir, mode="test")  # implement split="test" in your dataset
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 2. Define your checkpoint paths (all folds)
    ckpt_paths = [
        "C:/Users/Setup Game/PycharmProjects/RSNA Intracranial Aneurysm Detection/src/training/checkpoints/checkpoint_epoch1.pt",
        # "src/training/checkpoints/checkpoint_epoch2.pt",
        # "src/training/checkpoints/checkpoint_epoch2.pt",
        # add all folds you trained
    ]

    # 3. Run multi-fold inference
    preds = multi_fold_inference(ResNet3D18, ckpt_paths, test_loader, device)

    # 4. Optional: apply TTA
    # preds = tta_inference(model, test_loader, device)

    # 5. Collect SeriesInstanceUIDs
    #uids = test_dataset.get_series_uids()  # implement this in your dataset
    # Run multi-fold inference
    preds_15 = multi_fold_inference(Aneurysm3DNet, ckpt_paths, test_loader, device)

    # Create submission
    # List of SeriesInstanceUIDs from test dataset
    series_uids = [s['uid'] for s in test_dataset]
    submission_df = create_submission(preds_15, series_uids)
    # 6. Create submission DataFrame
    submission_df = create_submission(preds, uids)

    # 7. Save CSV
    submission_df.to_csv("submission.csv", index=False)
    print("✅ Submission CSV generated: submission.csv");exit()

    # Test DataLoader
    #test_dataset = YourAneurysmTestDataset()  # replace with your dataset
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Checkpoints for multi-fold inference
    ckpt_paths = [
        "checkpoints/fold0_best.pt",
        "checkpoints/fold1_best.pt",
        "checkpoints/fold2_best.pt"
    ]

    # List of SeriesInstanceUIDs from test dataset
    series_uids = [s['uid'] for s in test_dataset]

    # Run multi-fold inference
    preds_15 = multi_fold_inference(Aneurysm3DNet, ckpt_paths, test_loader, device)

    # Create submission
    submission_df = create_submission(preds_15, series_uids)

    # Save to CSV
    submission_df.to_csv("submission.csv", index=False)

    print("Submission CSV ready!")
    #display(submission_df)
