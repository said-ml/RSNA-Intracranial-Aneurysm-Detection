import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from src.dataset.aneurysm_dataset import AneurysmDataset
from src.models.resnet3d import Aneurysm3DNet as ResNet3D  # your backbone
from src.training.trainer import CVTrainer  # your updated custom trainer class
import os

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

# =============================
# Train loop
# =============================
if __name__ == '__main__':
    from multiprocessing import freeze_support
    # =============================
    # Dataset and DataLoader
    # =============================
    npz_dir = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    labels_csv = os.path.join(npz_dir, "labels.csv")

    train_dataset = AneurysmDataset(npz_dir, labels_csv=labels_csv, augment_fn=True, segmentation= not False)
    val_dataset = AneurysmDataset(npz_dir, labels_csv=labels_csv, augment_fn=False, segmentation= not False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # =============================
    # Model
    # =============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D(num_classes=14,
            segmentation= not False)  # both classification and segmentation
    model.to(device)

    # =============================
    # Loss and Optimizer
    # =============================
    criterion_clf = nn.BCEWithLogitsLoss()
    from src.training.losses import SoftDiceLoss
    criterion_seg = SoftDiceLoss()
    from src.training.losses import  FocalLoss
    #criterion =  FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # =============================
    # CVTrainer setup
    # =============================
    trainer = CVTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion_clf=criterion_clf,
        criterion_seg=criterion_seg,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=4,
        use_amp=True,
        max_epochs=5,
        save_dir="checkpoints"  # CVTrainer will handle saving
    )

    trainer.fit()   # CVTrainer now manages epochs, training, validation & checkpoint saving
