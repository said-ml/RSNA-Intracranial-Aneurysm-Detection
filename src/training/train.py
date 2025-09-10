

import numpy as np
import pandas as pd
import torch
import random
import os

#from examples.nbeats import batch_size
#from examples.nbeats import batch_size
# local imports
from torch.utils.data import DataLoader
from torch import nn, optim
from src.dataset.aneurysm_dataset import AneurysmDataset
from src.models.resnet3d import resnet50 as ResNet3D  # your backbone
#from src.models.convexnext import MultiTaskModel as ResNet3D  # your backbone
from src.models.resnet3d import resnet18 as ResNet3D#, SliceBased3DModel
from src.submission.sample_submission import labels
from src.training.trainer import CVTrainer  # your updated custom trainer class
from src.utils.util import  safe_collate,collate_segmentation, collate_classification


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
batch_size = 16

# =============================
# Train loop
# =============================
if __name__ == '__main__':
    from multiprocessing import freeze_support
    # =============================
    # Dataset and DataLoader
    # =============================  Strict data split to avoid data leakage   ============
    import os
    import torch
    from torch.utils.data import DataLoader, random_split

    from src.utils.util import  MultiLabelStratifiedSampler, BalancedBatchSampler
    # -----------------------
    # 1. Paths
    # -----------------------
    npz_dir = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    labels_csv = os.path.join(npz_dir, "labels.csv")
    labels_df = pd.read_csv(labels_csv)
    array_labels = labels_df.drop(columns=["SeriesInstanceUID"]).values

    print('Done')
    # -----------------------
    # 2. Full Dataset (before splitting)
    # -----------------------
    full_dataset = AneurysmDataset(
        npz_dir,
        labels_csv=labels_csv,
        augment_fn=True,
        segmentation=True
    )

    # -----------------------
    # 3. Train/Validation Split
    # -----------------------
    val_ratio = 0.2  # 20% validation
    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )

    # -----------------------
    # 4. DataLoaders
    # -----------------------
    StratifiedSampler = MultiLabelStratifiedSampler(labels=array_labels[train_dataset.indices], batch_size=16)
    StratifiedSampler = BalancedBatchSampler(labels=array_labels[train_dataset.indices], batch_size=16)

    from torch.utils.data import DataLoader, WeightedRandomSampler
    import numpy as np

    # weights: higher for rare classes
    class_counts = np.sum(array_labels[train_dataset.indices], axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = np.dot(array_labels[train_dataset.indices], class_weights)
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        #shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        # sampler=MultiLabelStratifiedSampler(labels=labels_csv.values)  <--- shuffle = not True
        persistent_workers=True
    )

    # -----------------------
    # 5. Verify
    # -----------------------
    print(f"Length of train_loader: {len(train_loader)} batches")
    print(f"Length of val_loader: {len(val_loader)} batches")

    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    assert train_indices.isdisjoint(val_indices), "❌ Data leakage between train/val!"
    print("✅ Train and Val datasets are strictly disjoint")
    # =============================
    # Model
    # =============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D(num_classes=14,
                     segmentation=not False)  # both classification and segmentation
    model.to(device)

    # =============================
    # Loss and Optimizer
    # =============================
    # criterion_clf = nn.BCEWithLogitsLoss()
    from src.training.losses import SoftDiceLoss

    criterion_seg = SoftDiceLoss()
    from src.training.losses import FocalLoss

    # criterion_clf = FocalLoss(alpha=0.43, gamma=3.0)

    ##################### This code is added ###############
    import torch
    import torch.nn as nn

    # Example: weighted BCE for multi-label
    pos_counts = torch.tensor([74, 90, 293, 250, 203, 271, 344, 46, 54, 84, 100, 110, 110, 1722], dtype=torch.float).to(
        device)
    neg_counts = 236076 - pos_counts
    weights = neg_counts / (pos_counts + 1e-5)  # inverse frequency weighting
    weights = weights / weights.sum() * len(weights)  # normalize

    criterion_clf = nn.BCEWithLogitsLoss(pos_weight=weights)

    from src.training.losses import HardCustomLoss
    from src.training.losses import DiceBCELoss

    criterion_clf = HardCustomLoss()
    criterion_seg = DiceBCELoss()
    criterion_clf = FocalLoss(alpha=0.25, gamma=2.0)#, reduction="mean")

    ##########################################################

    # criterion_clf =  FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5
                            , weight_decay=1e-5)

    # =============================
    # CVTrainer setup
    # =============================
    from src.metric.for_cv import ReliableCVMetric, LABEL_COLS
    from multiprocessing import freeze_support
    freeze_support()
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
        max_epochs=75,
        save_dir="checkpoints"  ,# CVTrainer will handle saving
        metric = ReliableCVMetric(label_cols=LABEL_COLS)
    )

    trainer.fit()   # CVTrainer now manages epochs, training, validation & checkpoint saving

    val_loss, metrics = trainer.validate()  # run validation
    from src.utils.util import pretty_print_cv
    pretty_print_cv(metrics);exit()

