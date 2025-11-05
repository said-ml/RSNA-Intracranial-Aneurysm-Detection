import torch
from torch.utils.data.dataloader import default_collate

import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Dynamic runtime settings
    cfg["runtime"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["runtime"]["use_amp"] = torch.cuda.is_available()
    cfg["runtime"]["persistent_workers"] = cfg["runtime"]["num_workers_train"] > 0

    return cfg
import numpy as np

import torch
from typing import List, Tuple


def collate_classification(batch: List[Tuple]):
    images = torch.stack([x[0] for x in batch])
    labels = torch.stack([
        l if torch.is_tensor(l) else torch.tensor(l, dtype=torch.float32)
        for _, l, _,_ in batch
    ])
    patient_ids = [x[2] for x in batch]
    return images, labels, patient_ids


def collate_segmentation(batch: List[Tuple]):
    images = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    labels = torch.stack([
        l if torch.is_tensor(l) else torch.tensor(l, dtype=torch.float32)
        for _, _, l, _ in batch
    ])
    patient_ids = [x[3] for x in batch]
    return images, masks, labels, patient_ids

def safe_collate(batch,segmentation= True):
        if segmentation:
            images = torch.stack([x[0] for x in batch])
            masks = torch.stack([x[1] for x in batch])
            labels = torch.stack([x[2] for x in batch])
            patient_ids = [x[3] for x in batch]
            return images, masks, labels, patient_ids
        else:
            images = torch.stack([x[0] for x in batch])
            labels = torch.stack([x[1] for x in batch])
            patient_ids = [x[2] for x in batch]
            return images, labels, patient_ids


def pretty_print_cv(metrics: dict):
    """
    Pretty-print CV metrics per class and compute weighted RSNA metric.

    Args:
        metrics (dict): Output from ReliableCVMetric or your CVTrainer.validate()
            Expected keys: 'per_class_auc_last' (dict)
    """
    per_class_auc = metrics.get("per_class_auc_last", {})
    if not per_class_auc:
        print("⚠ No per-class AUC metrics found.")
        return

    # RSNA weights
    weights = {k: 1 for k in per_class_auc.keys()}
    if "Aneurysm Present" in weights:
        weights["Aneurysm Present"] = 13

    # Compute weighted average
    weighted_sum = sum(per_class_auc[k] * weights[k] for k in per_class_auc)
    total_weight = sum(weights.values())
    weighted_auc = weighted_sum / total_weight

    # Sort classes by AUC for readability
    sorted_auc = sorted(per_class_auc.items(), key=lambda x: x[1], reverse=True)

    print("\n📊 Per-Class CV AUC:")
    print("Class".ljust(40), "AUC")
    print("-" * 50)
    for cls, auc in sorted_auc:
        flag = "Good" if auc >= 0.7 else "!" if auc >= 0.5 else "Bad"
        print(f"{cls.ljust(40)} {auc:.4f} {flag}")

    print("\n🔹 Weighted RSNA-style CV metric:", f"{weighted_auc:.4f}")
    print("Good : ≥0.7 | Medium: 0.5–0.7 | Bad : <0.5\n")

#################################### FOR IMBALANCED DATA #########################################

import numpy as np
from torch.utils.data import Sampler

class MultiLabelStratifiedSampler(Sampler):
    """
    Ensures each batch contains positives from as many classes as possible.
    Works well for multilabel imbalance in medical datasets.
    """
    def __init__(self, labels, batch_size, shuffle=True):
        """
        labels: numpy array [num_samples, num_classes]
        """
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.shuffle = shuffle

        # Precompute positive indices per class
        self.class_indices = [np.where(labels[:, c] == 1)[0] for c in range(labels.shape[1])]

    def __iter__(self):
        indices = []
        all_indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(all_indices)

        while len(indices) < self.num_samples:
            batch = set()

            # Try to insert one positive per class until batch is full
            for c_idx in self.class_indices:
                if len(batch) >= self.batch_size:
                    break
                if len(c_idx) > 0:
                    batch.add(np.random.choice(c_idx))

            # Fill the rest randomly
            while len(batch) < self.batch_size:
                batch.add(np.random.choice(all_indices))

            indices.extend(list(batch))

        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size=16):
        """
        labels: np.ndarray shape (N, C) binary multi-label matrix
        """
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = labels.shape[0]
        self.class_indices = {c: np.where(labels[:, c] == 1)[0] for c in range(labels.shape[1])}
        self.all_indices = np.arange(self.num_samples)

    def __iter__(self):
        while True:
            batch = []
            # try to include at least 1 positive per class if possible
            for c, indices in self.class_indices.items():
                if len(indices) > 0:
                    batch.append(np.random.choice(indices))
                if len(batch) >= self.batch_size:
                    break

            # fill the rest randomly
            while len(batch) < self.batch_size:
                batch.append(np.random.choice(self.all_indices))

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size
