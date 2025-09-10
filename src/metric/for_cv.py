import numpy as np
from sklearn.metrics import roc_auc_score

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

ANEURYSM_PRESENT_IDX = LABEL_COLS.index("Aneurysm Present")


def rsna_weighted_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RSNA Intracranial Aneurysm Detection metric.

    Args:
        y_true (np.ndarray): shape (N, 14), ground truth binary labels.
        y_pred (np.ndarray): shape (N, 14), predicted probabilities.

    Returns:
        float: Weighted AUC score.
    """
    aucs = []
    for i in range(len(LABEL_COLS)):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = np.nan  # happens if only one class is present
        aucs.append(auc)

    auc_present = aucs[ANEURYSM_PRESENT_IDX]
    auc_other = [auc for i, auc in enumerate(aucs) if i != ANEURYSM_PRESENT_IDX]

    # Weighted average
    final_score = (13 * auc_present + np.nansum(auc_other)) / 14
    return final_score, dict(zip(LABEL_COLS, aucs))



#######################--------- Reliable CV Metric---------####################
#from __future__ import annotations      ====> at the beginning of the file
#from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from collections import defaultdict


class ReliableCVMetric:
    """
    RSNA competition-trustworthy CV metric.
    Features:
    - StratifiedGroupKFold compatible
    - Patient-level aggregation before AUC
    - Correct weighted RSNA metric
    - Multi-fold, multi-seed support
    - Out-of-Fold (OOF) dataframe for stacking/analysis
    """

    def __init__(self, label_cols: List[str], aneurysm_col: str = "Aneurysm Present"):
        self.label_cols = label_cols
        self.aneurysm_col = aneurysm_col
        self.num_labels = len(label_cols)

        # Multi-fold, multi-seed storage
        self.oof_preds = defaultdict(list)   # key = (seed, fold)
        self.oof_targets = defaultdict(list)
        self.patient_indices = defaultdict(list)
        self.fold_scores = defaultdict(list) # key = seed
        self.per_class_auc = defaultdict(dict)  # key = (seed, fold)

    def reset(self):
        self.oof_preds.clear()
        self.oof_targets.clear()
        self.patient_indices.clear()
        self.fold_scores.clear()
        self.per_class_auc.clear()

    def update(
        self, preds: np.ndarray, targets: np.ndarray,
        fold: int, seed: int, patient_idx: Optional[np.ndarray] = None
    ):
        """Store OOF predictions for a fold & seed."""
        key = (seed, fold)
        self.oof_preds[key].append(preds)
        self.oof_targets[key].append(targets)
        if patient_idx is not None:
            self.patient_indices[key].append(patient_idx)

    def _aggregate_patient_level(
        self, preds: np.ndarray, targets: np.ndarray, patient_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Average predictions & targets at patient level."""
        df = pd.DataFrame(preds, columns=self.label_cols)
        df["patient_id"] = patient_ids
        df_targets = pd.DataFrame(targets, columns=[c + "_target" for c in self.label_cols])
        df = pd.concat([df, df_targets], axis=1)

        df_grouped = df.groupby("patient_id").mean()
        y_true = df_grouped[[c + "_target" for c in self.label_cols]].values
        y_pred = df_grouped[self.label_cols].values
        return y_pred, y_true

    def compute_fold_score(self, fold: int, seed: int) -> float:
        """Compute weighted RSNA metric for a fold (patient-level)."""
        key = (seed, fold)
        preds = np.concatenate(self.oof_preds[key], axis=0)
        targets = np.concatenate(self.oof_targets[key], axis=0)
        patient_ids = np.concatenate(self.patient_indices[key], axis=0)

        # Aggregate to patient-level
        preds, targets = self._aggregate_patient_level(preds, targets, patient_ids)

        # Per-class AUCs
        per_class_auc = {}
        for i, col in enumerate(self.label_cols):
            try:
                per_class_auc[col] = roc_auc_score(targets[:, i], preds[:, i])
            except ValueError:
                per_class_auc[col] = np.nan

        self.per_class_auc[key] = per_class_auc

        # Weighted RSNA metric (correct formula)
        w = np.ones(self.num_labels)
        aneurysm_idx = self.label_cols.index(self.aneurysm_col)
        w[aneurysm_idx] = 13
        auc_values = np.array([per_class_auc[col] for col in self.label_cols])
        weighted_auc = np.nansum(auc_values * w) / np.sum(w)

        self.fold_scores[seed].append(weighted_auc)
        return weighted_auc

    def compute_seed_average(self, seed: int) -> float:
        """Average across all folds for a given seed."""
        return float(np.mean(self.fold_scores[seed]))

    def compute_overall(self) -> float:
        """Average across all seeds."""
        return float(np.mean([self.compute_seed_average(seed) for seed in self.fold_scores]))

    def get_oof_dataframe(self) -> pd.DataFrame:
        """Return full OOF dataframe for stacking/debugging (not aggregated)."""
        dfs = []
        for key in self.oof_preds:
            preds = np.concatenate(self.oof_preds[key], axis=0)
            targets = np.concatenate(self.oof_targets[key], axis=0)
            patient_ids = np.concatenate(self.patient_indices[key], axis=0)
            df = pd.DataFrame(preds, columns=self.label_cols)
            df_targets = pd.DataFrame(targets, columns=[c + "_target" for c in self.label_cols])
            df = pd.concat([df, df_targets], axis=1)
            df["patient_id"] = patient_ids
            dfs.append(df)
        return pd.concat(dfs, axis=0).reset_index(drop=True)

    def summary(self) -> dict:
        """Return full CV summary with per-seed, overall, and aggregated per-class AUCs."""
        # Aggregate per-class AUCs across all folds/seeds
        all_per_class = defaultdict(list)
        for key, aucs in self.per_class_auc.items():
            for c, v in aucs.items():
                all_per_class[c].append(v)

        per_class_avg = {c: float(np.nanmean(vs)) for c, vs in all_per_class.items()}

        summary = {
            "per_seed_average": {seed: self.compute_seed_average(seed) for seed in self.fold_scores},
            "overall_average": self.compute_overall(),
            "per_class_auc_avg": per_class_avg,
        }
        return summary

    def compute_holdout(self, preds: np.ndarray, targets: np.ndarray, patient_ids: np.ndarray) -> float:
        """Compute weighted RSNA metric on a holdout set (patient-level)."""
        preds, targets = self._aggregate_patient_level(preds, targets, patient_ids)

        per_class_auc = {}
        for i, col in enumerate(self.label_cols):
            try:
                per_class_auc[col] = roc_auc_score(targets[:, i], preds[:, i])
            except ValueError:
                per_class_auc[col] = 0.5

        w = np.ones(self.num_labels)
        aneurysm_idx = self.label_cols.index(self.aneurysm_col)
        w[aneurysm_idx] = 13
        auc_values = np.array([per_class_auc[col] for col in self.label_cols])
        weighted_auc = np.nansum(auc_values * w) / np.sum(w)
        #weighted_auc = np.sum(auc_values * w) / np.sum(w)  #<============ this is added
        return weighted_auc



