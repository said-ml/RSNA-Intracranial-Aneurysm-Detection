# ============================
# Metric: Mean Weighted ColAUC
# ============================
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# local imports
from src.configurations import*

AP_COL = 'Aneurysm Present'
LOC_COLS = LABEL_COLS[:-1]

def mean_weighted_colwise_auc(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    """Implements the competition metric."""
    aucs = {}
    for c in LABEL_COLS:
        y_t = y_true_df[c].values
        y_p = y_pred_df[c].values
        if len(np.unique(y_t)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_t, y_p)
        aucs[c] = auc
    ap = aucs[AP_COL]
    others = float(np.mean([aucs[c] for c in LOC_COLS]))
    final = 0.5 * (ap + others)
    return final, aucs