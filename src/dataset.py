import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
# locals import
from src.configurations import*

class AneurysmDataset(Dataset):
    """Dataset bridges CSV rows to processed 3D volumes and label vectors, with memory caching."""

    def __init__(self, data_df: pd.DataFrame, series_dir: str, processor):# DICOMProcessor):
        self.data_df = data_df.reset_index(drop=True)
        self.series_dir = series_dir
        self.processor = processor

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        #row = self.data_df.iloc[idx]
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.data_df.iloc[idx]   # ✅ works with Subset indices

        series_id = row[ID_COL]
        series_path = os.path.join(self.series_dir, series_id)

        volume = self.processor.load_dicom_series(series_path)  # (D,H,W) in [0,1]
        # ensure a writeable array to avoid PyTorch warning
        if not volume.flags.writeable:
            volume = volume.copy()

        labels = row[LABEL_COLS].values.astype(np.float32)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # (1,D,H,W)
        labels_tensor = torch.from_numpy(labels)
        return volume_tensor, labels_tensor