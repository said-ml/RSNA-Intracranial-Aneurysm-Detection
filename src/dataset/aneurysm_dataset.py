
import os
from typing import Optional, Callable, Tuple, List, Union
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

#print(f' cuda is availabe {torch.cuda.is_available()} ')
#local imports
from src.data.augmentations import augment_3d_volume


#print('local imports is OK')
#exit()

class AneurysmDataset(Dataset):
    def __init__(self,
                 npz_dir: str,
                 labels_csv: Optional[str] = None,
                 mode: str = 'train',
                 augment_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
                 segmentation: bool = not False) -> None:
        """
        npz_dir: folder with .npz files containing 'image' and optionally 'mask'
        labels_csv: CSV path with SeriesInstanceUID + labels
        mode: 'train', 'val', 'test'/'infer'
        augment_fn: optional function for augmentations, signature: image, mask -> image, mask
        segmentation: whether to return masks
        """
        self.npz_dir: str = npz_dir
        self.files: List[str] = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
        self.mode: str = mode
        self.segmentation: bool = segmentation
        self.augment_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]] = augment_fn or  augment_3d_volume

        self.labels_map: dict[str, torch.Tensor] = {}
        if labels_csv and os.path.exists(labels_csv) and self.mode in ['train', 'val']:
            df: pd.DataFrame = pd.read_csv(labels_csv)
            label_cols: List[str] = list(df.columns[4:])
            df_labels: pd.DataFrame = df[['SeriesInstanceUID'] + label_cols].copy()

            for col in label_cols:
                df_labels[col] = pd.to_numeric(df_labels[col], errors='coerce').fillna(0.0)

            for _, row in df_labels.iterrows():
                label_values: List[float] = [float(v) for v in row[label_cols].values]
                self.labels_map[row['SeriesInstanceUID']] = torch.tensor(label_values, dtype=torch.float32)

            self.num_classes: int = len(label_cols)
        else:
            self.num_classes: int = 15  # fallback

    def __len__(self) -> int:


         return len(self.files)-1# TODO ( THE LATEST SAMPLE IS CORRUPTED)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, str]]:
        npz_path: str = os.path.join(self.npz_dir, self.files[idx])
        data: np.lib.npyio.NpzFile = np.load(npz_path)
        image: torch.Tensor = torch.tensor(data['image'], dtype=torch.float32).unsqueeze(0)  # [C=1,D,H,W]
        mask: Optional[torch.Tensor] = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0) if 'mask' in data else None

        if self.mode == 'train' and self.augment_fn is not None:
            image, mask =  augment_3d_volume(image, mask)

        series_uid: str = self.files[idx].replace(".npz", "")

        if self.mode in ['train', 'val']:
            labels: torch.Tensor = self.labels_map.get(series_uid, torch.zeros(self.num_classes, dtype=torch.float32))
            if self.segmentation:
                return image, mask, labels, series_uid
            else:
                return image, labels,series_uid
        else:  # test/inference
            return image, series_uid

    def __getitem1__(self, idx: int):
        npz_path = os.path.join(self.npz_dir, self.files[idx])
        data = np.load(npz_path)
        image = torch.tensor(data['image'], dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0) if 'mask' in data else None

        if self.mode == 'train' and self.augment_fn is not None:
            image, mask = augment_3d_volume(image, mask)

        series_uid = self.files[idx].replace(".npz", "")
        labels = self.labels_map.get(series_uid, torch.zeros(self.num_classes, dtype=torch.float32))

        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.float().view(-1)
        if self.segmentation:
            return image, mask, labels, series_uid
        else:
            return image, labels, series_uid


# ---------------- DataLoader wrapper ----------------
def get_dataloader(npz_dir: str,
                   labels_csv: Optional[str] = None,
                   mode: str = 'train',
                   batch_size: int = 8,
                   shuffle: bool = True,
                   augment_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
                   segmentation: bool = False,
                   num_workers: int = 4) -> DataLoader:
    dataset: AneurysmDataset = AneurysmDataset(npz_dir, labels_csv=labels_csv, mode=mode,
                                               augment_fn= augment_3d_volume, segmentation=segmentation)
    loader: DataLoader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle if mode=='train' else False,
                                    num_workers=num_workers,
                                    pin_memory=True)
    return loader



from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torch
class RobustAneurysmDataset(AneurysmDataset):
    """
    Subclass of AneurysmDataset that automatically creates
    train/val/test splits ensuring:
        - No data leakage
        - Optional stratified split
        - Honest CV
    """

    def __init__(self,
                 npz_dir: str,
                 labels_csv: Optional[str] = None,
                 mode: str = "train",
                 augment_fn: Optional[Callable] = None,
                 segmentation: bool = True,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        super().__init__(npz_dir, labels_csv, mode="train", augment_fn=augment_fn, segmentation=segmentation)
        self.random_seed = random_seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        if labels_csv is not None:
            # Stratified split based on first label column
            label_col = list(pd.read_csv(labels_csv).columns[4:])[0]
            df_labels = pd.read_csv(labels_csv)
            strat_labels = df_labels[label_col].values

            sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=random_seed)
            train_idx, temp_idx = next(sss_train_val.split(np.zeros(len(strat_labels)), strat_labels))

            temp_labels = strat_labels[temp_idx]
            val_ratio_adj = val_ratio / (val_ratio + test_ratio)
            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_adj, random_state=random_seed)
            val_idx_rel, test_idx_rel = next(sss_val_test.split(np.zeros(len(temp_labels)), temp_labels))
            self.train_indices = train_idx
            self.val_indices = temp_idx[val_idx_rel]
            self.test_indices = temp_idx[test_idx_rel]

            # Create subsets
            self.train_dataset = Subset(self, self.train_indices)
            self.val_dataset = Subset(self, self.val_indices)
            self.test_dataset = Subset(self, self.test_indices)

        # Set current mode dataset
        if mode == "train":
            self.current_dataset = self.train_dataset
        elif mode == "val":
            self.current_dataset = self.val_dataset
        elif mode == "test":
            self.current_dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown mode {mode}, must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    # ----------------------
    # DataLoader helper
    # ----------------------
    def get_loader(self, batch_size=8, shuffle=True, num_workers=4):
        is_train = self.current_dataset == self.train_dataset
        return DataLoader(
            self.current_dataset,
            batch_size=batch_size,
            shuffle=shuffle if is_train else False,
            num_workers=num_workers,
            pin_memory=True
        )

    # ----------------------
    # Disjoint indices check
    # ----------------------
    def disjoint_indices(self, verbose=True):
        """
        Checks that train, val, test indices are strictly disjoint.
        Returns True if disjoint, False otherwise.
        """
        train_set = set(self.train_indices)
        val_set = set(self.val_indices)
        test_set = set(self.test_indices)

        disjoint = True
        if not train_set.isdisjoint(val_set):
            disjoint = False
            if verbose:
                print("❌ Train and Val sets overlap!")
        if not train_set.isdisjoint(test_set):
            disjoint = False
            if verbose:
                print("❌ Train and Test sets overlap!")
        if not val_set.isdisjoint(test_set):
            disjoint = False
            if verbose:
                print("❌ Val and Test sets overlap!")

        if disjoint and verbose:
            print("✅ Train/Val/Test sets are strictly disjoint.")
        return disjoint


'''
import pandas as pd
import matplotlib.pyplot as plt

# Suppose you have a CSV or dataframe with labels
# Columns: ['PatientID', 'Left Infraclinoid ICA', 'Right Infraclinoid ICA', ..., 'Aneurysm Present']
labels_csv: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv"
df = pd.read_csv(labels_csv)

# Count positives per class
class_counts = df.iloc[:, 1:].sum()  # skip PatientID
print(class_counts)

# Visualize
plt.figure(figsize=(12,6))
class_counts.plot(kind='bar')
plt.title("Positive Samples per Class")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.show()
'''

# ---------------- Example usage ----------------
if __name__ == '__main__':
    npz_dir: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    labels_csv: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv"

    train_loader: DataLoader = get_dataloader(npz_dir, labels_csv, mode='train', batch_size=4, augment_fn= augment_3d_volume, segmentation=True)
    for batch in train_loader:
        images, masks, labels,_ = batch
        print(f"Images: {images.shape}, Masks: {masks.shape}, Labels: {labels.shape}")
        break

    test_loader: DataLoader = get_dataloader(npz_dir, mode='test', batch_size=2, shuffle=False, segmentation=False)
    for images, uids in test_loader:
        print(f"Images: {images.shape}, SeriesInstanceUIDs: {uids}")
        break

