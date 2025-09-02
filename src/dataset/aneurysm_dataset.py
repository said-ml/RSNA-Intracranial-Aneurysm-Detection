import os
from typing import Optional, Callable, Tuple, Union, List
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

print(f' cuda is availabe {torch.cuda.is_available()} ')
#local imports
from data.augmentations import augment_3d_volume


print('local imports is OK')
exit()

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
                return image, mask, labels
            else:
                return image, labels
        else:  # test/inference
            return image, series_uid


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


# ---------------- Example usage ----------------
if __name__ == '__main__':
    npz_dir: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    labels_csv: str = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv"

    train_loader: DataLoader = get_dataloader(npz_dir, labels_csv, mode='train', batch_size=4, augment_fn= augment_3d_volume, segmentation=True)
    for batch in train_loader:
        images, masks, labels = batch
        print(f"Images: {images.shape}, Masks: {masks.shape}, Labels: {labels.shape}")
        break

    test_loader: DataLoader = get_dataloader(npz_dir, mode='test', batch_size=2, shuffle=False, segmentation=False)
    for images, uids in test_loader:
        print(f"Images: {images.shape}, SeriesInstanceUIDs: {uids}")
        break


