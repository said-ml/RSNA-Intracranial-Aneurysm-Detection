import random
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
#from src.dataset.patch_dataset import PatchDataset     ===> DON'T NEEDED HERE IN THE CVTrainer class(src/training/trainer.py)


#print('local imports is OK')

##########################=================> Augmentation:
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch
import scipy.ndimage as ndi
import random
def augment(volume):
    volume = volume ** np.random.uniform(0.9, 1.1)  # sim contrast
    volume += np.random.normal(0, 0.01, volume.shape)  # small speckle
    return volume


def augment1(volume):
    """
    Apply mild 3D augmentations preserving vascular topology.
    volume: numpy array of shape (C, D, H, W) or (D, H, W)
    returns augmented numpy array
    """

    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()

    # --- Ensure correct shape ---
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]  # (1, D, H, W)

    # Copy to avoid modifying in-place
    vol = volume.copy()

    # -----------------------------
    # 1. Spatial augmentations
    # -----------------------------
    # a) Mild rotation (around z-axis)
    if random.random() < 0.5:
        angle = np.random.uniform(-10, 10)
        for c in range(vol.shape[0]):
            vol[c] = ndi.rotate(vol[c], angle, axes=(1, 2), reshape=False, order=1, mode='nearest')

    # b) Mild zoom (±5%)
    if random.random() < 0.5:
        zoom_factor = np.random.uniform(0.95, 1.05)
        vol_zoomed = ndi.zoom(vol, (1, 1, zoom_factor, zoom_factor), order=1)
        # Crop or pad back to original shape
        dz, dh, dw = vol_zoomed.shape[1:]
        D, H, W = vol.shape[1:]
        vol = _center_crop_or_pad(vol_zoomed, (vol.shape[0], D, H, W))

    # c) Elastic deformation (subtle)
    if random.random() < 0.3:
        alpha = np.random.uniform(15, 25)  # deformation intensity
        sigma = np.random.uniform(3, 5)    # smoothing
        vol = _elastic_deformation(vol, alpha, sigma)

    # -----------------------------
    # 2. Intensity augmentations
    # -----------------------------
    if random.random() < 0.5:
        # Brightness/contrast ±10%
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        vol = brightness * (vol - vol.mean()) * contrast + vol.mean()

    if random.random() < 0.4:
        # Gaussian noise (σ=0.01–0.03)
        sigma = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, sigma, vol.shape)
        vol = vol + noise

    # -----------------------------
    # 3. Random erasing (dropout-style)
    # -----------------------------
    if random.random() < 0.3:
        for _ in range(np.random.randint(1, 4)):
            D, H, W = vol.shape[1:]
            z = np.random.randint(0, D)
            h = np.random.randint(0, H)
            w = np.random.randint(0, W)
            dh = np.random.randint(8, 32)
            dw = np.random.randint(8, 32)
            vol[:, z, h:h+dh, w:w+dw] = 0

    return vol


def _elastic_deformation(volume, alpha, sigma):
    """Applies smooth elastic deformation to 3D volume."""
    random_state = np.random.RandomState(None)
    shape = volume.shape[1:]  # (D, H, W)
    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    deformed = np.empty_like(volume)
    for c in range(volume.shape[0]):
        deformed[c] = ndi.map_coordinates(volume[c], indices, order=1, mode='reflect').reshape(shape)
    return deformed


def _center_crop_or_pad(volume, target_shape):
    """Crops or pads volume to match target shape."""
    result = np.zeros(target_shape, dtype=volume.dtype)
    src_shape = volume.shape
    min_shape = [min(s, t) for s, t in zip(src_shape, target_shape)]
    src_start = [(s - m) // 2 for s, m in zip(src_shape, min_shape)]
    tgt_start = [(t - m) // 2 for t, m in zip(target_shape, min_shape)]

    slices_src = tuple(slice(src_start[i], src_start[i] + min_shape[i]) for i in range(len(src_shape)))
    slices_tgt = tuple(slice(tgt_start[i], tgt_start[i] + min_shape[i]) for i in range(len(target_shape)))
    result[slices_tgt] = volume[slices_src]
    return result


#############################################################################################################
#exit()
import torch
class AneurysmDataset(Dataset):
    def __init__(self,
                 npz_dir: str,
                 labels_csv: Optional[str] = None,
                 mode: str = 'train',
                 augment_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
                 segmentation: bool = False) -> None:
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
            self.num_classes: int = 14  # fallback

    def __len__(self) -> int:


         return len(self.files)-1# TODO ( THE LATEST SAMPLE IS CORRUPTED)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, str]]:
        npz_path: str = os.path.join(self.npz_dir, self.files[idx])
        data: np.lib.npyio.NpzFile = np.load(npz_path)
        image: torch.Tensor = torch.tensor(data['image'], dtype=torch.float32).unsqueeze(0)  # [C=1,D,H,W]
        mask: Optional[torch.Tensor] = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0) if 'mask' in data else None


        ###########################-------- This is a debugging shape ----------#####################
        # Ensure correct orientation
        image = image.squeeze()  # remove extra batch dim if exists
        if image.shape != (2, 7, 400, 400):
            image = image.permute(0, 2, 1, 3) if image.shape[2] == 7 else image  # adjust as needed

        mask = mask.squeeze()
        if mask.shape != (7, 400, 400):
            mask = mask.permute(1, 0, 2) if mask.shape[1] == 7 else mask
        ############################################################################################
        #############################################################################################
        if self.mode == 'train' and self.augment_fn is not None:
            image, mask =  image, mask#augment_3d_volume(image, mask)


        series_uid: str = self.files[idx].replace(".npz", "")

        if self.mode in ['train', 'val']:
            labels: torch.Tensor = self.labels_map.get(series_uid, torch.zeros(self.num_classes, dtype=torch.float32))
            if self.segmentation:
                return image, mask, labels, series_uid
            else:
                # augment
                #image = augment(image)
                #image = torch.tensor(image,dtype=torch.float32)
                return image, labels,series_uid
        else:  # test/inference
            return image, series_uid

    def __getitem2__(self, idx: int):
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

######################################------- Aneurysm Patch Dataset -------###############################
#==================================================>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
###########################################################################################################
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional, Callable, Tuple, List, Union

# Optional: your augmentations
# from your_augment_module import augment_3d_volume


class AneurysmPatchDataset(Dataset):
    def __init__(self,
                 npz_dir: str,
                 labels_csv: Optional[str] = None,
                 mode: str = 'train',
                 augment_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
                 segmentation: bool = True,
                 patch_size: Tuple[int, int, int] = (16, 256, 256),
                 num_patches_per_volume: int = 4,
                 patch_overlap: float = 0.5) -> None:
        """
        Args:
            npz_dir: folder with .npz files containing 'image' and optionally 'mask'
            labels_csv: optional CSV with SeriesInstanceUID + labels
            mode: 'train', 'val', or 'test'
            augment_fn: function(image, mask) -> image, mask
            segmentation: whether to return mask
            patch_size: (depth, height, width) of each 3D patch
            num_patches_per_volume: how many patches to sample per volume
            patch_overlap: fraction of overlap between adjacent patches
        """
        self.npz_dir = npz_dir
        self.files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
        self.mode = mode
        self.segmentation = segmentation
        self.augment_fn = augment_fn
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.patch_overlap = patch_overlap

        # load labels if available
        self.labels_map = {}
        if labels_csv and os.path.exists(labels_csv) and mode in ['train', 'val']:
            df = pd.read_csv(labels_csv)
            label_cols = list(df.columns[4:])
            for _, row in df.iterrows():
                uid = row['SeriesInstanceUID']
                ###################### added code to handle warning ###############
                values = torch.tensor(
                    row[label_cols]
                    .infer_objects(copy=False)  # Fix dtype before fillna
                    .fillna(0.0)  #  Now fill missing values
                    .astype(float)
                    .values,
                    dtype=torch.float32
                )

                #values = torch.tensor(values, dtype=torch.float32)       <====== all ready is a tensor
                #########################################################
                #values = torch.tensor(row[label_cols].fillna(0.0).values, dtype=torch.float32)
                self.labels_map[uid] = values
            self.num_classes = len(label_cols)
        else:
            self.num_classes = 14

    def __len__(self):
        # Each volume contributes multiple patches
        return len(self.files) * self.num_patches_per_volume

    def _sample_patch(self, volume: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Randomly extract one patch from the full 3D volume"""
        _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size

        # Ensure patch fits inside
        d0 = np.random.randint(0, max(1, D - pd + 1))
        h0 = np.random.randint(0, max(1, H - ph + 1))
        w0 = np.random.randint(0, max(1, W - pw + 1))

        image_patch = volume[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
        mask_patch = mask[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] if mask is not None else None

        return image_patch, mask_patch

    def __getitem__(self, idx: int):
        # Find which volume this patch belongs to
        volume_idx = idx // self.num_patches_per_volume
        npz_path = os.path.join(self.npz_dir, self.files[volume_idx])

        data = np.load(npz_path)
        image = torch.tensor(data['image'], dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        mask = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0) if 'mask' in data else None

        # fix orientation if necessary
        image = image.squeeze()
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, D, H, W)

        if mask is not None:
            mask = mask.squeeze()
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)

        # sample patch
        image_patch, mask_patch = self._sample_patch(image, mask)

        # optional augmentation
        if self.mode == 'train' and self.augment_fn is not None:
            image_patch, mask_patch = self.augment_fn(image_patch, mask_patch)

        # get label
        series_uid = self.files[volume_idx].replace(".npz", "")
        label = self.labels_map.get(series_uid, torch.zeros(self.num_classes, dtype=torch.float32))

        if self.segmentation:
            return image_patch, mask_patch, label, series_uid
        else:
            return image_patch, label, series_uid




############################################################################################################
############################################################################################################
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
                print(" Train and Val sets overlap!")
        if not train_set.isdisjoint(test_set):
            disjoint = False
            if verbose:
                print(" Train and Test sets overlap!")
        if not val_set.isdisjoint(test_set):
            disjoint = False
            if verbose:
                print(" Val and Test sets overlap!")

        if disjoint and verbose:
            print(" Train/Val/Test sets are strictly disjoint.")
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
###############################################################################################################################################################
###############################################################################################################################################################
import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------
# Custom collate_fn to handle variable-length arrays
# -----------------------------------------------
def rsna_collate_fn(batch):
    """
    Collate function for RSNA dataset that handles variable D (slice count)
    """
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    voxel_spacing = torch.stack([b["voxel_spacing"] for b in batch])
    slice_thickness = torch.stack([b["slice_thickness"] for b in batch])
    bbox = torch.stack([b["bbox"] for b in batch])





    coords = [b["coords"] for b in batch]
    # Keep variable-length arrays as lists
    ipp = [b["ImagePositionPatient"] for b in batch]
    iop = [b["ImageOrientationPatient"] for b in batch]
    uid = [b["uid"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "labels": labels,
        "voxel_spacing": voxel_spacing,
        "slice_thickness": slice_thickness,
        "bbox": bbox,
        "ImagePositionPatient": ipp,
        "ImageOrientationPatient": iop,
        "uid": uid,
        "coords": coords,
    }


def physical_to_voxel(phys_coord, ipp0, iop, pixel_spacing, slice_thickness):
    """
    Convert physical coordinates (X, Y, Z) → voxel coordinates (x, y, z).
    """
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    slice_cosine = np.cross(row_cosine, col_cosine)

    # Orientation * spacing matrix
    M = np.stack([
        row_cosine * pixel_spacing[1],
        col_cosine * pixel_spacing[0],
        slice_cosine * slice_thickness
    ], axis=1)

    # Convert
    voxel = np.linalg.inv(M) @ (phys_coord - ipp0)
    return voxel  # continuous voxel coordinates


def draw_sphere(mask, center, radius=3):
    """
    Draw a 3D spherical pseudo mask around voxel center (z,y,x).
    """
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    dist = (xx - center[2])**2 + (yy - center[1])**2 + (zz - center[0])**2
    mask[dist <= radius**2] = 1
    return mask


import torch
import numpy as np
import os, ast
import pandas as pd
from torch.utils.data import Dataset

def random_crop_3d(image, mask, pseudo_mask, crop_size):
    """Randomly crop a 3D patch from image, mask, and pseudo_mask."""
    #print(f'image shape: {image.shape}');exit()
    _,z, y, x = image.shape
    print(f'x={x}, y={y}, z={z}')#;exit()
    cz, cy, cx = crop_size
    print(f'cropped z={cz}, cropped y={cy}, cropped x={cx}')

    # Ensure crop fits in volume
    if z <= cz or y <= cy or x <= cx:
        # if volume smaller than crop, just center crop
        z1 = max((z - cz) // 2, 0)
        y1 = max((y - cy) // 2, 0)
        x1 = max((x - cx) // 2, 0)
    else:
        z1 = np.random.randint(0, z - cz)
        y1 = np.random.randint(0, y - cy)
        x1 = np.random.randint(0, x - cx)

    image_crop = image[z1:z1 + cz, y1:y1 + cy, x1:x1 + cx]
    mask_crop = mask[z1:z1 + cz, y1:y1 + cy, x1:x1 + cx]
    pseudo_crop = pseudo_mask[z1:z1 + cz, y1:y1 + cy, x1:x1 + cx]
    #print(f'image _cropped: {image_crop.shape}')
    #print(f'mask _cropped: {mask_crop.shape}')
    #print(f'pseudo_mask _cropped: {pseudo_crop.shape}');exit()
    return image_crop, mask_crop, pseudo_crop
def crop_3d(volume, crop_size=(24,  256, 256), start = None):

    """
    Safe 3D cropping for tomograms. Ensures valid shapes.
    """
    z, y, x = volume.shape[-3:]
    cz, cy, cx = crop_size

    if start is None:
        z_start = random.randint(0, max(0, z - cz))
        y_start = random.randint(0, max(0, y - cy))
        x_start = random.randint(0, max(0, x - cx))
    else:
        z_start, y_start, x_start = start

    z_end = min(z_start + cz, z)
    y_end = min(y_start + cy, y)
    x_end = min(x_start + cx, x)

    cropped = volume[..., z_start:z_end, y_start:y_end, x_start:x_end]

    # Pad if crop smaller (at borders)
    pad_z = cz - cropped.shape[-3]
    pad_y = cy - cropped.shape[-2]
    pad_x = cx - cropped.shape[-1]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        cropped = torch.nn.functional.pad(
            cropped,
            (0, pad_x, 0, pad_y, 0, pad_z),
            mode='constant',
            value=0,
        )

    return cropped, (z_start, y_start, x_start)


class RSNADataset(Dataset):
    def __init__(self, npz_dir, localizers_path=None, transform=None,
                 normalize=True, sphere_radius=3, crop_size=(36, 128, 128), train= not True):
        self.npz_files = sorted([
            os.path.join(npz_dir, f)
            for f in os.listdir(npz_dir)
            if f.endswith('.npz')
        ])
        self.localizers_df = pd.read_csv(localizers_path) if localizers_path else None
        self.transform = transform
        self.normalize = normalize
        self.sphere_radius = sphere_radius
        self.crop_size = crop_size
        self.train = train  # switch to False for full-volume validation/inference

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        file_path = self.npz_files[idx]
        uid = os.path.basename(file_path).replace(".npz", "")
        data = np.load(file_path, allow_pickle=True)

        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        labels = data["labels"].astype(np.float32)[3:]
        #print(f'labels=={labels}')
        #exit()
        voxel_spacing = data["voxel_spacing"].astype(np.float32)
        slice_thickness = float(data["slice_thickness"])
        bbox = data["bbox"].astype(np.int32)
        iop = data["ImageOrientationPatient"].astype(np.float32)
        ipp = data["ImagePositionPatient"].astype(np.float32)
        data.close()

        if self.normalize:
            image /= 255.0

        # Initialize pseudo mask
        pseudo_mask = np.zeros_like(image, dtype=np.uint8)

        # Build pseudo mask if localizers available
        if self.localizers_df is not None:
            match = self.localizers_df[self.localizers_df["SeriesInstanceUID"] == uid]
            if len(match) > 0:
                for coord_str in match["coordinates"]:
                    try:
                        coord_dict = ast.literal_eval(coord_str)
                        x = float(coord_dict.get("x", np.nan))
                        y = float(coord_dict.get("y", np.nan))
                        if not np.isnan(x) and not np.isnan(y):
                            phys_coord = np.array([x, y, ipp[2]])
                            voxel_coord = physical_to_voxel(
                                phys_coord, ipp, iop, voxel_spacing, slice_thickness
                            )
                            voxel_coord = np.round(voxel_coord[::-1]).astype(int)
                            if (
                                0 <= voxel_coord[0] < pseudo_mask.shape[0]
                                and 0 <= voxel_coord[1] < pseudo_mask.shape[1]
                                and 0 <= voxel_coord[2] < pseudo_mask.shape[2]
                            ):
                                pseudo_mask = draw_sphere(pseudo_mask, voxel_coord, self.sphere_radius)
                    except Exception:
                        continue

        # Random crop for training
        if self.train and self.crop_size is not None:
            image, mask, pseudo_mask = random_crop_3d(image, mask, pseudo_mask, self.crop_size)

        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)      # (1, D, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)        # (1, D, H, W)
        pseudo_mask = torch.from_numpy(pseudo_mask).unsqueeze(0)
        labels = torch.from_numpy(labels)

        image, start = crop_3d(image)
        mask, _ = crop_3d(mask, start=start)
        pseudo_mask, _ = crop_3d(pseudo_mask,  start=start)

        sample = {
            "image": image,
            "mask": mask,
            "pseudo_mask": pseudo_mask,
            "labels": labels,
            "uid": uid
        }
        if self.transform:
            sample = self.transform(sample)

        #return sample
        return image, labels, uid

##############################################################################################################################################################
###############################################################################################################################################################
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

