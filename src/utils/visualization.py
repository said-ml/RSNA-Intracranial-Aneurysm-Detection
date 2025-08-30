import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# local import
from src.dataset.aneurysm_dataset import AneurysmDataset


def show_aug_comparison(dataset, indices: list, slice_idx: int = None):
    """
    Show original vs augmented images side by side for given indices.

    Args:
        dataset: AneurysmDataset (augment=False for originals)
        indices: list of indices to visualize
        slice_idx: int, which slice along depth to show (default: middle)
    """
    n_samples = len(indices)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 10))

    # Row 1: original images (no augmentation)
    for i, idx in enumerate(indices):
        dataset.augment = False
        sample = dataset[idx]
        img, mask, _ = sample
        # Select a slice
        D = img.shape[1]
        s = slice_idx if slice_idx is not None else D // 2
        img_slice = img[0, s]  # [H,W]
        mask_slice = mask[0, s] if mask is not None else None

        axes[0, i].imshow(img_slice.cpu().numpy(), cmap='gray')
        if mask_slice is not None:
            axes[0, i].imshow(mask_slice.cpu().numpy(), cmap='Reds', alpha=0.3)
        axes[0, i].set_title(f"Original {idx} (slice {s})")
        axes[0, i].axis('off')

    # Row 2: augmented images
    for i, idx in enumerate(indices):
        dataset.augment = True
        sample = dataset[idx]
        img, mask, _ = sample
        # Select same slice
        D = img.shape[1]
        s = slice_idx if slice_idx is not None else D // 2
        img_slice = img[0, s]
        mask_slice = mask[0, s] if mask is not None else None

        axes[1, i].imshow(img_slice.cpu().numpy(), cmap='gray')
        if mask_slice is not None:
            axes[1, i].imshow(mask_slice.cpu().numpy(), cmap='Reds', alpha=0.3)
        axes[1, i].set_title(f"Augmented {idx} (slice {s})")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    npz_dir = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation"
    labels_csv = r"C:/Users/Setup Game/Music/Favorites/Downloads/rsna_preprocessed_segmentation/labels.csv"

    # Initialize dataset
    dataset = AneurysmDataset(npz_dir, labels_csv=labels_csv, augment_fn=False, segmentation=True)
    print(f"Dataset size: {len(dataset)}")

    # Visualize 4 samples
    sample_indices = [0, 1, 2, 3]
    show_aug_comparison(dataset, sample_indices)

