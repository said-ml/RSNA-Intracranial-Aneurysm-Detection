import torch
import numpy as np
import random
from typing import Optional, Tuple


def augment_3d_volume(image: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      intensity_jitter: float = 0.1,
                      flip_prob: float = 0.5,
                      rotate_prob: float = 0.5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply optimized 3D augmentations for RSNA datasets.

    Args:
        image: torch.Tensor [C,D,H,W]
        mask: optional torch.Tensor [C,D,H,W]
        intensity_jitter: float, max relative intensity change
        flip_prob: probability to apply flip along H/W axes
        rotate_prob: probability to apply 90-degree rotation along axial slices
    Returns:
        image, mask
    """

    # -------------------- 1. Random flips --------------------
    if random.random() < flip_prob:
        # Flip along W axis
        image = torch.flip(image, dims=[3])
        if mask is not None:
            mask = torch.flip(mask, dims=[3])
    if random.random() < flip_prob:
        # Flip along H axis
        image = torch.flip(image, dims=[2])
        if mask is not None:
            mask = torch.flip(mask, dims=[2])

    # -------------------- 2. Random 90-degree rotations --------------------
    if random.random() < rotate_prob:
        k = random.choice([1, 2, 3])
        # rotate each slice (D axis) along H and W
        image = torch.rot90(image, k=k, dims=[2, 3])
        if mask is not None:
            mask = torch.rot90(mask, k=k, dims=[2, 3])

    # -------------------- 3. Intensity jitter --------------------
    if intensity_jitter > 0:
        jitter = (torch.rand_like(image) - 0.5) * 2 * intensity_jitter
        image = image + jitter
        image = image.clamp(0.0, 1.0)  # assuming images are normalized to 0-1

    # -------------------- 4. Optional Gaussian noise --------------------
    if random.random() < 0.3:
        sigma = random.uniform(0.0, 0.05)
        noise = torch.randn_like(image) * sigma
        image = image + noise
        image = image.clamp(0.0, 1.0)

    return image, mask
