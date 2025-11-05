import numpy as np

def print_data_files(path:str)->None:
  # this example ==> path = '/home/saidkoussi/Downloads/rsna_48_384_384_all_elements/tmp_npz/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647.npz'
  data = np.load(path)
  for key in data.files:
      if key !='labels':
          continue
      print(f"{key}: shape={data[key][0:]}, dtype={data[key].dtype}")# <=== some elements has np.int data type
    #exit()
path = '/home/saidkoussi/Downloads/rsna_48_384_384_all_elements/tmp_npz/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647.npz'
#print_data_files(path = path);exit()

####################### ------ Build ready data loader to drop it  in training pipeline ------###########################
#########################################################################################################################
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_pseudo_mask(centers, shape, radius=3, sigma=1.5):
    """
    Create a pseudo mask volume with small Gaussian blobs around each aneurysm center.
    centers: list of (z, y, x)
    shape: (D, H, W)
    radius: radius (voxels) for spherical region
    """
    D, H, W = shape
    mask = np.zeros((len(centers), 3, D, H, W), dtype=np.float32)  # 3 for coordinate channels

    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")

    for i, (cz, cy, cx) in enumerate(centers):
        dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        blob = np.exp(-0.5 * (dist / sigma) ** 2)
        blob[dist > radius] = 0  # cut off far region

        # Assign same blob to all 3 coordinate channels (you can also encode positional maps)
        mask[i, 0] = blob
        mask[i, 1] = blob
        mask[i, 2] = blob

    return mask


class RSNADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)

        image = data["image"].astype(np.float32)  # (2, D, H, W)
        mask = data["mask"].astype(np.float32)[None]  # (1, D, H, W)
        labels = data["labels"].astype(np.float32)
        bbox = data["bbox"].astype(np.int32)
        uid = self.files[idx].replace(".npz", "")

        # === Pseudo-mask generation ===
        aneurysm_centers = []
        if labels.shape[0] >= 14:
            num_aneurysms = 13
            for i in range(num_aneurysms):
                # Adjust this extraction once you know where aneurysm centers are stored
                cz, cy, cx = np.random.randint(0, image.shape[1], 3)  # placeholder
                aneurysm_centers.append((cz, cy, cx))

        pseudo_mask = generate_pseudo_mask(
            aneurysm_centers, shape=image.shape[1:], radius=3, sigma=1.5
        )


        # we want last 14 labels not all 17
        labels = labels[3:]
        #print(f'labels.shape{labels.shape}')
        sample = {
            "image": torch.from_numpy(image),                # (2, D, H, W)
            "mask": torch.from_numpy(mask),                  # (1, D, H, W)
            "labels": torch.from_numpy(labels),              # (num_labels,)
            "pseudo_mask": torch.from_numpy(pseudo_mask),    # (num_aneurysms, 3, D, H, W)
            "uid": uid,
            "bbox": torch.from_numpy(bbox),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def rsna_collate_fn(batch):
    """
    Collate function for RSNA dataset with pseudo masks.
    """
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    bboxes = torch.stack([b["bbox"] for b in batch])

    # Pseudo masks may have different # of aneurysms, so pad dynamically
    max_aneurysms = max(b["pseudo_mask"].shape[0] for b in batch)
    D, H, W = batch[0]["image"].shape[-3:]
    padded_pseudo = torch.zeros((len(batch), max_aneurysms, 3, D, H, W))
    for i, b in enumerate(batch):
        n = b["pseudo_mask"].shape[0]
        padded_pseudo[i, :n] = b["pseudo_mask"]

    uids = [b["uid"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "labels": labels,
        "pseudo_mask": padded_pseudo,
        "bbox": bboxes,
        "uid": uids,
    }


if __name__ == "__main__":
    # Suppose you already have: images, masks, pseudo_masks, labels, uids


    import torch
    from torch.utils.data import DataLoader

    # ====== CONFIG ======
    npz_dir = "/home/saidkoussi/Downloads/rsna_48_384_384_all_elements/tmp_npz/"
    localizers_path = "/home/saidkoussi/Downloads/train_localizers.csv"

    from torch.utils.data import DataLoader

    #dataset = RSNADataset(npz_dir)
    #loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=rsna_collate_fn)

    #batch = next(iter(loader))
    #print(f'images shape = {batch["image"].shape}')  #
    #print(f' mask shape = {batch["mask"].shape}')  #
    #print(f' labels shape = {batch["labels"].shape}')  # (4,17)
    #print(f' pseud masks shape = {batch["pseudo_mask"].shape}')  # (4, 13, 3, 48, 384, 384)
    #print(f' bbox shape = {batch["bbox"].shape}')       #
    #exit()
# -----------------------------------------------




