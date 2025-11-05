from __future__ import annotations
from typing import Optional, List, Dict, Tuple,Any
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm  # clip_grad_norm is deprecated
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

#from src.data.preprocess import series_uids

try:
   from transformers import get_cosine_schedule_with_warmup
except:
    Transformers_not_installed = False

# local imports
from src.metric.for_cv import ReliableCVMetric
from src.training.losses import MultiTaskLoss
#from src.dataset.patch_dataset import PatchDataset     # <========= when we want to apply patch training
from src.training.losses import MultiTaskLoss
criteria = MultiTaskLoss()

import datetime
##############################################---------- TTA ------------------###########################
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate

def tta_predict(model, volume, device='cuda', tta_transforms=5):
    """
    Args:
        model: PyTorch model
        volume: input tensor, shape [C, D, H, W], single patient
        device: 'cuda' or 'cpu'
        tta_transforms: number of TTA augmentations
    Returns:
        avg_probs: averaged prediction across TTA
    """
    model.eval()
    volume = volume.to(device)
    probs_list = []

    # original prediction
    with torch.no_grad():
        probs = torch.sigmoid(model(volume))#.unsqueeze(0)))  # add batch dim
        probs_list.append(probs.cpu())

    for _ in range(tta_transforms):
        vol_aug = volume.cpu().numpy().copy()

        # small random rotation ±5°
        axes = [(1, 2), (1, 3), (2, 3)]
        for ax in axes:
            angle = np.random.uniform(-5, 5)
            vol_aug = rotate(vol_aug, angle, axes=ax, reshape=False, order=1, mode='nearest')

        # intensity scaling ±5%
        scale = np.random.uniform(0.95, 1.05)
        vol_aug = vol_aug * scale
        vol_aug = np.clip(vol_aug, 0, 1)

        vol_aug = torch.tensor(vol_aug, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(vol_aug))#.unsqueeze(0)))
            probs_list.append(probs.cpu())

    # average predictions
    avg_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
    return avg_probs.squeeze(0)  # remove batch dim


#########################################################################################################
#########################################################################################################
def is_memory_safe(threshold_gb=1.0):
    """Returns False if GPU memory left is below threshold."""
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    free = total - reserved
    #print(f"GPU Memory | Total: {total:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB")
    return free > threshold_gb


def get_max_norm(epoch, total_epochs):
    if epoch < total_epochs * 0.2:       # first 20%
        return 6.0
    elif epoch < total_epochs * 0.6:     # middle 20%
        return 4.0

    elif epoch < total_epochs * 0.8:  # middle 20%
        return 3.0

    else:                                # last 40%
        return 2.0

# inside your training loop:


class CVTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion_clf: Optional[nn.Module] = None,
        criterion_seg: Optional[nn.Module] = None,
        metric: Optional[nn.Module]=None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] =  not True,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 10,
        save_dir: Optional[str] = None,
        only_classification: bool = True,
        segmentation: bool =  False,
        use_clip_grad_norm:bool=   not True,
        patch_train: bool = True,

    )->None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion_cls = criterion_clf
        self.criterion_seg = criterion_seg
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.only_classification = only_classification
        self.segmentation = segmentation
        self.scaler = GradScaler() if use_amp else None
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.model.to(self.device)
        self.use_clip_grad_norm = use_clip_grad_norm
        if self.scheduler:
               from torch.optim import lr_scheduler
               self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)


        self.metric = metric
        if self.segmentation:
            self.criterion = MultiTaskLoss().to(self.device)
        self.patch_train = patch_train

    def train_one_epoch(self, epoch: int) -> None:
        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        self.model.train()
        for step, batch in pbar:
            image, labels, series_uid = batch
            #print(f'labels shape = {labels.shape}')
            images, clf_labels, series_uid = image.to(self.device).squeeze(1),  labels.to(self.device), series_uid
            running_loss = 0.0
            if not is_memory_safe(threshold_gb=1.0):
                #print(f" Memory threshold reached — stopping training at epoch {epoch}, {step}")
                return  # Stop training early

            #images = batch['image']
            #cls_labels = batch['labels']
            #seg_labels = batch['mask']
            #pseudo_seg = batch['pseudo_mask']
            #series_uids= batch['uid']
            # move all tensors from CPU to GPU (CUDA )
            images = images.to(self.device).squeeze(1)
            #cls_labels = cls_labels.to(self.device)
            #clf_labels = batch["labels"][:, 0].float().unsqueeze(1).to(self.device)
            #clf_labels = batch["labels"].float().to(self.device)
            #seg_labels = seg_labels.to(self.device).squeeze(1)
            #pseudo_seg = pseudo_seg.to(self.device).squeeze(1)[:,1:2,...]


            #print(f'images shape: {images.size()}')
            #print(f'cls_labels shape: {cls_labels.size()}')
            #print(f'seg_labels shape: {seg_labels.size()}')
            #print(f'pseud_seg shape: {pseudo_seg.size()}')#;exit()

            ########################------ WE write Down patch training split ----##########
            ################################################################################
            if self.patch_train:
                #patch_images, patch_seg_labels, patch_cls_labels, patch_series_uid = images
                pass

            #################################################################################

            #################################################################################
            #if self.segmentation:
               # cls_labels, seg_labels = labels
                #cls_labels = cls_labels.to(self.device)
                #seg_labels = seg_labels.to(self.device)
            #else:
               # cls_labels = labels.to(self.device)
                #seg_labels = None

            #  Ensure classification labels always [B, num_classes]
            if clf_labels.ndim == 1:
                cls_labels = cls_labels.unsqueeze(0)

            #with autocast(enabled=self.use_amp, device_type="cuda"):
            from torch import autocast
            from contextlib import nullcontext
            #with autocast('cuda') if self.use_amp else nullcontext():
            with torch.cuda.amp.autocast(enabled= True):
                # we begin our training with classification only
                if self.only_classification:
                    clf_out = self.model(images)
                    #print(f'preds mean={clf_out.mean().item()}, preds std={clf_out.std().item()}')

                    #cls_out, seg_out,  pseudo_seg_out = self.model(images)

                    # classification loss
                    loss = self.criterion_cls(clf_out, clf_labels)
                    #print("Initial Loss:", loss.item());exit()

                # segmentation loss (optional)

                if self.segmentation and seg_out is not None and seg_labels is not None:
                    #loss_seg = self.criterion_seg(seg_out, seg_labels)
                    ########++++++++++++=====> added code=========
                    #seg_out = seg_out.squeeze(1)
                    #print(f'seg_out: {seg_out.shape}'); print(f'seg_labels: {seg_labels.shape}')#;exit()
                    #print(f'seg_labels: {seg_labels.shape}')
                    seg_out = seg_out.squeeze(1)
                    #loss, cls_loss, seg_loss = self.criterion(cls_out, cls_labels, seg_out,seg_labels)
                    loss = criteria(clf_out, seg_out,pseudo_seg_out, cls_labels, seg_labels,pseudo_seg)

                    #print(f'loss seg={seg_loss}')
                    #print(f'classification loss={cls_loss}')
                    #print(f'loss={loss}')
                    #loss += 1.0 * seg_loss


                # scale for gradient accumulation
                #loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()

                #pass
            else:
                loss.backward()

            if self.use_clip_grad_norm:
                clip_grad_norm(self.model.parameters(), max_norm=10.0)
            #if self.use_clip_grad_norm:
                    #max_norm = get_max_norm(epoch, self.max_epochs)
                    #clip_grad_norm(self.model.parameters(), max_norm=max_norm)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            #####################################
            #####################################
            # safe gradient norm calculation
            #total_norm = 0.0
            #found_grad = False
            #for p in self.model.parameters():
                #if p.grad is not None:
                    #found_grad = True
                    #total_norm += p.grad.data.norm(2).item() ** 2
            #total_norm = total_norm ** 0.5 if found_grad else 0.0
            #print(f"Grad norm: {total_norm}")

            ###################################
            ###################################

            running_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_description(
                f"Epoch {epoch} | Step {step + 1} | Loss {running_loss / (step + 1):.4f}"
            )
            self.global_step += 1

    @torch.no_grad()
    def validate(self, fold=0, seed=0):
        self.model.eval()
        self.metric.reset()
        val_losses = []

        for batch in self.val_dataloader:
            image,  labels, series_uid = batch
            images, clf_labels, patient_idx = image.to(self.device), labels.to(self.device), series_uid
            ########## adde code <=========
            # inside validate loop
            #if self.segmentation:
            #images = batch['image']
            #cls_labels = batch['labels']
            #clf_labels = batch["labels"][:, 0].float().unsqueeze(1).to(self.device)
            #clf_labels = batch["labels"].float().to(self.device)
            #seg_labels = batch['mask']
            #pseudo_seg = batch['pseudo_mask']
            #patient_idx = batch['uid']
            # move all tensors from CPU to GPU (CUDA )
            images = images.to(self.device).squeeze(1)
            #cls_labels = cls_labels.to(self.device)
            #seg_labels = seg_labels.to(self.device).squeeze(1)
            #pseudo_seg = pseudo_seg.to(self.device).squeeze(1)[:,1:2,...]

            #inputs, masks, targets, patient_ids = batch
            #inputs=inputs.to(self.device)
            #targets = targets.to(self.device)
            #masks =masks.to(self.device)
            #outputs = self.model(inputs.to(self.device))
            ## addedcode
            #print(f'targets = {targets.shape}')
            #print('*****************************************************')
            #print(f'outputs = {outputs[0].shape}');


            ###########
            #if isinstance(outputs, tuple):
                #outputs = outputs[0]
            #assert outputs.shape == targets.shape
            #labels_clf, pred_masks = outputs
            #labels_clf, pred_masks =  labels_clf.to(self.device), pred_masks.to(self.device)

            #loss_clf = self.criterion_cls(labels_clf, targets)
            #pred_masks=pred_masks.squeeze(1)
            #loss_seg = self.criterion_seg(pred_masks, masks)
            #loss = loss_clf+0.5*loss_seg
            with torch.no_grad():
             #with torch.cuda.amp.autocast(enabled=True):
             if self.only_classification:
                    #print(f'image: {image.shape}, label: {labels.shape}')
                    #cls_out = tta_predict(self.model, image)
                    #cls_out = cls_out.to(self.device)
                    #cls_out, seg_out, pseudo_seg_out = self.model(images)
                    cls_out = self.model(images)
                    #print(cls_out.mean().item(), cls_out.std().item())
                    # classification loss
                    loss = self.criterion_cls(cls_out, clf_labels)
                    print(f'validataion loss: {loss:.4f}')

                    # segmentation loss (optional)

                    if self.segmentation and seg_out is not None and seg_labels is not None:
                        # loss_seg = self.criterion_seg(seg_out, seg_labels)
                        ########++++++++++++=====> added code=========
                        # seg_out = seg_out.squeeze(1)
                        # print(f'seg_out: {seg_out.shape}'); print(f'seg_labels: {seg_labels.shape}')#;exit()
                        # print(f'seg_labels: {seg_labels.shape}')
                        seg_out = seg_out.squeeze(1)
                        # loss, cls_loss, seg_loss = self.criterion(cls_out, cls_labels, seg_out,seg_labels)
                        loss = criteria(cls_out, seg_out, pseudo_seg_out, cls_labels, seg_labels, pseudo_seg)

                        # print(f'loss seg={seg_loss}')
                        # print(f'classification loss={cls_loss}')
                        # print(f'loss={loss}')
                        # loss += 1.0 * seg_loss

                    # scale for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

             else:
                inputs, targets, patient_ids = batch
                targets = targets.to(self.device)
                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # classification target shape should be [batch, num_classes] or [batch]
                if len(targets.shape) > 1: targets = targets.squeeze()
                loss = self.criterion_cls(outputs, targets)
            #print('Done')

            ###########################################
            # Unpack batch safely
            #if not self.segmentation:
                # batch = (inputs, masks, targets, patient_ids)

            val_losses.append(loss.item())

            # Update CV metric
            self.metric.update(
                preds=cls_out.detach().cpu().numpy(),
                targets=clf_labels.detach().cpu().numpy(),
                fold=fold,
                seed=seed,
                patient_idx=np.array(patient_idx),  # patient-level grouping
            )

        # Aggregate results
        val_loss = float(np.mean(val_losses))
        fold_score = self.metric.compute_fold_score(fold=fold, seed=seed)
        summary = self.metric.summary()

        # Logging
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Patient-level weighted CV (fold {fold}, seed {seed}): {summary['overall_average']:.4f}")
        print("Per-class patient-level AUCs:")
        for cls, auc in summary['per_class_auc_avg'].items():
            print(f"  {cls}: {auc:.4f}")

        return val_loss, summary

    def save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = os.path.join(self.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "val_loss": val_loss,
                "timestamp": str(datetime.datetime.now())
            }, path)
            print(f"✅ Best checkpoint saved: {path} | Val Loss: {val_loss:.4f}")

    def fit(self) -> None:
        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch [{epoch}/{self.max_epochs}]")
            self.train_one_epoch(epoch)
            torch.cuda.empty_cache()
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f} | Metrics: {val_metrics}")
            self.save_best_checkpoint(epoch, val_loss)
            torch.cuda.empty_cache()


