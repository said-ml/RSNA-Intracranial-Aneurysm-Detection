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

try:
   from transformers import get_cosine_schedule_with_warmup
except:
    Transformers_not_installed =False


from src.metric.for_cv import ReliableCVMetric
from src.training.losses import MultiTaskLoss



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
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = not True,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 10,
        save_dir: Optional[str] = None,
        segmentation: bool = not False,
        use_clip_grad_norm:bool=True,

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
        self.segmentation = segmentation
        self.scaler = GradScaler() if use_amp else None
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.model.to(self.device)
        self.use_clip_grad_norm = use_clip_grad_norm
        if scheduler:
               self.scheduler = get_cosine_schedule_with_warmup(
                                optimizer=self.optimizer,
                                num_warmup_steps=55     ,#self.warmup_steps,
                                num_training_steps=1150#self.total_steps,
                            )
        self.metric = metric
        if self.segmentation:
            self.criterion = MultiTaskLoss(alpha=.5, beta=1.0).to(self.device)
    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for step, batch in pbar:
            images, seg_labels, cls_labels,series_uid = batch
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            seg_labels = seg_labels.to(self.device)
            #if self.segmentation:
               # cls_labels, seg_labels = labels
                #cls_labels = cls_labels.to(self.device)
                #seg_labels = seg_labels.to(self.device)
            #else:
               # cls_labels = labels.to(self.device)
                #seg_labels = None

            #  Ensure classification labels always [B, num_classes]
            if cls_labels.ndim == 1:
                cls_labels = cls_labels.unsqueeze(0)

            #with autocast(enabled=self.use_amp, device_type="cuda"):
            from torch import autocast
            from contextlib import nullcontext
            with autocast('cuda') if self.use_amp else nullcontext():
                cls_out, seg_out = self.model(images)

                # classification loss
                loss = self.criterion_cls(cls_out, cls_labels)

                # segmentation loss (optional)

                if self.segmentation and seg_out is not None and seg_labels is not None:
                    #loss_seg = self.criterion_seg(seg_out, seg_labels)
                    ########++++++++++++=====> added code=========
                    seg_out = seg_out.squeeze(1)
                    loss, cls_loss, seg_loss = self.criterion(cls_out, cls_labels, seg_out,seg_labels)
                    #print(f'loss seg={seg_loss}')
                    #print(f'classification loss={cls_loss}')
                    #print(f'loss={loss}')
                    loss += 1.0 * seg_loss


                # scale for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            #if self.use_clip_grad_norm:
                #clip_grad_norm(self.model.parameters(), max_norm=5.0)
            if self.use_clip_grad_norm:
                    max_norm = get_max_norm(epoch, self.max_epochs)
                    clip_grad_norm(self.model.parameters(), max_norm=max_norm)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            running_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_description(
                f"Epoch {epoch} | Step {step + 1} | Loss {running_loss / (step + 1):.4f}"
            )
            self.global_step += 1

    @torch.no_grad()
    def validate1(self) -> Tuple[float, Dict[str, float]]:
        if self.val_dataloader is None:
            return 0.0, {}

        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        for batch in tqdm(self.val_dataloader, desc="Validation"):

           # if self.segmentation:
                #images, cls_labels, seg_labels = batch
                #images = images.to(self.device)
                #cls_labels = cls_labels.to(self.device)
                #seg_labels = seg_labels.to(self.device)
            #else:
                #images, cls_labels = batch
                #images = images.to(self.device)
                #cls_labels = cls_labels.to(self.device)
               # seg_labels = None

            # for batch in pbar:

            images, seg_labels, cls_labels,_ = batch
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            seg_labels = seg_labels.to(self.device)

            #added code
            #seg_labels = seg_labels.unsqueeze(1)  # add channel dimension
            ##
            #with autocast(enabled=self.use_amp, device_type='cuda'):
            from torch import autocast
            from contextlib import nullcontext
            with autocast('cuda') if self.use_amp else nullcontext():
            # training code

                cls_out, seg_out = self.model(images)
                loss = self.criterion_cls(cls_out, cls_labels)
                if self.segmentation and seg_out is not None and seg_labels is not None:
                    loss_seg = self.criterion_seg(seg_out, seg_labels)
                    loss += loss_seg
            val_loss += 1.0*loss.item() * images.size(0)

            all_labels.append(cls_labels.cpu().numpy())
            all_preds.append(torch.sigmoid(cls_out).cpu().numpy())

        val_loss /= len(self.val_dataloader.dataset)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        # Compute AUC
        auc = {}
        try:
            auc['AUC'] = roc_auc_score(all_labels, all_preds, average='macro')
        except ValueError:
            auc['AUC'] = float('nan')

        # Save best checkpoint
        if val_loss < self.best_val_loss and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }, best_path)
            self.best_val_loss = val_loss
            print(f"✅ Best checkpoint updated at {best_path} | Val Loss: {val_loss:.4f}")

        return val_loss, auc

    from typing import Tuple, Dict
    from contextlib import nullcontext
    from tqdm import tqdm
    import os
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score

    @torch.no_grad()
    def validate1(self) -> Tuple[float, Dict[str, float]]:
        """
        Run validation and update all metrics from ReliableCVMetric.
        Returns:
            val_loss: float
            metrics: dict[str, float] (includes per-class AUC, overall RSNA metric)
        """
        if self.val_dataloader is None:
            return 0.0, {}

        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            images, seg_labels, cls_labels = batch
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            seg_labels = seg_labels.to(self.device)

            # autocast for mixed precision if enabled
            from torch import autocast
            from contextlib import nullcontext
            with autocast('cuda') if self.use_amp else nullcontext():
                cls_out, seg_out = self.model(images)
                loss = self.criterion_cls(cls_out, cls_labels)
                if self.segmentation and seg_out is not None and seg_labels is not None:
                    loss_seg = self.criterion_seg(seg_out, seg_labels)
                    loss += loss_seg

            val_loss += loss.item() * images.size(0)

            # store preds for metrics
            all_labels.append(cls_labels.cpu().numpy())
            all_preds.append(torch.sigmoid(cls_out).cpu().numpy())

        val_loss /= len(self.val_dataloader.dataset)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        metrics = {}
        # Compute per-class AUC
        if hasattr(self, 'metric') and isinstance(self.metric, ReliableCVMetric):
            # Update ReliableCVMetric with a single fold/seed for validation
            self.metric.reset()
            self.metric.update(all_preds, all_labels, fold=0, seed=0)
            self.metric.compute_fold_score(fold=0, seed=0)
            metrics = self.metric.summary()
        else:
            # fallback: compute simple macro AUC
            try:
                metrics['AUC'] = roc_auc_score(all_labels, all_preds, average='macro')
            except ValueError:
                metrics['AUC'] = float('nan')

        # Save best checkpoint
        if val_loss < self.best_val_loss and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }, best_path)
            self.best_val_loss = val_loss
            print(f"✅ Best checkpoint updated at {best_path} | Val Loss: {val_loss:.4f}")

        return val_loss, metrics

    @torch.no_grad()
    def validate(self, fold=0, seed=0):
        self.model.eval()
        self.metric.reset()
        val_losses = []

        for batch in self.val_dataloader:
            ########## adde code <=========
            # inside validate loop
            if self.segmentation:
                inputs, masks, targets, patient_ids = batch
                inputs=inputs.to(self.device)
                targets = targets.to(self.device)
                masks =masks.to(self.device)
                outputs = self.model(inputs.to(self.device))
                ## addedcode
                #print(f'targets = {targets.shape}')
                #print('*****************************************************')
                #print(f'outputs = {outputs[0].shape}');


                ###########
                #if isinstance(outputs, tuple):
                    #outputs = outputs[0]
                #assert outputs.shape == targets.shape
                labels_clf, pred_masks = outputs
                labels_clf, pred_masks =  labels_clf.to(self.device), pred_masks.to(self.device)

                loss_clf = self.criterion_cls(labels_clf, targets)
                pred_masks=pred_masks.squeeze(1)
                loss_seg = self.criterion_seg(pred_masks, masks)
                loss = loss_clf+0.5*loss_seg
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
            '''
            print("BATCH TYPES:", [type(x) for x in batch])
            images, targets, patient_ids= batch  # unpack

            print("DEBUG targets:", type(targets), getattr(targets, "shape", None))
            outputs = self.model(images.to(self.device))

            try:
                inputs, masks, targets, patient_ids = batch
                inputs, masks, targets = (
                    inputs.to(self.device),
                    masks.to(self.device),
                    targets.to(self.device),
                )
                outputs = self.model(inputs)
            #else:
            except:
                # batch = (inputs, targets, patient_ids)
                inputs, targets, patient_ids = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

            # Compute loss
            #print(f'outputs  ={outputs}////////')
            #print(f'targets shape ={targets.shape}');exit()
            #if isinstance(outputs, tuple) :#and len(targets) == 1:
                #outputs = outputs[1]
            #print("outputs:", outputs.shape, "targets:", type(targets), getattr(targets, "shape", None))
            '''
            #loss = self.criterion_cls(outputs, targets)
            val_losses.append(loss.item())

            # Update CV metric
            self.metric.update(
                preds=labels_clf.detach().cpu().numpy(),
                targets=targets.detach().cpu().numpy(),
                fold=fold,
                seed=seed,
                patient_idx=np.array(patient_ids),  # patient-level grouping
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

    def fit(self) -> None:
        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch [{epoch}/{self.max_epochs}]")
            self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f} | Metrics: {val_metrics}")



##################################################################################################################
"""
from typing import Optional, Tuple, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score
import numpy as np


class CVTr1ainer:

    #HuggingFace-style Trainer for Computer Vision Kaggle Competitions.
    #Supports mixed precision, gradient accumulation, classification + segmentation.
    #Tracks validation AUC and saves the best checkpoint only.
    #
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        seg_criterion: Optional[nn.Module] = None,   # for segmentation masks
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 10,
        log_every: int = 50,
        use_clip_grad_norm:bool = True,
        save_dir: Optional[str] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.seg_criterion = seg_criterion
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.log_every = log_every
        self.save_dir = save_dir
        self.scaler = GradScaler("cuda") if use_amp else None
        self.global_step = 0
        self.best_val_loss = float("inf")   # for best checkpoint tracking
        self.use_clip_grad_norm = use_clip_grad_norm
        self.model.to(self.device)

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for step, batch in pbar:
            if len(batch) == 2:
                images, labels = batch
                masks = None
            else:  # (images, labels, masks)
                images, labels, masks = batch
                masks = masks.to(self.device)

            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)

                if isinstance(outputs, tuple):
                    cls_out, seg_out = outputs
                else:
                    cls_out, seg_out = outputs, None

                # classification loss
                loss = self.criterion(cls_out, labels)

                # segmentation loss (optional)
                if seg_out is not None and masks is not None and self.seg_criterion is not None:
                    seg_loss = self.seg_criterion(seg_out, masks)
                    loss = loss + 10*seg_loss

                loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.use_clip_grad_norm:
                  clip_grad_norm(self.model.parameters(),max_norm=1.0)  # adjust max_norm to prevent exploding or diminishing logits, in general set to 1.0


            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            running_loss += loss.item() * self.gradient_accumulation_steps
            if (step + 1) % self.log_every == 0:
                pbar.set_description(
                    f"Epoch {epoch} | Step {step+1} | Loss {running_loss/(step+1):.4f}"
                )

            self.global_step += 1

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        if self.val_dataloader is None:
            return 0.0, {}

        self.model.eval()
        val_loss = 0.0
        total = 0
        all_labels = []
        all_preds = []

        for batch in self.val_dataloader:
            if len(batch) == 2:
                images, labels = batch
                masks = None
            else:
                images, labels, masks = batch
                masks = masks.to(self.device)

            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    cls_out, seg_out = outputs
                else:
                    cls_out, seg_out = outputs, None

                # classification loss
                loss = self.criterion(cls_out, labels)

                # segmentation loss (optional)
                if seg_out is not None and masks is not None and self.seg_criterion is not None:
                    seg_loss = self.seg_criterion(seg_out, masks)
                    loss = loss + seg_loss

            val_loss += loss.item() * images.size(0)
            total += images.size(0)

            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(torch.sigmoid(cls_out).detach().cpu().numpy())

        val_loss /= total
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        # compute AUC (macro)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="macro")
        except ValueError:
            auc = float("nan")

        metrics = {"AUC": auc}
        return val_loss, metrics

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)

        # Save only the best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = os.path.join(self.save_dir, f"best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                    if self.optimizer
                    else None,
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                    "val_loss": val_loss,
                },
                path,
            )
            print(f"✅ Best checkpoint updated at {path} | Val Loss: {val_loss:.4f}")

    def fit(self) -> None:
        for epoch in range(1, self.max_epochs + 1):
            self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f} | Metrics: {val_metrics}")
            self.save_checkpoint(epoch, val_loss)
"""
