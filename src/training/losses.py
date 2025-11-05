import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        prob = prob.clamp(self.eps, 1 - self.eps)

        targets = targets.type_as(prob)

        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            pt_neg = (1 - prob).clamp(min=self.clip)
        else:
            pt_neg = 1 - prob

        # Loss for positives and negatives
        loss_pos = targets * torch.log(prob)
        loss_neg = (1 - targets) * torch.log(pt_neg)

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = prob * targets + pt_neg * (1 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt) ** one_sided_gamma

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)  # pt = sigmoid(logits) if target=1 else (1-sigmoid)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


class SmoothBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.
    Args:
        smoothing: label smoothing factor (0.0 = no smoothing).
        pos_weight: tensor of positive weights for imbalance (like BCEWithLogitsLoss).
        reduction: 'mean' or 'sum'
    """
    def __init__(self, smoothing=0.0, pos_weight=None, reduction='mean'):
        super(SmoothBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets: (batch, num_classes), logits: (batch, num_classes)
        with torch.no_grad():
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        return loss


    #######################---- all losses needed for rsna ------####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Binary Classification Losses
# ----------------------
class BCEWithLogits(nn.Module):
    """Standard BCE with logits loss for scan-level presence."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight.to(logits.device))

class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ----------------------
# Soft Dice / Segmentation Losses
# ----------------------
class SoftDiceLoss(nn.Module):
    """Soft Dice loss for segmentation of aneurysm masks."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: logits -> apply sigmoid
        ########## addedcode for debuuging-----##########
        if preds.ndim == 4:  # (B, D, H, W)
            preds = preds.unsqueeze(1)  # -> (B, 1, D, H, W)
        if targets.ndim == 4:
            targets = targets.unsqueeze(1)

        ##################################################
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(1, 2, 3, 4))
        union = preds.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""

    def __init__(self, smooth=1.0, bce_weight=0.5):
        super().__init__()
        self.dice = SoftDiceLoss(smooth)
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        dice_loss = self.dice(preds, targets)
        bce_loss = self.bce(preds, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# ----------------------
# Regression / Localization Loss
# ----------------------
class L1SmoothLoss(nn.Module):
    """Smooth L1 loss for regression (centroid or bounding box coordinates)."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, preds, targets):
        diff = torch.abs(preds - targets)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()

# ----------------------
# Weighted Multi-task Loss
# ----------------------
class MultiTaskLoss(nn.Module):
    """

    Multi-task loss for RSNA Intracranial Aneurysm Detection.
    Supports:
        - Classification (scan-level presence)
        - Segmentation (mask-based aneurysm regions)
        - Localization (centroid or bounding box regression)

    """

    def __init__(self,
                 cls_weight: float = 0.1,
                 seg_weight: float = 0.0,
                 loc_weight: float = 1.0,
                 ):
        super().__init__()
        self.cls_loss = BCEWithLogits()
        self.seg_loss = DiceBCELoss()
        self.loc_loss = L1SmoothLoss()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.loc_weight = loc_weight

    def forward(self, logits_cls, logits_seg, cords_pred, targets_cls, targets_seg, targets_cords):
        loss_cls = self.cls_loss(logits_cls, targets_cls)
        loss_seg = self.seg_loss(logits_seg, targets_seg)
        ############ added #######################
        #print(f'pseudo seg ={targets_cords.shape}')
        #print(f'pred pseudo seg ={cords_pred.shape}');exit()
        #loss_loc  = self.seg_loss(cords_pred, targets_cords)
        ####################
        loss_loc = self.loc_loss(cords_pred.float(), targets_cords.float())
        return self.cls_weight * loss_cls + self.seg_weight * loss_seg + self.loc_weight * loss_loc


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class HardCustomLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        # Raw pos_weight values = neg/pos
        raw_weights = torch.tensor([
            53.41,  # L-ICA (Infraclinoid)
            43.73,  # R-ICA (Infraclinoid)
            12.74,  # L-ICA (Supraclinoid)
            15.10,  # R-ICA (Supraclinoid)
            18.83,  # L-MCA
            13.86,  # R-MCA
            10.71,  # AComm
            86.52,  # L-ACA
            73.56,  # R-ACA
            46.93,  # L-PComm
            39.26,  # R-PComm
            35.60,  # Basilar Tip
            35.60,  # Post. Circulation
            1.34    # Any Aneurysm
        ], dtype=torch.float32, device=device)

        # 🔥 Normalize so mean = 1 (avoids exploding gradients)
        self.register_buffer("pos_weight", raw_weights / raw_weights.mean())

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size, num_classes)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none"
        )
        return bce_loss.mean(dim=1).mean()




##############################################################################################################

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return 1 - (num / den).mean()


class MultiTaskLoss2(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        """
        alpha = weight for segmentation
        beta = weight for classification
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, cls_pred, cls_true, seg_pred, seg_true):
        cls_loss = self.bce_logits(cls_pred, cls_true)
        seg_loss = self.bce_logits(seg_pred, seg_true) + dice_loss(seg_pred, seg_true)
        return self.alpha * seg_loss + self.beta * cls_loss, cls_loss, seg_loss
