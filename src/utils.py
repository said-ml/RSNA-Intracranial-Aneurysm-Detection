
import os
import numpy as np
from scipy import ndimage
from typing import Tuple
from typing import List
from collections import OrderedDict
import pydicom
# local imports
from src.configurations import*
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)  # avoid zeros/negatives
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) using safe zoom."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# ==========================
# DICOM Series Processor
# ==========================
class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes (CPU only in workers, memory LRU only)."""

    def __init__(
        self,
        target_size: Tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: Tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
        lru_capacity: int = LRU_CAPACITY,
    ):
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip
        self.memory_cache = OrderedDict()
        self.lru_capacity = lru_capacity

    # ---- LRU helpers (memory only) ----
    def _cache_put(self, key: str, vol: np.ndarray):
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)

    def _cache_get(self, key: str):
        if key in self.memory_cache:
            vol = self.memory_cache[key]
            self.memory_cache.move_to_end(key)
            return vol
        return None

    # ---- Main API (no disk cache) ----
    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Return (D,H,W) float32 volume in [0,1]. No disk I/O cache is used."""
        series_id = os.path.basename(series_path)

        # 1) Memory cache
        m = self._cache_get(series_id)
        if m is not None and isinstance(m, np.ndarray) and m.shape == self.target_size:
            return m

        # 2) Build from DICOM (CPU path only)
        try:
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.endswith(".dcm"):
                        try:
                            ds = pydicom.dcmread(os.path.join(root, f), force=True)
                            if hasattr(ds, "PixelData"):
                                dicoms.append(ds)
                        except Exception as e:
                            print(f"[DICOM read] {e}")
                            continue
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")

            dicoms = self._sort_slices(dicoms)
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

            # choose base HxW (most frequent) WITHOUT decoding pixel_array
            base_h, base_w = self._choose_base_shape(dicoms)

            modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()  # 'CT' or 'MR'
            vol_slices = []

            for ds in dicoms:
                arr = ds.pixel_array
                # standardize to (N,H,W) where N=number of frames (1 if 2D)
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape(n, h, w)
                    frames = arr
                else:
                    frames = arr[np.newaxis, ...]  # shape (1,H,W)

                for sl in frames:
                    sl = sl.astype(np.float32)

                    # Handle MONOCHROME1 inversion
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl

                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    sl = sl * slope + intercept

                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if len(vol_slices) == 0:
                raise ValueError("No valid slices extracted.")

            volume = np.stack(vol_slices, axis=0).astype(np.float32)  # (D,H,W)

            # Normalize by modality -> [0,1]
            volume = self._normalize_by_modality(volume, modality_tag)

            # Isotropic resample (mm-based) on CPU
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                newD = max(1, int(round(z * dz / self.target_spacing_mm)))
                newH = max(1, int(round(y * dy / self.target_spacing_mm)))
                newW = max(1, int(round(x * dx / self.target_spacing_mm)))
                volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Resize to target grid
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

            # Put to memory cache and return
            self._cache_put(series_id, volume)
            return volume

        except Exception as e:
            print(f"[Processor] Error: {e}")
            vol = np.zeros(self.target_size, dtype=np.float32)
            self._cache_put(series_id, vol)
            return vol

    # ---- helpers ----
    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]; col = orient[3:]
            normal = np.cross(row, col)
            def sort_key(ds):
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))
            return sorted(ds_list, key=sort_key)
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
        try:
            dy, dx = map(float, ds_sorted[0].PixelSpacing)
        except Exception:
            ps = getattr(ds_sorted[0], "PixelSpacing", [1.0, 1.0])
            dy, dx = float(ps[0]), float(ps[1])

        if has_multiframe:
            dz = float(getattr(ds_sorted[0], "SpacingBetweenSlices", getattr(ds_sorted[0], "SliceThickness", 1.0)))
        else:
            zs = []
            for i in range(1, len(ds_sorted)):
                p0 = np.array(getattr(ds_sorted[i-1], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                p1 = np.array(getattr(ds_sorted[i], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                d = np.linalg.norm(p1 - p0)
                if d > 0:
                    zs.append(d)
            if zs:
                dz = float(np.median(zs))
            else:
                dz = float(getattr(ds_sorted[0], "SliceThickness", 1.0))

        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        shapes = []
        for ds in ds_list:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
            except Exception:
                arr = ds.pixel_array
                h, w = arr.shape[-2], arr.shape[-1]
            shapes.append((h, w))
        vals, counts = np.unique(shapes, return_counts=True, axis=0)
        base = tuple(vals[counts.argmax()])
        return int(base[0]), int(base[1])

    def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
        volume = np.nan_to_num(volume, copy=False)
        if modality_tag == "CT":
            c, w = self.cta_window
            lo, hi = c - w / 2.0, c + w / 2.0
            v = np.clip(volume, lo, hi)
            v = (v - lo) / (hi - lo + 1e-6)
            return v.astype(np.float32, copy=False)
        else:
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            v = (volume - mean) / std
            zc = float(self.mri_z_clip)
            v = np.clip(v, -zc, zc)
            v = (v + zc) / (2.0 * zc)  # map to [0,1]
            return v.astype(np.float32, copy=False)


##################### ------------ scaling huge logits that generated from trained models -------------####################
from torch import nn
from torch import optim

class TempScaler(nn.Module):
    def __init__(self, init_temp=2.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, logits):
        temp = self.log_temp.exp()
        return logits / temp

def find_temperature(val_logits, val_labels, device="cpu", init_temp=2.0):
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device).float()
    scaler = TempScaler(init_temp).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([scaler.log_temp], lr=0.1, max_iter=200)

    def _eval():
        optimizer.zero_grad()
        loss = criterion(scaler(val_logits), val_labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    T = scaler.log_temp.exp().item()
    print("Optimal temperature:", T)
    return T
