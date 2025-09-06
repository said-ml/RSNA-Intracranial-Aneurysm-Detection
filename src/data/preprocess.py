# ================================================================
# RSNA Reduction Pipeline — Compact mode for Kaggle 19GB target
# - TARGET_SHAPE=(16,256,256), 2 windows, uint8 quantization
# - multiprocessing + batch zipping + robust handling
# ================================================================

import os, glob, math, json, traceback, zipfile, shutil, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
import nibabel as nib

# Optional numba
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------------------
# CONFIG (edit before running)
# ---------------------
DATA_DIR       = "/kaggle/input/rsna-intracranial-aneurysm-detection"
SERIES_DIR     = os.path.join(DATA_DIR, "series")
SEGS_DIR       = os.path.join(DATA_DIR, "segmentations")
TRAIN_CSV      = os.path.join(DATA_DIR, "train.csv")   # optional

OUT_DIR        = "/kaggle/working/rsna-reduced-compact"
BATCH_ZIP_DIR  = "/kaggle/working/rsna-zips-compact"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BATCH_ZIP_DIR, exist_ok=True)

# Compact settings to fit ~19 GB on Kaggle
TARGET_SHAPE   = (7,400,400)   # (1, H, W)
WINDOWS        = [(40, 80), (600, 2800)]   # 2 windows (soft-tissue, bone)
BBOX_MARGIN    = 10
SAVE_UINT8     = True   # convert normalized [0,1] -> uint8 0..255
KEEP_NEGATIVE_FRACTION = 0.10   # keep only 10% negatives

# Parallelism / batching
N_WORKERS      = 6    # adjust to Kaggle CPU (6-8 good)
BATCH_SIZE     = 600  # how many series to process before zipping

# Debug safe mode (run a tiny subset first)
TEST_MODE = False
DEBUG_COUNT = 8

# ---------------------
# HELPERS
# ---------------------
def dcm_key_sorter(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if "InstanceNumber" in ds: return int(ds.InstanceNumber)
        if "ImagePositionPatient" in ds and len(ds.ImagePositionPatient) == 3:
            return float(ds.ImagePositionPatient[2])
    except Exception:
        pass
    return path

def load_dicom_volume(series_dir, min_slices=8):
    """Robustly load DICOM series -> (D,H,W) float32 HU"""
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")), key=dcm_key_sorter)
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")
    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                try:
                    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                except Exception:
                    pass
            # Collapse common extra dims: (1,H,W) or (H,W,1) or (H,W)
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.shape[-1] == 1:
                    arr = arr[...,0]
                else:
                    arr = arr[...,0]   # fallback take first channel
            if arr.ndim != 2:
                continue
            slices.append(arr)
        except Exception:
            # skip problematic slice
            continue
    if len(slices) < min_slices:
        raise ValueError(f"Too few usable slices ({len(slices)}) in {series_dir}")
    vol = np.stack(slices, axis=0)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        if vol.ndim > 3:
            vol = vol[..., 0]
        else:
            raise ValueError(f"Unable to coerce volume to 3D, got shape {vol.shape}")
    return vol.astype(np.float32)

def load_nii_mask_safe(seg_path):
    """Load NIfTI mask robustly -> (D,H,W) uint8, or None on failure"""
    try:
        nii = nib.load(seg_path)
        m = np.array(nii.get_fdata())
        m = np.squeeze(m)
        m = (m > 0).astype(np.uint8)
        # If (H,W,D) -> transpose
        if m.ndim == 3 and m.shape[0] < m.shape[-1]:
            m = np.transpose(m, (2,0,1))
        elif m.ndim > 3:
            m = m[...,0]
        elif m.ndim < 3:
            m = m.reshape((1,)+m.shape)
        return m.astype(np.uint8)
    except Exception:
        return None

# optional numba clip; else numpy.clip
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _clip_flat(src, lower, upper, out):
        for i in prange(src.size):
            v = src.flat[i]
            if v < lower:
                out.flat[i] = lower
            elif v > upper:
                out.flat[i] = upper
            else:
                out.flat[i] = v
        return out

    def apply_window(vol, WL, WW):
        lower = WL - WW/2.0
        upper = WL + WW/2.0
        clipped = np.empty_like(vol)
        clipped = _clip_flat(vol, lower, upper, clipped)
        scaled = (clipped - lower) / max(1e-12, (upper - lower))
        return scaled.astype(np.float32)
else:
    def apply_window(vol, WL, WW):
        lower = WL - WW/2.0
        upper = WL + WW/2.0
        clipped = np.clip(vol, lower, upper)
        scaled = (clipped - lower) / max(1e-12, (upper - lower))
        return scaled.astype(np.float32)

def clamp_bbox(z0,z1,y0,y1,x0,x1,D,H,W):
    z0 = max(0,int(z0)); y0 = max(0,int(y0)); x0 = max(0,int(x0))
    z1 = min(D,int(z1)); y1 = min(H,int(y1)); x1 = min(W,int(x1))
    if z1 <= z0: z0,z1 = 0,D
    if y1 <= y0: y0,y1 = 0,H
    if x1 <= x0: x0,x1 = 0,W
    return z0,z1,y0,y1,x0,x1

def get_bbox_from_mask(mask, margin=10):
    coords = np.argwhere(mask>0)
    if coords.size == 0: return None
    zmin,ymin,xmin = coords.min(axis=0)
    zmax,ymax,xmax = coords.max(axis=0)
    return (zmin-margin, zmax+margin, ymin-margin, ymax+margin, xmin-margin, xmax+margin)

def center_bbox(D,H,W, crop_dhw=None):
    if crop_dhw is None:
        return (0,D,0,H,0,W)
    cd,ch,cw = crop_dhw
    z0 = max(0,(D-cd)//2); z1 = min(D,z0+cd)
    y0 = max(0,(H-ch)//2); y1 = min(H,y0+ch)
    x0 = max(0,(W-cw)//2); x1 = min(W,x0+cw)
    return (z0,z1,y0,y1,x0,x1)

def resample_volume_numpy(vol, bbox, target_shape):
    """Crop using bbox and resample using scipy.zoom if available, else nearest sampling."""
    z0,z1,y0,y1,x0,x1 = clamp_bbox(*bbox, *vol.shape)
    cropped = vol[z0:z1, y0:y1, x0:x1]
    if cropped.size == 0:
        return np.zeros(target_shape, dtype=vol.dtype)
    try:
        from scipy.ndimage import zoom
        dz = target_shape[0] / cropped.shape[0]
        dh = target_shape[1] / cropped.shape[1]
        dw = target_shape[2] / cropped.shape[2]
        res = zoom(cropped, (dz, dh, dw), order=1).astype(vol.dtype)
        return res
    except Exception:
        iz = np.linspace(0, cropped.shape[0]-1, target_shape[0]).astype(np.int32)
        iy = np.linspace(0, cropped.shape[1]-1, target_shape[1]).astype(np.int32)
        ix = np.linspace(0, cropped.shape[2]-1, target_shape[2]).astype(np.int32)
        out = cropped[iz][:, iy][:, :, ix]
        return out.astype(vol.dtype)

# ---------------------
# PROCESS WORKER
# ---------------------
def process_series(uid, series_dir, seg_dir, out_dir, target_shape, windows, bbox_margin,
                   save_uint8, keep_negative_fraction, labels_map):
    """
    Process one series -> returns (uid, out_path or None, status_str)
    status_str in {"ok","skipped_negative","error:..."}
    """
    try:
        series_path = os.path.join(series_dir, uid)
        if not os.path.isdir(series_path):
            raise FileNotFoundError(f"series dir missing: {series_path}")

        vol = load_dicom_volume(series_path)
        D,H,W = vol.shape

        # try load mask
        seg_path = os.path.join(seg_dir, f"{uid}.nii")
        mask = None
        bbox = None
        if os.path.exists(seg_path):
            mask = load_nii_mask_safe(seg_path)
            if mask is not None and mask.shape != (D,H,W):
                # try safe resample with scipy (order=0) else simple mapping
                try:
                    from scipy.ndimage import zoom
                    fz = (D / mask.shape[0], H / mask.shape[1], W / mask.shape[2])
                    mask = zoom(mask.astype(np.float32), fz, order=0) > 0.5
                    mask = mask.astype(np.uint8)
                except Exception:
                    # fallback depth mapping + pad/crop
                    if mask.shape[0] != D:
                        idx = np.linspace(0, mask.shape[0]-1, D).astype(int)
                        mask = mask[idx]
                    mh, mw = mask.shape[1], mask.shape[2]
                    if mh != H or mw != W:
                        pad_h = max(0, H - mh); pad_w = max(0, W - mw)
                        if pad_h>0 or pad_w>0:
                            pad_top = pad_h//2; pad_bottom = pad_h - pad_top
                            pad_left = pad_w//2; pad_right = pad_w - pad_left
                            mask = np.pad(mask, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right)), mode='constant')
                        else:
                            start_h = (mh - H)//2; start_w = (mw - W)//2
                            mask = mask[:, start_h:start_h+H, start_w:start_w+W]
                    mask = mask.astype(np.uint8)
            bbox = get_bbox_from_mask(mask, margin=bbox_margin) if mask is not None else None

        if bbox is None:
            bbox = center_bbox(D,H,W, crop_dhw=None)

        # multi-window normalized channels (C,D,H,W) in [0,1]
        chans = []
        for wl, ww in windows:
            chans.append(apply_window(vol, wl, ww))
        vol_cdhw = np.stack(chans, axis=0)

        # crop & resample each channel to target_shape
        C = vol_cdhw.shape[0]
        out_image = np.zeros((C, target_shape[0], target_shape[1], target_shape[2]), dtype=np.float32)
        for c in range(C):
            out_image[c] = resample_volume_numpy(vol_cdhw[c], bbox, target_shape)

        # mask resampling
        if mask is not None and mask.sum() > 0:
            mask_res = resample_volume_numpy(mask.astype(np.float32), bbox, target_shape)
            mask_res = (mask_res > 0.5).astype(np.uint8)
        else:
            mask_res = np.zeros((target_shape[0], target_shape[1], target_shape[2]), dtype=np.uint8)

        # labels
        labels = None
        if labels_map is not None:
            labels = labels_map.get(uid)
            if labels is not None:
                labels = np.asarray(labels, dtype=np.float32)

        # skip negatives probabilistically
        if labels is not None:
            if labels.sum() == 0 and (random.random() > keep_negative_fraction):
                return uid, None, "skipped_negative"
        else:
            if mask_res.sum() == 0 and (random.random() > keep_negative_fraction):
                return uid, None, "skipped_negative"

        # quantize to uint8 if requested
        if save_uint8:
            # out_image in [0,1] -> scale to 0..255
            img_u8 = (np.clip(out_image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            image_to_save = img_u8
        else:
            image_to_save = out_image.astype(np.float16)

        out_path = os.path.join(out_dir, f"{uid}.npz")
        # Save image (uint8 or float16), mask (uint8), labels (float32 or None)
        np.savez_compressed(out_path, image=image_to_save, mask=mask_res, labels=labels)
        return uid, out_path, "ok"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return uid, None, f"error: {repr(e)} | {tb}"

# ---------------------
# Load labels_map (numeric only)
# ---------------------
labels_map = None
label_cols = []
if os.path.exists(TRAIN_CSV):
    try:
        df = pd.read_csv(TRAIN_CSV)
        non_label_cols = set(["SeriesInstanceUID","StudyInstanceUID","PatientID","Sex","Age"])
        label_cols = [c for c in df.columns if c not in non_label_cols]
        df_labels = df[["SeriesInstanceUID"] + label_cols].copy()
        df_labels[label_cols] = df_labels[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        labels_map = {row.SeriesInstanceUID: row[label_cols].values.astype(np.float32) for _, row in df_labels.iterrows()}
        print("Loaded labels for", len(labels_map), "series. Label cols:", label_cols)
    except Exception as e:
        print("Failed to load labels:", e)
        labels_map = None

# ---------------------
# Batch runner
# ---------------------
series_uids = sorted([os.path.basename(p) for p in glob.glob(os.path.join(SERIES_DIR, "*"))])
print("Total series:", len(series_uids))
if TEST_MODE:
    series_uids = series_uids[:DEBUG_COUNT]
    print("TEST_MODE ON — processing only first", DEBUG_COUNT, "series")

def run_batch(uids, batch_idx):
    out_paths = []
    batch_failures = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_series, uid, SERIES_DIR, SEGS_DIR, OUT_DIR, TARGET_SHAPE, WINDOWS, BBOX_MARGIN,
                             SAVE_UINT8, KEEP_NEGATIVE_FRACTION, labels_map): uid for uid in uids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"batch {batch_idx}"):
            uid = futures[fut]
            try:
                uid_ret, out_path, status = fut.result()
                if out_path:
                    out_paths.append(out_path)
                else:
                    if status and status.startswith("error"):
                        batch_failures.append((uid_ret, status))
            except Exception as e:
                batch_failures.append((uid, f"future_exception: {repr(e)}"))
    # zip outputs if any
    if len(out_paths) > 0:
        zip_name = os.path.join(BATCH_ZIP_DIR, f"rsna_compact_batch_{batch_idx}.zip")
        with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in out_paths:
                try:
                    zf.write(p, arcname=os.path.basename(p))
                except Exception as e:
                    batch_failures.append((os.path.basename(p), f"zip_write_failed: {repr(e)}"))
        # remove .npz to free space
        for p in out_paths:
            try:
                os.remove(p)
            except Exception:
                pass
    return out_paths, batch_failures

# Process
batch_uids = [series_uids[i:i+BATCH_SIZE] for i in range(0, len(series_uids), BATCH_SIZE)]
all_outs = []
all_failures = []
for bi, uids in enumerate(batch_uids):
    print(f"Processing batch {bi+1}/{len(batch_uids)} | size={len(uids)}")
    outs, fails = run_batch(uids, bi+1)
    all_outs.extend(outs)
    all_failures.extend(fails)
    print(f"Batch {bi+1} done: produced {len(outs)} files, failures {len(fails)}")
    import gc; gc.collect()

# Save failures and labels CSV for produced files
if all_failures:
    with open(os.path.join(OUT_DIR, "failures.json"), "w") as f:
        json.dump(all_failures, f, indent=2)
    print("Saved failures.json with", len(all_failures), "items")

if labels_map is not None:
    saved_uids = []
    for zip_fp in sorted(glob.glob(os.path.join(BATCH_ZIP_DIR, "*.zip"))):
        with zipfile.ZipFile(zip_fp, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                if name.endswith(".npz"):
                    saved_uids.append(Path(name).stem)
    rows = []
    for uid in saved_uids:
        lbl = labels_map.get(uid)
        if lbl is not None:
            rows.append([uid] + lbl.tolist())
    if rows:
        cols = ["SeriesInstanceUID"] + label_cols
        pd.DataFrame(rows, columns=cols).to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
        print("Saved labels.csv for produced set.")

print("All done. Zips at:", BATCH_ZIP_DIR)
print("Total produced files (zipped):", len(all_outs), "failures:", len(all_failures))


#he above data processing is used to transform .dcm files.npz files to reduce data with difference target size
# next we must assemble the 8 .zip files into one unzip file

import os, glob, zipfile
import numpy as np
from pathlib import Path

BATCH_ZIP_DIR  = "/kaggle/working/rsna-zips-compact"  # i'll change this later
MERGED_OUT     = "/kaggle/working/rsna_all_data.npz"  # i'll change this later



def merge_npz_from_zips(zip_dir, merged_out):
    images, masks, labels, uids = [], [], [], []

    zip_files = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))
    print(f"Found {len(zip_files)} zip batches")

    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path, "r") as zf:
            for fname in zf.namelist():
                if not fname.endswith(".npz"):
                    continue
                uid = Path(fname).stem
                with zf.open(fname) as f:
                    data = np.load(f, allow_pickle=True)
                    images.append(data["image"])
                    masks.append(data["mask"])
                    if "labels" in data:
                        labels.append(data["labels"])
                    else:
                        labels.append(None)
                    uids.append(uid)

    images = np.array(images)   # shape: (N, C, D, H, W)
    masks  = np.array(masks)    # shape: (N, D, H, W)
    labels = np.array(labels, dtype=object)  # ragged safe
    uids   = np.array(uids)

    np.savez_compressed(merged_out,
                        images=images,
                        masks=masks,
                        labels=labels,
                        uids=uids)
    print(f"Saved merged dataset: {merged_out} | size={len(images)} series")

merge_npz_from_zips(BATCH_ZIP_DIR, MERGED_OUT)
