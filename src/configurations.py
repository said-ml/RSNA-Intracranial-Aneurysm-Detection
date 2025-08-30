
import torch
if torch.cuda.is_available():
    print('cuda is here')



ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Paths
TRAIN_CSV_PATH = "C:/Users/Setup Game/Music/Favorites/Downloads/rsna_subset/train.csv"
SERIES_DIR = "C:/Users/Setup Game/Music/Favorites/Downloads/rsna_subset/series"

# Processing / Model config
TARGET_SIZE = (64, 64, 64)      # final (D,H,W)
TARGET_SPACING_MM = 1.0         # isotropic resample
CTA_WINDOW = (300.0, 700.0)     # (center, width) for CT (CTA)
MRI_Z_CLIP = 3.0                # clip z-score to ±3σ
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()  # enable mixed precision on GPU

# Training knobs
DO_TRAIN = True                 # This notebook is for training
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-5
ANEURYSM_PRESENT_BOOST = 1.0
PATIENCE = 2  # early stopping

# Runtime practicality
TRAIN_MAX_SERIES = 128
VAL_MAX_SERIES = 64

# Avoid CUDA in DataLoader workers (to prevent fork-CUDA issue)
NUM_WORKERS_TRAIN = 0 if torch.cuda.is_available() else 0
NUM_WORKERS_VAL = 0
PIN_MEMORY = not False
PERSISTENT_WORKERS = True if NUM_WORKERS_TRAIN > 0 else False

# Memory-only LRU cache during training (no disk cache)
LRU_CAPACITY = 8

AP_COL = 'Aneurysm Present'
LOC_COLS = LABEL_COLS[:-1]