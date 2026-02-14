#scripts/ml/config.py
"""
config.py
=========
Hardware config and hyperparameters.

Targets:
  - Linux desktop: i9-14900K (32 cores) + RTX A4000 16GB VRAM  → CUDA, batch=1024
  - MacBook Pro  : M2 Pro 12-core, 16GB unified                 → MPS,  batch=256
"""

import torch
import os

# ── Device detection ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE       = torch.device('cuda')
    BATCH_SIZE   = 256
    NUM_WORKERS  = 0
    PIN_MEMORY   = True
    PREFETCH     = 4
    USE_AMP      = False         # CUDA: full AMP with GradScaler
    USE_COMPILE  = False         # torch.compile (Triton/CUDA kernels)
elif torch.backends.mps.is_available():
    DEVICE       = torch.device('mps')
    BATCH_SIZE   = 256           # M2 Pro: 16GB shared, stay safe
    NUM_WORKERS  = 6             # Performance cores on M2 Pro
    PIN_MEMORY   = False         # MPS does not support pinned memory
    PREFETCH     = 2
    USE_AMP      = False         # MPS AMP not stable as of PyTorch 2.2
    USE_COMPILE  = False         # torch.compile MPS backend not mature
else:
    DEVICE       = torch.device('cpu')
    BATCH_SIZE   = 64
    NUM_WORKERS  = os.cpu_count() or 4
    PIN_MEMORY   = False
    PREFETCH     = 2
    USE_AMP      = False
    USE_COMPILE  = False

# ── Architecture ──────────────────────────────────────────────────────────────
WINDOW_SIZE   = 100             # lookback snapshots
NUM_CLASSES   = 3               # DOWN=0, FLAT=1, UP=2
D_FEATURES    = 40              # raw LOB features (10 levels × 4: bp,bs,ap,as)
D_HAWKES      = 4               # u_L(k1), u_L(k2), delta_hat, event_density
D_TOTAL       = D_FEATURES + D_HAWKES   # 44 for full Hawkes configs

# ── DeepLOB (Config A) ────────────────────────────────────────────────────────
DEEPLOB_INCEPTION_CHANNELS = 256
DEEPLOB_LSTM_HIDDEN        = 64
DEEPLOB_LSTM_LAYERS        = 1

# ── Transformer (Configs B-F) ─────────────────────────────────────────────────
TRANSFORMER_D_MODEL    = 128
TRANSFORMER_N_HEADS    = 4
TRANSFORMER_N_LAYERS   = 2
HAWKES_CONV_CHANNELS   = 32
HAWKES_CONV_KERNEL     = 3

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-5
LR_PATIENCE  = 5               # ReduceLROnPlateau patience
ES_PATIENCE  = 15              # Early stopping patience
VAL_SPLIT    = 0.15            # Chronological: last 15% of each train day

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER       = 'nvda'

# ── Path auto-detection ───────────────────────────────────────────────────────
# Resolves both ~/HFT/... (Linux) and ~/HFT/HFT/... (Mac clone) layouts.
def _find_project_root() -> str:
    """Walk up from this file to find the directory that contains data/."""
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):  # at most 6 levels up
        if os.path.isdir(os.path.join(candidate, 'data', 'datasets', 'clean', 'daily')):
            return candidate
        candidate = os.path.dirname(candidate)
    # Fallback: assume script lives inside the project somewhere
    return os.path.expanduser('~/HFT/HFT')

_PROJECT_ROOT = _find_project_root()
DATA_DIR     = os.path.join(_PROJECT_ROOT, 'data', 'datasets', 'clean', 'daily')
MODEL_DIR    = os.path.join(_PROJECT_ROOT, 'models')
RESULTS_DIR  = os.path.join(_PROJECT_ROOT, 'results')

# DoubleAlpha fitted from NVDA train days (Section 1 of Preprocessing doc)
# DO NOT change these without re-fitting from training data
ALPHA_DOWN   = 0.000052        # DOWN threshold (22nd percentile of return dist)
ALPHA_UP     = 0.000049        # UP threshold   (78th percentile)
# Multi-day alpha pkl path (created by fit_alpha.py)
ALPHA_PKL    = os.path.join(MODEL_DIR, 'nvda_double_alpha.pkl')

DOWNSAMPLE_STRIDE = 4 if DEVICE.type == 'cuda' else 32
# CUDA  (Linux A4000, 64GB RAM) : stride=4  → full resolution, ~48 GB peak
# MPS   (M2 Pro,       16GB)    : stride=32 → 1 sample per ~150ms, ~3-5 GB peak
# stride=32 preserves fast microstructure for 100ms prediction
# stride=128 (suggested elsewhere) is TOO COARSE -- 600ms sampling for a 100ms model

TRAIN_DATES = [
    '20260105', '20260106', '20260107', '20260108', '20260109',
    '20260112', '20260113', '20260114', '20260115', '20260116',
]
TEST_DATES = [
    '20260120', '20260121', '20260122', '20260123',
    '20260126', '20260127', '20260128', '20260129',
]
# Jan 17 is the validation day for tuning conf_threshold and IB_threshold
VAL_TUNING_DATE = '20260117'

# ── Backtester (Section 5 of Research Plan) ───────────────────────────────────
CONF_THRESHOLD   = 0.60        # Tuned on Jan 17 only, frozen before test
IB_THRESHOLD     = 0.02        # delta_hat threshold for burst suppression
STOP_LOSS_TICKS  = 3           # stop-loss width in ticks
MAX_DAILY_DD     = 0.02        # 2% of start-of-day capital hard stop
POSITION_SIZE    = 100         # fixed shares per trade

TICKER_INSTRUMENT_IDS = {
    'nvda': 11667,
    'spy':  15144,
    'tsla': 16244,
    'pltr': 12716,
    'amd':  773,
}

# ── Snapshot binary format ────────────────────────────────────────────────────
SNAPSHOT_BYTES  = 248
SNAPSHOT_DTYPE  = '<d' + 'Q'*10 + 'i'*10 + 'Q'*10 + 'i'*10
PRICE_SCALE     = 1e4          # prices stored as int × 1e-4


def print_hardware_summary():
    print("=" * 55)
    print(f"  Device   : {DEVICE}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Workers  : {NUM_WORKERS}")
    print(f"  AMP      : {USE_AMP}")
    print(f"  Compile  : {USE_COMPILE}")
    if DEVICE.type == 'cuda':
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU      : {name}")
        print(f"  VRAM     : {vram:.1f} GB")
        print(f"  CUDA     : {torch.version.cuda}")
    elif DEVICE.type == 'mps':
        import platform
        print(f"  Platform : {platform.processor()}")
        print(f"  Backend  : Apple MPS (Metal)")
    print("=" * 55)


if __name__ == '__main__':
    print_hardware_summary()