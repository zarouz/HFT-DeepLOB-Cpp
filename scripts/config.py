"""
config.py -- hardware config for i9-14900K + RTX A4000
"""
import torch
import os

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS  = 12
PIN_MEMORY   = True
PREFETCH     = 4
USE_AMP      = True
USE_COMPILE  = True

WINDOW_SIZE  = 100
NUM_CLASSES  = 3
D_FEATURES   = 40

BATCH_SIZE   = 1024
EPOCHS       = 100
LR           = 0.0001
WEIGHT_DECAY = 1e-5
LR_PATIENCE  = 5
ES_PATIENCE  = 15
VAL_SPLIT    = 0.15

TICKER       = 'nvda'
DATA_DIR     = os.path.expanduser('~/HFT/data/datasets/clean/daily')
MODEL_DIR    = os.path.expanduser('~/HFT/models')
ALPHA        = 0.0002

TRAIN_DATES = [
    '20260105', '20260106', '20260107', '20260108', '20260109',
    '20260112', '20260113', '20260114', '20260115', '20260116',
]
TEST_DATES = [
    '20260120', '20260121', '20260122', '20260123', '20260126',
    '20260127', '20260128', '20260129',
]

def check_hardware():
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available, running on CPU")
        return None, 0
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
    if vram_gb < 8:
        print("  WARNING: <8GB VRAM. Reduce BATCH_SIZE to 256.")
    return gpu_name, vram_gb