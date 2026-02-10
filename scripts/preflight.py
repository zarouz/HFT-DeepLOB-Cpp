import os, sys, importlib

print("=" * 50)
print("PREFLIGHT CHECK")
print("=" * 50)

# Check packages
for pkg in ["torch", "numpy", "sklearn", "scipy"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  [OK] {pkg} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  [FAIL] {pkg} not found -- pip install {pkg}")

# Check CUDA
import torch
print(f"  [INFO] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Check training files
data_dir = os.path.expanduser("~/HFT/data/datasets/clean/daily")
TRAIN_DATES = ["20260105","20260106","20260107","20260108","20260109",
               "20260112","20260113","20260114","20260115","20260116"]
missing = []
for d in TRAIN_DATES:
    f = os.path.join(data_dir, f"nvda_{d}_dataset.bin")
    if os.path.exists(f):
        sz = os.path.getsize(f) / 1e6
        print(f"  [OK] nvda_{d}_dataset.bin  ({sz:.0f} MB)")
    else:
        print(f"  [MISSING] nvda_{d}_dataset.bin")
        missing.append(d)

if missing:
    print(f"\n  WARNING: {len(missing)} files missing")
else:
    print("\n  All 10 training files found. Ready to train.")
print("=" * 50)