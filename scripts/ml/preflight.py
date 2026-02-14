"""
preflight.py
============
Pre-training checklist. Run before every training session.
All items must show ✅ before running train_nvda.py.

Checks:
  1. Package versions (torch, numpy, scipy)
  2. Device detection (CUDA / MPS / CPU)
  3. All 10 training data files present and non-empty
  4. All 8 test data files present and non-empty
  5. DoubleAlpha pkl exists (or warns to run fit_alpha.py)
  6. DeepLOB model forward pass (batch=4)
  7. Feature builder single-day smoke test (tiny subset)
  8. Estimated RAM and VRAM budget

Usage: python scripts/ml/preflight.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# CRITICAL: insert scripts/ml/ at position 0 so Python finds our adaptive_labeler
# before the old one at scripts/adaptive_labeler.py
sys.path.insert(0, _HERE)

PASS = "✅"
FAIL = "❌"
WARN = "⚠ "

failures = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    symbol = PASS if ok else (WARN if warn_only else FAIL)
    print(f"  {symbol} {label}" + (f"  {detail}" if detail else ""))
    if not ok and not warn_only:
        failures.append(label)


# ── 1. Packages ───────────────────────────────────────────────────────────────
print("\n[1] Package versions")
try:
    import torch
    check("PyTorch", True, torch.__version__)
except ImportError:
    check("PyTorch", False, "pip install torch")

try:
    import numpy as np
    check("NumPy", True, np.__version__)
except ImportError:
    check("NumPy", False, "pip install numpy")

try:
    import scipy
    check("SciPy", True, scipy.__version__)
except ImportError:
    check("SciPy", False, "pip install scipy")

# ── 2. Device ─────────────────────────────────────────────────────────────────
print("\n[2] Device")
from config import (DEVICE, BATCH_SIZE, USE_AMP, USE_COMPILE, print_hardware_summary,
                    DOWNSAMPLE_STRIDE, WINDOW_SIZE, D_FEATURES)

if DEVICE.type == 'cuda':
    check("CUDA device", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not found")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("VRAM >= 8GB", vram >= 8, f"{vram:.1f} GB")
elif DEVICE.type == 'mps':
    check("Apple MPS", torch.backends.mps.is_available(), "M-series GPU")
    check("AMP disabled for MPS", not USE_AMP, "(correct)")
    check("torch.compile disabled for MPS", not USE_COMPILE, "(correct)")
else:
    check("CPU fallback", True, "training will be very slow", warn_only=True)

print(f"  → Batch size: {BATCH_SIZE}  AMP: {USE_AMP}  Compile: {USE_COMPILE}")

# ── 3. Training data files ─────────────────────────────────────────────────────
print("\n[3] Training data files")
from config import DATA_DIR, TICKER, TRAIN_DATES, TEST_DATES, SNAPSHOT_BYTES

all_train_ok = True
for date in TRAIN_DATES:
    p = os.path.join(DATA_DIR, f'{TICKER}_{date}_dataset.bin')
    exists = os.path.exists(p)
    if exists:
        n = os.path.getsize(p) // SNAPSHOT_BYTES
        ok = n > 10000
        check(f"{TICKER}_{date}", ok, f"{n:,} snapshots")
        if not ok:
            all_train_ok = False
    else:
        check(f"{TICKER}_{date}", False, "file not found")
        all_train_ok = False

# ── 4. Test data files ────────────────────────────────────────────────────────
print("\n[4] Test data files")
for date in TEST_DATES:
    p = os.path.join(DATA_DIR, f'{TICKER}_{date}_dataset.bin')
    exists = os.path.exists(p)
    if exists:
        n = os.path.getsize(p) // SNAPSHOT_BYTES
        check(f"{TICKER}_{date}", n > 10000, f"{n:,} snapshots")
    else:
        check(f"{TICKER}_{date}", False, "file not found")

# ── 5. DoubleAlpha pkl ────────────────────────────────────────────────────────
print("\n[5] DoubleAlpha")
from config import ALPHA_PKL
from adaptive_labeler import DoubleAlpha

if os.path.exists(ALPHA_PKL):
    da = DoubleAlpha.load(ALPHA_PKL)
    check("nvda_double_alpha.pkl", True, f"{da}")
else:
    check("nvda_double_alpha.pkl", False,
          "run: python scripts/ml/fit_alpha.py", warn_only=True)
    print("    (will be auto-created at start of train_nvda.py)")

# ── 6. Model forward pass ─────────────────────────────────────────────────────
print("\n[6] Model forward pass")
try:
    from deeplob_model import build_model, count_parameters
    from config import WINDOW_SIZE, D_FEATURES, NUM_CLASSES

    model = build_model('A').to(DEVICE)
    params = count_parameters(model)
    x_test = torch.randn(4, 1, WINDOW_SIZE, D_FEATURES, device=DEVICE)
    with torch.no_grad():
        out = model(x_test)
    check("DeepLOB Config A forward pass", out.shape == (4, NUM_CLASSES),
          f"out={out.shape}  params={params:,}")
    del model, x_test, out

    # Also verify Config E (Hawkes Transformer)
    from config import D_HAWKES
    model_e = build_model('E').to(DEVICE)
    x_lob = torch.randn(4, WINDOW_SIZE, D_FEATURES, device=DEVICE)
    x_haw = torch.randn(4, WINDOW_SIZE, D_HAWKES, device=DEVICE)
    with torch.no_grad():
        out_e = model_e(x_lob, x_haw)
    check("HawkesTransformer Config E forward pass", out_e.shape == (4, NUM_CLASSES),
          f"out={out_e.shape}  params={count_parameters(model_e):,}")
    del model_e, x_lob, x_haw, out_e

except Exception as e:
    check("Model forward pass", False, str(e))

# ── 7. Feature builder smoke test (first 50k snapshots) ──────────────────────
print("\n[7] Feature builder smoke test")
try:
    import importlib.util as _ilu
    _al_spec = _ilu.spec_from_file_location('adaptive_labeler_ml', os.path.join(_HERE, 'adaptive_labeler.py'))
    _al_mod = _ilu.module_from_spec(_al_spec); _al_spec.loader.exec_module(_al_mod)
    load_snapshots = _al_mod.load_snapshots
    filter_session = _al_mod.filter_session
    from feature_label_builder import build_raw_features, build_sliding_windows
    import numpy as np

    first_path = os.path.join(DATA_DIR, f'{TICKER}_{TRAIN_DATES[0]}_dataset.bin')
    if os.path.exists(first_path):
        arr_full = load_snapshots(first_path)
        arr_mini = filter_session(arr_full)[:50_000]
        feats = build_raw_features(arr_mini)
        check("build_raw_features", feats.shape == (len(arr_mini), 40),
              f"shape={feats.shape}")
        check("feature dtype", feats.dtype == np.float32, str(feats.dtype))

        # Quick label check with known NVDA alpha
        DA = _al_mod.DoubleAlpha
        label_fixed = _al_mod.label_fixed
        da_test = DA(alpha_down=0.000052, alpha_up=0.000049, n_samples=0, asymmetry=0.065)
        labels = label_fixed(arr_mini, da_test)
        valid = labels != -1
        check("label_fixed", valid.sum() > 1000,
              f"{valid.sum():,} valid labels from 50k snapshots")
    else:
        check("Smoke test file", False, f"{first_path} not found")
except Exception as e:
    check("Feature builder smoke test", False, str(e))

# ── 8. RAM / VRAM budget ──────────────────────────────────────────────────────
print("\n[8] Memory budget")
# Actual NVDA session snapshots (from preflight output above):
#   avg across 10 training days ≈ 8.1M snapshots/day total in file
#   after 09:30-16:00 session filter (~5.5h of 6.5h file = ~85%): ~7.0M/day
#   after valid-label filter (~98.6%): ~6.9M/day
# Windows at stride=S: N_valid / S per day
NVDA_SESSION_SNAPSHOTS_PER_DAY = 7_000_000   # post session-filter estimate
STRIDE        = DOWNSAMPLE_STRIDE
WINDOWS_PER_DAY = NVDA_SESSION_SNAPSHOTS_PER_DAY // STRIDE
WINDOW_BYTES  = WINDOW_SIZE * D_FEATURES * 4   # float32, bytes per window
DAYS          = len(TRAIN_DATES)
array_gb      = (WINDOWS_PER_DAY * WINDOW_BYTES * DAYS) / 1e9

print(f"  Snapshot rate (est)  : ~{NVDA_SESSION_SNAPSHOTS_PER_DAY/1e6:.1f}M/day post-session-filter")
print(f"  Windows at stride={STRIDE}  : ~{WINDOWS_PER_DAY/1e6:.2f}M/day  ×{DAYS} days = {WINDOWS_PER_DAY*DAYS/1e6:.0f}M total")
print(f"  X array size         : {array_gb:.0f} GB  (float32, no labels)")

if array_gb <= 12:
    print(f"  ✅ Fits in RAM (stride={STRIDE})")
elif array_gb <= 32:
    print(f"  ⚠  {array_gb:.0f} GB > 16 GB unified -- use stride={STRIDE*2} or day-by-day streaming")
    print(f"     → Set DOWNSAMPLE_STRIDE={STRIDE*2} in config.py for Mac, keep {STRIDE} for Linux A4000")
else:
    print(f"  ⚠  {array_gb:.0f} GB -- too large for single array; training will use streaming DataLoader")
    print(f"     → Set DOWNSAMPLE_STRIDE={STRIDE*4} in config.py, or use the streaming loader in train_nvda.py")

# Note: the training script streams one day at a time into a buffer and builds
# the DataLoader incrementally, so the full array never needs to be in RAM at once.
# The above is worst-case if you call build_multi_day_dataset() and hold everything.
print(f"  ℹ  train_nvda.py streams one day at a time -- peak RAM is ~{array_gb/DAYS:.1f} GB/day")

if DEVICE.type == 'cuda':
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    batch_vram_gb = BATCH_SIZE * 100 * 40 * 4 / 1e9
    check("Batch fits in VRAM", batch_vram_gb < vram_gb * 0.5,
          f"batch={batch_vram_gb*1000:.1f} MB  VRAM={vram_gb:.1f} GB")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if failures:
    print(f"  ❌ PREFLIGHT FAILED -- {len(failures)} issue(s):")
    for f in failures:
        print(f"     • {f}")
    print("\n  Fix all ❌ items before running train_nvda.py")
else:
    print("  ✅ All checks passed -- ready to train!")
    print(f"\n  Run: python scripts/ml/train_nvda.py")
print("=" * 55)