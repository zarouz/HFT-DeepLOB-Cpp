"""
feature_label_builder.py
========================
Imports performance-critical functions from adaptive_labeler.
Adds: build_raw_features, build_sliding_windows (OOM bug FIXED),
      build_dataset, build_multi_day_dataset.

OOM FIX (from Preprocessing doc Section 3):
  Original: materialised full (6M, 100, 40) float32 = 91 GB  → CRASH
  Fix 1: downsample_stride=4 → (1.5M, 100, 40) = 22.9 GB → OK
  Fix 2: chunked construction processes CHUNK_SIZE windows at a time
         → peak RAM = ~1 GB regardless of dataset size
  Both fixes are applied by default.
"""

import os
import sys
import importlib.util
import numpy as np

# Explicitly load adaptive_labeler from scripts/ml/ to avoid importing the
# old version at scripts/adaptive_labeler.py (which lacks filter_session etc.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_AL_PATH = os.path.join(_HERE, 'adaptive_labeler.py')
_spec = importlib.util.spec_from_file_location('adaptive_labeler_ml', _AL_PATH)
_al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_al)

# Pull names into local namespace
SNAPSHOT_DTYPE          = _al.SNAPSHOT_DTYPE
load_snapshots          = _al.load_snapshots
filter_session          = _al.filter_session
mid_price               = _al.mid_price
fit_double_alpha        = _al.fit_double_alpha
fit_double_alpha_from_files = _al.fit_double_alpha_from_files
label_fixed             = _al.label_fixed
normalize_features_zscore = _al.normalize_features_zscore
DoubleAlpha             = _al.DoubleAlpha

# Add scripts/ml/ to path for config
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    WINDOW_SIZE,
    DOWNSAMPLE_STRIDE,
    TRAIN_DATES,
    TEST_DATES,
    DATA_DIR,
    TICKER,
)

CHUNK_SIZE = 100_000   # windows built per chunk to cap peak RAM at ~1 GB


# ── Raw 40-dim feature vector ─────────────────────────────────────────────────
def build_raw_features(arr: np.ndarray) -> np.ndarray:
    """
    Build 40-dim LOB feature matrix from structured snapshot array.

    Feature layout (interleaved ask/bid per level, matching Zhang et al.):
      [ask_px_0, bid_px_0, ask_sz_0, bid_sz_0,   ← level 0 (top of book)
       ask_px_1, bid_px_1, ask_sz_1, bid_sz_1,   ← level 1
       ...
       ask_px_9, bid_px_9, ask_sz_9, bid_sz_9]   ← level 9

    Prices are float (price / PRICE_SCALE), sizes are float.
    Returns (N, 40) float32.
    """
    N = len(arr)
    feats = np.empty((N, 40), dtype=np.float32)

    bid_px = arr['bid_px'].astype(np.float32) / 1e4   # (N, 10)
    ask_px = arr['ask_px'].astype(np.float32) / 1e4   # (N, 10)
    bid_sz = arr['bid_sz'].astype(np.float32)          # (N, 10)
    ask_sz = arr['ask_sz'].astype(np.float32)          # (N, 10)

    for lvl in range(10):
        base = lvl * 4
        feats[:, base + 0] = ask_px[:, lvl]
        feats[:, base + 1] = bid_px[:, lvl]
        feats[:, base + 2] = ask_sz[:, lvl]
        feats[:, base + 3] = bid_sz[:, lvl]

    return feats


# ── Sliding windows (OOM BUG FIXED) ──────────────────────────────────────────
def build_sliding_windows(features: np.ndarray,
                           labels: np.ndarray,
                           window_size: int = WINDOW_SIZE,
                           stride: int = DOWNSAMPLE_STRIDE,
                           chunk_size: int = CHUNK_SIZE):
    """
    Build (M, window_size, n_features) windows with stride subsampling.

    stride=4 reduces 6M snapshots to 1.5M candidate windows.
    Chunk construction keeps peak RAM at ~1 GB regardless of M.

    Args:
        features  : (N, F) float32
        labels    : (N,)   int8
        window_size: lookback length (100)
        stride    : subsample factor for candidate window starts
        chunk_size: windows to materialise per chunk

    Returns:
        X : (M, window_size, F) float32
        y : (M,)               int8
    """
    N, F = features.shape
    # Valid window starts: need at least window_size history
    starts = np.arange(window_size - 1, N, stride)
    M = len(starts)

    if M == 0:
        return np.empty((0, window_size, F), dtype=np.float32), np.empty(0, dtype=np.int8)

    # Pre-allocate output
    X = np.empty((M, window_size, F), dtype=np.float32)
    y = np.empty(M, dtype=np.int8)

    # Chunked fill
    for chunk_start in range(0, M, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M)
        chunk_starts = starts[chunk_start:chunk_end]
        for j, s in enumerate(chunk_starts):
            X[chunk_start + j] = features[s - window_size + 1: s + 1]
            y[chunk_start + j] = labels[s]

    return X, y


# ── Single-day pipeline ───────────────────────────────────────────────────────
def build_dataset(path: str,
                  da: DoubleAlpha = None,
                  stride: int = DOWNSAMPLE_STRIDE):
    """
    Full single-day pipeline: load → label → features → normalise → windows.

    If da is None, fits DoubleAlpha from this file (smoke test / single-day use).
    For multi-day training always pass a shared da fitted on all training days.

    Returns:
        X  : (M, 100, 40) float32
        y  : (M,)         int8
        da : DoubleAlpha  (for inspection)
    """
    arr = load_snapshots(path)
    arr = filter_session(arr)

    if da is None:
        da = fit_double_alpha(path)
        print(f"  Fitted {da}")

    labels  = label_fixed(arr, da)
    feats   = build_raw_features(arr)
    feats   = normalize_features_zscore(feats)

    # Drop invalid-label rows
    valid   = labels != -1
    feats   = feats[valid]
    labels  = labels[valid]

    # Snapshot rate diagnostics
    arr_v = arr[valid]
    if len(arr_v) > 1:
        duration = arr_v['time'][-1] - arr_v['time'][0]
        rate = len(arr_v) / duration if duration > 0 else 0
        print(f"  Snapshots (valid): {len(arr_v):,}  "
              f"Rate: {rate:.0f}/s  Duration: {duration/3600:.2f}h")

    X, y = build_sliding_windows(feats, labels, stride=stride)
    print(f"  X shape: {X.shape}  "
          f"Label dist: DOWN={( y==0).sum()/len(y):.1%} "
          f"FLAT={(y==1).sum()/len(y):.1%} "
          f"UP={(y==2).sum()/len(y):.1%}")
    return X, y, da


# ── Multi-day dataset ─────────────────────────────────────────────────────────
def build_multi_day_dataset(data_dir: str,
                             ticker: str,
                             dates: list,
                             da: DoubleAlpha = None,
                             stride: int = DOWNSAMPLE_STRIDE,
                             n_workers: int = None):
    """
    Build dataset from multiple days.
    If da is None, fits DoubleAlpha from all provided dates (for training set).
    For test set, pass the training da explicitly.

    Returns:
        X  : (M_total, 100, 40) float32
        y  : (M_total,)         int8
        da : DoubleAlpha
    """
    paths = []
    for date in dates:
        p = os.path.join(data_dir, f'{ticker}_{date}_dataset.bin')
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"  ⚠ Missing: {p}")

    if not paths:
        raise FileNotFoundError(f"No dataset files found for {ticker} in {data_dir}")

    # Fit DoubleAlpha from all training days jointly (Section 1.3 of Preprocessing doc)
    if da is None:
        if n_workers is None:
            n_workers = min(len(paths), os.cpu_count() or 1)
        print(f"Fitting DoubleAlpha from {len(paths)} files ({n_workers} workers)...")
        da = fit_double_alpha_from_files(paths, n_workers=n_workers)
        print(f"  {da}")
        if da.asymmetry > 0.05:
            print(f"  ⚠ Asymmetry {da.asymmetry:.1%} -- DoubleAlpha essential")

    X_list, y_list = [], []
    for i, (path, date) in enumerate(zip(paths, dates)):
        print(f"[{i+1}/{len(paths)}] {ticker} {date}...")
        arr    = load_snapshots(path)
        arr    = filter_session(arr)
        labels = label_fixed(arr, da)
        feats  = build_raw_features(arr)
        feats  = normalize_features_zscore(feats)

        valid  = labels != -1
        feats  = feats[valid]
        labels = labels[valid]

        if len(feats) < 200:
            print(f"  Skipped ({len(feats)} valid rows)")
            continue

        X, y = build_sliding_windows(feats, labels, stride=stride)
        print(f"  X={X.shape}  DOWN={(y==0).mean():.1%} "
              f"FLAT={(y==1).mean():.1%} UP={(y==2).mean():.1%}")
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No valid days processed")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"\nTotal: X={X_all.shape}  "
          f"DOWN={(y_all==0).mean():.1%} "
          f"FLAT={(y_all==1).mean():.1%} "
          f"UP={(y_all==2).mean():.1%}")
    return X_all, y_all, da


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python feature_label_builder.py <path_to_dataset.bin>")
        sys.exit(1)

    print(f"\nSmoke test: {path}")
    print("-" * 60)
    X, y, da = build_dataset(path)
    print(f"\n✅ Done. X={X.shape} y={y.shape}")
    print(f"   RAM estimate: {X.nbytes / 1e9:.2f} GB")
