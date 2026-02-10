"""
feature_label_builder.py
========================
Builds features and labels from 248-byte LOBSnapshot binary files.

Uses DoubleAlpha from adaptive_labeler (separate alpha_down / alpha_up).
Uses adaptive_labeler.normalize_features_zscore (scipy-based, ~1s for 6M rows).
Uses adaptive_labeler.load_snapshots (fast numpy path).

Usage:
  python scripts/feature_label_builder.py data/datasets/clean/daily/nvda_20260105_dataset.bin

  from feature_label_builder import build_multi_day_dataset, TRAIN_DATES, TEST_DATES
  X, y, ts = build_multi_day_dataset(DATA_DIR, 'nvda', TRAIN_DATES)
"""

import numpy as np
import os
from typing import Tuple, List

from adaptive_labeler import (
    load_snapshots,
    mid_price,
    compute_future_mid,
    normalize_features_zscore,
    fit_double_alpha_from_files,
    fit_double_alpha,
    label_fixed,
    DoubleAlpha,
    HORIZON_S,
    FUTURE_WIN_S,
)

NUM_LEVELS = 10

TRAIN_DATES = [
    '20260105', '20260106', '20260107', '20260108', '20260109',
    '20260112', '20260113', '20260114', '20260115', '20260116',
]
TEST_DATES = [
    '20260120', '20260121', '20260122', '20260123', '20260126',
    '20260127', '20260128', '20260129',
]


def build_raw_features(snapshots: np.ndarray) -> np.ndarray:
    """
    40-feature DeepLOB vector per snapshot.
    Interleaved: [ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ...]
    Output shape: (N, 40)
    """
    N        = len(snapshots)
    features = np.zeros((N, 40), dtype=np.float32)
    for lvl in range(NUM_LEVELS):
        base = lvl * 4
        features[:, base + 0] = snapshots[:, 21 + lvl]   # ask price
        features[:, base + 1] = snapshots[:, 31 + lvl]   # ask size
        features[:, base + 2] = snapshots[:, 1  + lvl]   # bid price
        features[:, base + 3] = snapshots[:, 11 + lvl]   # bid size
    return features


def build_sliding_windows(
    features_normalized: np.ndarray,
    labels:              np.ndarray,
    valid_mask:          np.ndarray,
    window_size:         int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chunked sliding window construction to stay within 64GB RAM.
    Processes 100k windows at a time (~1.5GB per chunk).
    """
    N_feat   = len(features_normalized)
    feat_end = np.arange(window_size - 1, N_feat)
    orig_idx = feat_end + window_size

    in_bounds = orig_idx < len(labels)
    feat_end  = feat_end[in_bounds]
    orig_idx  = orig_idx[in_bounds]

    keep     = valid_mask[orig_idx] & (labels[orig_idx] >= 0)
    feat_end = feat_end[keep]
    orig_idx = orig_idx[keep]

    if len(feat_end) == 0:
        return np.empty((0, window_size, 40), dtype=np.float32), np.empty(0, dtype=np.int64)

    M = len(feat_end)
    X = np.empty((M, window_size, 40), dtype=np.float32)
    y = labels[orig_idx].astype(np.int64)

    CHUNK = 100_000
    for start in range(0, M, CHUNK):
        end     = min(start + CHUNK, M)
        col_idx = feat_end[start:end, None] - np.arange(window_size - 1, -1, -1)[None, :]
        X[start:end] = features_normalized[col_idx]
        if start % 1_000_000 == 0 and start > 0:
            print(f"    windows: {start:,}/{M:,}", flush=True)

    return X, y


def extract_timestamps(snapshots, features_normalized, labels, valid_mask, window_size=100, n=None):
    N_feat    = len(features_normalized)
    feat_end  = np.arange(window_size - 1, N_feat)
    orig_idx  = feat_end + window_size
    in_bounds = orig_idx < len(snapshots)
    orig_idx  = orig_idx[in_bounds]
    feat_end  = feat_end[in_bounds]
    keep      = valid_mask[orig_idx] & (labels[orig_idx] >= 0)
    orig_idx  = orig_idx[keep]
    ts        = snapshots[orig_idx, 0].astype(np.float64)
    return ts[:n] if n is not None else ts


def print_label_distribution(y: np.ndarray, name: str = ""):
    total = len(y)
    if total == 0:
        print(f"  {name}: EMPTY"); return
    c = {k: int((y == k).sum()) for k in [0, 1, 2]}
    print(f"\n  Label distribution {name} (total={total:,}):")
    print(f"    DOWN       (0): {c[0]:8,}  ({100*c[0]/total:5.1f}%)")
    print(f"    STATIONARY (1): {c[1]:8,}  ({100*c[1]/total:5.1f}%)")
    print(f"    UP         (2): {c[2]:8,}  ({100*c[2]/total:5.1f}%)")


def build_dataset(
    filepath:    str,
    da:          DoubleAlpha = None,
    window_size: int         = 100,
    verbose:     bool        = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: .bin -> (X, y, timestamps)

    da: pre-fitted DoubleAlpha. If None, fits from this file alone.
        Always pass a pre-fitted da for multi-day training so all days
        use the same alpha_down / alpha_up thresholds.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Building dataset: {os.path.basename(filepath)}")
        print(f"{'='*60}")

    snapshots = load_snapshots(filepath)
    if verbose:
        dur  = (snapshots[-1, 0] - snapshots[0, 0]) / 60.0
        rate = len(snapshots) / (snapshots[-1, 0] - snapshots[0, 0])
        print(f"  Loaded {len(snapshots):,} snapshots  ({dur:.1f} min, {rate:.0f}/sec)")

    if da is None:
        if verbose: print(f"  Fitting DoubleAlpha from this file...")
        da = fit_double_alpha([snapshots], target_pct=0.22, verbose=verbose)
    else:
        if verbose:
            print(f"  DoubleAlpha: alpha_down={da.alpha_down:.6f}  alpha_up={da.alpha_up:.6f}")

    if verbose: print(f"  Labelling...")
    labels, valid_mask = label_fixed(snapshots, da)
    if verbose: print_label_distribution(labels[valid_mask], os.path.basename(filepath))

    raw_features = build_raw_features(snapshots)

    if verbose: print(f"  Normalizing (scipy rolling z-score, window={window_size})...")
    features_norm = normalize_features_zscore(raw_features, window=window_size)

    if verbose: print(f"  Building sliding windows...")
    X, y = build_sliding_windows(features_norm, labels, valid_mask, window_size)

    if len(X) == 0:
        raise ValueError("No valid windows produced.")

    ts = extract_timestamps(snapshots, features_norm, labels, valid_mask, window_size, n=len(y))

    if verbose: print(f"\n  Done: X={X.shape}  y={y.shape}")
    return X, y, ts


def build_multi_day_dataset(
    bin_dir:     str,
    ticker:      str,
    date_range:  List[str],
    da:          DoubleAlpha = None,
    window_size: int         = 100,
    verbose:     bool        = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-day pipeline. Fits DoubleAlpha across ALL files first if da is None.
    This ensures consistent alpha_down/alpha_up across all training days.
    """
    filepaths = []
    for date in date_range:
        fp = os.path.join(bin_dir, f"{ticker}_{date}_dataset.bin")
        if os.path.exists(fp):
            filepaths.append(fp)
        elif verbose:
            print(f"  SKIP (missing): {fp}")

    if not filepaths:
        raise ValueError(f"No files found for {ticker} in {bin_dir}")

    if da is None:
        if verbose:
            print(f"\n  Fitting DoubleAlpha across {len(filepaths)} files (parallel)...")
        da = fit_double_alpha_from_files(filepaths, target_pct=0.22, verbose=verbose)

    X_parts, y_parts, ts_parts = [], [], []
    for fp in filepaths:
        try:
            X, y, ts = build_dataset(fp, da=da, window_size=window_size, verbose=verbose)
            X_parts.append(X); y_parts.append(y); ts_parts.append(ts)
        except Exception as e:
            print(f"  ERROR {fp}: {e}")

    if not X_parts:
        raise ValueError(f"No valid data for {ticker}")

    X_all  = np.concatenate(X_parts,  axis=0)
    y_all  = np.concatenate(y_parts,  axis=0)
    ts_all = np.concatenate(ts_parts, axis=0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-DAY: {ticker.upper()} ({len(filepaths)} days)")
        print_label_distribution(y_all, ticker.upper())
        print(f"Final: X={X_all.shape}  y={y_all.shape}")
        print(f"{'='*60}")

    return X_all, y_all, ts_all


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/feature_label_builder.py <path_to_dataset.bin>")
        sys.exit(1)
    X, y, ts = build_dataset(sys.argv[1], da=None)
    print(f"\nReady.  X={X.shape}  y={y.shape}")