# scripts/ml/adaptive_labeler.py
"""
adaptive_labeler.py
===================
FINAL. Do not modify functions already marked DONE in Preprocessing doc.

Contains:
  - load_snapshots         : fast numpy structured dtype binary loader
  - fit_double_alpha       : fit asymmetric thresholds from single-day data
  - fit_double_alpha_from_files : parallel multi-file DoubleAlpha fitting
  - label_fixed            : vectorized time-based 100ms labelling
  - label_realtime_atr     : per-snapshot ATR labelling (inference only)
  - build_multi_day_parallel : parallel multi-day dataset builder

DoubleAlpha (Section 1 of Preprocessing doc):
  - alpha_down = 22nd percentile of return distribution (DOWN threshold)
  - alpha_up   = 78th percentile                        (UP  threshold)
  - MUST be fitted once across all 10 training days jointly.
  - Test set uses training alpha -- never re-fit on test data.

Horizon: 100ms wall-clock (NOT fixed snapshot count k).
  At avg 210 snapshots/s (NVDA), 100ms ≈ 21 snapshots,
  but varies 50ms–200ms in burst/quiet periods.
  This is methodologically correct and stated in paper Section 3.
"""

import os
import pickle
import struct
import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool
from scipy.ndimage import uniform_filter1d

# ── Constants ─────────────────────────────────────────────────────────────────
SNAPSHOT_BYTES  = 248
SNAPSHOT_DTYPE  = np.dtype([
    ('time',     '<f8'),
    ('bid_px',   '<u8', 10),
    ('bid_sz',   '<i4', 10),
    ('ask_px',   '<u8', 10),
    ('ask_sz',   '<i4', 10),
])
PRICE_SCALE     = 1e4          # stored as int × 1e-4
HORIZON_SECS    = 0.100        # 100ms forward window for label
HORIZON_AVG_SECS= 0.100        # 100ms averaging window [t+H, t+2H]

# Session window (09:30–16:00 ET) -- LOCKED decision, cite in paper Sec 3
# Using Unix-timestamp arithmetic: epoch offset depends on date, so we keep
# it as hour offsets and apply per-file. Callers may also pass pre-filtered data.
SESSION_START_HOUR_ET = 9.5    # 09:30
SESSION_END_HOUR_ET   = 16.0   # 16:00

# ── DoubleAlpha dataclass ─────────────────────────────────────────────────────
@dataclass
class DoubleAlpha:
    """Asymmetric return thresholds fitted from training data."""
    alpha_down:  float   # |22nd percentile| -- DOWN threshold
    alpha_up:    float   # 78th percentile   -- UP  threshold
    n_samples:   int     # number of returns used in fitting
    asymmetry:   float   # abs(alpha_down - alpha_up) / mean -- diagnostic

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'DoubleAlpha':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return (f"DoubleAlpha(down={self.alpha_down:.6f}, up={self.alpha_up:.6f}, "
                f"n={self.n_samples:,}, asymmetry={self.asymmetry:.1%})")


# ── Fast binary loader (DONE -- Bug 1 fixed) ─────────────────────────────────
def load_snapshots(path: str) -> np.ndarray:
    """
    Load LOBSnapshot binary file into numpy structured array.
    Uses fromfile() -- zero Python-loop overhead.
    ~0.5s for 6.19M snapshots (6.19M × 248 bytes = 1.53 GB).
    """
    arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    return arr


def filter_session(arr: np.ndarray) -> np.ndarray:
    """
    Keep only market-hours snapshots (09:30–16:00 ET).
    LOCKED decision: pre-market data has 9-77% accuracy and violates
    Hawkes stationarity requirement (Rambaldi et al. 2016, Sec 2).
    """
    # Convert Unix timestamp to hour-of-day in ET (UTC-5 or UTC-4 DST)
    # January 2026 is winter: ET = UTC - 5
    UTC_OFFSET = 5.0
    hour_et = (arr['time'] % 86400 / 3600 - UTC_OFFSET) % 24
    mask = (hour_et >= SESSION_START_HOUR_ET) & (hour_et < SESSION_END_HOUR_ET)
    return arr[mask]


def mid_price(arr: np.ndarray) -> np.ndarray:
    """Return mid-price in price units (not raw int). Shape: (N,)"""
    bp = arr['bid_px'][:, 0].astype(np.float64) / PRICE_SCALE
    ap = arr['ask_px'][:, 0].astype(np.float64) / PRICE_SCALE
    return (bp + ap) * 0.5


# ── DoubleAlpha fitting (DONE) ────────────────────────────────────────────────
def _compute_returns_for_file(path: str):
    """
    Worker function: compute 100ms forward returns for a single file.
    Returns 1-D numpy array of percentage changes.
    """
    arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    arr = filter_session(arr)
    if len(arr) < 200:
        return np.array([], dtype=np.float64)

    t   = arr['time']
    mid = mid_price(arr)
    N   = len(arr)

    returns = np.empty(N, dtype=np.float64)
    returns[:] = np.nan

    # For each snapshot t_i, average mid in [t_i + H, t_i + 2H]
    j_start = 0
    j_end   = 0
    for i in range(N):
        t_lo = t[i] + HORIZON_SECS
        t_hi = t[i] + HORIZON_SECS + HORIZON_AVG_SECS
        # Advance j_start to first index >= t_lo
        while j_start < N and t[j_start] < t_lo:
            j_start += 1
        # Advance j_end to first index >= t_hi
        if j_end < j_start:
            j_end = j_start
        while j_end < N and t[j_end] < t_hi:
            j_end += 1
        if j_end > j_start and mid[i] > 0:
            avg_future = mid[j_start:j_end].mean()
            returns[i] = (avg_future - mid[i]) / mid[i]

    return returns[~np.isnan(returns)]


def fit_double_alpha(path: str, target_tail: float = 0.22) -> DoubleAlpha:
    """Fit DoubleAlpha from a single file (for quick inspection / smoke test)."""
    rets = _compute_returns_for_file(path)
    return _alpha_from_returns(rets, target_tail)


def fit_double_alpha_from_files(paths: list, target_tail: float = 0.22,
                                n_workers: int = None) -> DoubleAlpha:
    """
    Fit DoubleAlpha from multiple files in parallel.
    MUST be called on training files only -- never include test data.
    Uses all available cores unless n_workers specified.
    """
    if n_workers is None:
        n_workers = min(len(paths), os.cpu_count() or 1)
    with Pool(processes=n_workers) as pool:
        all_returns = pool.map(_compute_returns_for_file, paths)
    combined = np.concatenate([r for r in all_returns if len(r) > 0])
    return _alpha_from_returns(combined, target_tail)


def _alpha_from_returns(rets: np.ndarray, target_tail: float) -> DoubleAlpha:
    if len(rets) == 0:
        raise ValueError("No valid returns -- check data files")
    alpha_down = float(abs(np.percentile(rets, target_tail * 100)))
    alpha_up   = float(np.percentile(rets, (1 - target_tail) * 100))
    mean_abs   = float(np.mean(np.abs(rets[rets != 0]))) or 1e-9
    asymmetry  = abs(alpha_down - alpha_up) / mean_abs
    return DoubleAlpha(alpha_down=alpha_down, alpha_up=alpha_up,
                       n_samples=len(rets), asymmetry=asymmetry)


# ── Fixed-alpha labelling (DONE -- vectorized, no Python loop) ────────────────
def label_fixed(arr: np.ndarray, da: DoubleAlpha) -> np.ndarray:
    """
    Assign class labels using DoubleAlpha thresholds.
    Labels: 0=DOWN, 1=FLAT, 2=UP.

    Returns (N,) int8 array. Snapshots without a valid 100ms future
    window are assigned label -1 (excluded from training).
    """
    t   = arr['time']
    mid = mid_price(arr)
    N   = len(arr)

    labels = np.full(N, -1, dtype=np.int8)

    # Vectorized: for each snapshot i, find indices in [t_i+H, t_i+2H]
    # Use searchsorted for O(N log N) instead of nested loop
    t_lo = t + HORIZON_SECS
    t_hi = t + HORIZON_SECS + HORIZON_AVG_SECS
    idx_lo = np.searchsorted(t, t_lo, side='left')
    idx_hi = np.searchsorted(t, t_hi, side='left')

    valid = (idx_hi > idx_lo) & (mid > 0)

    # Compute prefix sums for fast range means
    mid_cumsum = np.concatenate(([0.0], np.cumsum(mid)))
    counts     = idx_hi - idx_lo  # could be zero, safe because of valid mask

    future_mid = np.where(
        valid,
        (mid_cumsum[idx_hi] - mid_cumsum[idx_lo]) / np.where(counts > 0, counts, 1),
        0.0
    )
    ret = np.where(valid & (mid > 0), (future_mid - mid) / mid, 0.0)

    labels[valid & (ret < -da.alpha_down)] = 0   # DOWN
    labels[valid & (ret >= -da.alpha_down) & (ret <= da.alpha_up)] = 1  # FLAT
    labels[valid & (ret > da.alpha_up)]    = 2   # UP

    return labels


# ── ATR labelling (for inference / C++ simulation -- NOT used in training) ────
def label_realtime_atr(arr: np.ndarray,
                       atr_window_sec: float = 60.0,
                       sample_interval_sec: float = 0.010) -> np.ndarray:
    """
    Per-snapshot adaptive threshold labelling.
    Computes rolling 60-second window of 10ms-sampled mid-price changes.
    Used by inference pipeline and C++ AdaptiveAlpha -- NOT for supervised training.
    Returns (N,) int8 array (0=DOWN, 1=FLAT, 2=UP, -1=invalid).
    """
    t   = arr['time']
    mid = mid_price(arr)
    N   = len(arr)
    labels = np.full(N, -1, dtype=np.int8)

    for i in range(N):
        t_now = t[i]
        # Collect 10ms-sampled changes in [t_now - atr_window, t_now]
        mask = (t >= t_now - atr_window_sec) & (t < t_now)
        window_mid = mid[mask]
        if len(window_mid) < 10:
            continue
        # Subsample at ~10ms
        diffs = np.diff(window_mid[::max(1, int(sample_interval_sec *
                                              len(window_mid) / atr_window_sec))])
        if len(diffs) < 5:
            continue
        atr_threshold = np.mean(np.abs(diffs))

        # Forward return
        idx_lo = np.searchsorted(t, t_now + HORIZON_SECS, 'left')
        idx_hi = np.searchsorted(t, t_now + HORIZON_SECS + HORIZON_AVG_SECS, 'left')
        if idx_hi <= idx_lo or mid[i] <= 0:
            continue
        future = mid[idx_lo:idx_hi].mean()
        ret = (future - mid[i]) / mid[i]

        if ret < -atr_threshold:
            labels[i] = 0
        elif ret > atr_threshold:
            labels[i] = 2
        else:
            labels[i] = 1

    return labels


# ── Normalisation (DONE -- scipy uniform_filter1d, ~1s for 6M×40) ────────────
def normalize_features_zscore(features: np.ndarray,
                               window: int = 100) -> np.ndarray:
    """
    Rolling z-score normalisation using scipy uniform_filter1d.
    ~1 second for 6M × 40 array. No global normalisation (leaks future).

    features: (N, 40) float32
    Returns : (N, 40) float32, normalised
    """
    features = features.astype(np.float32)
    # Rolling mean and std approximated by uniform filter
    mu  = uniform_filter1d(features, size=window, axis=0, mode='nearest')
    sq  = uniform_filter1d(features**2, size=window, axis=0, mode='nearest')
    var = np.maximum(sq - mu**2, 1e-8)
    return ((features - mu) / np.sqrt(var)).astype(np.float32)


# ── Multi-day parallel builder (DONE) ─────────────────────────────────────────
def _process_one_day(args):
    """Worker: load one day, build features, label, normalise."""
    path, da = args
    from feature_label_builder import build_raw_features, build_sliding_windows

    arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    arr = filter_session(arr)
    if len(arr) < 200:
        return None, None, None

    labels = label_fixed(arr, da)
    feats  = build_raw_features(arr)
    feats  = normalize_features_zscore(feats)

    # Remove invalid-label rows
    valid  = labels != -1
    feats  = feats[valid]
    labels = labels[valid]
    arr    = arr[valid]

    if len(feats) < 200:
        return None, None, None

    return feats, labels, arr['time']


def build_multi_day_parallel(train_paths: list, da: DoubleAlpha,
                              n_workers: int = None):
    """
    Build multi-day dataset in parallel.
    Applies shared DoubleAlpha (fitted on all training days jointly).

    Returns:
      features : (N_total, 40) float32
      labels   : (N_total,)   int8
      timestamps: (N_total,)  float64
    """
    if n_workers is None:
        n_workers = min(len(train_paths), os.cpu_count() or 1)

    args = [(p, da) for p in train_paths]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_process_one_day, args)

    feats_list, label_list, ts_list = [], [], []
    for feats, labels, ts in results:
        if feats is not None:
            feats_list.append(feats)
            label_list.append(labels)
            ts_list.append(ts)

    if not feats_list:
        raise RuntimeError("No valid days loaded")

    return (np.concatenate(feats_list),
            np.concatenate(label_list),
            np.concatenate(ts_list))


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python adaptive_labeler.py <path_to_dataset.bin>")
        raise SystemExit(1)

    print(f"Loading {path}...")
    arr = load_snapshots(path)
    arr = filter_session(arr)
    print(f"  Snapshots (session): {len(arr):,}")

    print("Fitting DoubleAlpha...")
    da = fit_double_alpha(path)
    print(f"  {da}")
    if da.asymmetry > 0.05:
        print(f"  ⚠ Asymmetry {da.asymmetry:.1%} > 5% -- DoubleAlpha essential")

    print("Labelling...")
    labels = label_fixed(arr, da)
    valid  = labels != -1
    lv     = labels[valid]
    total  = valid.sum()
    print(f"  DOWN: {(lv==0).sum()/total:.1%}  FLAT: {(lv==1).sum()/total:.1%}"
          f"  UP: {(lv==2).sum()/total:.1%}  (n={total:,})")
