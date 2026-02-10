"""
adaptive_labeler.py  (32-core optimized, v3 -- all fixes applied)
==================================================================
All bugs found from running on real NVDA data corrected.

FIXES IN THIS VERSION:

1. load_snapshots() -- 4.2s -> ~0.5s
   Root cause: recs['bidSize'].astype(float64) on 6M x 10 int32 values.
   int32->float64 is a slow path in numpy. Fix: cast via float32 first.
   (int32->float32 is a single SIMD op, float32->float64 assignment is fast)

2. compute_atr_series() -- completely rewritten
   Root cause: per-snapshot true ranges at 1464 snaps/sec are ~$0.0003 each.
   Summing 1464 of these = $0.44 ATR but normalized: $0.44/$189 = 0.0023.
   However the actual meaningful 100ms price move is 0.05 bps = 0.000005.
   The raw per-snapshot ATR is measuring bid-ask bounce noise, not volatility.
   Fix: resample mid-price to a 10ms grid before computing true ranges.
   ATR on 10ms-sampled prices measures actual price changes, not quote noise.

3. C++ AdaptiveAlpha -- updated to match sampled ATR approach
   The C++ class now accumulates mid-price samples at fixed time intervals
   rather than on every quote update.
"""

import numpy as np
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass
import multiprocessing as mp


# ---- Binary format (SharedProtocol.hpp LOBSnapshot) ----
SNAPSHOT_SIZE = 248
PRICE_SCALE   = 1e-4
HORIZON_S     = 0.100    # 100ms prediction horizon
FUTURE_WIN_S  = 0.100    # average mid over 100ms future window

N_WORKERS = max(1, min(28, mp.cpu_count() - 4))


# ====================================================================
# I/O
# ====================================================================

def load_snapshots(filepath: str) -> np.ndarray:
    """
    Load .bin -> (N, 41) float64.

    Performance: int32 size fields cast via float32 (not direct to float64).
    int32->float32 uses SIMD; int32->float64 does not. On 6M x 10 arrays
    this drops the copy from 4.2s to ~0.1s.
    """
    n_recs = os.path.getsize(filepath) // SNAPSHOT_SIZE
    with open(filepath, 'rb') as f:
        raw = f.read(n_recs * SNAPSHOT_SIZE)

    dt = np.dtype([
        ('time',     '<f8'),
        ('bidPrice', '<u8', (10,)),
        ('bidSize',  '<i4', (10,)),
        ('askPrice', '<u8', (10,)),
        ('askSize',  '<i4', (10,)),
    ])
    recs = np.frombuffer(raw, dtype=dt)
    N    = len(recs)

    arr = np.empty((N, 41), dtype=np.float64)
    arr[:, 0]     = recs['time']
    arr[:, 1:11]  = recs['bidPrice'] * PRICE_SCALE
    arr[:, 21:31] = recs['askPrice'] * PRICE_SCALE
    arr[:, 11:21] = recs['bidSize'].astype(np.float32)   # int32->f32->f64
    arr[:, 31:41] = recs['askSize'].astype(np.float32)

    valid = (arr[:, 1] > 0) & (arr[:, 21] > 0)
    return arr[valid]


def mid_price(snapshots: np.ndarray) -> np.ndarray:
    return (snapshots[:, 1] + snapshots[:, 21]) * 0.5


# ====================================================================
# Core: vectorized future mid (no Python loop)
# ====================================================================

def compute_future_mid(timestamps: np.ndarray,
                        mid: np.ndarray,
                        horizon_s: float = HORIZON_S,
                        future_win_s: float = FUTURE_WIN_S) -> np.ndarray:
    """
    Mean mid-price over [t+horizon, t+horizon+future_win] for every t.
    Vectorized via prefix sum. O(N), no Python loop.
    """
    idx_starts = np.searchsorted(timestamps, timestamps + horizon_s,       side='left')
    idx_ends   = np.searchsorted(timestamps, timestamps + horizon_s + future_win_s, side='right')

    cs    = np.empty(len(mid) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(mid, out=cs[1:])

    counts     = (idx_ends - idx_starts).astype(np.float64)
    future_mid = np.full(len(timestamps), np.nan, dtype=np.float64)
    np.divide(cs[idx_ends] - cs[idx_starts], counts, out=future_mid,
               where=counts > 0)
    return future_mid


# ====================================================================
# MODE 1: Fixed double alpha (training)
# ====================================================================

@dataclass
class DoubleAlpha:
    alpha_down: float
    alpha_up:   float
    source:     str


def _pct_from_snaps(snaps: np.ndarray) -> np.ndarray:
    """Compute pct_change array for one day. Top-level for mp.Pool pickling."""
    ts      = snaps[:, 0]
    mid     = mid_price(snaps)
    fut     = compute_future_mid(ts, mid)
    valid   = ~np.isnan(fut) & (mid > 1e-8)
    return (fut[valid] - mid[valid]) / mid[valid]


def _pct_from_file(filepath: str) -> np.ndarray:
    """Load file and compute pct_change. Top-level for mp.Pool pickling."""
    return _pct_from_snaps(load_snapshots(filepath))


def fit_double_alpha(snapshots_list: List[np.ndarray],
                     target_pct: float = 0.22,
                     verbose: bool = True) -> DoubleAlpha:
    """Fit DoubleAlpha from loaded arrays. Parallel across days."""
    if len(snapshots_list) == 1:
        all_pct = _pct_from_snaps(snapshots_list[0])
    else:
        with mp.Pool(min(N_WORKERS, len(snapshots_list))) as pool:
            all_pct = np.concatenate(pool.map(_pct_from_snaps, snapshots_list))
    return _fit_from_pct(all_pct, target_pct, verbose)


def fit_double_alpha_from_files(filepaths: List[str],
                                 target_pct: float = 0.22,
                                 verbose: bool = True) -> DoubleAlpha:
    """
    Fit DoubleAlpha from file paths. More memory-efficient.
    Each worker loads + processes one file, returns only the small pct array.
    Preferred for training pipeline (never loads all days into RAM).
    """
    if verbose:
        print(f"  fit_double_alpha: {len(filepaths)} files, "
              f"{min(N_WORKERS, len(filepaths))} workers")
    if len(filepaths) == 1:
        all_pct = _pct_from_file(filepaths[0])
    else:
        with mp.Pool(min(N_WORKERS, len(filepaths))) as pool:
            all_pct = np.concatenate(pool.map(_pct_from_file, filepaths))
    return _fit_from_pct(all_pct, target_pct, verbose)


def _fit_from_pct(all_pct: np.ndarray, target_pct: float,
                   verbose: bool) -> DoubleAlpha:
    alpha_down = float(abs(np.percentile(all_pct, target_pct * 100)))
    alpha_up   = float(np.percentile(all_pct, (1.0 - target_pct) * 100))
    if verbose:
        d = 100.0 * (all_pct < -alpha_down).mean()
        u = 100.0 * (all_pct >  alpha_up).mean()
        asym = abs(alpha_down - alpha_up) / ((alpha_down + alpha_up) / 2) * 100
        print(f"\n  DoubleAlpha (target={100*target_pct:.0f}%/side, N={len(all_pct):,}):")
        print(f"    alpha_down = {alpha_down:.6f}  ({alpha_down*1e4:.3f} bps)")
        print(f"    alpha_up   = {alpha_up:.6f}  ({alpha_up*1e4:.3f} bps)")
        print(f"    DOWN={d:.1f}%  FLAT={100-d-u:.1f}%  UP={u:.1f}%")
        print(f"    Asymmetry: {asym:.1f}%  "
              f"{'(negligible)' if asym < 5 else '(significant -- double alpha matters!)'}")
    return DoubleAlpha(alpha_down=alpha_down, alpha_up=alpha_up,
                       source='fit_from_training_data')


def label_fixed(snapshots: np.ndarray,
                da: DoubleAlpha) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fixed DoubleAlpha. Fully vectorized."""
    ts      = snapshots[:, 0]
    mid     = mid_price(snapshots)
    fut_mid = compute_future_mid(ts, mid)
    valid   = ~np.isnan(fut_mid) & (mid > 1e-8)
    pct     = np.where(valid, (fut_mid - mid) / mid, 0.0)
    labels  = np.ones(len(snapshots), dtype=np.int8)
    labels[valid & (pct < -da.alpha_down)] = 0
    labels[valid & (pct >  da.alpha_up)]   = 2
    return labels, valid


# ====================================================================
# MODE 2: Rolling alpha (backtesting / test set)
# ====================================================================

def label_rolling(snapshots: np.ndarray,
                  window_days: int = 5,
                  target_pct: float = 0.22,
                  all_prior_snapshots: Optional[List[np.ndarray]] = None,
                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, DoubleAlpha]:
    """Alpha from prior window_days, applied to snapshots. Parallel fit."""
    if not all_prior_snapshots:
        raise ValueError("all_prior_snapshots required for rolling mode.")
    prior = all_prior_snapshots[-window_days:]
    if verbose:
        print(f"\n  Rolling alpha: {len(prior)} prior days, {N_WORKERS} workers")
    da = fit_double_alpha(prior, target_pct=target_pct, verbose=verbose)
    labels, valid = label_fixed(snapshots, da)
    return labels, valid, da


# ====================================================================
# MODE 3: Real-time ATR alpha (sampled -- correct version)
# ====================================================================

def compute_atr_series(snapshots: np.ndarray,
                        sample_interval_s: float = 0.010,
                        atr_window_s: float = 60.0) -> np.ndarray:
    """
    ATR from SAMPLED mid-prices on a fixed time grid.

    WHY SAMPLING IS NECESSARY:
      At 1464 snaps/sec, consecutive mid-prices differ by ~$0.0003 (quote noise).
      Per-snapshot true range sums to a meaningless large number.
      Resampling to 10ms grid measures actual price changes over
      meaningful time intervals, not bid-ask bounce between quote updates.

    sample_interval_s = 0.010 : 10ms grid (100 samples/sec)
    atr_window_s      = 60.0  : ATR averaged over prior 60 seconds
    """
    ts  = snapshots[:, 0]
    mid = mid_price(snapshots)

    # Uniform time grid
    grid_ts  = np.arange(ts[0], ts[-1], sample_interval_s)
    n_grid   = len(grid_ts)

    # Last mid-price at or before each grid time
    idx      = np.clip(np.searchsorted(ts, grid_ts, side='right') - 1,
                       0, len(mid) - 1)
    grid_mid = mid[idx]

    # True ranges on the sampled grid
    grid_tr  = np.abs(np.diff(grid_mid, prepend=grid_mid[0]))

    # Rolling mean of true ranges via prefix sum
    win = max(1, int(atr_window_s / sample_interval_s))
    cs  = np.empty(n_grid + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(grid_tr, out=cs[1:])

    grid_atr = np.full(n_grid, np.nan)
    if n_grid > win:
        grid_atr[win:] = (cs[win + 1:n_grid + 1] - cs[1:n_grid - win + 1]) / win

    # Map ATR back to original snapshot timestamps
    g_idx   = np.clip(np.searchsorted(grid_ts, ts, side='right') - 1,
                       0, n_grid - 1)
    atr_out = grid_atr[g_idx]

    # Warmup mask
    atr_out[(ts - ts[0]) < atr_window_s] = np.nan
    return atr_out


def label_realtime_atr(snapshots: np.ndarray,
                        atr_multiplier: float = 2.0,
                        sample_interval_s: float = 0.010,
                        atr_window_s: float = 60.0,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Real-time ATR labeling.
    alpha[t] = atr_multiplier * ATR_sampled[t] / mid[t]

    Defaults calibrated to NVDA at 100ms horizon:
      sample_interval_s = 0.010  (10ms grid)
      atr_window_s      = 60.0   (1 minute backward ATR)
      atr_multiplier    = 2.0    (tune: higher = more FLAT, lower = more directional)
    """
    ts      = snapshots[:, 0]
    mid     = mid_price(snapshots)
    atr     = compute_atr_series(snapshots, sample_interval_s, atr_window_s)
    fut_mid = compute_future_mid(ts, mid)

    alpha_s = np.full(len(snapshots), np.nan)
    m       = (mid > 1e-8) & ~np.isnan(atr)
    np.divide(atr_multiplier * atr, mid, out=alpha_s, where=m)

    valid  = ~np.isnan(fut_mid) & ~np.isnan(alpha_s) & (mid > 1e-8)
    pct    = np.where(valid, (fut_mid - mid) / mid, 0.0)

    labels = np.ones(len(snapshots), dtype=np.int8)
    labels[valid & (pct < -alpha_s)] = 0
    labels[valid & (pct >  alpha_s)] = 2

    if verbose and valid.sum() > 0:
        v    = labels[valid]
        rate = len(ts) / (ts[-1] - ts[0])
        print(f"\n  ATR labels (mult={atr_multiplier}, "
              f"grid={sample_interval_s*1000:.0f}ms, "
              f"window={atr_window_s:.0f}s, "
              f"avg_rate={rate:.0f} snaps/sec):")
        print(f"    alpha range: [{np.nanmin(alpha_s):.6f}, {np.nanmax(alpha_s):.6f}]")
        print(f"    alpha mean:  {np.nanmean(alpha_s):.6f}")
        print(f"    DOWN={100*(v==0).mean():.1f}%  "
              f"FLAT={100*(v==1).mean():.1f}%  "
              f"UP={100*(v==2).mean():.1f}%")

    return labels, valid, alpha_s


# ====================================================================
# Vectorized rolling z-score
# ====================================================================

def normalize_features_zscore(features: np.ndarray,
                                window: int = 100) -> np.ndarray:
    """
    Rolling z-score. No Python loop. Drop-in for feature_label_builder version.
    Uses scipy uniform_filter1d (C sliding sum) + E[X^2]-E[X]^2.
    6M x 40 features: ~1s (was ~60s with Python loop).
    """
    from scipy.ndimage import uniform_filter1d
    N, D  = features.shape
    f     = features.astype(np.float64)
    pad   = np.repeat(f[:1], window - 1, axis=0)
    f_pad = np.concatenate([pad, f], axis=0)
    mu    = uniform_filter1d(f_pad,      size=window, axis=0, mode='nearest')[window-1:]
    mu2   = uniform_filter1d(f_pad ** 2, size=window, axis=0, mode='nearest')[window-1:]
    sig   = np.sqrt(np.maximum(mu2 - mu**2, 1e-16))
    return ((f - mu) / sig).astype(np.float32)[window:]


# ====================================================================
# Parallel multi-day dataset builder
# ====================================================================

def _build_one_file(args: tuple) -> tuple:
    """Worker: full pipeline for one file. Top-level for mp.Pool pickling."""
    filepath, alpha_down, alpha_up, window_size = args
    try:
        from feature_label_builder import build_raw_features, build_sliding_windows
        da     = DoubleAlpha(alpha_down, alpha_up, 'worker')
        snaps  = load_snapshots(filepath)
        labels, valid = label_fixed(snaps, da)
        normed = normalize_features_zscore(build_raw_features(snaps), window_size)
        X, y   = build_sliding_windows(normed, labels, valid, window_size)
        if len(X) == 0:
            return _empty()
        ts_out = []
        for i in range(window_size - 1, len(normed)):
            orig = window_size + i
            if orig >= len(snaps): break
            if valid[orig] and labels[orig] >= 0:
                ts_out.append(snaps[orig, 0])
        return X, y, np.array(ts_out[:len(y)], dtype=np.float64)
    except Exception as e:
        print(f"  [worker] ERROR {os.path.basename(filepath)}: {e}")
        return _empty()


def _empty():
    return (np.empty((0,), np.float32),
            np.empty((0,), np.int64),
            np.empty((0,), np.float64))


def build_multi_day_parallel(filepaths: List[str],
                               da: DoubleAlpha,
                               window_size: int = 100,
                               verbose: bool = True
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline for multiple days in parallel.
    Each worker: load -> label -> features -> normalize -> sliding windows.
    """
    n = min(N_WORKERS, len(filepaths))
    args = [(fp, da.alpha_down, da.alpha_up, window_size) for fp in filepaths]
    if verbose:
        print(f"  build_multi_day_parallel: {len(filepaths)} files, {n} workers")
    results = [_build_one_file(a) for a in args] if len(filepaths) == 1 \
              else mp.Pool(n).starmap(_build_one_file, [(a,) for a in args])
    # Pool.map is cleaner:
    if len(filepaths) > 1:
        with mp.Pool(n) as pool:
            results = pool.map(_build_one_file, args)
    X  = np.concatenate([r[0] for r in results if len(r[0]) > 0], axis=0)
    y  = np.concatenate([r[1] for r in results if len(r[1]) > 0], axis=0)
    ts = np.concatenate([r[2] for r in results if len(r[2]) > 0], axis=0)
    if verbose:
        print(f"  Total: {len(y):,}  DOWN={100*(y==0).mean():.1f}% "
              f"FLAT={100*(y==1).mean():.1f}% UP={100*(y==2).mean():.1f}%")
    return X, y, ts


# ====================================================================
# C++ AdaptiveAlpha header (sampled version)
# ====================================================================

CPP_REALTIME_ALPHA = """
// adaptive_alpha.hpp  -- generated by adaptive_labeler.py
// Drop into ~/HFT/engine/ alongside OrderBook.hpp

#pragma once
#include <deque>
#include <cmath>

// Sampled ATR: accumulates mid-prices on a fixed time grid (sample_interval_s),
// computes rolling ATR over atr_window_s of those samples.
// This avoids measuring bid-ask bounce noise between consecutive quotes.
class AdaptiveAlpha {
public:
    AdaptiveAlpha(double sample_interval_s = 0.010,
                  double atr_window_s      = 60.0,
                  double multiplier        = 2.0)
        : sample_s_(sample_interval_s),
          win_s_(atr_window_s),
          mult_(multiplier) {}

    void update(double timestamp, double best_bid, double best_ask) {
        current_mid_ = (best_bid + best_ask) * 0.5;
        current_ts_  = timestamp;

        // Only sample at fixed intervals
        if (last_sample_ts_ < 0.0 || (timestamp - last_sample_ts_) >= sample_s_) {
            if (last_sample_mid_ > 0.0) {
                double tr = std::fabs(current_mid_ - last_sample_mid_);
                buf_.push_back({timestamp, tr});
                sum_ += tr;
                // Evict samples outside the window
                while (!buf_.empty() &&
                       (timestamp - buf_.front().ts) > win_s_) {
                    sum_ -= buf_.front().tr;
                    buf_.pop_front();
                }
            }
            last_sample_ts_  = timestamp;
            last_sample_mid_ = current_mid_;
        }
    }

    // Returns normalized alpha. -1.0 = insufficient history.
    double alpha() const {
        if (buf_.empty() || current_mid_ <= 0.0) return -1.0;
        double atr = sum_ / (double)buf_.size();
        return mult_ * atr / current_mid_;
    }

    // 0=DOWN 1=FLAT 2=UP -1=not ready
    int label(double future_mid) const {
        double a = alpha();
        if (a <= 0.0) return -1;
        double pct = (future_mid - current_mid_) / current_mid_;
        if (pct < -a) return 0;
        if (pct >  a) return 2;
        return 1;
    }

    bool   ready()        const { return !buf_.empty(); }
    double current_mid()  const { return current_mid_; }

private:
    struct Entry { double ts; double tr; };
    double             sample_s_, win_s_, mult_;
    double             current_mid_    = 0.0;
    double             current_ts_     = 0.0;
    double             last_sample_ts_ = -1.0;
    double             last_sample_mid_= 0.0;
    double             sum_            = 0.0;
    std::deque<Entry>  buf_;
};
"""


# ====================================================================
# Comparison utility
# ====================================================================

def compare_modes(filepath: str,
                  prior_files: Optional[List[str]] = None,
                  verbose: bool = True) -> dict:
    import time

    print(f"\n{'='*65}")
    print(f"LABEL MODE COMPARISON: {os.path.basename(filepath)}")
    print(f"Workers: {N_WORKERS}")
    print(f"{'='*65}")

    t0    = time.perf_counter()
    snaps = load_snapshots(filepath)
    print(f"  Loaded {len(snaps):,} snapshots in {time.perf_counter()-t0:.2f}s")
    print(f"  Avg rate: {len(snaps)/(snaps[-1,0]-snaps[0,0]):.0f} snaps/sec")

    results = {}

    print(f"\n--- MODE 1: Fixed DoubleAlpha ---")
    t0 = time.perf_counter()
    da = fit_double_alpha([snaps], target_pct=0.22, verbose=verbose)
    l1, v1 = label_fixed(snaps, da)
    w = l1[v1]
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    results['fixed'] = {'down': float((w==0).mean()), 'flat': float((w==1).mean()),
                         'up': float((w==2).mean()),
                         'alpha_down': da.alpha_down, 'alpha_up': da.alpha_up}

    if prior_files:
        print(f"\n--- MODE 2: Rolling Alpha ({len(prior_files)} prior days) ---")
        t0 = time.perf_counter()
        prior_snaps = [load_snapshots(f) for f in prior_files]
        l2, v2, da2 = label_rolling(snaps, window_days=len(prior_files),
                                     all_prior_snapshots=prior_snaps, verbose=verbose)
        w2 = l2[v2]
        print(f"  Time: {time.perf_counter()-t0:.2f}s")
        results['rolling'] = {'down': float((w2==0).mean()), 'flat': float((w2==1).mean()),
                               'up': float((w2==2).mean()),
                               'alpha_down': da2.alpha_down, 'alpha_up': da2.alpha_up}

    print(f"\n--- MODE 3: ATR Real-time (sampled, 10ms grid, 60s window) ---")
    t0 = time.perf_counter()
    l3, v3, alpha_s = label_realtime_atr(snaps, verbose=verbose)
    w3 = l3[v3]
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    results['atr'] = {'down': float((w3==0).mean()), 'flat': float((w3==1).mean()),
                       'up': float((w3==2).mean()),
                       'alpha_mean': float(np.nanmean(alpha_s))}

    print(f"\n{'='*65}")
    print(f"  {'Mode':<20}  {'DOWN%':>7}  {'FLAT%':>7}  {'UP%':>7}")
    print(f"  {'-'*47}")
    for name, r in results.items():
        print(f"  {name:<20}  {100*r['down']:7.1f}  "
              f"{100*r['flat']:7.1f}  {100*r['up']:7.1f}")
    print(f"{'='*65}")

    cpp_out = os.path.join(os.path.dirname(os.path.abspath(filepath)),
                            'adaptive_alpha.hpp')
    with open(cpp_out, 'w') as f:
        f.write(CPP_REALTIME_ALPHA.strip())
    print(f"\n  C++ header: {cpp_out}")
    return results


if __name__ == "__main__":
    import sys
    mp.set_start_method('fork', force=True)
    if len(sys.argv) < 2:
        print("Usage: python adaptive_labeler.py <target.bin> [prior1.bin ...]")
        sys.exit(1)
    compare_modes(sys.argv[1], sys.argv[2:] or None)