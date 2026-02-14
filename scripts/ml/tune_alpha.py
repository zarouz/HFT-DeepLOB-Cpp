"""
tune_alpha.py
=============
Sweep alpha/tail percentile values on one or more files to find the
right threshold before training.

Usage:
  # Single file (quick smoke test)
  python scripts/ml/tune_alpha.py data/datasets/clean/daily/nvda_20260105_dataset.bin

  # Multi-file (recommended before final training)
  python scripts/ml/tune_alpha.py  # uses all TRAIN_DATES from config.py

From Preprocessing doc:
  NVDA result  : alpha_down=0.000052, alpha_up=0.000049, asymmetry=6.5%
  Target dist  : ~22% DOWN | 56% FLAT | 22% UP
  Key check    : alpha must exceed half-spread to avoid training on noise
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from adaptive_labeler import (
    load_snapshots, filter_session, mid_price, label_fixed,
    fit_double_alpha, fit_double_alpha_from_files,
    SNAPSHOT_DTYPE,
)
from config import DATA_DIR, TICKER, TRAIN_DATES


TAIL_PERCENTS = [15, 18, 20, 22, 25, 28, 30]   # % in each tail to sweep


def compute_half_spread(path: str) -> float:
    """Compute mean half-spread from a file (alpha must exceed this)."""
    arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    arr = filter_session(arr)
    bp  = arr['bid_px'][:, 0].astype(np.float64) / 1e4
    ap  = arr['ask_px'][:, 0].astype(np.float64) / 1e4
    mid = (bp + ap) * 0.5
    spread = (ap - bp) / np.where(mid > 0, mid, 1)
    return float(np.median(spread) / 2)


def sweep_single_file(path: str):
    """Run alpha sweep on a single file."""
    print(f"\nFile: {os.path.basename(path)}")
    print("-" * 65)

    arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    arr = filter_session(arr)
    if len(arr) < 500:
        print("  Too few snapshots -- skip")
        return

    half_spread = compute_half_spread(path)
    print(f"  Snapshots (session): {len(arr):,}")
    print(f"  Mean half-spread   : {half_spread:.6f}  ({half_spread*100:.4f}%)")
    print(f"  (alpha must exceed {half_spread:.6f} to label profitable moves)\n")

    print(f"  {'Tail%':>5}  {'alpha_down':>10}  {'alpha_up':>9}  "
          f"{'asym':>6}  {'DOWN':>6}  {'FLAT':>6}  {'UP':>6}  {'OK?':>4}")
    print(f"  {'-'*65}")

    for tail in TAIL_PERCENTS:
        from adaptive_labeler import _compute_returns_for_file, _alpha_from_returns
        rets = _compute_returns_for_file(path)
        da = _alpha_from_returns(rets, tail / 100)
        labels = label_fixed(arr, da)
        valid  = labels != -1
        lv     = labels[valid]
        n      = valid.sum()
        if n == 0:
            continue

        ok = '✅' if da.alpha_down > half_spread and da.alpha_up > half_spread else '❌'
        print(f"  {tail:>5}  {da.alpha_down:>10.6f}  {da.alpha_up:>9.6f}  "
              f"{da.asymmetry:>6.1%}  {(lv==0).sum()/n:>6.1%}  "
              f"{(lv==1).sum()/n:>6.1%}  {(lv==2).sum()/n:>6.1%}  {ok:>4}")

    # Recommend closest to 22/56/22
    print(f"\n  → Recommended: tail=22% (target 22/56/22 distribution)")
    da22 = fit_double_alpha(path, target_tail=0.22)
    print(f"  → {da22}")
    if da22.asymmetry > 0.05:
        print(f"  ⚠  Asymmetry {da22.asymmetry:.1%} > 5% -- DoubleAlpha thresholds are essential")
    else:
        print(f"  ✓  Asymmetry {da22.asymmetry:.1%} < 5% -- symmetric alpha would also work")


def sweep_multi_file(paths: list):
    """Fit DoubleAlpha from all paths jointly and show stats per file."""
    print(f"\nMulti-file sweep: {len(paths)} files")
    print("=" * 65)

    n_workers = min(len(paths), os.cpu_count() or 1)
    for tail in [20, 22, 25]:
        from adaptive_labeler import fit_double_alpha_from_files
        da = fit_double_alpha_from_files(paths, target_tail=tail/100, n_workers=n_workers)
        print(f"\nTail={tail}%: {da}")
        # Show distribution per file
        for path in paths:
            arr = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
            arr = filter_session(arr)
            lbl = label_fixed(arr, da)
            v   = lbl[lbl != -1]
            if len(v) > 0:
                name = os.path.basename(path).replace('_dataset.bin', '')
                print(f"  {name}: DOWN={(v==0).mean():.1%} FLAT={(v==1).mean():.1%} "
                      f"UP={(v==2).mean():.1%} n={len(v):,}")

    # Save recommended alpha
    da_final = fit_double_alpha_from_files(paths, target_tail=0.22, n_workers=n_workers)
    print(f"\n✅ Recommended DoubleAlpha: {da_final}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Single file mode
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
        sweep_single_file(path)
    else:
        # Multi-file mode: use all training days
        paths = []
        for date in TRAIN_DATES:
            p = os.path.join(DATA_DIR, f'{TICKER}_{date}_dataset.bin')
            if os.path.exists(p):
                paths.append(p)
        if not paths:
            print(f"No training files found in {DATA_DIR}")
            print("Usage: python tune_alpha.py <path_to_dataset.bin>")
            sys.exit(1)
        print(f"Found {len(paths)} training files")
        # Show single-file sweep on first file for reference
        sweep_single_file(paths[0])
        # Then full multi-file fit
        sweep_multi_file(paths)
