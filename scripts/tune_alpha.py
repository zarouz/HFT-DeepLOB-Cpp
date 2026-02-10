"""
tune_alpha.py
=============
Find the optimal alpha threshold for your NVDA data BEFORE training.

Run this first. A bad alpha makes your entire training pipeline garbage.

What it does:
  Tries alpha values from 0.00005 to 0.002
  Reports class distribution for each
  Recommends the alpha closest to balanced 33/33/33% split

Why this matters:
  If STATIONARY is 85% of your data, a model that always predicts
  STATIONARY gets 85% accuracy but is completely useless.
  You want roughly equal class proportions so the model actually learns
  to distinguish price movements.

Usage:
  python scripts/tune_alpha.py data/datasets/clean/daily/nvda_20260105_dataset.bin
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_label_builder import load_snapshots, build_time_based_labels, HORIZON_S

ALPHAS_TO_TEST = [
    0.00005, 0.0001, 0.00015, 0.0002, 0.00025,
    0.0003, 0.0004, 0.0005, 0.0007, 0.001, 0.002
]


def evaluate_alpha(snapshots: np.ndarray, alpha: float) -> dict:
    labels, valid = build_time_based_labels(
        snapshots, horizon_s=HORIZON_S, alpha=alpha
    )
    valid_labels = labels[valid]
    total = len(valid_labels)
    if total == 0:
        return None

    counts = {0: (valid_labels == 0).sum(),
              1: (valid_labels == 1).sum(),
              2: (valid_labels == 2).sum()}
    pcts = {k: 100.0 * v / total for k, v in counts.items()}

    # Imbalance score: max deviation from 33.3% (lower is better)
    imbalance = max(abs(pcts[k] - 33.33) for k in [0, 1, 2])

    return {
        'alpha':     alpha,
        'total':     total,
        'down_pct':  pcts[0],
        'flat_pct':  pcts[1],
        'up_pct':    pcts[2],
        'imbalance': imbalance,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tune_alpha.py path/to/nvda_YYYYMMDD_dataset.bin")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\nLoading: {filepath}")
    snapshots = load_snapshots(filepath)
    print(f"Loaded {len(snapshots):,} snapshots\n")

    print(f"{'Alpha':>10}  {'DOWN%':>7}  {'FLAT%':>7}  {'UP%':>7}  {'Imbalance':>10}")
    print("-" * 55)

    results = []
    for alpha in ALPHAS_TO_TEST:
        r = evaluate_alpha(snapshots, alpha)
        if r is None:
            continue
        results.append(r)
        flag = " <-- GOOD" if r['imbalance'] < 10 else ""
        flag = " <-- BEST" if r['imbalance'] == min(x['imbalance'] for x in results) and len(results) > 1 else flag
        print(f"  {alpha:8.5f}  {r['down_pct']:7.1f}  {r['flat_pct']:7.1f}  {r['up_pct']:7.1f}  {r['imbalance']:10.2f}{flag}")

    best = min(results, key=lambda x: x['imbalance'])
    print(f"\n{'='*55}")
    print(f"RECOMMENDATION: alpha = {best['alpha']}")
    print(f"  DOWN: {best['down_pct']:.1f}%  FLAT: {best['flat_pct']:.1f}%  UP: {best['up_pct']:.1f}%")
    print(f"\nUpdate ALPHA in train_nvda.py before running training.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()