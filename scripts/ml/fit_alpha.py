"""
fit_alpha.py
============
Fit DoubleAlpha from all 10 training days jointly and pickle to models/.
Must be run once before train_nvda.py (train_nvda.py also auto-creates it).

Usage: python scripts/ml/fit_alpha.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import DATA_DIR, TICKER, TRAIN_DATES, ALPHA_PKL, MODEL_DIR, NUM_WORKERS
from adaptive_labeler import fit_double_alpha_from_files

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    paths = [
        os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
        for d in TRAIN_DATES
        if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
    ]

    if not paths:
        print(f"Error: No training files found in {DATA_DIR}")
        return

    print(f"Fitting DoubleAlpha from {len(paths)} training files ({NUM_WORKERS} workers)...")
    # On macOS, n_workers determines how many spawn processes start.
    da = fit_double_alpha_from_files(paths, target_tail=0.22, n_workers=NUM_WORKERS)
    print(f"  {da}")

    da.save(ALPHA_PKL)
    print(f"  Saved → {ALPHA_PKL}")

    if da.asymmetry > 0.05:
        print(f"  ⚠ Asymmetry {da.asymmetry:.1%} -- DoubleAlpha essential (see Preprocessing doc Sec 1.1)")
    else:
        print(f"  ✓ Asymmetry {da.asymmetry:.1%} -- within expected range")

if __name__ == '__main__':
    # This guard is required for multiprocessing on macOS
    main()