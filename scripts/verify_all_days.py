#!/usr/bin/env python3
"""
Quick check that all days processed correctly.
"""

import os

TICKERS = ["nvda", "spy", "tsla", "pltr", "amd"]
DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116",
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

DAILY_DIR = "../data/converted/daily"

print("="*60)
print("VERIFICATION: Checking all daily files")
print("="*60)

for ticker in TICKERS:
    print(f"\n{ticker.upper()}:")
    
    for date in DAYS:
        orders = os.path.join(DAILY_DIR, f"{ticker}_{date}_orders.bin")
        truth = os.path.join(DAILY_DIR, f"{ticker}_{date}_truth.bin")
        dataset = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
        
        o_exists = "✅" if os.path.exists(orders) else "❌"
        t_exists = "✅" if os.path.exists(truth) else "❌"
        d_exists = "✅" if os.path.exists(dataset) else "❌"
        
        if os.path.exists(dataset):
            count = os.path.getsize(dataset) // 248
            print(f"  {date}: Orders {o_exists} Truth {t_exists} Dataset {d_exists} ({count:,} snapshots)")
        else:
            print(f"  {date}: Orders {o_exists} Truth {t_exists} Dataset {d_exists}")

print("\n" + "="*60)
