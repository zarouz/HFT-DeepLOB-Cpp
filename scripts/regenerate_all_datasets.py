# scripts/regenerate_all_datasets.py -> stick to this

#!/usr/bin/env python3
"""
Regenerate all datasets with the FIXED processor.
"""

import subprocess
import os
from multiprocessing import Pool, cpu_count

TICKERS = ["nvda", "spy", "tsla", "pltr", "amd"]
DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116",
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

DAILY_DIR = "../data/converted/daily"
ENGINE_DIR = "../engine"

def process_single_day(args):
    ticker, date = args
    
    orders_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_orders.bin")
    truth_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_truth.bin")
    output_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
    
    if not os.path.exists(orders_file) or not os.path.exists(truth_file):
        return (ticker, date, 0, "SKIP")
    
    try:
        result = subprocess.run(
            [os.path.join(ENGINE_DIR, "process_day_fixed"), 
             orders_file, truth_file, output_file],
            capture_output=True,
            timeout=120
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            size = os.path.getsize(output_file)
            records = size // 248
            return (ticker, date, records, "OK")
        else:
            error = result.stderr.decode()[:50] if result.stderr else "Unknown"
            return (ticker, date, 0, f"FAIL: {error}")
            
    except Exception as e:
        return (ticker, date, 0, f"ERROR: {str(e)[:50]}")

if __name__ == "__main__":
    print("="*70)
    print("REGENERATING ALL DATASETS WITH FIXED PROCESSOR")
    print("="*70)
    print()
    
    work_items = []
    for ticker in TICKERS:
        for date in DAYS:
            work_items.append((ticker, date))
    
    n_cores = cpu_count()
    print(f"Processing {len(work_items)} days using {n_cores} cores...")
    print()
    
    with Pool(processes=n_cores) as pool:
        results = list(pool.imap_unordered(process_single_day, work_items))
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for ticker in TICKERS:
        print(f"\n{ticker.upper()}:")
        total = 0
        for t, date, count, status in sorted(results):
            if t == ticker and count > 0:
                print(f"  {date}: {count:,} snapshots")
                total += count
        if total > 0:
            print(f"  Total: {total:,} snapshots")
    
    success = sum(1 for _, _, c, _ in results if c > 0)
    print(f"\n✅ Successfully processed: {success}/{len(work_items)} days")
    print("="*70)
