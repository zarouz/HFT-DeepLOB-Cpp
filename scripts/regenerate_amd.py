#!/usr/bin/env python3
"""
Regenerate AMD datasets only.
"""

import subprocess
import os
from multiprocessing import Pool

DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116",
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

DAILY_DIR = "../data/converted/daily"
ENGINE_DIR = "../engine"

def process_single_day(date):
    ticker = "amd"
    
    orders_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_orders.bin")
    truth_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_truth.bin")
    output_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
    
    if not os.path.exists(orders_file) or not os.path.exists(truth_file):
        return (date, 0, "SKIP - no input files")
    
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
            return (date, records, "OK")
        else:
            error = result.stderr.decode()[:50] if result.stderr else "Unknown"
            return (date, 0, f"FAIL: {error}")
            
    except Exception as e:
        return (date, 0, f"ERROR: {str(e)[:50]}")

if __name__ == "__main__":
    print("="*60)
    print("REGENERATING AMD DATASETS")
    print("="*60)
    print()
    
    with Pool(processes=18) as pool:
        results = list(pool.map(process_single_day, DAYS))
    
    print("\nAMD Results:")
    total = 0
    for date, count, status in sorted(results):
        if count > 0:
            print(f"  {date}: {count:,} snapshots")
            total += count
        elif "SKIP" not in status:
            print(f"  {date}: {status}")
    
    print(f"\nTotal: {total:,} snapshots")
    success = sum(1 for _, c, _ in results if c > 0)
    print(f"✅ Successfully processed: {success}/18 days")
    print("="*60)
