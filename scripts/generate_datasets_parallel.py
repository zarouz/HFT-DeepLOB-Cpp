# scripts/generate_datasets_parallel.py
#!/usr/bin/env python3
"""
Parallel C++ dataset generation using Python multiprocessing.
No GNU parallel needed!
"""

import subprocess
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

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
    """Process one day through C++ engine."""
    ticker, date = args
    
    orders_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_orders.bin")
    truth_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_truth.bin")
    output_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
    
    # Check if input files exist
    if not os.path.exists(orders_file) or not os.path.exists(truth_file):
        return (ticker, date, 0, "SKIP - files not found")
    
    # Run C++ processor
    try:
        result = subprocess.run(
            [os.path.join(ENGINE_DIR, "process_day"), 
             orders_file, truth_file, output_file],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            size = os.path.getsize(output_file)
            records = size // 248
            return (ticker, date, records, "OK")
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            return (ticker, date, 0, f"FAILED - {error_msg[:50]}")
            
    except subprocess.TimeoutExpired:
        return (ticker, date, 0, "TIMEOUT")
    except Exception as e:
        return (ticker, date, 0, f"ERROR - {str(e)[:50]}")

if __name__ == "__main__":
    print("="*70)
    print("PARALLEL C++ DATASET GENERATION")
    print("="*70)
    
    # Build C++ processor first
    print("\nBuilding C++ processor...")
    build_result = subprocess.run(
        ["g++", "-std=c++20", "-O3", "ProcessDay.cpp", "-o", "process_day"],
        cwd=ENGINE_DIR,
        capture_output=True
    )
    
    if build_result.returncode != 0:
        print("ERROR: Failed to build C++ processor")
        print(build_result.stderr.decode())
        exit(1)
    
    print("✅ C++ processor built\n")
    
    # Create work items
    work_items = []
    for ticker in TICKERS:
        for date in DAYS:
            work_items.append((ticker, date))
    
    n_cores = cpu_count()
    print(f"CPU cores: {n_cores}")
    print(f"Total jobs: {len(work_items)} (5 tickers × 18 days)")
    print(f"Processing...\n")
    
    # Process in parallel
    with Pool(processes=n_cores) as pool:
        results = list(pool.imap_unordered(process_single_day, work_items))
    
    # Summarize
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for ticker in TICKERS:
        print(f"\n{ticker.upper()}:")
        total = 0
        for t, date, count, status in sorted(results):
            if t == ticker:
                if count > 0:
                    print(f"  {date}: {count:,} snapshots - {status}")
                    total += count
                elif "SKIP" not in status:
                    print(f"  {date}: {status}")
        
        if total > 0:
            print(f"  Total: {total:,} snapshots")
    
    # Count successes
    success_count = sum(1 for _, _, count, _ in results if count > 0)
    print(f"\n✅ Successfully processed: {success_count}/{len(work_items)} days")
    print("="*70)
