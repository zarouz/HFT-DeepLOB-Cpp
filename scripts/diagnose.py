"""
Two diagnostics:

1. ATR window recalibration
   Your snapshot rate is 210/sec (6.19M snaps over ~29,200 seconds = 8.1 hours).
   Wait -- that's wrong. 6M / 210 = 29,000 seconds = 8 hours.
   But a trading day is 6.5 hours = 23,400 seconds.
   210 snaps/sec is the AVERAGE including the slow open/close periods.
   Peak NVDA rate is 3,000-8,000/sec during active mid-day.
   
   For ATR we want a 1-second backward window.
   At 210/sec average: window_snapshots = 210
   But this is highly variable. Better: use TIME-BASED ATR window, 
   not snapshot-count-based.

2. Load time diagnosis
   5.45s for 6M snapshots via np.frombuffer is 15x slower than expected (~0.3s).
   Possible causes:
   a) File is on NFS/network mount
   b) numpy fallback to non-optimized path
   c) OS page cache cold (first read)
"""

import numpy as np
import time
import os

# Check if file is on a network mount
def check_mount(path):
    import subprocess
    result = subprocess.run(['df', '-T', path], capture_output=True, text=True)
    return result.stdout

# Benchmark raw read speed
def benchmark_read(filepath, n_times=3):
    times = []
    sizes = []
    for _ in range(n_times):
        t0 = time.perf_counter()
        with open(filepath, 'rb') as f:
            raw = f.read()
        times.append(time.perf_counter() - t0)
        sizes.append(len(raw))
    return sizes[0], times

# Benchmark np.frombuffer parse speed
def benchmark_parse(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    n_recs = len(raw) // 248
    dt = np.dtype([
        ('time',     '<f8'),
        ('bidPrice', '<u8', (10,)),
        ('bidSize',  '<i4', (10,)),
        ('askPrice', '<u8', (10,)),
        ('askSize',  '<i4', (10,)),
    ])
    
    t0 = time.perf_counter()
    recs = np.frombuffer(raw, dtype=dt)
    t_parse = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    N = len(recs)
    arr = np.empty((N, 41), dtype=np.float64)
    arr[:, 0]     = recs['time']
    arr[:, 1:11]  = recs['bidPrice'] * 1e-4
    arr[:, 11:21] = recs['bidSize'].astype(np.float64)
    arr[:, 21:31] = recs['askPrice'] * 1e-4
    arr[:, 31:41] = recs['askSize'].astype(np.float64)
    t_copy = time.perf_counter() - t0
    
    return t_parse, t_copy, arr

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else \
        "data/datasets/clean/daily/nvda_20260105_dataset.bin"
    
    print(f"File: {filepath}")
    print(f"Size: {os.path.getsize(filepath)/1e9:.2f} GB")
    
    # Mount info
    import subprocess
    r = subprocess.run(['df', '-T', filepath], capture_output=True, text=True)
    print(f"\nMount info:\n{r.stdout}")
    
    # Read benchmark (3 runs -- 2nd/3rd will use page cache)
    print("Read benchmark (3 runs):")
    sz, times = benchmark_read(filepath, 3)
    for i, t in enumerate(times):
        mb_s = (sz / 1e6) / t
        print(f"  Run {i+1}: {t:.2f}s  ({mb_s:.0f} MB/s)  "
              f"{'[cold]' if i==0 else '[cached]'}")
    
    # Parse benchmark
    print("\nParse benchmark (frombuffer + array construction):")
    t_parse, t_copy, arr = benchmark_parse(filepath)
    print(f"  frombuffer:  {t_parse:.3f}s")
    print(f"  array copy:  {t_copy:.3f}s")
    print(f"  total parse: {t_parse+t_copy:.3f}s")
    print(f"  records:     {len(arr):,}")
    
    # Snapshot rate analysis
    duration = arr[-1, 0] - arr[0, 0]
    avg_rate = len(arr) / duration
    print(f"\nSnapshot rate analysis:")
    print(f"  Duration:  {duration:.0f}s ({duration/3600:.2f}h)")
    print(f"  Avg rate:  {avg_rate:.0f} snaps/sec")
    
    # Rate by hour bucket
    print(f"\n  Rate by 30-min bucket:")
    t_start = arr[0, 0]
    bucket_s = 1800
    n_buckets = int(duration / bucket_s) + 1
    for b in range(n_buckets):
        t0_b = t_start + b * bucket_s
        t1_b = t0_b + bucket_s
        mask = (arr[:, 0] >= t0_b) & (arr[:, 0] < t1_b)
        n = mask.sum()
        if n > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(t0_b)
            rate = n / bucket_s
            print(f"    {dt.strftime('%H:%M')}: {rate:6.0f}/sec  ({n:,} snaps)")
    
    # Recommended ATR window
    print(f"\nATR window recommendation:")
    print(f"  For a 1-second backward window at avg rate {avg_rate:.0f}/sec:")
    print(f"    window_snapshots = {int(avg_rate)}")
    print(f"  For a time-based ATR (more robust), use compute_atr_time_based()")
    print(f"  The current window=5000 is {5000/avg_rate:.1f}s of data -- way too wide")