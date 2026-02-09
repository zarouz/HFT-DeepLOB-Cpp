#!/usr/bin/env python3
"""
Regenerate AMD Jan 27 only.
"""

import subprocess
import os

date = "20260127"
ticker = "amd"

DAILY_DIR = "../data/converted/daily"
ENGINE_DIR = "../engine"

orders_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_orders.bin")
truth_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_truth.bin")
output_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")

print(f"Regenerating AMD {date}...")

result = subprocess.run(
    [os.path.join(ENGINE_DIR, "process_day_fixed"), 
     orders_file, truth_file, output_file],
    capture_output=True,
    timeout=120
)

if result.returncode == 0:
    size = os.path.getsize(output_file)
    records = size // 248
    print(f"✅ Success: {records:,} snapshots")
else:
    error = result.stderr.decode()
    print(f"Output: {error}")
    print(f"Return code: {result.returncode}")
