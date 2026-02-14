#!/usr/bin/env python3
import os

TICKERS = ["nvda", "spy", "tsla", "pltr", "amd"]

TRAIN_DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116"
]

TEST_DAYS = [
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

DAILY_DIR = "../data/converted/daily"
OUTPUT_DIR = "../data/datasets/clean"

def concatenate_ticker(ticker):
    print(f"\n{ticker.upper()}:")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_file = os.path.join(OUTPUT_DIR, f"{ticker}_train.bin")
    test_file = os.path.join(OUTPUT_DIR, f"{ticker}_test.bin")
    
    train_count = 0
    with open(train_file, 'wb') as out_f:
        for date in TRAIN_DAYS:
            day_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
            if os.path.exists(day_file):
                with open(day_file, 'rb') as in_f:
                    data = in_f.read()
                    out_f.write(data)
                    train_count += len(data) // 248
    
    test_count = 0
    with open(test_file, 'wb') as out_f:
        for date in TEST_DAYS:
            day_file = os.path.join(DAILY_DIR, f"{ticker}_{date}_dataset.bin")
            if os.path.exists(day_file):
                with open(day_file, 'rb') as in_f:
                    data = in_f.read()
                    out_f.write(data)
                    test_count += len(data) // 248
    
    print(f"  Train: {train_count:,} snapshots")
    print(f"  Test:  {test_count:,} snapshots")

if __name__ == "__main__":
    print("="*60)
    print("CONCATENATING DATASETS")
    print("="*60)
    
    for ticker in TICKERS:
        concatenate_ticker(ticker)
    
    print("\n✅ Done!")
