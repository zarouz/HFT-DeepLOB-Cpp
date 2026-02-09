#!/usr/bin/env python3
"""
Parallel processing of all days - uses all CPU cores.
i9-14900K: 24 cores = 24x speedup!
"""

import databento as db
import struct
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

TICKERS = {
    "nvda": 11667,
    "spy":  15144,
    "tsla": 16244,
    "pltr": 12716,
    "amd":  773,
}

TRADING_DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116",
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

MBO_DIR = "../data/raw/mbo"
MBP_DIR = "../data/raw/mbo10"
OUTPUT_DIR = "../data/converted/daily"

PRICE_NULL = 9223372036854775807

def convert_day_mbo(ticker, inst_id, date):
    """Convert one day of MBO data."""
    mbo_file = os.path.join(MBO_DIR, f"xnas-itch-{date}.mbo.dbn")
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_{date}_orders.bin")
    
    if not os.path.exists(mbo_file):
        return (ticker, date, 'MBO', 0)
    
    count = 0
    
    try:
        with open(output_file, 'wb') as out_f:
            data = db.DBNStore.from_file(mbo_file)
            
            for msg in data:
                if not hasattr(msg, "instrument_id") or msg.instrument_id != inst_id:
                    continue
                
                raw_price = msg.price
                price_1e4 = 0
                if raw_price != PRICE_NULL and raw_price > 0:
                    price_1e4 = int(raw_price / 100000)
                
                etype = 0
                
                if msg.action == 'A':
                    if price_1e4 <= 0: continue
                    etype = 1
                elif msg.action == 'M':
                    if price_1e4 <= 0: continue
                    etype = 1
                elif msg.action == 'C':
                    etype = 2
                elif msg.action == 'F':
                    etype = 4
                elif msg.action == 'T':
                    continue
                elif msg.action == 'D':
                    etype = 3
                elif msg.action == 'R':
                    etype = 3
                else:
                    continue
                
                direction = 1 if msg.side == 'B' else -1
                ts = float(msg.ts_event) / 1e9
                
                packed = struct.pack('<diQiqi',
                    ts, etype, msg.order_id, msg.size, price_1e4, direction
                )
                out_f.write(packed)
                count += 1
    except Exception as e:
        print(f"ERROR: {ticker} {date} MBO: {e}")
        return (ticker, date, 'MBO', -1)
    
    return (ticker, date, 'MBO', count)

def convert_day_mbp(ticker, inst_id, date):
    """Convert one day of MBP-10 data."""
    mbp_file = os.path.join(MBP_DIR, f"xnas-itch-{date}.mbp-10.dbn")
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_{date}_truth.bin")
    
    if not os.path.exists(mbp_file):
        return (ticker, date, 'MBP', 0)
    
    count = 0
    
    try:
        with open(output_file, 'wb') as out_f:
            data = db.DBNStore.from_file(mbp_file)
            
            for msg in data:
                if not hasattr(msg, "instrument_id") or msg.instrument_id != inst_id:
                    continue
                
                if not hasattr(msg, "levels") or len(msg.levels) == 0:
                    continue
                
                bid_px_raw = msg.levels[0].bid_px
                ask_px_raw = msg.levels[0].ask_px
                
                if bid_px_raw == PRICE_NULL or bid_px_raw <= 0:
                    continue
                if ask_px_raw == PRICE_NULL or ask_px_raw <= 0:
                    continue
                
                best_bid_px = int(bid_px_raw / 100000)
                best_bid_sz = int(msg.levels[0].bid_sz)
                best_ask_px = int(ask_px_raw / 100000)
                best_ask_sz = int(msg.levels[0].ask_sz)
                ts = float(msg.ts_event) / 1e9
                
                if best_bid_px <= 0 or best_ask_px <= 0:
                    continue
                if best_bid_px >= best_ask_px:
                    continue
                
                packed = struct.pack('<dQiQi',
                    ts, best_bid_px, best_bid_sz, best_ask_px, best_ask_sz
                )
                out_f.write(packed)
                count += 1
    except Exception as e:
        print(f"ERROR: {ticker} {date} MBP: {e}")
        return (ticker, date, 'MBP', -1)
    
    return (ticker, date, 'MBP', count)

def process_day(args):
    """Process one day for one ticker (both MBO and MBP)."""
    ticker, inst_id, date = args
    
    mbo_result = convert_day_mbo(ticker, inst_id, date)
    mbp_result = convert_day_mbp(ticker, inst_id, date)
    
    return (mbo_result, mbp_result)

if __name__ == "__main__":
    print("="*70)
    print("PARALLEL DAILY PROCESSING - ALL TICKERS")
    print("="*70)
    
    # Get CPU count
    n_cores = cpu_count()
    print(f"CPU cores available: {n_cores}")
    print(f"Using: {n_cores} parallel workers\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create work items (ticker, inst_id, date) for each day
    work_items = []
    for ticker, inst_id in TICKERS.items():
        for date in TRADING_DAYS:
            work_items.append((ticker, inst_id, date))
    
    print(f"Total jobs: {len(work_items)} (5 tickers × 18 days)")
    print(f"Processing...\n")
    
    # Process in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_day, work_items)
    
    # Summarize results
    print("\n" + "="*70)
    print("RESULTS BY TICKER")
    print("="*70)
    
    for ticker in TICKERS.keys():
        total_orders = 0
        total_truth = 0
        
        print(f"\n{ticker.upper()}:")
        for mbo_res, mbp_res in results:
            if mbo_res[0] == ticker:
                date = mbo_res[1]
                mbo_count = mbo_res[3]
                mbp_count = mbp_res[3]
                
                if mbo_count > 0 and mbp_count > 0:
                    print(f"  {date}: {mbo_count:,} orders, {mbp_count:,} truth")
                    total_orders += mbo_count
                    total_truth += mbp_count
        
        print(f"  Total: {total_orders:,} orders, {total_truth:,} truth snapshots")
    
    print("\n" + "="*70)
    print("✅ ALL DAYS PROCESSED IN PARALLEL")
    print("="*70)
