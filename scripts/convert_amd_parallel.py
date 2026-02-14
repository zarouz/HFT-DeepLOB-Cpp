#!/usr/bin/env python3
"""
Convert AMD data in parallel (one process per day).
"""

import databento as db
import struct
import os
from multiprocessing import Pool, cpu_count

TICKER = "amd"
INSTRUMENT_ID = 773

DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",
    "20260112", "20260113", "20260114", "20260115", "20260116",
    "20260120", "20260121", "20260122", "20260123",
    "20260126", "20260127", "20260128", "20260129"
]

MBO_DIR = "../data/mbo/dbn"
MBP_DIR = "../data/mbo10/dbn"
OUTPUT_DIR = "../data/converted/daily"
PRICE_NULL = 9223372036854775807

def convert_day(date):
    """Convert one day of AMD data."""
    
    # MBO
    mbo_file = os.path.join(MBO_DIR, f"xnas-itch-{date}.mbo.dbn")
    orders_out = os.path.join(OUTPUT_DIR, f"{TICKER}_{date}_orders.bin")
    
    orders_count = 0
    if os.path.exists(mbo_file):
        with open(orders_out, 'wb') as out_f:
            data = db.DBNStore.from_file(mbo_file)
            for msg in data:
                if not hasattr(msg, "instrument_id") or msg.instrument_id != INSTRUMENT_ID:
                    continue
                
                raw_price = msg.price
                price_1e4 = 0 if raw_price == PRICE_NULL or raw_price <= 0 else int(raw_price / 100000)
                
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
                elif msg.action in ['D', 'R']:
                    etype = 3
                else:
                    continue
                
                direction = 1 if msg.side == 'B' else -1
                ts = float(msg.ts_event) / 1e9
                
                out_f.write(struct.pack('<diQiqi',
                    ts, etype, msg.order_id, msg.size, price_1e4, direction))
                orders_count += 1
    
    # MBP
    mbp_file = os.path.join(MBP_DIR, f"xnas-itch-{date}.mbp-10.dbn")
    truth_out = os.path.join(OUTPUT_DIR, f"{TICKER}_{date}_truth.bin")
    
    truth_count = 0
    if os.path.exists(mbp_file):
        with open(truth_out, 'wb') as out_f:
            data = db.DBNStore.from_file(mbp_file)
            for msg in data:
                if not hasattr(msg, "instrument_id") or msg.instrument_id != INSTRUMENT_ID:
                    continue
                if not hasattr(msg, "levels") or len(msg.levels) == 0:
                    continue
                
                bid_raw = msg.levels[0].bid_px
                ask_raw = msg.levels[0].ask_px
                
                if bid_raw == PRICE_NULL or bid_raw <= 0 or ask_raw == PRICE_NULL or ask_raw <= 0:
                    continue
                
                bid_px = int(bid_raw / 100000)
                ask_px = int(ask_raw / 100000)
                
                if bid_px <= 0 or ask_px <= 0 or bid_px >= ask_px:
                    continue
                
                ts = float(msg.ts_event) / 1e9
                
                out_f.write(struct.pack('<dQiQi',
                    ts, bid_px, int(msg.levels[0].bid_sz), 
                    ask_px, int(msg.levels[0].ask_sz)))
                truth_count += 1
    
    return (date, orders_count, truth_count)

if __name__ == "__main__":
    print("="*60)
    print("PARALLEL AMD CONVERSION (18 days)")
    print("="*60)
    
    n_cores = min(cpu_count(), 18)
    print(f"Using {n_cores} cores\n")
    
    with Pool(processes=n_cores) as pool:
        results = pool.map(convert_day, DAYS)
    
    print("\nResults:")
    total_orders = 0
    total_truth = 0
    for date, orders, truth in sorted(results):
        print(f"  {date}: {orders:,} orders, {truth:,} truth")
        total_orders += orders
        total_truth += truth
    
    print(f"\nTotal: {total_orders:,} orders, {total_truth:,} truth")
    print("="*60)
