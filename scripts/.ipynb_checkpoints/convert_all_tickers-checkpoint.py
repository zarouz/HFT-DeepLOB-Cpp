#!/usr/bin/env python3
"""
Batch Convert All Tickers - Databento to Binary Format
=======================================================
Converts MBO and MBP-10 data for all 5 tickers across all trading days.

Output:
- data/converted/{ticker}_orders.bin (all days concatenated)
- data/converted/{ticker}_truth.bin (all days concatenated)
"""

import databento as db
import struct
import os
from pathlib import Path
import sys

# ============================================================================
# TICKER CONFIGURATION
# ============================================================================
TICKER_CONFIG = {
    "spy":  {"instrument_id": 15144},
    "nvda": {"instrument_id": 11667},
    "tsla": {"instrument_id": 16244},
    "pltr": {"instrument_id": 12716},
    "amd":  {"instrument_id": 773},
}

# All trading days (Jan 5-29, 2026) - excluding weekends
TRADING_DAYS = [
    "20260105", "20260106", "20260107", "20260108", "20260109",  # Week 1
    "20260112", "20260113", "20260114", "20260115", "20260116",  # Week 2
    "20260120", "20260121", "20260122", "20260123",              # Week 3 (Mon-Thu)
    "20260126", "20260127", "20260128", "20260129"               # Week 4
]

MBO_DIR = "data/raw/mbo"
MBP_DIR = "data/raw/mbo10"
OUTPUT_DIR = "data/converted"

PRICE_NULL = 9223372036854775807

# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def convert_mbo_day(dbn_path, instrument_id, output_handle):
    """
    Convert one day of MBO data and append to output file handle.
    Returns: number of records written
    """
    if not os.path.exists(dbn_path):
        print(f"  WARNING: File not found: {dbn_path}")
        return 0
    
    count = 0
    skipped_trades = 0
    skipped_invalid = 0
    
    try:
        data = db.DBNStore.from_file(dbn_path)
        
        for msg in data:
            # Filter by instrument
            if not hasattr(msg, "instrument_id") or msg.instrument_id != instrument_id:
                continue
            
            # Scale price from 1e9 to 1e4
            raw_price = msg.price
            price_1e4 = 0
            if raw_price != PRICE_NULL and raw_price > 0:
                price_1e4 = int(raw_price / 100000)
            
            etype = 0
            
            # Action mapping (critical for accuracy)
            if msg.action == 'A':
                if price_1e4 <= 0:
                    skipped_invalid += 1
                    continue
                etype = 1  # ADD
                
            elif msg.action == 'M':
                if price_1e4 <= 0:
                    skipped_invalid += 1
                    continue
                etype = 1  # MODIFY (C++ handles overwrite)
                
            elif msg.action == 'C':
                etype = 2  # CANCEL
                
            elif msg.action == 'F':
                etype = 4  # FILL
                
            elif msg.action == 'T':
                skipped_trades += 1
                continue  # Skip trade summaries (duplicates)
                
            elif msg.action == 'D':
                etype = 3  # DELETE
                
            elif msg.action == 'R':
                etype = 3  # RESET/CLEAR
                
            else:
                skipped_invalid += 1
                continue
            
            # Pack data: <diQiqi (36 bytes)
            direction = 1 if msg.side == 'B' else -1
            ts = float(msg.ts_event) / 1e9
            
            packed = struct.pack('<diQiqi',
                ts,            # double time (8)
                etype,         # int32_t eventType (4)
                msg.order_id,  # uint64_t orderId (8)
                msg.size,      # int32_t size (4)
                price_1e4,     # uint64_t price (8)
                direction      # int32_t direction (4)
            )
            output_handle.write(packed)
            count += 1
            
    except Exception as e:
        print(f"  ERROR processing {dbn_path}: {e}")
        import traceback
        traceback.print_exc()
        return count
    
    return count


def convert_mbp_day(dbn_path, instrument_id, output_handle):
    """
    Convert one day of MBP-10 data and append to output file handle.
    Returns: number of records written
    """
    if not os.path.exists(dbn_path):
        print(f"  WARNING: File not found: {dbn_path}")
        return 0
    
    count = 0
    skipped = 0
    
    try:
        data = db.DBNStore.from_file(dbn_path)
        
        for msg in data:
            # Filter by instrument
            if not hasattr(msg, "instrument_id") or msg.instrument_id != instrument_id:
                continue
            
            if not hasattr(msg, "levels") or len(msg.levels) == 0:
                continue
            
            # Scale prices from 1e9 to 1e4
            best_bid_px = int(msg.levels[0].bid_px / 100000)
            best_bid_sz = int(msg.levels[0].bid_sz)
            best_ask_px = int(msg.levels[0].ask_px / 100000)
            best_ask_sz = int(msg.levels[0].ask_sz)
            ts = float(msg.ts_event) / 1e9
            
            # Skip invalid/NULL prices
            if best_bid_px <= 0 or best_ask_px <= 0:
                skipped += 1
                continue
            
            # Pack data: <dQiQi (32 bytes)
            packed = struct.pack('<dQiQi',
                ts,            # double time (8)
                best_bid_px,   # uint64_t bidPrice (8)
                best_bid_sz,   # int32_t bidSize (4)
                best_ask_px,   # uint64_t askPrice (8)
                best_ask_sz    # int32_t askSize (4)
            )
            output_handle.write(packed)
            count += 1
            
    except Exception as e:
        print(f"  ERROR processing {dbn_path}: {e}")
        import traceback
        traceback.print_exc()
        return count
    
    return count


def convert_ticker(ticker, instrument_id):
    """
    Convert all days for one ticker.
    """
    print("\n" + "="*70)
    print(f"CONVERTING {ticker.upper()} (Instrument ID: {instrument_id})")
    print("="*70)
    
    orders_path = os.path.join(OUTPUT_DIR, f"{ticker}_orders.bin")
    truth_path = os.path.join(OUTPUT_DIR, f"{ticker}_truth.bin")
    
    total_orders = 0
    total_truth = 0
    
    # Open output files in append mode (concatenate all days)
    with open(orders_path, 'wb') as orders_file, \
         open(truth_path, 'wb') as truth_file:
        
        for date in TRADING_DAYS:
            mbo_file = os.path.join(MBO_DIR, f"xnas-itch-{date}.mbo.dbn")
            mbp_file = os.path.join(MBP_DIR, f"xnas-itch-{date}.mbp-10.dbn")
            
            print(f"\n  Processing {date}...")
            
            # Convert MBO
            print(f"    MBO: {mbo_file}")
            n_orders = convert_mbo_day(mbo_file, instrument_id, orders_file)
            total_orders += n_orders
            print(f"      → {n_orders:,} orders")
            
            # Convert MBP
            print(f"    MBP: {mbp_file}")
            n_truth = convert_mbp_day(mbp_file, instrument_id, truth_file)
            total_truth += n_truth
            print(f"      → {n_truth:,} truth snapshots")
    
    # Verify output
    orders_size = os.path.getsize(orders_path) / (1024 * 1024)
    truth_size = os.path.getsize(truth_path) / (1024 * 1024)
    
    print("\n" + "-"*70)
    print(f"COMPLETE: {ticker.upper()}")
    print(f"  Orders:  {total_orders:,} records ({orders_size:.1f} MB)")
    print(f"  Truth:   {total_truth:,} records ({truth_size:.1f} MB)")
    print(f"  Output:  {orders_path}")
    print(f"           {truth_path}")
    print("-"*70)


def verify_output():
    """Quick sanity check on generated files."""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    for ticker in TICKER_CONFIG.keys():
        orders_path = os.path.join(OUTPUT_DIR, f"{ticker}_orders.bin")
        truth_path = os.path.join(OUTPUT_DIR, f"{ticker}_truth.bin")
        
        if os.path.exists(orders_path) and os.path.exists(truth_path):
            orders_count = os.path.getsize(orders_path) // 36
            truth_count = os.path.getsize(truth_path) // 32
            
            print(f"\n{ticker.upper()}:")
            print(f"  Orders: {orders_count:,} records")
            print(f"  Truth:  {truth_count:,} records")
            
            # Read first timestamp from each
            with open(orders_path, 'rb') as f:
                data = f.read(36)
                if len(data) == 36:
                    ts = struct.unpack('<d', data[:8])[0]
                    print(f"  First order timestamp: {ts:.6f}")
            
            with open(truth_path, 'rb') as f:
                data = f.read(32)
                if len(data) == 32:
                    ts = struct.unpack('<d', data[:8])[0]
                    print(f"  First truth timestamp: {ts:.6f}")
        else:
            print(f"\n{ticker.upper()}: MISSING FILES")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "LATENCYLAB BATCH CONVERTER" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if input directories exist
    if not os.path.exists(MBO_DIR):
        print(f"ERROR: MBO directory not found: {MBO_DIR}")
        sys.exit(1)
    
    if not os.path.exists(MBP_DIR):
        print(f"ERROR: MBP directory not found: {MBP_DIR}")
        sys.exit(1)
    
    print(f"\nInput:")
    print(f"  MBO:  {MBO_DIR}")
    print(f"  MBP:  {MBP_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Days:   {len(TRADING_DAYS)} trading days")
    print(f"Tickers: {', '.join(TICKER_CONFIG.keys()).upper()}")
    
    # Convert all tickers
    for ticker, config in TICKER_CONFIG.items():
        convert_ticker(ticker, config["instrument_id"])
    
    # Verify
    verify_output()
    
    print("\n" + "="*70)
    print("ALL CONVERSIONS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run validation: python3 scripts/AuditBinaries.py")
    print("  2. Test C++ pipeline on one ticker")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()