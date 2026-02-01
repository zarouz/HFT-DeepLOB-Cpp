#!/usr/bin/env python3
"""
Databento MBO/MBP-10 to Binary Converter
=========================================
Converts Databento DBN files to binary format for the C++ HFT engine.

Output files:
- msft_orders.bin: MBO events (36 bytes each)
- msft_truth.bin: MBP-10 BBO snapshots (32 bytes each)
"""

import databento as db
import struct
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
MBO_INPUT = "XNAS-20260201-VK77TMM7WD/xnas-itch-20260114.mbo.dbn"
MBP_INPUT = "MBO10/xnas-itch-20260114.mbp-10.dbn"

ORDER_OUTPUT = "msft_orders.bin"
TRUTH_OUTPUT = "msft_truth.bin"

TARGET_ID = 10888  # MSFT instrument ID
PRICE_NULL = 9223372036854775807  # Databento NULL sentinel

# ============================================================================
# ACTION MAPPING (Critical for >98% Accuracy)
# ============================================================================
# Databento ITCH Actions:
#   'A' = Add Order        -> etype 1 (ADD)
#   'M' = Modify Order     -> etype 1 (ADD with overwrite) - C++ detects existing ID
#   'C' = Cancel (partial) -> etype 2 (CANCEL - reduce size by delta)
#   'F' = Fill (execution) -> etype 4 (TRADE - reduce size by fill amount)
#   'T' = Trade Summary    -> SKIP (duplicate of 'F', causes double-counting)
#   'D' = Delete Order     -> etype 3 (DELETE - remove entire order)
#   'R' = Clear/Reset      -> etype 3 (DELETE)
# ============================================================================

def convert_mbo():
    """Convert MBO (Market-By-Order) data to binary format."""
    print("=" * 50)
    print("CONVERTING MBO DATA")
    print("=" * 50)
    
    if not os.path.exists(MBO_INPUT):
        print(f"ERROR: Could not find file {MBO_INPUT}")
        print(f"Please ensure the Databento data file exists.")
        return False

    print(f"Input:  {MBO_INPUT}")
    print(f"Output: {ORDER_OUTPUT}")
    print()
    
    count = 0
    skipped_trades = 0
    skipped_invalid = 0
    
    with open(ORDER_OUTPUT, "wb") as out_f:
        try:
            data = db.DBNStore.from_file(MBO_INPUT)
            
            for msg in data:
                # Filter by instrument
                if not hasattr(msg, "instrument_id") or msg.instrument_id != TARGET_ID:
                    continue
                    
                # Scale price from 1e9 to 1e4
                raw_price = msg.price
                price_1e4 = 0
                if raw_price != PRICE_NULL and raw_price > 0:
                    price_1e4 = int(raw_price / 100000)

                etype = 0
                
                # --- ACTION MAPPING ---
                if msg.action == 'A':
                    # ADD: New order - must have valid price
                    if price_1e4 <= 0:
                        skipped_invalid += 1
                        continue
                    etype = 1
                    
                elif msg.action == 'M':
                    # MODIFY: Order update - use etype 1, C++ handles overwrite
                    if price_1e4 <= 0:
                        skipped_invalid += 1
                        continue
                    etype = 1
                    
                elif msg.action == 'C':
                    # CANCEL: Partial cancellation
                    etype = 2
                    
                elif msg.action == 'F':
                    # FILL: Execution occurred
                    etype = 4
                    
                elif msg.action == 'T':
                    # TRADE SUMMARY: Skip to avoid double-counting
                    skipped_trades += 1
                    continue
                    
                elif msg.action == 'D':
                    # DELETE: Full order removal
                    etype = 3
                    
                elif msg.action == 'R':
                    # CLEAR/RESET: Treat as delete
                    etype = 3
                    
                else:
                    skipped_invalid += 1
                    continue

                # Pack data (Little Endian, matches SharedProtocol.hpp)
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
                out_f.write(packed)
                count += 1
                
                if count % 500000 == 0:
                    print(f"  Processed: {count:,} orders...")
                    
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    file_size = os.path.getsize(ORDER_OUTPUT) / (1024 * 1024)
    print()
    print(f"SUCCESS: {count:,} orders written")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Skipped trades (T): {skipped_trades:,}")
    print(f"  Skipped invalid: {skipped_invalid:,}")
    print()
    return True


def convert_mbp():
    """Convert MBP-10 (Market-By-Price) data to truth binary format."""
    print("=" * 50)
    print("CONVERTING MBP-10 TRUTH DATA")
    print("=" * 50)
    
    if not os.path.exists(MBP_INPUT):
        print(f"ERROR: Could not find file {MBP_INPUT}")
        print(f"Please ensure the Databento data file exists.")
        return False

    print(f"Input:  {MBP_INPUT}")
    print(f"Output: {TRUTH_OUTPUT}")
    print()
    
    count = 0
    skipped = 0
    
    with open(TRUTH_OUTPUT, "wb") as out_f:
        try:
            data = db.DBNStore.from_file(MBP_INPUT)
            
            for msg in data:
                # Filter by instrument
                if not hasattr(msg, "instrument_id") or msg.instrument_id != TARGET_ID:
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

                # Pack data (Little Endian, matches SharedProtocol.hpp)
                packed = struct.pack('<dQiQi',
                    ts,            # double time (8)
                    best_bid_px,   # uint64_t bidPrice (8)
                    best_bid_sz,   # int32_t bidSize (4)
                    best_ask_px,   # uint64_t askPrice (8)
                    best_ask_sz    # int32_t askSize (4)
                )
                out_f.write(packed)
                count += 1
                
                if count % 200000 == 0:
                    print(f"  Processed: {count:,} snapshots...")
                    
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    file_size = os.path.getsize(TRUTH_OUTPUT) / (1024 * 1024)
    print()
    print(f"SUCCESS: {count:,} truth snapshots written")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Skipped invalid: {skipped:,}")
    print()
    return True


def verify_binary_files():
    """Quick verification of generated binary files."""
    print("=" * 50)
    print("VERIFYING BINARY FILES")
    print("=" * 50)
    
    # Check orders file
    if os.path.exists(ORDER_OUTPUT):
        size = os.path.getsize(ORDER_OUTPUT)
        records = size // 36
        print(f"{ORDER_OUTPUT}: {size:,} bytes ({records:,} records)")
        
        # Read first record
        with open(ORDER_OUTPUT, 'rb') as f:
            data = f.read(36)
            if len(data) == 36:
                ts, etype, oid, sz, px, dir = struct.unpack('<diQiqi', data)
                print(f"  First record: time={ts:.6f}, type={etype}, price={px}")
    else:
        print(f"{ORDER_OUTPUT}: NOT FOUND")
    
    # Check truth file
    if os.path.exists(TRUTH_OUTPUT):
        size = os.path.getsize(TRUTH_OUTPUT)
        records = size // 32
        print(f"{TRUTH_OUTPUT}: {size:,} bytes ({records:,} records)")
        
        # Read first record
        with open(TRUTH_OUTPUT, 'rb') as f:
            data = f.read(32)
            if len(data) == 32:
                ts, bid_px, bid_sz, ask_px, ask_sz = struct.unpack('<dQiQi', data)
                print(f"  First record: time={ts:.6f}, bid={bid_px}, ask={ask_px}")
    else:
        print(f"{TRUTH_OUTPUT}: NOT FOUND")
    
    print()


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  DATABENTO TO BINARY CONVERTER                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    
    success = True
    success = convert_mbo() and success
    success = convert_mbp() and success
    
    if success:
        verify_binary_files()
        print("All conversions completed successfully!")
    else:
        print("Some conversions failed. Check errors above.")
        sys.exit(1)