import databento as db
import struct
import os

# --- CONFIGURATION ---
MBO_INPUT = "XNAS-20260201-VK77TMM7WD/xnas-itch-20260114.mbo.dbn"
MBP_INPUT = "MBO10/xnas-itch-20260114.mbp-10.dbn"

ORDER_OUTPUT = "msft_orders.bin"
TRUTH_OUTPUT = "msft_truth.bin"

TARGET_ID = 10888
PRICE_NULL = 9223372036854775807

def convert_mbo():
    print(f"--- GENERATING ORDERS BINARY (MBO) ---")
    if not os.path.exists(MBO_INPUT):
        print(f"ERROR: Could not find file {MBO_INPUT}")
        return

    print(f"Reading: {MBO_INPUT}")
    count = 0
    with open(ORDER_OUTPUT, "wb") as out_f:
        try:
            data = db.DBNStore.from_file(MBO_INPUT)
            for msg in data:
                if hasattr(msg, "instrument_id") and msg.instrument_id == TARGET_ID:
                    
                    # 1. Safe Price Calculation
                    raw_price = msg.price
                    price_1e4 = 0
                    if raw_price != PRICE_NULL:
                        price_1e4 = int(raw_price / 100000)

                    etype = 0
                    
                    # --- ACTION MAPPING (CRITICAL LOGIC) ---
                    if msg.action == 'A': 
                        # ADD: Must have valid price
                        if price_1e4 <= 0: continue
                        etype = 1   
                        
                    elif msg.action == 'T': 
                        # TRADE: Do NOT filter by price. Use OrderID to reduce size.
                        etype = 4   
                        
                    elif msg.action == 'C':
                        # CANCEL: Do NOT filter by price. Use OrderID to reduce size.
                        etype = 2   
                        
                    else:
                        # MODIFY (M), CLEAR (R): Force Delete.
                        # Do NOT filter by price.
                        etype = 3   

                    # 2. Pack Data (Little Endian '<')
                    direction = 1 if msg.side == 'B' else -1
                    ts = float(msg.ts_event) / 1e9 

                    packed = struct.pack('<diQiqi', ts, etype, msg.order_id, msg.size, price_1e4, direction)
                    out_f.write(packed)
                    count += 1
                    
                    if count % 200000 == 0: print(f"Encoded {count} orders...")
                    
        except Exception as e:
            print(f"Error reading DBN file: {e}")
    print(f"SUCCESS: Saved {count} orders to {ORDER_OUTPUT}\n")

def convert_mbp():
    print(f"--- GENERATING TRUTH BINARY (MBP-10) ---")
    if not os.path.exists(MBP_INPUT):
        print(f"ERROR: Could not find file {MBP_INPUT}")
        return

    print(f"Reading: {MBP_INPUT}")
    count = 0
    with open(TRUTH_OUTPUT, "wb") as out_f:
        try:
            data = db.DBNStore.from_file(MBP_INPUT)
            for msg in data:
                if hasattr(msg, "instrument_id") and msg.instrument_id == TARGET_ID:
                    if not hasattr(msg, "levels"): continue

                    # Scaling 1e9 -> 1e4
                    best_bid_px = int(msg.levels[0].bid_px / 100000)
                    best_bid_sz = int(msg.levels[0].bid_sz)
                    best_ask_px = int(msg.levels[0].ask_px / 100000)
                    best_ask_sz = int(msg.levels[0].ask_sz)
                    ts = float(msg.ts_event) / 1e9

                    if best_bid_px <= 0 or best_ask_px <= 0: continue

                    # Pack Data (Little Endian '<')
                    packed = struct.pack('<dQiqi', ts, best_bid_px, best_bid_sz, best_ask_px, best_ask_sz)
                    out_f.write(packed)
                    count += 1
                    
                    if count % 200000 == 0: print(f"Encoded {count} truth snapshots...")
        except Exception as e:
            print(f"Error reading DBN file: {e}")

    print(f"SUCCESS: Saved {count} snapshots to {TRUTH_OUTPUT}\n")

if __name__ == "__main__":
    convert_mbo()
    convert_mbp()