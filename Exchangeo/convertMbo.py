import databento as db
import struct

# Shared Protocol Constants
# etype: 1=Add, 2=Cancel, 3=Delete, 4=Trade
# direction: 1=Bid, -1=Ask

def convert_databento_to_binary(input_dbn, output_bin):
    # Load the DBN file
    stored_data = db.DBNStore.from_file(input_dbn)
    
    with open(output_bin, "wb") as f:
        for msg in stored_data:
            # Skip non-MBO messages if any
            if not hasattr(msg, 'action'):
                continue

            # 1. Map Actions to Engine Event Types
            if msg.action == 'A':
                etype = 1
            elif msg.action == 'C':
                etype = 2
            elif msg.action == 'M':
                # IMPORTANT: Modify overwrites the existing order in our C++ map
                etype = 1 
            elif msg.action in ['F', 'T']:
                etype = 4
            elif msg.action == 'D':
                etype = 3
            else:
                continue

            # 2. Map Side
            # Databento: 'B' for Bid, 'A' for Ask
            direction = 1 if msg.side == 'B' else -1

            # 3. Packing according to SharedProtocol.hpp
            # double time;         (8 bytes) - 'd'
            # int32_t eventType;   (4 bytes) - 'i'
            # uint64_t orderId;    (8 bytes) - 'Q'
            # int32_t size;        (4 bytes) - 'i'
            # int64_t price;       (8 bytes) - 'q'
            # int32_t direction;   (4 bytes) - 'i'
            # Total: 36 bytes
            
            # Note: msg.ts_event is in nanoseconds. We convert to seconds for the 'double'
            timestamp = msg.ts_event / 1e9
            
            binary_data = struct.pack(
                "diQiqi", 
                timestamp, 
                etype, 
                msg.order_id, 
                int(msg.size), 
                int(msg.price), 
                direction
            )
            f.write(binary_data)

if __name__ == "__main__":
    convert_databento_to_binary("XNAS-20260201-VK77TMM7WD/xnas-itch-20260114.mbo.dbn", "msft_orders.bin")
    print("Conversion Complete: msft_orders.bin created.")