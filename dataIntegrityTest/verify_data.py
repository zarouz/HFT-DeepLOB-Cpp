import pandas as pd
import sys

# CONFIGURATION
# Adjust filenames if needed
MSG_FILE = "/Users/karthikyadav/Desktop/Startup/HFT/TradingSystem/Exchange/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv"
BOOK_FILE = "/Users/karthikyadav/Desktop/Startup/HFT/TradingSystem/Exchange/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

def verify_integrity():
    print(f"Loading {MSG_FILE}...")
    # Load raw CSVs (No headers in LOBSTER files usually)
    messages = pd.read_csv(MSG_FILE, header=None, names=["Time", "Type", "ID", "Size", "Price", "Dir"])
    book = pd.read_csv(BOOK_FILE, header=None)
    
    # LOBSTER Orderbook File: 
    # Col 0: Ask Price 1, Col 1: Ask Size 1, Col 2: Bid Price 1, Col 3: Bid Size 1
    
    # SIMPLE ORDERBOOK ENGINE (Dictionary based)
    # Price -> Size
    bids = {} 
    asks = {}
    
    # INITIAL SNAPSHOT (Bootstrap from Line 0 of Orderbook)
    print("Bootstrapping from Snapshot (Line 0)...")
    first_row = book.iloc[0]
    
    # Load top 10 levels from first row to prime the book
    # Format: AskP1, AskS1, BidP1, BidS1, AskP2...
    for i in range(10):
        offset = i * 4
        ask_p = first_row[offset]
        ask_s = first_row[offset+1]
        bid_p = first_row[offset+2]
        bid_s = first_row[offset+3]
        
        if ask_s > 0: asks[ask_p] = ask_s
        if bid_s > 0: bids[bid_p] = bid_s

    # PERFORMANCE METRICS
    matches = 0
    errors = 0
    
    print("Starting Offline Replay...")
    
    # Iterate through messages starting from INDEX 1 (Since Line 0 was snapshot)
    # We compare against Book Row 1
    
    total_rows = len(messages)
    
    for i in range(1, total_rows):
        msg = messages.iloc[i]
        
        # --- 1. ENGINE LOGIC (Same as C++) ---
        event_type = msg['Type']
        price = msg['Price']
        size = msg['Size']
        direction = msg['Dir'] # 1=Buy(Bid), -1=Sell(Ask)
        
        # Ignore Hidden Executions (Type 5) and Halts (Type 7)
        if event_type not in [5, 7]:
            is_bid = (direction == 1)
            delta = 0
            
            if event_type == 1: # ADD
                delta = size
            elif event_type in [2, 3, 4]: # CANCEL, DELETE, EXECUTE
                delta = -size
            
            # Update Book
            if is_bid:
                curr = bids.get(price, 0)
                bids[price] = curr + delta
                if bids[price] <= 0: del bids[price]
            else:
                curr = asks.get(price, 0)
                asks[price] = curr + delta
                if asks[price] <= 0: del asks[price]

        # --- 2. VERIFICATION (Against Book Row i) ---
        expected_row = book.iloc[i]
        expected_bid_p = expected_row[2] # Bid Price 1
        expected_ask_p = expected_row[0] # Ask Price 1
        
        # Get Engine Best Bid/Ask
        # Default to 0 if empty
        engine_bid = max(bids.keys()) if bids else 0
        engine_ask = min(asks.keys()) if asks else 0
        
        if engine_bid == expected_bid_p and engine_ask == expected_ask_p:
            matches += 1
        else:
            errors += 1
            if errors < 5: # Print first 5 errors only
                print(f"Mismatch at Row {i}:")
                print(f"  Engine: Bid {engine_bid} / Ask {engine_ask}")
                print(f"  Truth:  Bid {expected_bid_p} / Ask {expected_ask_p}")
                print(f"  Event: Type {event_type} Size {size} Price {price}")

        if i % 50000 == 0:
            print(f"Processed {i}/{total_rows} | Accuracy: {matches/(matches+errors)*100:.2f}%")

    print(f"\nFINAL RESULTS:")
    print(f"Matches: {matches}")
    print(f"Errors:  {errors}")
    print(f"Accuracy: {matches/(matches+errors)*100:.2f}%")

if __name__ == "__main__":
    verify_integrity()