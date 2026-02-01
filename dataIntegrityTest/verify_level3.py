import pandas as pd
import sys

# CONFIGURATION
MSG_FILE = "/Users/karthikyadav/Desktop/Startup/HFT/TradingSystem/Exchange/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv"
BOOK_FILE = "/Users/karthikyadav/Desktop/Startup/HFT/TradingSystem/Exchange/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"


def verify_integrity():
    print(f"Loading {MSG_FILE}...")
    messages = pd.read_csv(MSG_FILE, header=None, names=["Time", "Type", "ID", "Size", "Price", "Dir"])
    book = pd.read_csv(BOOK_FILE, header=None)
    
    # --- LEVEL 3 DATA STRUCTURES ---
    # 1. Order Map: ID -> {Price, Size, Direction}
    active_orders = {}
    
    # 2. Price Levels: Price -> Total Size (Derived from Order Map)
    #    Split into Bids (Buy) and Asks (Sell)
    bids = {} 
    asks = {}

    def update_book(price, size_delta, is_bid):
        target = bids if is_bid else asks
        if price not in target: target[price] = 0
        target[price] += size_delta
        if target[price] <= 0: 
            if price in target: del target[price]

    # --- BOOTSTRAP (Crucial Step) ---
    # Since we don't have the Order IDs for the initial snapshot (LOBSTER anonymizes them until they update),
    # we will "Fake" the IDs for the initial state just to get the math right.
    print("Bootstrapping from Snapshot...")
    first_row = book.iloc[0]
    
    # We assign negative IDs to snapshot orders so they don't clash with real updates
    fake_id_counter = -1
    
    for i in range(10):
        offset = i * 4
        ask_p, ask_s = first_row[offset], first_row[offset+1]
        bid_p, bid_s = first_row[offset+2], first_row[offset+3]
        
        if ask_s > 0:
            active_orders[fake_id_counter] = {'price': ask_p, 'size': ask_s, 'dir': -1}
            update_book(ask_p, ask_s, False)
            fake_id_counter -= 1
            
        if bid_s > 0:
            active_orders[fake_id_counter] = {'price': bid_p, 'size': bid_s, 'dir': 1}
            update_book(bid_p, bid_s, True)
            fake_id_counter -= 1

    matches = 0
    errors = 0
    
    print("Starting Level 3 Replay...")
    
    # We iterate starting at 1 because Book Row 0 is the start state.
    # Message 1 takes us to Book Row 1.
    for i in range(1, len(messages)):
        msg = messages.iloc[i]
        
        m_type = msg['Type']
        m_id = msg['ID']
        m_size = msg['Size']
        m_price = msg['Price']
        m_dir = msg['Dir']
        
        # --- LOGIC CORE ---
        if m_type == 1: # ADD
            active_orders[m_id] = {'price': m_price, 'size': m_size, 'dir': m_dir}
            update_book(m_price, m_size, m_dir == 1)
            
        elif m_type in [2, 3, 4, 5]: # CANCEL, DELETE, EXECUTE (Visible & Hidden)
            # LOBSTER is tricky: Sometimes it gives an ID we haven't seen 
            # (if it was part of the initial snapshot or hidden).
            
            # Case A: We know this Order ID
            if m_id in active_orders:
                order = active_orders[m_id]
                price_to_update = order['price']
                is_bid = (order['dir'] == 1)
                
                # Calculate removal amount
                remove_size = m_size
                if m_type == 3: # Delete (Remove all remaining)
                     remove_size = order['size']

                # Update Level
                update_book(price_to_update, -remove_size, is_bid)
                
                # Update/Remove Order
                order['size'] -= remove_size
                if order['size'] <= 0:
                    del active_orders[m_id]
            
            # Case B: We DON'T know this ID (It was in the Snapshot or Hidden)
            else:
                # Fallback to Price-Level logic using the message price
                # This is necessary because LOBSTER snapshot orders have no public IDs
                update_book(m_price, -m_size, m_dir == 1)

        # --- VERIFICATION ---
        expected = book.iloc[i]
        
        # Get Best Bid/Ask
        my_best_bid = max(bids.keys()) if bids else 0
        my_best_ask = min(asks.keys()) if asks else 0
        
        # LOBSTER uses -9999999999 for empty levels
        true_best_ask = expected[0]
        true_best_bid = expected[2]
        
        if my_best_bid == true_best_bid and my_best_ask == true_best_ask:
            matches += 1
        else:
            # Filter out "Empty Book" noise (-9999...)
            if true_best_bid > 0 and true_best_ask > 0:
                errors += 1
                if errors < 3:
                    print(f"Row {i} FAIL | MyAsk: {my_best_ask} vs TrueAsk: {true_best_ask}")

        if i % 50000 == 0:
            acc = matches / (matches + errors) * 100
            print(f"Progress {i} | Accuracy: {acc:.2f}%")

    print(f"Final Accuracy: {matches/(matches+errors)*100:.2f}%")

if __name__ == "__main__":
    verify_integrity()