# scripts/AuditBinaries.py
import struct
import os
import sys

ORDER_FMT = '<diQiqi'  # 36 bytes
TRUTH_FMT = '<dQiQi'   # 32 bytes

def audit_ticker(ticker):
    """Audit synchronization for one ticker."""
    
    order_file = f"data/converted/{ticker}_orders.bin"
    truth_file = f"data/converted/{ticker}_truth.bin"
    
    if not os.path.exists(order_file):
        print(f"SKIP {ticker}: {order_file} not found")
        return
    
    if not os.path.exists(truth_file):
        print(f"SKIP {ticker}: {truth_file} not found")
        return
    
    print(f"\n{'='*60}")
    print(f"AUDITING {ticker.upper()}")
    print(f"{'='*60}")
    
    # Get stats
    order_size = struct.calcsize(ORDER_FMT)
    truth_size = struct.calcsize(TRUTH_FMT)
    
    with open(order_file, 'rb') as of, open(truth_file, 'rb') as tf:
        # First record
        o_chunk = of.read(order_size)
        t_chunk = tf.read(truth_size)
        
        o_first = struct.unpack(ORDER_FMT, o_chunk)[0] if o_chunk else 0
        t_first = struct.unpack(TRUTH_FMT, t_chunk)[0] if t_chunk else 0
        
        # Last record
        of.seek(-order_size, os.SEEK_END)
        tf.seek(-truth_size, os.SEEK_END)
        
        o_chunk = of.read(order_size)
        t_chunk = tf.read(truth_size)
        
        o_last = struct.unpack(ORDER_FMT, o_chunk)[0] if o_chunk else 0
        t_last = struct.unpack(TRUTH_FMT, t_chunk)[0] if t_chunk else 0
        
        # Count
        of.seek(0, os.SEEK_END)
        tf.seek(0, os.SEEK_END)
        
        o_count = of.tell() // order_size
        t_count = tf.tell() // truth_size
    
    print(f"Orders (MBO):  {o_count:,} records")
    print(f"  First: {o_first:.6f}")
    print(f"  Last:  {o_last:.6f}")
    print(f"  Span:  {o_last - o_first:.2f} seconds")
    
    print(f"\nTruth  (MBP):  {t_count:,} records")
    print(f"  First: {t_first:.6f}")
    print(f"  Last:  {t_last:.6f}")
    print(f"  Span:  {t_last - t_first:.2f} seconds")
    
    print(f"\nSynchronization:")
    print(f"  Start offset: {abs(o_first - t_first):.6f} seconds")
    print(f"  End offset:   {abs(o_last - t_last):.6f} seconds")
    
    if abs(o_last - t_last) > 1.0:
        print("  ⚠️  WARNING: Files may be out of sync")
    else:
        print("  ✅ Files are synchronized")

if __name__ == "__main__":
    tickers = ["spy", "nvda", "tsla", "pltr", "amd"]
    
    print("\n" + "="*60)
    print("FILE SYNCHRONIZATION AUDIT")
    print("="*60)
    
    for ticker in tickers:
        audit_ticker(ticker)
    
    print("\n" + "="*60)