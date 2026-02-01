import struct
import os

ORDER_FILE = "msft_orders.bin"
TRUTH_FILE = "msft_truth.bin"

# <diQiqi -> double, int32, uint64, int32, int64, int32 (36 bytes)
ORDER_FMT = '<diQiqi'
# <dQiqi -> double, uint64, int32, uint64, int32 (32 bytes)
TRUTH_FMT = '<dQiqi'

def get_stats(file_path, fmt):
    size = struct.calcsize(fmt)
    first_ts = None
    last_ts = 0
    count = 0
    
    with open(file_path, "rb") as f:
        # Read first record
        chunk = f.read(size)
        if chunk:
            first_ts = struct.unpack(fmt, chunk)[0]
            count += 1
        
        # Seek to the end to get the last record
        f.seek(-size, os.SEEK_END)
        chunk = f.read(size)
        if chunk:
            last_ts = struct.unpack(fmt, chunk)[0]
            
        # Count total records
        f.seek(0)
        file_size = os.path.getsize(file_path)
        count = file_size // size
        
    return first_ts, last_ts, count

print("--- FILE SYNCHRONIZATION AUDIT ---")
o_start, o_end, o_count = get_stats(ORDER_FILE, ORDER_FMT)
t_start, t_end, t_count = get_stats(TRUTH_FILE, TRUTH_FMT)

print(f"Orders (MBO):  {o_count} records | Start: {o_start:.6f} | End: {o_end:.6f}")
print(f"Truth  (MBP):  {t_count} records | Start: {t_start:.6f} | End: {t_end:.6f}")
print("-" * 40)
print(f"Start Offset: {abs(o_start - t_start):.6f} seconds")
print(f"End Offset:   {abs(o_end - t_end):.6f} seconds")

if abs(o_end - t_end) > 1.0:
    print("\n[CONCLUSION]: MBO and MBP are out of sync at the end.")
    print("Your engine is processing 'Extra' time that the Truth file ignores.")