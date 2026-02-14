#!/bin/bash
# Process all tickers using direct processor (fast mode)

cd engine

TICKERS=("nvda" "spy" "tsla" "pltr" "amd")

echo "=========================================="
echo "BATCH PROCESSING ALL TICKERS"
echo "=========================================="
echo ""

for ticker in "${TICKERS[@]}"; do
  echo "================================================"
  echo "Processing ${ticker^^}"
  echo "================================================"
  
  ORDERS="../data/converted/${ticker}_orders.bin"
  TRUTH="../data/converted/${ticker}_truth.bin"
  OUTPUT="../data/converted/${ticker}_dataset.bin"
  
  # Check if files exist
  if [ ! -f "$ORDERS" ]; then
    echo "ERROR: $ORDERS not found, skipping..."
    continue
  fi
  
  if [ ! -f "$TRUTH" ]; then
    echo "ERROR: $TRUTH not found, skipping..."
    continue
  fi
  
  # Run direct processor
  time ./direct_processor "$ORDERS" "$TRUTH" "$OUTPUT"
  
  echo ""
  echo "Dataset saved: $OUTPUT"
  echo ""
  
  # Quick validation
  if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    COUNT=$(stat -c%s "$OUTPUT")
    RECORDS=$((COUNT / 248))
    echo "Output size: $SIZE ($RECORDS snapshots)"
  fi
  
  echo ""
done

echo "=========================================="
echo "ALL TICKERS PROCESSED"
echo "=========================================="
