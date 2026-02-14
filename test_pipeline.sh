#!/bin/bash
# Test the C++ pipeline with one ticker

TICKER=${1:-nvda}

echo "=========================================="
echo "Testing Pipeline with ${TICKER^^}"
echo "=========================================="

cd engine

# Start components in background
echo "Starting checker..."
./checker ../data/converted/${TICKER}_truth.bin > ../logs/${TICKER}_checker.log 2>&1 &
CHECKER_PID=$!

sleep 1

echo "Starting recorder..."
./recorder ../data/converted/${TICKER}_dataset.bin > ../logs/${TICKER}_recorder.log 2>&1 &
RECORDER_PID=$!

sleep 1

echo "Starting engine..."
./engine > ../logs/${TICKER}_engine.log 2>&1 &
ENGINE_PID=$!

sleep 2

echo "Starting sender..."
./sender ../data/converted/${TICKER}_orders.bin

echo ""
echo "Waiting for processing to complete..."
sleep 5

# Stop all components
echo "Stopping components..."
kill $CHECKER_PID $RECORDER_PID $ENGINE_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Pipeline test complete!"
echo "=========================================="
echo "Check logs in logs/ directory"
echo "Dataset saved to: data/converted/${TICKER}_dataset.bin"
