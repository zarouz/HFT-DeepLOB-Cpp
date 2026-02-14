#!/bin/bash
# Build all C++ components

echo "Building LatencyLab C++ Components..."

g++ -std=c++20 -O3 MarketDataSender.cpp -o sender
g++ -std=c++20 -O3 MarketDataReceiver.cpp -o engine -pthread
g++ -std=c++20 -O3 DataRecorder.cpp -o recorder
g++ -std=c++20 -O3 IntegrityChecker.cpp -o checker
g++ -std=c++20 -O3 OfflineValidator.cpp -o validator
g++ -std=c++20 -O3 MismatchAnalyzer.cpp -o analyzer

echo "Build complete!"
ls -lh sender engine recorder checker validator analyzer
