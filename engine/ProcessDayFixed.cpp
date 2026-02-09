#include "OrderBook.hpp"
#include "SharedProtocol.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <orders.bin> <truth.bin> <output.bin>\n";
    return 1;
  }
  
  const char* orders_file = argv[1];
  const char* truth_file = argv[2];
  const char* output_file = argv[3];
  
  // Load truth
  std::ifstream tf(truth_file, std::ios::binary);
  std::vector<TruthMessage> truth;
  TruthMessage tm;
  while (tf.read((char*)&tm, sizeof(TruthMessage))) {
    truth.push_back(tm);
  }
  tf.close();
  
  if (truth.empty()) {
    std::cerr << "No truth data\n";
    return 1;
  }
  
  // CRITICAL FIX: Get market open time from first truth record
  double market_open = truth.front().time;
  double market_close = truth.back().time;
  
  // Process orders
  std::ifstream of(orders_file, std::ios::binary);
  std::ofstream outf(output_file, std::ios::binary | std::ios::trunc);
  
  OrderBook engine;
  MDMessage msg;
  
  size_t truthIdx = 0;
  long long processed = 0, saved = 0, matched = 0, failed = 0;
  long long skipped_premarket = 0;
  
  std::vector<LOBSnapshot> buffer;
  buffer.reserve(10000);
  
  while (of.read((char*)&msg, sizeof(MDMessage))) {
    if (msg.time < 1.0) continue;
    
    // CRITICAL FIX: Skip pre-market orders
    if (msg.time < market_open - 60.0) {  // 60 seconds before truth starts
      skipped_premarket++;
      continue;
    }
    
    // Stop processing after market close
    if (msg.time > market_close + 60.0) {
      break;
    }
    
    engine.processPacket(msg);
    processed++;
    
    // Advance truth pointer
    while (truthIdx < truth.size() - 1 &&
           truth[truthIdx + 1].time <= msg.time) {
      truthIdx++;
    }
    
    const LOBSnapshot& snap = engine.getSnapshot();
    
    // Only validate if we're within market hours
    if (msg.time >= market_open && msg.time <= market_close) {
      if (snap.bidPrice[0] == truth[truthIdx].bidPrice &&
          snap.askPrice[0] == truth[truthIdx].askPrice) {
        matched++;
      } else {
        failed++;
      }
    }
    
    buffer.push_back(snap);
    saved++;
    
    if (buffer.size() >= 10000) {
      outf.write((char*)buffer.data(), buffer.size() * sizeof(LOBSnapshot));
      buffer.clear();
    }
  }
  
  if (!buffer.empty()) {
    outf.write((char*)buffer.data(), buffer.size() * sizeof(LOBSnapshot));
  }
  
  double acc = (matched + failed > 0) ? (100.0 * matched / (matched + failed)) : 0.0;
  
  // Silent unless error
  if (acc < 95.0) {
    std::cerr << "WARNING: " << acc << "% accuracy (skipped " << skipped_premarket << " pre-market orders)\n";
  }
  
  return (acc >= 95.0) ? 0 : 1;
}
