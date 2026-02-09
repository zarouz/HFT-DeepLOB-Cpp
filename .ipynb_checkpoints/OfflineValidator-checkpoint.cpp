#include "SharedProtocol.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#define DATASET_FILE "dataset.bin"
#define TRUTH_FILE "msft_truth.bin"

int main() {
  std::ifstream df(DATASET_FILE, std::ios::binary);
  std::ifstream tf(TRUTH_FILE, std::ios::binary);

  if (!df) {
    std::cerr << "ERROR: Cannot open " << DATASET_FILE << std::endl;
    return 1;
  }
  if (!tf) {
    std::cerr << "ERROR: Cannot open " << TRUTH_FILE << std::endl;
    return 1;
  }

  std::cout << "=== OFFLINE VALIDATOR ===" << std::endl;
  std::cout << "LOBSnapshot:  " << sizeof(LOBSnapshot) << " bytes" << std::endl;
  std::cout << "TruthMessage: " << sizeof(TruthMessage) << " bytes"
            << std::endl;

  // Load truth
  std::vector<TruthMessage> truth;
  TruthMessage tm;
  while (tf.read((char *)&tm, sizeof(TruthMessage))) {
    if (tm.bidPrice > 0 && tm.askPrice > 0)
      truth.push_back(tm);
  }
  std::cout << "Truth loaded: " << truth.size() << " records" << std::endl;

  // Scan dataset
  LOBSnapshot snap;
  long total = 0, valid = 0;
  while (df.read((char *)&snap, sizeof(LOBSnapshot))) {
    total++;
    if (snap.time > 1.0)
      valid++;
  }
  std::cout << "Dataset: " << total << " total, " << valid
            << " valid timestamps" << std::endl;

  // Reset for validation
  df.clear();
  df.seekg(0);

  const double SYNC_GATE = 0.005; // 5ms
  size_t truthIdx = 0;
  long checked = 0, matches = 0;

  while (df.read((char *)&snap, sizeof(LOBSnapshot))) {
    if (snap.time < 1.0)
      continue;
    if (snap.time < truth.front().time - 1.0)
      continue;
    if (snap.time > truth.back().time + 1.0)
      break;

    while (truthIdx < truth.size() - 1 &&
           truth[truthIdx + 1].time <= snap.time) {
      truthIdx++;
    }

    double timeDiff = std::abs(snap.time - truth[truthIdx].time);
    if (timeDiff < SYNC_GATE) {
      checked++;
      if (snap.bidPrice[0] == truth[truthIdx].bidPrice &&
          snap.askPrice[0] == truth[truthIdx].askPrice) {
        matches++;
      }
    }
  }

  double accuracy = (checked > 0) ? (100.0 * matches / checked) : 0.0;

  std::cout << "\n=== RESULTS ===" << std::endl;
  std::cout << "Checked:  " << checked << std::endl;
  std::cout << "Matched:  " << matches << std::endl;
  std::cout << "ACCURACY: " << std::fixed << std::setprecision(2) << accuracy
            << "%" << std::endl;

  if (accuracy >= 98.0)
    std::cout << "✅ TARGET ACHIEVED" << std::endl;
  else
    std::cout << "⚠️  Below 98% target" << std::endl;

  return 0;
}