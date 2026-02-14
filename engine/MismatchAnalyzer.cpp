// engine/MismatchAnalyzer.cpp
#include "SharedProtocol.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

int main() {
  std::ifstream df("dataset.bin", std::ios::binary);
  std::ifstream tf("msft_truth.bin", std::ios::binary);

  if (!df || !tf) {
    std::cerr << "Cannot open files" << std::endl;
    return 1;
  }

  // Load truth
  std::vector<TruthMessage> truth;
  TruthMessage tm;
  while (tf.read((char *)&tm, sizeof(TruthMessage))) {
    if (tm.bidPrice > 0 && tm.askPrice > 0)
      truth.push_back(tm);
  }

  std::cout << "=== MISMATCH ANALYSIS ===" << std::endl;
  std::cout << "Truth records: " << truth.size() << std::endl << std::endl;

  LOBSnapshot snap;
  const double SYNC_GATE = 0.005;
  size_t truthIdx = 0;

  long checked = 0, matches = 0;
  long bidOnlyFail = 0, askOnlyFail = 0, bothFail = 0;
  long bidHigher = 0, bidLower = 0, askHigher = 0, askLower = 0;

  int printCount = 0;

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

      bool bidOK = (snap.bidPrice[0] == truth[truthIdx].bidPrice);
      bool askOK = (snap.askPrice[0] == truth[truthIdx].askPrice);

      if (bidOK && askOK) {
        matches++;
      } else {
        if (!bidOK && askOK)
          bidOnlyFail++;
        else if (bidOK && !askOK)
          askOnlyFail++;
        else
          bothFail++;

        if (!bidOK) {
          if (snap.bidPrice[0] > truth[truthIdx].bidPrice)
            bidHigher++;
          else
            bidLower++;
        }
        if (!askOK) {
          if (snap.askPrice[0] > truth[truthIdx].askPrice)
            askHigher++;
          else
            askLower++;
        }

        // Print first 20 mismatches with detail
        if (printCount < 20) {
          std::cout << "MISMATCH #" << (printCount + 1)
                    << " at T=" << std::fixed << std::setprecision(6)
                    << snap.time << std::endl;
          std::cout << "  Engine: Bid=" << snap.bidPrice[0] << " ("
                    << snap.bidSize[0] << ") "
                    << "Ask=" << snap.askPrice[0] << " (" << snap.askSize[0]
                    << ")" << std::endl;
          std::cout << "  Truth:  Bid=" << truth[truthIdx].bidPrice << " ("
                    << truth[truthIdx].bidSize << ") "
                    << "Ask=" << truth[truthIdx].askPrice << " ("
                    << truth[truthIdx].askSize << ")" << std::endl;

          int64_t bidDiff =
              (int64_t)snap.bidPrice[0] - (int64_t)truth[truthIdx].bidPrice;
          int64_t askDiff =
              (int64_t)snap.askPrice[0] - (int64_t)truth[truthIdx].askPrice;
          std::cout << "  Delta:  Bid=" << bidDiff << " Ask=" << askDiff
                    << std::endl;
          std::cout << std::endl;
          printCount++;
        }
      }
    }
  }

  long totalFail = bidOnlyFail + askOnlyFail + bothFail;

  std::cout << "=== SUMMARY ===" << std::endl;
  std::cout << "Checked:     " << checked << std::endl;
  std::cout << "Matched:     " << matches << " (" << std::fixed
            << std::setprecision(2) << (100.0 * matches / checked) << "%)"
            << std::endl;
  std::cout << "Failed:      " << totalFail << " ("
            << (100.0 * totalFail / checked) << "%)" << std::endl;
  std::cout << std::endl;
  std::cout << "Failure breakdown:" << std::endl;
  std::cout << "  Bid only wrong:  " << bidOnlyFail << " ("
            << (100.0 * bidOnlyFail / totalFail) << "%)" << std::endl;
  std::cout << "  Ask only wrong:  " << askOnlyFail << " ("
            << (100.0 * askOnlyFail / totalFail) << "%)" << std::endl;
  std::cout << "  Both wrong:      " << bothFail << " ("
            << (100.0 * bothFail / totalFail) << "%)" << std::endl;
  std::cout << std::endl;
  std::cout << "Direction of errors:" << std::endl;
  std::cout << "  Bid too high: " << bidHigher << "  Bid too low: " << bidLower
            << std::endl;
  std::cout << "  Ask too high: " << askHigher << "  Ask too low: " << askLower
            << std::endl;

  return 0;
}