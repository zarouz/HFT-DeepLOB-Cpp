// engine/ValidateAllDays.cpp
#include "OrderBook.hpp"
#include "SharedProtocol.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
  const char *tickers[] = {"nvda", "spy", "tsla", "pltr", "amd"};
  const char *dates[] = {
      "20260105", "20260106", "20260107", "20260108", "20260109", "20260112",
      "20260113", "20260114", "20260115", "20260116", "20260120", "20260121",
      "20260122", "20260123", "20260126", "20260127", "20260128", "20260129"};

  int total_days = 0;
  int passed_days = 0;
  int skipped_days = 0;

  std::cout << "VALIDATING ALL BINARY FILES (5 tickers × 18 days = 90 tests)\n";
  std::cout
      << "=============================================================\n\n";

  for (auto ticker : tickers) {
    std::cout << ticker << ":\n";

    for (auto date : dates) {
      total_days++;

      std::string orders_file = "../data/converted/daily/" +
                                std::string(ticker) + "_" + std::string(date) +
                                "_orders.bin";
      std::string truth_file = "../data/converted/daily/" +
                               std::string(ticker) + "_" + std::string(date) +
                               "_truth.bin";

      std::ifstream of(orders_file, std::ios::binary);
      std::ifstream tf(truth_file, std::ios::binary);

      if (!of || !tf) {
        std::cout << "  " << date << ": SKIP (files missing)\n";
        skipped_days++;
        continue;
      }

      // Load truth
      std::vector<TruthMessage> truth;
      TruthMessage tm;
      while (tf.read((char *)&tm, sizeof(TruthMessage))) {
        truth.push_back(tm);
      }

      if (truth.empty()) {
        std::cout << "  " << date << ": FAIL (no truth data)\n";
        continue;
      }

      // Process orders
      OrderBook engine;
      MDMessage msg;

      size_t truthIdx = 0;
      long long match = 0, fail = 0;

      while (of.read((char *)&msg, sizeof(MDMessage))) {
        if (msg.time < 1.0)
          continue;

        engine.processPacket(msg);

        while (truthIdx < truth.size() - 1 &&
               truth[truthIdx + 1].time <= msg.time) {
          truthIdx++;
        }

        if (engine.getBestBidPrice() == truth[truthIdx].bidPrice &&
            engine.getBestAskPrice() == truth[truthIdx].askPrice) {
          match++;
        } else {
          fail++;
        }
      }

      double acc = (match + fail > 0) ? (100.0 * match / (match + fail)) : 0.0;

      if (acc >= 95.0) {
        std::cout << "  " << date << ": ✅ " << std::fixed
                  << std::setprecision(1) << acc << "%\n";
        passed_days++;
      } else if (acc >= 90.0) {
        std::cout << "  " << date << ": ⚠️  " << std::fixed
                  << std::setprecision(1) << acc << "% (marginal)\n";
        passed_days++;
      } else {
        std::cout << "  " << date << ": ❌ " << std::fixed
                  << std::setprecision(1) << acc << "% (FAIL)\n";
      }
    }
    std::cout << "\n";
  }

  std::cout
      << "=============================================================\n";
  std::cout << "Total days tested: " << total_days << "\n";
  std::cout << "Passed (≥95%):     " << passed_days << "\n";
  std::cout << "Skipped (missing): " << skipped_days << "\n";
  std::cout << "Failed (<95%):     "
            << (total_days - passed_days - skipped_days) << "\n";

  double success_rate =
      (double)passed_days / (total_days - skipped_days) * 100.0;
  std::cout << "\nSuccess rate: " << std::fixed << std::setprecision(1)
            << success_rate << "%\n";

  if (success_rate >= 90.0) {
    std::cout << "\n✅ SAFE TO DELETE .dbn FILES\n";
    std::cout << "Your binary files are validated and complete.\n";
    return 0;
  } else {
    std::cout << "\n⚠️  KEEP .dbn FILES FOR NOW\n";
    std::cout << "Success rate below 90% - investigate failures first.\n";
    return 1;
  }
}
