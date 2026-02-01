#include "SharedProtocol.hpp" // <--- FIXED: Removed "../"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <vector>

#define PORT 6000

int main() {
  // --- DEBUG: SIZE CHECK ---
  std::cout << "Checking Struct Sizes..." << std::endl;
  std::cout << "MDMessage Size:    " << sizeof(MDMessage) << " (Expected 36)"
            << std::endl;
  std::cout << "TruthMessage Size: " << sizeof(TruthMessage) << " (Expected 32)"
            << std::endl;

  if (sizeof(MDMessage) != 36 || sizeof(TruthMessage) != 32) {
    std::cerr << "CRITICAL ERROR: Struct padding is still active! Check "
                 "SharedProtocol.hpp"
              << std::endl;
    return 1;
  }
  // YOUR PATH
  std::string truthPath = "/Users/karthikyadav/Desktop/Startup/HFT/"
                          "TradingSystem/Exchange/msft_truth.bin";
  std::ifstream file(truthPath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Missing " << truthPath << std::endl;
    return 1;
  }

  std::cout << "Loading 865k Truth Snapshots into RAM..." << std::endl;
  std::vector<TruthMessage> truthLog;
  TruthMessage tMsg;
  while (file.read(reinterpret_cast<char *>(&tMsg), sizeof(TruthMessage))) {
    truthLog.push_back(tMsg);
  }
  std::cout << "Truth Loaded. Ready." << std::endl;

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(PORT);
  bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr));

  std::cout << ">>> MSFT INTEGRITY MONITOR ONLINE <<<" << std::endl;

  EngineState state;
  size_t truthIdx = 0;
  long long match = 0, fail = 0;

  while (true) {
    int n = recvfrom(sockfd, &state, sizeof(EngineState), 0, NULL, NULL);
    if (n > 0) {
      // SYNC LOGIC: Advance Truth pointer until timestamps align
      while (truthIdx < truthLog.size() - 1 &&
             truthLog[truthIdx + 1].time <= state.lastTime) {
        truthIdx++;
      }

      const auto &truth = truthLog[truthIdx];

      // Only check if timestamps are reasonably close (within 1 second)
      bool pass = (state.bidPrice == truth.bidPrice) &&
                  (state.askPrice == truth.askPrice);

      if (pass)
        match++;
      else
        fail++;

      if ((match + fail) % 50000 == 0) {
        std::cout << "\033[2J\033[1;1H";
        double acc = 100.0 * match / (match + fail);
        std::string color = (acc > 95) ? "\033[1;32m" : "\033[1;31m";

        std::cout << "--- MSFT PROFESSIONAL VALIDATION ---" << std::endl;
        std::cout << "Accuracy: " << color << std::fixed << std::setprecision(2)
                  << acc << "%" << "\033[0m" << std::endl;
        std::cout << "Processed: " << (match + fail) << " states" << std::endl;
        std::cout << "Time:      " << std::fixed << state.lastTime << std::endl;

        if (!pass) {
          std::cout << "\n[MISMATCH]" << std::endl;
          std::cout << "Engine: Bid " << state.bidPrice << " / Ask "
                    << state.askPrice << std::endl;
          std::cout << "Truth:  Bid " << truth.bidPrice << " / Ask "
                    << truth.askPrice << std::endl;
        } else {
          std::cout << "\n[SYNCED]" << std::endl;
          std::cout << "Spread: $"
                    << (state.askPrice - state.bidPrice) / 10000.0 << std::endl;
        }
      }
    }
  }
  return 0;
}