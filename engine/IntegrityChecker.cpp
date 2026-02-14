// engine/IntegrityChecker.cpp

#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define LISTEN_PORT 6000

int main(int argc, char *argv[]) {
  // Accept truth file as command-line argument
  const char *truth_file =
      (argc > 1) ? argv[1] : "../data/converted/nvda_truth.bin";

  std::ifstream file(truth_file, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open " << truth_file << std::endl;
    return 1;
  }

  std::vector<TruthMessage> truthLog;
  TruthMessage tMsg;
  while (file.read((char *)&tMsg, sizeof(TruthMessage))) {
    if (tMsg.bidPrice > 0 && tMsg.askPrice > 0)
      truthLog.push_back(tMsg);
  }
  file.close();

  if (truthLog.empty()) {
    std::cerr << "ERROR: No valid truth records" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << truthLog.size() << " truth snapshots from "
            << truth_file << std::endl;

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  int optval = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  struct sockaddr_in servaddr{};
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(LISTEN_PORT);

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    std::cerr << "Bind failed on port " << LISTEN_PORT << std::endl;
    return 1;
  }

  std::cout << ">>> CHECKER ONLINE (Port " << LISTEN_PORT << ") <<<"
            << std::endl;

  EngineState state;
  size_t truthIdx = 0;
  long long match = 0, fail = 0;

  while (true) {
    if (recvfrom(sockfd, &state, sizeof(EngineState), 0, NULL, NULL) ==
        sizeof(EngineState)) {
      while (truthIdx < truthLog.size() - 1 &&
             truthLog[truthIdx + 1].time <= state.lastTime) {
        truthIdx++;
      }

      const auto &truth = truthLog[truthIdx];
      bool pass = (state.bidPrice == truth.bidPrice) &&
                  (state.askPrice == truth.askPrice);

      if (pass)
        match++;
      else
        fail++;

      if ((match + fail) % 50000 == 0) {
        std::cout << "\033[2J\033[1;1H";
        double acc = 100.0 * match / (match + fail);
        std::string color = (acc >= 98.0)   ? "\033[1;32m"
                            : (acc >= 90.0) ? "\033[1;33m"
                                            : "\033[1;31m";

        std::cout << "=== HFT VALIDATION ===" << std::endl;
        std::cout << "Processed: " << (match + fail) << std::endl;
        std::cout << "Accuracy:  " << color << std::fixed
                  << std::setprecision(2) << acc << "%" << "\033[0m"
                  << std::endl;
        std::cout << "Engine: Bid " << state.bidPrice << " / Ask "
                  << state.askPrice << std::endl;
        std::cout << "Truth:  Bid " << truth.bidPrice << " / Ask "
                  << truth.askPrice << std::endl;
      }
    }
  }

  close(sockfd);
  return 0;
}
