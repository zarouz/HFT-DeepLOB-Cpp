// engine/MarketDataSenderFast.cpp
#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define PORT 5000
#define TARGET_IP "127.0.0.1"
#define BATCH_SIZE 10000 // Send in batches

int main(int argc, char *argv[]) {
  const char *input_file =
      (argc > 1) ? argv[1] : "../data/converted/nvda_orders.bin";

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

  // Increase socket buffer
  int bufSize = 16 * 1024 * 1024;
  setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &bufSize, sizeof(bufSize));

  struct sockaddr_in servaddr{};
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = inet_addr(TARGET_IP);

  std::ifstream file(input_file, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open " << input_file << std::endl;
    return 1;
  }

  std::cout << ">>> FAST SENDER ONLINE (" << input_file << " -> Port " << PORT
            << ") <<<" << std::endl;

  std::vector<MDMessage> batch;
  batch.reserve(BATCH_SIZE);

  MDMessage msg;
  long long count = 0;

  while (file.read((char *)&msg, sizeof(MDMessage))) {
    batch.push_back(msg);

    if (batch.size() >= BATCH_SIZE) {
      // Send batch
      for (const auto &m : batch) {
        sendto(sockfd, &m, sizeof(MDMessage), 0, (struct sockaddr *)&servaddr,
               sizeof(servaddr));
      }
      count += batch.size();
      batch.clear();

      if (count % 1000000 == 0)
        std::cout << "\rSent: " << count / 1000000 << "M" << std::flush;
    }
  }

  // Send remaining
  for (const auto &m : batch) {
    sendto(sockfd, &m, sizeof(MDMessage), 0, (struct sockaddr *)&servaddr,
           sizeof(servaddr));
  }
  count += batch.size();

  std::cout << "\n>>> COMPLETE. Total: " << count << " <<<" << std::endl;
  close(sockfd);
  return 0;
}
