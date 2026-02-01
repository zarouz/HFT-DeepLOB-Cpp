#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define LISTEN_PORT 7000
#define DEFAULT_OUTPUT "dataset.bin"

volatile sig_atomic_t running = 1;
void signalHandler(int) { running = 0; }

int main() {
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  int optval = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  struct timeval tv = {0, 100000};
  setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  int bufSize = 16 * 1024 * 1024;
  setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &bufSize, sizeof(bufSize));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(LISTEN_PORT);

  if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    std::cerr << "Bind failed on port " << LISTEN_PORT << std::endl;
    return 1;
  }

  std::ofstream outFile(DEFAULT_OUTPUT, std::ios::binary | std::ios::trunc);
  std::vector<LOBSnapshot> buffer;
  buffer.reserve(1000);

  std::cout << ">>> RECORDER ONLINE (Port " << LISTEN_PORT << " -> "
            << DEFAULT_OUTPUT << ") <<<" << std::endl;

  LOBSnapshot snap;
  long total = 0;

  while (running) {
    if (recvfrom(sockfd, &snap, sizeof(LOBSnapshot), 0, nullptr, nullptr) ==
        sizeof(LOBSnapshot)) {
      if (snap.time < 1.0)
        continue; // Skip invalid timestamps
      buffer.push_back(snap);
      if (buffer.size() >= 1000) {
        outFile.write((char *)buffer.data(),
                      buffer.size() * sizeof(LOBSnapshot));
        total += buffer.size();
        buffer.clear();
        if (total % 100000 == 0)
          std::cout << "\rCaptured: " << total << std::flush;
      }
    }
  }

  if (!buffer.empty()) {
    outFile.write((char *)buffer.data(), buffer.size() * sizeof(LOBSnapshot));
    total += buffer.size();
  }

  std::cout << "\n>>> STOPPED. Saved: " << total << " snapshots <<<"
            << std::endl;
  close(sockfd);
  return 0;
}