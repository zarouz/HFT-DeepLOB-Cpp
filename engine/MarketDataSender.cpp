#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#define PORT 5000
#define TARGET_IP "127.0.0.1"
#define THROTTLE_US 1

int main(int argc, char* argv[]) {
  // Accept input file as command-line argument
  const char* input_file = (argc > 1) ? argv[1] : "../data/converted/nvda_orders.bin";
  
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in servaddr{};
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = inet_addr(TARGET_IP);

  std::ifstream file(input_file, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open " << input_file << std::endl;
    return 1;
  }

  std::cout << ">>> SENDER ONLINE (" << input_file << " -> Port " << PORT
            << ") <<<" << std::endl;

  MDMessage msg;
  long long count = 0;

  while (file.read((char *)&msg, sizeof(MDMessage))) {
    sendto(sockfd, &msg, sizeof(MDMessage), 0, (struct sockaddr *)&servaddr,
           sizeof(servaddr));
    std::this_thread::sleep_for(std::chrono::microseconds(THROTTLE_US));
    count++;
    if (count % 100000 == 0)
      std::cout << "\rSent: " << count << std::flush;
  }

  std::cout << "\n>>> COMPLETE. Total: " << count << " <<<" << std::endl;
  close(sockfd);
  return 0;
}
