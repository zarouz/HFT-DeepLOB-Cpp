#include "../SharedProtocol.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>

#define PORT 5000
#define TARGET_IP "127.0.0.1"

int main() {
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = inet_addr(TARGET_IP);

  // Ensure this path points to your binary file
  std::string binPath = "/Users/karthikyadav/Desktop/Startup/HFT/TradingSystem/"
                        "Exchange/msft_orders.bin";
  std::ifstream file(binPath, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open " << binPath << std::endl;
    return 1;
  }

  std::cout << ">>> BINARY BLASTER ONLINE (Reliability Mode) <<<" << std::endl;
  MDMessage msg;
  long long count = 0;

  while (file.read(reinterpret_cast<char *>(&msg), sizeof(MDMessage))) {
    sendto(sockfd, &msg, sizeof(MDMessage), 0,
           (const struct sockaddr *)&servaddr, sizeof(servaddr));

    // --- RELIABILITY FIX ---
    // Sleep 1 microsecond EVERY packet.
    // This prevents the OS UDP buffer from overflowing and dropping packets.
    std::this_thread::sleep_for(std::chrono::microseconds(1));

    count++;
    if (count % 100000 == 0)
      std::cout << "Sent: " << count << " orders\r" << std::flush;
  }

  std::cout << "\n>>> COMPLETE. Total: " << count << " <<<" << std::endl;
  return 0;
}