#include "LockFreeQueue.hpp"
#include "Orderbook/OrderBook.hpp"
#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>

#define LISTEN_PORT 5000
#define MONITOR_PORT 6000
#define QUEUE_SIZE 65536

LockFreeQueue<MDMessage, QUEUE_SIZE> ringBuffer;
std::atomic<bool> running(true);

void engineWorker() {
  int monitorSock = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in monitorAddr;
  memset(&monitorAddr, 0, sizeof(monitorAddr));
  monitorAddr.sin_family = AF_INET;
  monitorAddr.sin_port = htons(MONITOR_PORT);
  monitorAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

  OrderBook engine;
  MDMessage packet;
  EngineState state;

  while (running) {
    if (ringBuffer.pop(packet)) {
      engine.processPacket(packet);

      state.lastTime = packet.time;
      state.bidPrice = engine.getBestBidPrice();
      state.bidSize = engine.getBestBidSize();
      state.askPrice = engine.getBestAskPrice();
      state.askSize = engine.getBestAskSize();

      sendto(monitorSock, &state, sizeof(EngineState), 0,
             (const struct sockaddr *)&monitorAddr, sizeof(monitorAddr));
    } else {
      std::this_thread::yield();
    }
  }
}

int main() {
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in servaddr, cliaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(LISTEN_PORT);
  bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr));

  std::thread worker(engineWorker);
  std::cout << ">>> ENGINE ONLINE (MSFT High-Freq) <<<" << std::endl;

  MDMessage packet;
  socklen_t len = sizeof(cliaddr);
  long long count = 0;

  while (true) {
    int n = recvfrom(sockfd, &packet, sizeof(MDMessage), 0,
                     (struct sockaddr *)&cliaddr, &len);
    if (n > 0) {
      ringBuffer.push(packet);
      count++;
      if (count % 100000 == 0)
        std::cout << "Ingest: " << count << "\r" << std::flush;
    }
  }
  return 0;
}