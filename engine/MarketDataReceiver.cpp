// engine/MarketDataReceiver.cpp
#include "LockFreeQueue.hpp"
#include "OrderBook.hpp"
#include "SharedProtocol.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#define LISTEN_PORT 5000
#define MONITOR_PORT 6000
#define RECORDER_PORT 7000
#define QUEUE_SIZE 65536

LockFreeQueue<MDMessage, QUEUE_SIZE> ringBuffer;
std::atomic<bool> running(true);

void engineWorker() {
  int monitorSock = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in monitorAddr{};
  monitorAddr.sin_family = AF_INET;
  monitorAddr.sin_port = htons(MONITOR_PORT);
  monitorAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

  int recorderSock = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in recorderAddr{};
  recorderAddr.sin_family = AF_INET;
  recorderAddr.sin_port = htons(RECORDER_PORT);
  recorderAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

  OrderBook engine;
  MDMessage packet;
  EngineState state;
  bool hasValidTimestamp = false;

  while (running) {
    if (ringBuffer.pop(packet)) {
      engine.processPacket(packet);

      if (!hasValidTimestamp) {
        if (packet.time > 1.0) {
          hasValidTimestamp = true;
          std::cout << ">>> First valid timestamp: " << std::fixed
                    << packet.time << " <<<" << std::endl;
        } else
          continue;
      }

      const LOBSnapshot &snapshot = engine.getSnapshot();
      sendto(recorderSock, &snapshot, sizeof(LOBSnapshot), 0,
             (struct sockaddr *)&recorderAddr, sizeof(recorderAddr));

      state.lastTime = packet.time;
      state.bidPrice = engine.getBestBidPrice();
      state.bidSize = engine.getBestBidSize();
      state.askPrice = engine.getBestAskPrice();
      state.askSize = engine.getBestAskSize();
      sendto(monitorSock, &state, sizeof(EngineState), 0,
             (struct sockaddr *)&monitorAddr, sizeof(monitorAddr));
    } else {
      std::this_thread::yield();
    }
  }

  close(monitorSock);
  close(recorderSock);
}

int main() {
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  int optval = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  int bufSize = 16 * 1024 * 1024;
  setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &bufSize, sizeof(bufSize));

  struct sockaddr_in servaddr{}, cliaddr{};
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(LISTEN_PORT);

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    std::cerr << "Bind failed on port " << LISTEN_PORT << std::endl;
    return 1;
  }

  std::thread worker(engineWorker);
  std::cout << ">>> ENGINE ONLINE (Port " << LISTEN_PORT << " -> "
            << MONITOR_PORT << ", " << RECORDER_PORT << ") <<<" << std::endl;

  MDMessage packet;
  socklen_t len = sizeof(cliaddr);
  long long count = 0;

  while (true) {
    if (recvfrom(sockfd, &packet, sizeof(MDMessage), 0,
                 (struct sockaddr *)&cliaddr, &len) == sizeof(MDMessage)) {
      while (!ringBuffer.push(packet))
        std::this_thread::yield();
      count++;
      if (count % 100000 == 0)
        std::cout << "\rIngested: " << count << std::flush;
    }
  }

  running = false;
  worker.join();
  close(sockfd);
  return 0;
}