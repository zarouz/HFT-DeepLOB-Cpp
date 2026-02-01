#ifndef ORDERBOOK_HPP
#define ORDERBOOK_HPP

#include "../SharedProtocol.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

struct OrderInfo {
  uint64_t price;
  int size;
  int direction;
};

class OrderBook {
private:
  const size_t MAX_PRICE = 10000000;

  std::vector<int> bidVol;
  std::vector<int> askVol;
  uint64_t maxBid;
  uint64_t minAsk;
  std::unordered_map<uint64_t, OrderInfo> orderMap;

public:
  OrderBook() {
    bidVol.resize(MAX_PRICE, 0);
    askVol.resize(MAX_PRICE, 0);
    maxBid = 0;
    minAsk = MAX_PRICE;
  }

  void processPacket(const MDMessage &msg) {
    if (msg.price >= MAX_PRICE || msg.price == 0)
      return;

    // --- EVENT 1: ADD ---
    if (msg.eventType == 1) {
      orderMap[msg.orderId] = {msg.price, msg.size, msg.direction};
      updateLevel(msg.price, msg.size, msg.direction);
    }

    // --- EVENT 2: PARTIAL CANCEL / TRADE ---
    // (Uses msg.size as a Delta)
    else if (msg.eventType == 2 || msg.eventType == 4) {
      auto it = orderMap.find(msg.orderId);
      if (it != orderMap.end()) {
        int delta = -msg.size;
        updateLevel(it->second.price, delta, it->second.direction);
        it->second.size += delta;
        if (it->second.size <= 0)
          orderMap.erase(it);
      }
    }

    // --- EVENT 3: FORCE DELETE (Crucial Fix) ---
    // (Removes order completely, ignores msg.size)
    else if (msg.eventType == 3) {
      auto it = orderMap.find(msg.orderId);
      if (it != orderMap.end()) {
        // Remove the FULL current size of the order
        updateLevel(it->second.price, -it->second.size, it->second.direction);
        orderMap.erase(it);
      }
    }
  }

private:
  inline void updateLevel(uint64_t price, int delta, int direction) {
    if (direction == 1) { // BID
      bidVol[price] += delta;
      if (bidVol[price] > 0) {
        if (price > maxBid)
          maxBid = price;
      } else if (price == maxBid) {
        while (maxBid > 0 && bidVol[maxBid] <= 0)
          maxBid--;
      }
    } else { // ASK
      askVol[price] += delta;
      if (askVol[price] > 0) {
        if (price < minAsk)
          minAsk = price;
      } else if (price == minAsk) {
        while (minAsk < MAX_PRICE && askVol[minAsk] <= 0)
          minAsk++;
      }
    }
  }

public:
  uint64_t getBestBidPrice() const { return maxBid; }
  int getBestBidSize() const { return (maxBid > 0) ? bidVol[maxBid] : 0; }
  uint64_t getBestAskPrice() const { return minAsk; }
  int getBestAskSize() const {
    return (minAsk < MAX_PRICE) ? askVol[minAsk] : 0;
  }
};
#endif