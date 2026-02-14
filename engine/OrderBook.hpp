// engine/SharedProtocol.hpp
#ifndef ORDERBOOK_HPP
#define ORDERBOOK_HPP

#include "SharedProtocol.hpp"
#include <algorithm>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

struct OrderInfo {
  uint64_t price;
  int32_t size;
  int32_t direction;
};

class OrderBook {
private:
  static const size_t MAX_PRICE = 10000000;
  std::vector<int32_t> bidVol;
  std::vector<int32_t> askVol;
  std::set<uint64_t, std::greater<uint64_t>> activeBids;
  std::set<uint64_t> activeAsks;
  std::unordered_map<uint64_t, OrderInfo> orderMap;
  LOBSnapshot currentSnapshot;

public:
  OrderBook() : bidVol(MAX_PRICE, 0), askVol(MAX_PRICE, 0) {
    std::memset(&currentSnapshot, 0, sizeof(LOBSnapshot));
  }

  void processPacket(const MDMessage &msg) {
    if (msg.price >= MAX_PRICE)
      return;
    if (msg.eventType == 1 && msg.price == 0)
      return;

    switch (msg.eventType) {
    case 1:
      handleAddOrModify(msg);
      break;
    case 2:
    case 4:
      handleReduce(msg);
      break;
    case 3:
      handleDelete(msg);
      break;
    }
    currentSnapshot.time = msg.time;
    rebuildSnapshot();
  }

  const LOBSnapshot &getSnapshot() const { return currentSnapshot; }
  uint64_t getBestBidPrice() const { return currentSnapshot.bidPrice[0]; }
  int32_t getBestBidSize() const { return currentSnapshot.bidSize[0]; }
  uint64_t getBestAskPrice() const { return currentSnapshot.askPrice[0]; }
  int32_t getBestAskSize() const { return currentSnapshot.askSize[0]; }

private:
  void handleAddOrModify(const MDMessage &msg) {
    auto it = orderMap.find(msg.orderId);
    if (it != orderMap.end()) {
      updateLevel(it->second.price, -it->second.size, it->second.direction);
    }
    orderMap[msg.orderId] = {msg.price, msg.size, msg.direction};
    updateLevel(msg.price, msg.size, msg.direction);
  }

  void handleReduce(const MDMessage &msg) {
    auto it = orderMap.find(msg.orderId);
    if (it == orderMap.end())
      return;
    int32_t actualDelta = std::min(msg.size, it->second.size);
    updateLevel(it->second.price, -actualDelta, it->second.direction);
    it->second.size -= actualDelta;
    if (it->second.size <= 0)
      orderMap.erase(it);
  }

  void handleDelete(const MDMessage &msg) {
    auto it = orderMap.find(msg.orderId);
    if (it == orderMap.end())
      return;
    updateLevel(it->second.price, -it->second.size, it->second.direction);
    orderMap.erase(it);
  }

  inline void updateLevel(uint64_t price, int32_t delta, int32_t direction) {
    if (price == 0 || price >= MAX_PRICE)
      return;
    if (direction == 1) {
      bidVol[price] += delta;
      if (bidVol[price] > 0)
        activeBids.insert(price);
      else {
        bidVol[price] = 0;
        activeBids.erase(price);
      }
    } else {
      askVol[price] += delta;
      if (askVol[price] > 0)
        activeAsks.insert(price);
      else {
        askVol[price] = 0;
        activeAsks.erase(price);
      }
    }
  }

  void rebuildSnapshot() {
    int i = 0;
    for (auto it = activeBids.begin(); it != activeBids.end() && i < 10;
         ++it, ++i) {
      currentSnapshot.bidPrice[i] = *it;
      currentSnapshot.bidSize[i] = bidVol[*it];
    }
    while (i < 10) {
      currentSnapshot.bidPrice[i] = 0;
      currentSnapshot.bidSize[i] = 0;
      i++;
    }

    i = 0;
    for (auto it = activeAsks.begin(); it != activeAsks.end() && i < 10;
         ++it, ++i) {
      currentSnapshot.askPrice[i] = *it;
      currentSnapshot.askSize[i] = askVol[*it];
    }
    while (i < 10) {
      currentSnapshot.askPrice[i] = 0;
      currentSnapshot.askSize[i] = 0;
      i++;
    }
  }
};

#endif