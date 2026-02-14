// engine/SharedProtocol.hpp
#ifndef SHARED_PROTOCOL_HPP
#define SHARED_PROTOCOL_HPP

#include <cstdint>

#pragma pack(push, 1)

// MDMessage (36 Bytes) - Engine Input
struct MDMessage {
  double time;
  int32_t eventType;
  uint64_t orderId;
  int32_t size;
  uint64_t price;
  int32_t direction;
} __attribute__((packed));

// LOBSnapshot (248 Bytes) - MBP-10 for DeepLOB
struct LOBSnapshot {
  double time;
  uint64_t bidPrice[10];
  int32_t bidSize[10];
  uint64_t askPrice[10];
  int32_t askSize[10];
} __attribute__((packed));

// EngineState (32 Bytes) - BBO for validation
struct EngineState {
  double lastTime;
  uint64_t bidPrice;
  int32_t bidSize;
  uint64_t askPrice;
  int32_t askSize;
} __attribute__((packed));

// TruthMessage (32 Bytes) - Ground truth
struct TruthMessage {
  double time;
  uint64_t bidPrice;
  int32_t bidSize;
  uint64_t askPrice;
  int32_t askSize;
} __attribute__((packed));

#pragma pack(pop)
/*static_assert: These are safety checks.
If you accidentally change a struct so that its size is no longer what you
expect (e.g., changing int32_t to int64_t), the code will refuse to compile.
This prevents weird runtime bugs.*/
static_assert(sizeof(MDMessage) == 36, "MDMessage must be 36 bytes");
static_assert(sizeof(LOBSnapshot) == 248, "LOBSnapshot must be 248 bytes");
static_assert(sizeof(EngineState) == 32, "EngineState must be 32 bytes");
static_assert(sizeof(TruthMessage) == 32, "TruthMessage must be 32 bytes");

#endif