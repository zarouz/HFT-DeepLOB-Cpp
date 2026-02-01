#ifndef SHARED_PROTOCOL_HPP
#define SHARED_PROTOCOL_HPP

#include <cstdint>

// FORCE 1-BYTE ALIGNMENT (No Padding)
#pragma pack(push, 1)

// 1. INPUT: Raw Order (36 Bytes)
// Python: struct.pack('diQiqi', ...)
struct MDMessage {
  double time;             // 8 bytes
  int eventType;           // 4 bytes
  uint64_t orderId;        // 8 bytes
  int size;                // 4 bytes
  uint64_t price;          // 8 bytes
  int direction;           // 4 bytes
} __attribute__((packed)); // GCC/Clang specific safety

// 2. OUTPUT: Engine State (32 Bytes)
struct EngineState {
  double lastTime;
  uint64_t bidPrice;
  int bidSize;
  uint64_t askPrice;
  int askSize;
} __attribute__((packed));

// 3. TRUTH: Reference Snapshot (32 Bytes)
// Python: struct.pack('dQiqi', ...)
struct TruthMessage {
  double time;       // 8 bytes
  uint64_t bidPrice; // 8 bytes
  int bidSize;       // 4 bytes
  uint64_t askPrice; // 8 bytes
  int askSize;       // 4 bytes
} __attribute__((packed));

#pragma pack(pop)

#endif