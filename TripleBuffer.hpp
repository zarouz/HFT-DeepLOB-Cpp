#ifndef TRIPLE_BUFFER_HPP
#define TRIPLE_BUFFER_HPP

#include <atomic>
#include <cstring>

template <typename T> class TripleBuffer {
private:
  alignas(64) T buffers[3];

  alignas(64) std::atomic<uint8_t> writeIdx{0};
  alignas(64) std::atomic<uint8_t> readIdx{1};
  alignas(64) std::atomic<uint8_t> middleIdx{2};

  alignas(64) std::atomic<bool> newData{false};

public:
  TripleBuffer() { memset(buffers, 0, sizeof(buffers)); }

  T *startWrite() { return &buffers[writeIdx.load(std::memory_order_relaxed)]; }

  void finishWrite() {
    uint8_t currentWrite = writeIdx.load(std::memory_order_relaxed);
    uint8_t currentMiddle = middleIdx.load(std::memory_order_relaxed);

    writeIdx.store(currentMiddle, std::memory_order_release);
    middleIdx.store(currentWrite, std::memory_order_release);

    newData.store(true, std::memory_order_release);
  }

  bool hasNewData() const { return newData.load(std::memory_order_acquire); }

  const T *read() {
    if (newData.load(std::memory_order_acquire)) {
      uint8_t currentRead = readIdx.load(std::memory_order_relaxed);
      uint8_t currentMiddle = middleIdx.load(std::memory_order_acquire);

      readIdx.store(currentMiddle, std::memory_order_release);
      middleIdx.store(currentRead, std::memory_order_release);

      newData.store(false, std::memory_order_release);
    }
    return &buffers[readIdx.load(std::memory_order_relaxed)];
  }
};

#endif