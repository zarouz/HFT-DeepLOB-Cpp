#ifndef LOCKFREEQUEUE_HPP
#define LOCKFREEQUEUE_HPP

#include <atomic>
#include <vector>

template <typename T, size_t Size> class LockFreeQueue {
private:
  std::vector<T> buffer;
  alignas(64) std::atomic<size_t> head{0};
  alignas(64) std::atomic<size_t> tail{0};

public:
  LockFreeQueue() : buffer(Size) {}

  bool push(const T &item) {
    const size_t currentHead = head.load(std::memory_order_relaxed);
    const size_t nextHead = (currentHead + 1) % Size;
    if (nextHead == tail.load(std::memory_order_acquire))
      return false;
    buffer[currentHead] = item;
    head.store(nextHead, std::memory_order_release);
    return true;
  }

  bool pop(T &item) {
    const size_t currentTail = tail.load(std::memory_order_relaxed);
    if (currentTail == head.load(std::memory_order_acquire))
      return false;
    item = buffer[currentTail];
    tail.store((currentTail + 1) % Size, std::memory_order_release);
    return true;
  }
};

#endif