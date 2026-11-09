#pragma once
// Host stand-in for lib/hal/HalMemory.h: "PSRAM" is the ordinary heap.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

class HalMemory {
 public:
  struct PsramDeleter {
    void operator()(uint8_t* buffer) const { delete[] buffer; }
  };
  using PsramBuffer = std::unique_ptr<uint8_t[], PsramDeleter>;
  static PsramBuffer allocatePsram(size_t bytes) { return PsramBuffer(new (std::nothrow) uint8_t[bytes]); }
};
