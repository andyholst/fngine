#pragma once

// Next-layer dense-weight prefetch into L2. A decode/verify layer waits for the CPU expert tier
// after routing; DRAM is otherwise idle then, so a parallel graph branch pulls the next layer's
// weights (in consumption order) into the 96 MiB L2 and the next layer's projections read them at
// L2 bandwidth. Prefetches are hints: correctness never depends on them.
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::model {

struct L2Range {
  const void* ptr = nullptr;
  uint64_t bytes = 0;
};

inline constexpr int kMaxL2Ranges = 40;

struct L2PrefetchList {
  L2Range ranges[kMaxL2Ranges];
  int32_t count = 0;
};

// Prefetches list->ranges in order, at most `budget` bytes, when gate is null or *gate > 0.
// Ranges are trimmed to 16-byte boundaries.
cudaError_t l2_prefetch(const L2PrefetchList* list, uint64_t budget, const int32_t* gate,
                        cudaStream_t stream);

}  // namespace fngine::model
