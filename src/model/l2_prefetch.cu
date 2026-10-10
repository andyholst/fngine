#include "model/l2_prefetch.h"

namespace fngine::model {
namespace {

constexpr uint64_t kPiece = 32u << 10;  // bytes per bulk prefetch
constexpr int kThreads = 64;
constexpr int kBlocks = 32;

__global__ void __launch_bounds__(kThreads)
l2_prefetch_kernel(const L2PrefetchList* __restrict__ list, uint64_t budget, const int32_t* gate) {
  if (gate && *gate <= 0) return;
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(policy));
  const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t nthreads = uint64_t(gridDim.x) * blockDim.x;
  const int n = list->count;
  uint64_t issued = 0;      // bytes of the list covered so far
  uint64_t first_piece = 0; // global index of the current range's first piece
  for (int r = 0; r < n && issued < budget; ++r) {
    const uintptr_t lo = (reinterpret_cast<uintptr_t>(list->ranges[r].ptr) + 15) & ~uintptr_t{15};
    const uintptr_t hi = (reinterpret_cast<uintptr_t>(list->ranges[r].ptr) + list->ranges[r].bytes) &
                         ~uintptr_t{15};
    if (hi <= lo) continue;
    uint64_t bytes = hi - lo;
    if (bytes > budget - issued) bytes = (budget - issued) & ~uint64_t{15};
    const uint64_t pieces = (bytes + kPiece - 1) / kPiece;
    for (uint64_t i = (tid + nthreads - first_piece % nthreads) % nthreads; i < pieces; i += nthreads) {
      const uint64_t off = i * kPiece;
      const uint32_t len = uint32_t(bytes - off < kPiece ? bytes - off : kPiece);
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(lo + off),
                   "r"(len), "l"(policy)
                   : "memory");
    }
    first_piece += pieces;
    issued += bytes;
  }
}

}  // namespace

cudaError_t l2_prefetch(const L2PrefetchList* list, uint64_t budget, const int32_t* gate,
                        cudaStream_t stream) {
  if (!list || budget == 0) return cudaSuccess;
  l2_prefetch_kernel<<<kBlocks, kThreads, 0, stream>>>(list, budget, gate);
  return cudaGetLastError();
}

}  // namespace fngine::model
