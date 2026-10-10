#pragma once

// Device-side helpers of adaptive expert placement: stream-ordered, allocation-free slot-table
// updates and swap-candidate selection. (Decode/verify routing is counted by
// kernels::moe_select_publish into device u32 [48][512] counters; prompt routing is not counted,
// because expert placement optimizes the decode traffic, which routes differently from prompt text.)
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::model {

// (index, value) pairs for one launch: two map changes per layer and swap boundary.
struct SlotWrites {
  static constexpr int kMax = 96;
  int32_t n = 0;
  int32_t index[kMax];
  int32_t value[kMax];
};

// table[index[i]] = value[i] for i < writes.n, ordered on `s`. The pairs travel as a kernel
// parameter, so no host buffer has to outlive the call.
cudaError_t set_slot_entries(int32_t* table, const SlotWrites& writes, cudaStream_t s);

// A layer's hottest CPU expert b and coldest GPU expert a by routing EMA (-1: none).
struct SwapCandidate {
  int32_t b, a;
  float ema_b, ema_a;
};

// For every (layer, expert) of the 48 x 512: ema = ema * keep + (counts - last) (u32 modular
// deltas), last = counts. Then out[l] = the hottest expert with slot_of < 0 and the coldest with
// slot_of >= 0 (ties: lower id). Ordered on `s`; `out` is device-mapped host memory, written by the
// kernel rather than a copy engine (swap transfers occupy those).
cudaError_t swap_candidates(const uint32_t* counts, uint32_t* last, float* ema, const int32_t* slot_of,
                            float keep, SwapCandidate* out, cudaStream_t s);

// Loads the kernels above. Under CUDA lazy loading a kernel's first launch loads its module and
// waits for the device; launched while a step graph spins on the host exchange, that deadlocks.
cudaError_t preload_swap_kernels();

}  // namespace fngine::model
