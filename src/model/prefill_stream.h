#pragma once

// Prefill expert streaming (plan M10): during prefill, the warmest CPU-resident experts of each
// layer are copied over PCIe into a VRAM ring and computed by the GPU grouped GEMMs, while the
// CPU pool keeps the rarer ones. The split per layer comes from a startup cost model:
//   B_pcie  pinned RAM-tier -> device bytes/s (128 MiB copy),
//   B_cpu   BF16-dequantized CPU expert weight bytes/s (32 real cold experts, T=1),
//   F_cpu   marginal CPU BF16 GEMM FLOP/s per routed token (same experts at T=128).
// Both CPU measurements use the prompt kernels, not decode's Q8/VNNI arithmetic.
// For a chunk of T tokens, layer l's CPU-resident experts are sorted by prompt routing frequency
// (the pack's prefill profile; placement score without one), descending, and S_l is the longest
// prefix (<= kRingSlots experts) with
//   sum_S bytes / B_pcie <= sum_rest (P_e * bytes / B_cpu + E_e * flops / F_cpu) + T * kCopyOverlap,
// E_e = frequency share of e within the layer * T * 10 (expected routed tokens) and
// P_e = 1 - exp(-E_e) (probability that the CPU has to read the expert at all). The copies also
// overlap the layer's GPU attention/dense work, which the CPU cannot; kCopyOverlap (seconds per
// token) is the measured optimum of that allowance (flat between 1 and 3 us/token at T = 4096).
//
// Device side: per-layer tables expert_blob[512] (GPU slot address for resident experts, ring
// slot address for S_l, nullptr = CPU) consumed by moe_route and the grouped GEMMs, plus a fixed
// table for the MTP experts. The ring has kRingBuffers buffers x kRingSlots blob slots; layer l
// uses buffer l % kRingBuffers. Copies run on a low-priority stream: a buffer is refilled only
// after the layer that last used it finished (event recorded after its combine), and the main
// stream waits immediately before grouped GEMM (routing/shared computation can overlap copies).
#include "io/pack.h"
#include "model/placement.h"
#include "model/weights.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace fngine::model {

struct PrefillStreamOptions {
  bool enabled = true;             // false: tables only (resident experts), no ring, no copies
  std::vector<int> worker_cpus;    // CPU expert workers (cost model threads are pinned there)
  // Optional placement loan. It stays expert storage outside prefill; construction must not write it.
  void* ring_memory = nullptr;
  uint64_t ring_memory_bytes = 0;
};

struct PrefillCostModel {
  double pcie_bytes_per_s = 0;  // B_pcie
  double cpu_bytes_per_s = 0;   // B_cpu
  double cpu_flops_per_s = 0;   // F_cpu (marginal, per routed token)
};

class PrefillStream {
 public:
  // Two buffers: a layer's copies run while the previous layer computes. 192 slots cover the
  // largest per-layer stream set the planner picks at 4096-token chunks (~170).
  static constexpr int kRingBuffers = 2;
  static constexpr int kRingSlots = 192;
  static constexpr double kCopyOverlap = 2.5e-6;  // seconds per chunk token (see above)

  // Device bytes of the ring (0 when disabled); loan-backed by default, fixed in --no-borrow.
  static uint64_t ring_bytes(const io::Pack& pack, bool enabled);

  // Builds the tables and uses the supplied loan (or allocates a static ring), creates the copy
  // stream and measures the cost model. Pageable RAM uses placement's pinned bounce buffers.
  PrefillStream(const io::Pack& pack, const ExpertPlacement& placement, const ExpertBlobs& mtp,
                const PrefillStreamOptions& options);
  ~PrefillStream();
  PrefillStream(const PrefillStream&) = delete;
  PrefillStream& operator=(const PrefillStream&) = delete;

  // Marks the tables stale after slot_of / host blobs changed (between steps); the next
  // begin_chunk re-derives the plan and uploads the tables. Decode never reads them.
  void invalidate() noexcept { plan_tokens_ = 0; }

  // Chunk protocol (host side, all launches on `main`):
  //   begin_chunk(T, main)          plans S_l for T (cached per T until invalidate) and issues the first copies
  //   before layer l: prefetch(l)   issues copies of layers up to l + kRingBuffers - 1
  //   Forward::moe(l): wait_ready(l, main) before grouped GEMM, release(l, main) after combine
  void begin_chunk(int32_t T, cudaStream_t main);
  void prefetch(int32_t layer);
  void wait_ready(int32_t layer, cudaStream_t main);
  void release(int32_t layer, cudaStream_t main);
  // Quiesce every host read before swaps recycle RAM blobs or a prompt restores the loan.
  void synchronize();
  // Ring bytes written by copies since the previous call (one range per used buffer), then
  // restarts the count: a borrowed loan restores only the expert slots these overlap.
  std::vector<DeviceRange> take_ring_written();

  // Device [512] blob tables.
  [[nodiscard]] const uint8_t* const* table(int32_t layer) const noexcept;
  [[nodiscard]] const uint8_t* const* mtp_table() const noexcept;

  [[nodiscard]] bool streaming() const noexcept { return ring_ != nullptr; }
  [[nodiscard]] const PrefillCostModel& cost_model() const noexcept { return cost_; }
  // Streamed experts / bytes of the current plan.
  [[nodiscard]] int32_t streamed_experts() const noexcept;
  [[nodiscard]] uint64_t streamed_bytes() const noexcept;
  [[nodiscard]] std::string summary() const;

 private:
  struct Copy {
    const void* src;
    void* dst;
    uint64_t bytes;
  };
  struct LayerPlan {
    std::vector<int32_t> experts;  // S_l, descending score
    std::vector<Copy> copies;
    uint64_t bytes = 0;
    double cpu_only_s = 0;  // predicted CPU time of all CPU experts (no streaming)
    double cpu_rest_s = 0;  // predicted CPU time of the experts left on the CPU
    double pcie_s = 0;      // predicted copy time of S_l
  };

  void measure_cost_model(const std::vector<int>& worker_cpus);
  void plan_for(int32_t T);
  void upload_tables();

  const io::Pack* pack_ = nullptr;
  const ExpertPlacement* placement_ = nullptr;
  ExpertBlobs mtp_{};
  PrefillCostModel cost_{};
  uint64_t slot_bytes_ = 0;
  DeviceBuffer ring_buf_;
  uint64_t ring_size_ = 0;
  uint8_t* ring_ = nullptr;
  DeviceBuffer tables_;  // [49][512] pointers: 48 layers, then MTP
  std::vector<const uint8_t*> host_tables_;
  cudaStream_t copy_stream_ = nullptr;
  std::array<cudaEvent_t, kRingBuffers> copy_done_{};
  std::array<cudaEvent_t, kRingBuffers> buffer_free_{};
  std::array<bool, kRingBuffers> buffer_used_{};
  std::array<LayerPlan, 48> plan_{};
  int32_t plan_tokens_ = 0;  // T of plan_ (0 = none)
  int32_t issued_ = 0;       // layers whose copies were issued in the current chunk
  std::array<std::size_t, kRingBuffers> ring_written_{};  // slots written per buffer
};

}  // namespace fngine::model
