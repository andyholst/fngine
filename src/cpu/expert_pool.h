#pragma once

// CPU expert pool: pinned worker threads that compute the cold (CPU-resident) routed experts of
// every MoE layer through the host-mapped GPU<->CPU exchange (core/exchange.h).
//
// Step protocol (one decode/verify step or one prefill chunk, sequence number `seq` != 0):
//   engine:  pool.begin(seq, mode)  -> wakes the workers (they park on a futex after
//            PoolConfig::park_after_ms without a step), then launches the GPU step.
//   workers: for l = 0..47 in order, acquire-spin (_mm_pause) on control.ready[l] == seq.
//            Decode, n_cold == 0: advance without writing done[l] (the GPU skips its wait).
//            Prefill, n_cold == 0: release-store done[l] = seq once every worker has read the
//            header (the shared region must not be rewritten for layer l+1 before that; the GPU
//            always waits in prefill mode).
//            otherwise: compute and the last finisher release-stores control.done[l] = seq.
//   Decode mode reads DecodeExchange::publish[l] / writes result[l] (rows = step tokens; y is
//   written for every token < T, zero for tokens without cold pairs). Prefill mode reads the
//   compacted PrefillExchangeView (rows = header[1]) and writes y for every compacted row.
//   Workers abandon the step when control.abort != 0 (call wait_idle() before reusing).
//
// Per cold layer: quantize + pack each published row's BF16 x once (Q8_K or Q8_0 per the
// gate/up format, iq_cpu.h layout); group cold (row, expert) pairs by expert id.
// Phase A: tasks (expert, 32 intermediate rows, <= 128 of the expert's tokens), enumerated
// largest expert first so the long tail is made of small tasks; each task decodes its gate/up
// rows 4 + 4 at a time and runs the register-blocked GEMM over its tokens -> h = silu(g) * u ->
// one packed Q8_0 block of h per token. Spin barrier.
// Phase B: 80 tasks of 32 output rows, each over every cold expert in ascending id order
// (down rows decoded 8 at a time, all of the expert's tokens per GEMM):
// y[row][n] = sum_pairs w * down_n . h (deterministic order, independent of thread count).
#include "core/exchange.h"
#include "core/model_dims.h"
#include "cpu/iq_cpu.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fngine::cpu {

// Layout of one layer's expert blobs (pack experts.bin): gate | up | down rows, GGUF bytes.
struct ExpertLayerDesc {
  WeightFormat gu_format = WeightFormat::IQ2_S;   // gate/up rows: [640][2560]
  WeightFormat down_format = WeightFormat::IQ4_NL;  // down rows: [2560][640]; must be Q8_0-family
  std::size_t gate_offset = 0;  // byte offsets inside a blob
  std::size_t up_offset = 0;
  std::size_t down_offset = 0;
};

enum class PoolMode : uint32_t { Decode = 1, Prefill = 2 };

struct PoolConfig {
  // One worker per entry, pinned to that logical CPU (-1 = unpinned).
  std::vector<int> worker_cpus = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};
  std::array<ExpertLayerDesc, kExchangeLayers> layers{};
  // [kExchangeLayers * 512]: host blob pointer of (layer, expert), nullptr when not on the host.
  // Caller-owned; may be rewritten between steps (never while a step runs).
  const void* const* host_slot_of = nullptr;
  DecodeExchange* decode = nullptr;  // host-mapped; its control block is used in both modes
  PrefillExchangeView prefill{};     // capacity 0 -> prefill mode unavailable
  int park_after_ms = 50;
};

struct LayerTiming {
  uint64_t steps = 0;       // steps in which this layer had cold pairs
  uint64_t rows = 0;        // published rows (tokens) summed over those steps
  uint64_t pairs = 0;       // cold (row, expert) pairs
  uint64_t experts = 0;     // distinct cold experts
  uint64_t wait_ns = 0;     // worker 0: from reaching the layer to ready[l] (all steps)
  uint64_t compute_ns = 0;  // worker 0 seeing ready[l] -> done[l] stored
};

class ExpertPool {
 public:
  explicit ExpertPool(const PoolConfig& cfg);  // throws std::invalid_argument on a bad config
  ~ExpertPool();
  ExpertPool(const ExpertPool&) = delete;
  ExpertPool& operator=(const ExpertPool&) = delete;

  // Starts step `seq` (nonzero, different from the previous one). Waits for the previous step's
  // workers to finish first (they finish as soon as the GPU published all 48 ready cells).
  void begin(uint32_t seq, PoolMode mode);
  // Blocks until every worker finished (or abandoned) the current step.
  void wait_idle();
  // True once a step hit an inconsistent publish (expert id out of range, missing host blob,
  // too many rows). The layer still completes (that pair contributes zero) so the GPU never
  // hangs; the engine must treat the step as failed.
  [[nodiscard]] bool failed() const;
  [[nodiscard]] std::string failure() const;

  [[nodiscard]] int workers() const { return n_workers_; }
  [[nodiscard]] std::array<LayerTiming, kExchangeLayers> timings() const;
  void reset_timings();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int n_workers_ = 0;
};

}  // namespace fngine::cpu
