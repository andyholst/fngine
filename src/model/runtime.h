#pragma once

// ModelRuntime owns everything one loaded model needs: pack, dense weights, expert placement,
// per-sequence state, workspace, the host-mapped GPU<->CPU exchange, the CPU expert pool, the
// n-gram store and the Forward composer. It runs eager prefill chunks; decode/verify graphs are
// built on top of it by the step graph (model/step_graph.h).
#include "core/arena.h"
#include "core/host_memory.h"
#include "cpu/expert_pool.h"
#include "io/ngram_store.h"
#include "io/pack.h"
#include "kernels/linear/linear.h"
#include "kernels/qsa/qsa.h"
#include "model/config.h"
#include "model/adaptive_experts.h"
#include "model/forward.h"
#include "model/placement.h"
#include "model/state.h"
#include "model/weights.h"
#include "model/workspace.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace fngine::model {

struct RuntimeOptions {
  std::filesystem::path pack;
  int32_t max_context = 131072;
  int32_t prefill_chunk = 4096;
  kernels::QsaKvFormat kv = kernels::QsaKvFormat::K8V8;
  bool dense_bf16 = false;
  kernels::Fp8LinearPath fp8_path = kernels::Fp8LinearPath::Auto;
  int cpu_workers = 0;  // 0: every worker core of the CPU placement policy; N: the first N
  uint64_t vram_reserve_mib = 512;
  uint64_t ngram_cache_mib = 1024;
  bool mtp = true;                     // load MTP experts and maintain the MTP KV cache
  int32_t logit_rows = 16;             // >= 16; validation uses more
  std::optional<uint64_t> expert_vram_cap;  // bytes (tests / sharing the GPU)
  // In-session adaptive expert swaps (one spare GPU slot per layer), planned from the routing of
  // recent decode/verify steps (SwapOptions).
  bool adaptive_experts = true;
  // Persist routing counts to <pack>/routing-stats.bin (flush on destruction and every 10 min).
  bool persist_routing_stats = false;
  // Prefill: stream the warmest CPU-resident experts of each layer to the GPU over PCIe (M10).
  bool prefill_stream = true;
  bool prefill_borrow = true;          // lend cold-ranked GPU slots to prompt scratch/ring
  // Apply the pack's optional ablation section (abliterated packs); false = control run.
  bool ablation = true;
  // Decode/verify: next-layer dense weights prefetched into L2 during each layer's CPU-tier wait
  // (MiB per layer; 0 disables). FNGINE_L2_PREFETCH_MIB overrides it.
  uint64_t l2_prefetch_mib = 80;  // best of a 0..96 MiB sweep on decode (8.92 vs 9.47 ms/step)
};

// Startup resource problems (insufficient host memory, ...) are reported as fngine::StartupError
// (include/fngine/types.h), which the CLI maps to exit code 2.
struct PrefillHooks {
  // Called (after a stream sync) once the embedding (layer = -1) and each main layer finished for
  // the chunk at rows [0, T) = positions [pos0, pos0 + T); X is ws().X (device FP32 [T][4][2560]).
  std::function<void(int32_t layer, int32_t pos0, int32_t T)> after_layer;
  // When set, logits are computed for every position of the chunk in slices of logit_rows and
  // passed as device FP32 [rows][248320] (valid until the callback returns).
  std::function<void(int32_t pos0, int32_t rows, const float* logits)> on_logits;
  // Called after every completed chunk with the new position (e.g. to take session snapshots).
  std::function<void(int32_t position)> after_chunk;
};

class ModelRuntime {
 public:
  explicit ModelRuntime(const RuntimeOptions& options);
  ~ModelRuntime();
  ModelRuntime(const ModelRuntime&) = delete;
  ModelRuntime& operator=(const ModelRuntime&) = delete;

  [[nodiscard]] const RuntimeOptions& options() const noexcept { return options_; }
  [[nodiscard]] const io::Pack& pack() const noexcept { return *pack_; }
  [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
  [[nodiscard]] const DenseWeights& weights() const noexcept { return weights_; }
  [[nodiscard]] const ExpertPlacement& placement() const noexcept { return placement_; }
  [[nodiscard]] ModelState& state() noexcept { return *state_; }
  [[nodiscard]] Workspace& ws() noexcept { return *active_ws_; }
  [[nodiscard]] Forward& forward() noexcept { return *forward_; }
  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] io::NgramStore& ngram() noexcept { return *ngram_; }
  [[nodiscard]] cpu::ExpertPool& pool() noexcept { return *pool_; }
  [[nodiscard]] DecodeExchange* exchange_host() const noexcept;
  [[nodiscard]] DecodeExchange* exchange_device() const noexcept;
  [[nodiscard]] uint8_t* ple_staging_host() const noexcept;
  [[nodiscard]] const uint8_t* ple_staging_device() const noexcept;

  // New sequence: zero all state, position 0, n-gram history at sequence start.
  void reset_sequence();
  // Before work that writes KV at positions < `tokens` (plus a page of margin): elastic expert
  // slots in that part of the context are vacated. Between steps.
  void ensure_context(int32_t tokens);
  [[nodiscard]] int32_t position() const noexcept { return position_; }
  [[nodiscard]] const io::NgramHistory& ngram_history() const noexcept { return history_; }

  // Eager prefill of `tokens` at the current position (chunks of prefill_chunk). When MTP is
  // enabled, the MTP layer runs over every prompt position whose next token is known
  // (`next_token` supplies the token after the last one, if known; otherwise the last position
  // is left for the MTP bridge). Afterwards ws().X retains the final pre-mixer row (or the last
  // chunk in static mode); `last_rows()` gives the valid row count in the current workspace.
  void prefill(std::span<const int32_t> tokens, const PrefillHooks* hooks = nullptr,
               std::optional<int32_t> next_token = std::nullopt);
  [[nodiscard]] int32_t last_rows() const noexcept { return last_rows_; }

  // Next nonzero exchange sequence number.
  uint32_t next_seq();
  // Throws if the GPU flagged an exchange error or the CPU pool failed.
  void check_exchange();

  // Advance position/history after tokens were committed by a decode/verify step.
  void commit(std::span<const int32_t> tokens);

  // Right after enqueueing a decode/verify step on stream(): every SwapOptions::every_steps such
  // steps, enqueues the swap-candidate kernel behind it; starts the swap copies planned at the
  // previous boundary, which then overlap the step.
  void overlap_swaps();
  // Between steps (no graph, prefill chunk or CPU pool step running): completes expert-swap
  // transfers (slot-map changes) and queues swaps from the candidates of the last step. Called by
  // StepRunner::step and after every prefill chunk.
  void step_boundary();
  // Copies the routing counters to the host statistics (e.g. between requests).
  void collect_routing_stats();
  // Raw cumulative device routing counters u32 [48][512] (wrapping; synchronizes the stream).
  [[nodiscard]] std::vector<uint32_t> routing_counters();
  // Validation/tests: queues `n` random expert swaps and runs them to completion now (between
  // steps). Returns the number of completed swaps so far.
  uint64_t force_random_swaps(int32_t n, uint64_t seed);
  [[nodiscard]] uint64_t expert_swaps() const noexcept { return swapper_ ? swapper_->completed() : 0; }
  [[nodiscard]] bool adaptive_experts() const noexcept { return swapper_ && swapper_->usable(); }
  [[nodiscard]] RoutingStats& routing_stats() noexcept { return *routing_stats_; }

  // Session snapshots: the non-position-addressed sequence state (GDN recurrent + conv state,
  // PLE conv state, indexer raw-key rings) plus position and n-gram history. KV caches and
  // indexer block keys are position-addressed and stay valid below the snapshot position as long
  // as the token prefix is unchanged.
  [[nodiscard]] std::size_t snapshot_bytes() const;
  // Copies the state into `host` (pinned, snapshot_bytes()) — synchronous.
  void snapshot_to(void* host) const;
  // Restores a snapshot taken at `position` with n-gram history `history` — synchronous.
  void restore_from(const void* host, int32_t position, const io::NgramHistory& history);

  [[nodiscard]] std::string summary() const;
  [[nodiscard]] uint64_t fixed_vram_bytes() const noexcept { return fixed_vram_; }
  [[nodiscard]] PrefillStream& prefill_stream() noexcept { return *prefill_stream_; }

 private:
  void prefill_chunk(const int32_t* tokens, int32_t T, bool has_next, int32_t next,
                     const PrefillHooks* hooks);
  void begin_prefill_workspace(std::size_t prompt_tokens);
  void end_prefill_workspace(bool preserve_output);

  RuntimeOptions options_;
  std::unique_ptr<io::Pack> pack_;
  ModelConfig config_;
  cudaStream_t stream_ = nullptr;
  cudaStream_t prefetch_stream_ = nullptr;  // lowest priority; L2 prefetch graph branches
  cudaStream_t aux_stream_ = nullptr;       // decode/verify parallel branches inside a layer
  DenseWeights weights_;
  ExpertPlacement placement_;
  std::unique_ptr<ModelState> state_;
  std::unique_ptr<WorkspaceBuffer> workspace_;
  Workspace borrowed_workspace_{};
  WorkspaceOptions prefill_wsopt_{};  // full prefill chunk; per-prompt views shrink max_tokens
  uint64_t borrowed_bytes_ = 0;       // bytes of the current borrowed view
  Workspace* active_ws_ = nullptr;
  std::unique_ptr<kernels::CublasLtContext> cublas_;
  MappedHostBuffer exchange_buf_;
  MappedHostBuffer prefill_buf_;
  MappedHostBuffer ple_buf_;
  std::unique_ptr<PinnedHostBuffer> input_staging_;
  PrefillExchangeView prefill_host_{};
  PrefillExchangeView prefill_dev_{};
  DeviceBuffer routing_counts_;  // u32 [48][512]
  DeviceBuffer ablation_;        // f32 [48][2560] refusal directions (abliterated packs)
  std::unique_ptr<RoutingStats> routing_stats_;
  std::unique_ptr<ExpertSwapper> swapper_;
  int32_t steps_since_plan_ = 0;  // decode/verify steps since the last swap planning round
  std::unique_ptr<PrefillStream> prefill_stream_;
  std::unique_ptr<cpu::ExpertPool> pool_;
  std::unique_ptr<io::NgramStore> ngram_;
  std::unique_ptr<Forward> forward_;
  io::NgramHistory history_{};
  int32_t position_ = 0;
  int32_t last_rows_ = 0;
  int32_t context_ = 0;  // positions the sequence may hold without vacating elastic slots
  int32_t last_mtp_rows_ = 0;
  uint32_t seq_ = 0;
  uint64_t fixed_vram_ = 0;
};

}  // namespace fngine::model
