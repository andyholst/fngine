#pragma once

// Persistent routing telemetry (<pack>/routing-stats.bin) and in-session adaptive expert swaps
// between the GPU slot arrays and the RAM tier. Both run on the engine thread between steps.
#include "core/arena.h"
#include "core/host_memory.h"
#include "model/placement.h"
#include "model/routing_counts.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <utility>
#include <vector>

namespace fngine::model {

// Accumulates the device routing counters (u32 [48][512], incremented by
// kernels::moe_select_publish for every decode/verify routed pair) into host
// u64 counts and persists them: stored = stored * 0.5 + new at every flush.
class RoutingStats {
 public:
  static constexpr double kDecay = 0.5;
  static constexpr std::chrono::seconds kFlushInterval{600};

  // `file` empty: counts are collected but never written.
  RoutingStats(std::filesystem::path file, const uint32_t* device_counts);

  // Copies the device counters (ordered on `s`, synchronizes it) and returns the counts added
  // since the previous collect() [48*512]. Flushes pending counts after 10 minutes, including
  // when this collection adds no counts (e.g. an idle server maintenance tick).
  const std::vector<uint64_t>& collect(cudaStream_t s);
  // Writes the decayed file (<file>.part + rename) and clears the pending counts. Returns false
  // (and logs) on I/O failure; pending counts are kept then.
  bool flush();

  [[nodiscard]] uint64_t pending_total() const noexcept;
  [[nodiscard]] const std::filesystem::path& file() const noexcept { return file_; }

 private:
  std::filesystem::path file_;
  const uint32_t* device_ = nullptr;
  PinnedHostBuffer host_;
  std::vector<uint32_t> last_;
  std::vector<uint64_t> delta_;
  std::vector<uint64_t> pending_;
  std::chrono::steady_clock::time_point last_flush_;
};

// Planning follows decode steps. Every `every_steps` steps a kernel folds the device routing
// counters into a per-step decayed EMA (the placement's GPU experts start at `prior`) and picks
// each layer's hottest CPU expert and coldest GPU expert (experts in prefill loan slots
// included); of the idle layers whose pair clears `min_ratio`, the `max_per_plan` largest EMA
// gains start.
struct SwapOptions {
  int32_t every_steps = 1;   // decode/verify steps per planning round
  int32_t max_per_plan = 4;  // swaps started per round, at most one per layer (<= 48)
  double decay = 0.999;      // per-step decay of the routing-count EMA
  double min_ratio = 2.0;    // swap b (CPU) for a (GPU) only when ema[b] > ema[a] * min_ratio
  double prior = 1.0;        // starting EMA of the placement's GPU experts
};

// Moves a hot CPU expert b into a GPU slot of its layer, the cold GPU expert a out:
//   a in a regular slot: H2D b into the layer's spare; next boundary, upload complete:
//     slot_of[b] = spare and a's blob is copied D2H into b's former RAM blob; next boundary,
//     download complete: a is CPU-computed from that blob and its slot becomes the spare.
//   a in a backed (loan or elastic) slot: at the boundary that starts the swap a falls back to its
//     RAM backup; H2D b into a's slot; next boundary, upload complete: slot_of[b] = that slot, b's
//     RAM blob becomes its backup.
//   no a (fill): H2D b into a free backed slot (elastic slots freed when the context shrank).
// Slot maps change only at boundaries; the copies start in start(), which may overlap a step
// (target slots are unreferenced, and a stays resident while it is read). One swap per layer is
// in progress at a time; layers proceed in parallel. Unregistered RAM uses placement's shared pair
// of pinned staging buffers in both directions.
class ExpertSwapper {
 public:
  // `routing_counts`: the device u32 [48*512] counters of moe_select_publish.
  ExpertSwapper(ExpertPlacement& placement, cudaStream_t compute, const uint32_t* routing_counts,
                const SwapOptions& options);
  ~ExpertSwapper();
  ExpertSwapper(const ExpertSwapper&) = delete;
  ExpertSwapper& operator=(const ExpertSwapper&) = delete;

  // False when the placement has no spare slots.
  [[nodiscard]] bool usable() const noexcept { return usable_; }
  [[nodiscard]] const SwapOptions& options() const noexcept { return options_; }

  // Enqueues on the compute stream the EMA update with the counters' growth over the `steps`
  // decode steps since the previous call and the per-layer candidate selection.
  void enqueue_plan(int32_t steps);
  // Step boundary (compute, CPU workers and prefill copies idle, prompt borrowing restored):
  // completes finished transfers and, after enqueue_plan, replaces the queued swaps that have not
  // started with up to max_per_plan of its candidates. Rejects a busy compute stream. Returns true
  // on a slot-map change.
  bool boundary();
  // Starts queued uploads and pending downloads; also while a step runs.
  void start();
  // boundary() requirements: completes finished transfers and starts the next ones.
  bool advance();
  // Queues one swap of CPU expert b with GPU expert a in `layer` (a = -1: fill a free backed slot).
  void queue(int32_t layer, int32_t b, int32_t a);
  // Queues `n` random (b, a) swaps over random layers (validation / tests).
  void queue_random(int32_t n, uint64_t seed);
  // Same boundary requirements as advance(). Runs every queued and in-flight swap to completion
  // (synchronizes the copy stream). Returns true when a slot map changed.
  bool drain();
  // Between steps (also while prompt storage is borrowed), before the sequence may write `tokens`
  // positions: cancels swaps into elastic slots the context takes, then ExpertPlacement::vacate.
  // Returns true on a slot-map change.
  bool vacate(int32_t tokens);

  [[nodiscard]] uint64_t completed() const noexcept { return completed_; }
  [[nodiscard]] bool idle() const noexcept;

 private:
  // UploadPending -> Upload (b -> target) -> [regular target: DownloadPending -> Download (a)] -> Idle.
  enum class Stage : uint8_t { Idle, UploadPending, Upload, DownloadPending, Download };
  struct LayerSwap {
    Stage stage = Stage::Idle;
    int32_t b = -1, a = -1;
    int32_t target = -1;  // slot b is uploaded into
    bool backed = false;  // target is a backed slot (else the spare)
    std::byte* ram = nullptr;  // a's download destination
    cudaEvent_t event = nullptr;
    std::deque<std::pair<int32_t, int32_t>> queue;  // (b, a)
  };
  // Boundary transitions: finished copies, then the next request of an idle layer. True while
  // the layer has a swap in progress.
  bool complete_layer(int32_t layer, bool& changed);
  // Copy starts; true while the layer has a swap in progress.
  bool start_layer(int32_t layer);
  void finish_layer(LayerSwap& ls, bool completed);
  void queue_candidates();
  void require_boundary() const;

  ExpertPlacement& placement_;
  cudaStream_t compute_ = nullptr;
  cudaStream_t copy_ = nullptr;
  const uint32_t* counts_ = nullptr;
  SwapOptions options_;
  bool usable_ = false;
  bool planned_ = false;  // enqueue_plan's candidates await boundary()
  std::array<LayerSwap, 48> layers_{};
  DeviceBuffer stats_;  // float ema [48*512], then u32 last counters [48*512]
  MappedHostBuffer candidates_;  // SwapCandidate [48]
  uint64_t completed_ = 0;
};

}  // namespace fngine::model
