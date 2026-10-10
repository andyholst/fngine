#include "model/adaptive_experts.h"

#include "core/model_dims.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fngine::model {
namespace {

namespace fs = std::filesystem;
constexpr std::size_t kPairs = std::size_t(dims::kLayers) * dims::kExperts;
constexpr int kMaxPerPlan = dims::kLayers;  // bound on SwapOptions::max_per_plan: one swap per layer

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

uint64_t saturated_add(uint64_t a, uint64_t b) noexcept {
  return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max() : a + b;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// RoutingStats

RoutingStats::RoutingStats(fs::path file, const uint32_t* device_counts)
    : file_(std::move(file)),
      device_(device_counts),
      host_(kPairs * sizeof(uint32_t)),
      last_(kPairs, 0),
      delta_(kPairs, 0),
      pending_(kPairs, 0),
      last_flush_(std::chrono::steady_clock::now()) {
  if (!device_) throw std::invalid_argument("routing stats: null device counters");
}

const std::vector<uint64_t>& RoutingStats::collect(cudaStream_t s) {
  auto* cur = static_cast<uint32_t*>(host_.data());
  check(cudaMemcpyAsync(cur, device_, kPairs * sizeof(uint32_t), cudaMemcpyDeviceToHost, s), "routing counts D2H");
  check(cudaStreamSynchronize(s), "routing counts D2H");
  for (std::size_t i = 0; i < kPairs; ++i) {
    delta_[i] = uint32_t(cur[i] - last_[i]);  // modular: the device counters wrap
    last_[i] = cur[i];
    pending_[i] = saturated_add(pending_[i], delta_[i]);
  }
  if (!file_.empty() && std::chrono::steady_clock::now() - last_flush_ >= kFlushInterval) flush();
  return delta_;
}

uint64_t RoutingStats::pending_total() const noexcept {
  return std::accumulate(pending_.begin(), pending_.end(), uint64_t(0), saturated_add);
}

bool RoutingStats::flush() {
  last_flush_ = std::chrono::steady_clock::now();
  if (file_.empty() || pending_total() == 0) return true;
  std::vector<uint64_t> stored(kPairs, 0);
  std::error_code ec;
  const bool exists = fs::exists(file_, ec);
  if (ec) {
    spdlog::warn("cannot inspect routing stats {}: {}", file_.string(), ec.message());
    return false;
  }
  if (exists) {
    const auto size = fs::file_size(file_, ec);
    if (ec || size != kPairs * sizeof(uint64_t)) {
      spdlog::warn("cannot merge routing stats {}: unreadable or invalid size", file_.string());
      return false;
    }
    std::ifstream in(file_, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(stored.data()), std::streamsize(kPairs * sizeof(uint64_t)))) {
      spdlog::warn("cannot read routing stats {}; keeping pending counts", file_.string());
      return false;
    }
  }
  uint64_t total = 0;
  for (std::size_t i = 0; i < kPairs; ++i) {
    stored[i] = saturated_add(stored[i] / 2, pending_[i]);
    total = saturated_add(total, stored[i]);
  }
  const fs::path part = fs::path(file_.string() + ".part");
  {
    std::ofstream out(part, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(stored.data()), std::streamsize(kPairs * sizeof(uint64_t)));
    out.flush();
    out.close();
    if (!out) {
      spdlog::warn("cannot write routing stats {}", part.string());
      fs::remove(part, ec);
      return false;
    }
  }
  fs::rename(part, file_, ec);
  if (ec) {
    spdlog::warn("cannot rename {} to {}: {}", part.string(), file_.string(), ec.message());
    fs::remove(part, ec);
    return false;
  }
  spdlog::info("routing stats: flushed {} new routed pairs to {} (stored total {})", pending_total(),
               file_.string(), total);
  std::fill(pending_.begin(), pending_.end(), 0);
  return true;
}

// ---------------------------------------------------------------------------------------------
// ExpertSwapper

ExpertSwapper::ExpertSwapper(ExpertPlacement& placement, cudaStream_t compute, const uint32_t* routing_counts,
                             const SwapOptions& options)
    : placement_(placement), compute_(compute), counts_(routing_counts), options_(options) {
  if (options_.every_steps <= 0 || options_.max_per_plan < 0 || options_.max_per_plan > kMaxPerPlan ||
      !std::isfinite(options_.decay) || options_.decay <= 0.0 || options_.decay > 1.0 ||
      !std::isfinite(options_.min_ratio) || options_.min_ratio < 1.0 || !std::isfinite(options_.prior) ||
      options_.prior < 0.0)
    throw std::invalid_argument("expert swaps: invalid period, limit, decay, gain ratio or prior");
  if (!counts_) throw std::invalid_argument("expert swaps: null routing counters");
  const PlacementPlan& plan = placement_.plan();
  usable_ = plan.spares;
  if (!usable_) return;
  check(preload_swap_kernels(), "swap kernels");
  // EMA prior for the placement's GPU experts; the counters' current values are the baseline.
  std::vector<float> ema(kPairs, 0.0f);
  for (std::size_t i = 0; i < kPairs; ++i)
    if (plan.slot_of[i] >= 0) ema[i] = float(options_.prior);
  stats_ = DeviceBuffer(kPairs * (sizeof(float) + sizeof(uint32_t)));
  stats_.copy_from_host(ema.data(), kPairs * sizeof(float));
  check(cudaMemcpyAsync(static_cast<std::byte*>(stats_.p) + kPairs * sizeof(float), counts_,
                        kPairs * sizeof(uint32_t), cudaMemcpyDeviceToDevice, compute_), "swap counter baseline");
  check(cudaStreamSynchronize(compute_), "swap counter baseline");
  candidates_ = MappedHostBuffer(sizeof(SwapCandidate) * dims::kLayers);
  int least = 0, greatest = 0;
  check(cudaDeviceGetStreamPriorityRange(&least, &greatest), "stream priority range");
  check(cudaStreamCreateWithPriority(&copy_, cudaStreamNonBlocking, least), "swap copy stream");
  try {
    for (auto& l : layers_) check(cudaEventCreateWithFlags(&l.event, cudaEventDisableTiming), "swap event");
  } catch (...) {
    for (auto& l : layers_) if (l.event) cudaEventDestroy(l.event);
    cudaStreamDestroy(copy_);
    throw;
  }
}

ExpertSwapper::~ExpertSwapper() {
  if (copy_) cudaStreamSynchronize(copy_);
  for (auto& l : layers_) {
    if (l.event) cudaEventDestroy(l.event);
  }
  if (copy_) cudaStreamDestroy(copy_);
}

bool ExpertSwapper::idle() const noexcept {
  return std::all_of(layers_.begin(), layers_.end(),
                     [](const LayerSwap& l) { return l.stage == Stage::Idle && l.queue.empty(); });
}

void ExpertSwapper::queue(int32_t layer, int32_t b, int32_t a) {
  if (!usable_) return;
  if (layer < 0 || layer >= placement_.plan().expert_layers || b < 0 || b >= dims::kExperts || a < -1 ||
      a >= dims::kExperts)
    throw std::out_of_range("ExpertSwapper::queue");
  layers_[std::size_t(layer)].queue.emplace_back(b, a);
}

void ExpertSwapper::queue_random(int32_t n, uint64_t seed) {
  if (!usable_) return;
  auto next = [&seed] {
    uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  const int32_t layers = placement_.plan().expert_layers;
  if (layers == 0 || n <= 0) return;
  int32_t queued = 0;
  for (int64_t tries = 0; queued < n && tries < 64ll * n; ++tries) {
    const int32_t l = int32_t(next() % uint64_t(layers));
    const PlacementPlan& p = placement_.plan();
    if (p.gpu_slots[std::size_t(l)] == 0 || p.cpu_slots[std::size_t(l)] == 0) continue;
    std::vector<int32_t> cpu, gpu;
    const auto& ls = layers_[std::size_t(l)];
    for (int32_t e = 0; e < dims::kExperts; ++e) {
      if ((ls.stage != Stage::Idle && (e == ls.a || e == ls.b)) ||
          std::any_of(ls.queue.begin(), ls.queue.end(),
                      [e](const auto& pair) { return pair.first == e || pair.second == e; })) continue;
      if (placement_.slot_of(l, e) >= 0) gpu.push_back(e);
      else if (placement_.host_blob(l, e)) cpu.push_back(e);
    }
    if (cpu.empty() || gpu.empty()) continue;
    queue(l, cpu[next() % cpu.size()], gpu[next() % gpu.size()]);
    ++queued;
  }
}

void ExpertSwapper::enqueue_plan(int32_t steps) {
  if (!usable_) return;
  auto* ema = static_cast<float*>(stats_.p);
  auto* last = reinterpret_cast<uint32_t*>(ema + kPairs);
  const float keep = float(std::pow(options_.decay, double(std::max(steps, 0))));
  check(swap_candidates(counts_, last, ema, placement_.slot_of_device(), keep,
                        static_cast<SwapCandidate*>(candidates_.device_ptr()), compute_),
        "swap candidates");
  planned_ = true;
}

void ExpertSwapper::queue_candidates() {
  planned_ = false;
  struct Candidate {
    float gain;
    int32_t layer, b, a;
  };
  std::array<Candidate, dims::kLayers> candidates;
  int n = 0;
  const auto* picks = static_cast<const SwapCandidate*>(candidates_.data());
  const float ratio = float(options_.min_ratio);
  for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) {
    LayerSwap& ls = layers_[std::size_t(l)];
    ls.queue.clear();
    // One swap in progress per layer; a busy layer is planned next round.
    if (ls.stage != Stage::Idle) continue;
    const SwapCandidate& c = picks[l];
    // Transitions after the kernel ran may have moved either expert.
    if (c.b < 0 || !(c.ema_b > 0.0f) || placement_.slot_of(l, c.b) >= 0 || !placement_.host_blob(l, c.b)) continue;
    if (placement_.free_slots(l) > 0)
      candidates[std::size_t(n++)] = {c.ema_b, l, c.b, -1};
    else if (c.a >= 0 && placement_.slot_of(l, c.a) >= 0 && c.ema_b > c.ema_a * ratio)
      candidates[std::size_t(n++)] = {c.ema_b - c.ema_a, l, c.b, c.a};
  }
  const int k = std::min(n, options_.max_per_plan);
  std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.begin() + n,
                    [](const Candidate& x, const Candidate& y) {
                      return x.gain > y.gain || (x.gain == y.gain && x.layer < y.layer);
                    });
  for (int i = 0; i < k; ++i) {
    const Candidate& c = candidates[std::size_t(i)];
    layers_[std::size_t(c.layer)].queue.emplace_back(c.b, c.a);
  }
}

void ExpertSwapper::finish_layer(LayerSwap& ls, bool completed) {
  completed_ += completed;
  ls.stage = Stage::Idle;
  ls.b = ls.a = ls.target = -1;
  ls.backed = false;
  ls.ram = nullptr;
}

bool ExpertSwapper::complete_layer(int32_t l, bool& changed) {
  LayerSwap& ls = layers_[std::size_t(l)];
  if (ls.stage == Stage::Upload) {
    const cudaError_t q = cudaEventQuery(ls.event);
    if (q == cudaErrorNotReady) return true;
    check(q, "expert swap upload");
    changed = true;
    if (ls.backed) {
      placement_.fill(l, ls.b, ls.target);
      finish_layer(ls, true);
    } else {
      ls.ram = placement_.assign_spare_to(l, ls.b);
      ls.stage = Stage::DownloadPending;
      return true;
    }
  } else if (ls.stage == Stage::Download) {
    const cudaError_t q = cudaEventQuery(ls.event);
    if (q == cudaErrorNotReady) return true;
    check(q, "expert swap download");
    // The placement's fallback D2H host callback precedes this event, so even pageable RAM
    // contains the complete victim blob before publishing it to CPU workers.
    placement_.release_to_ram(l, ls.a, ls.ram);
    changed = true;
    finish_layer(ls, true);
  } else if (ls.stage != Stage::Idle) {
    return true;
  }
  // Idle: the next valid request. Dropping a victim from a backed slot changes the map, so it
  // happens here; the copies start in start_layer.
  while (!ls.queue.empty()) {
    const auto [b, a] = ls.queue.front();
    ls.queue.pop_front();
    if (placement_.slot_of(l, b) >= 0 || !placement_.host_blob(l, b)) continue;  // stale request
    int32_t target = -1;
    bool backed = true;
    if (a < 0) {
      target = placement_.claim_free_slot(l);
    } else if (const int32_t slot = placement_.slot_of(l, a); slot >= 0 && placement_.backed_slot(l, slot)) {
      target = placement_.drop(l, a);
      changed = true;
    } else if (slot >= 0) {
      target = placement_.spare_slot(l);
      backed = false;
    }
    if (target < 0) continue;
    ls.b = b;
    ls.a = a;
    ls.target = target;
    ls.backed = backed;
    ls.stage = Stage::UploadPending;
    return true;
  }
  return false;
}

bool ExpertSwapper::start_layer(int32_t l) {
  LayerSwap& ls = layers_[std::size_t(l)];
  const std::size_t stride = std::size_t(placement_.blob_stride(l));
  if (ls.stage == Stage::UploadPending) {
    placement_.enqueue_upload(placement_.slot_address(l, ls.target), placement_.host_blob(l, ls.b), stride, copy_);
    check(cudaEventRecord(ls.event, copy_), "expert swap event");
    ls.stage = Stage::Upload;
  } else if (ls.stage == Stage::DownloadPending) {
    placement_.enqueue_download(ls.ram, placement_.device_blob(l, ls.a), stride, copy_);
    check(cudaEventRecord(ls.event, copy_), "expert swap event");
    ls.stage = Stage::Download;
  }
  return ls.stage != Stage::Idle;
}

bool ExpertSwapper::vacate(int32_t tokens) {
  if (usable_) {
    for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) {
      LayerSwap& ls = layers_[std::size_t(l)];
      if (!ls.backed || (ls.stage != Stage::UploadPending && ls.stage != Stage::Upload) ||
          !placement_.retires(l, ls.target, tokens))
        continue;
      // The context takes the slot: b stays on the CPU (a, if any, already left it).
      if (ls.stage == Stage::Upload) check(cudaEventSynchronize(ls.event), "expert swap upload");
      finish_layer(ls, false);
    }
  }
  return placement_.vacate(tokens, compute_);
}

void ExpertSwapper::require_boundary() const {
  if (placement_.borrowed())
    throw std::logic_error("expert swap requires prompt borrowing to be restored");
  const cudaError_t status = cudaStreamQuery(compute_);
  if (status == cudaErrorNotReady)
    throw std::logic_error("expert swap requires an idle compute stream at a step boundary");
  check(status, "expert swap boundary");
}

bool ExpertSwapper::boundary() {
  if (!usable_) return false;
  require_boundary();  // also: enqueue_plan's kernel has completed
  bool changed = false;
  // Completing transfers first frees their layers for this round's candidates.
  for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) complete_layer(l, changed);
  if (planned_) queue_candidates();
  placement_.publish_slot_writes(compute_);
  return changed;
}

void ExpertSwapper::start() {
  if (!usable_) return;
  for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) start_layer(l);
}

bool ExpertSwapper::advance() {
  if (!usable_) return false;
  require_boundary();
  bool changed = false;
  for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) {
    complete_layer(l, changed);
    start_layer(l);
  }
  placement_.publish_slot_writes(compute_);
  return changed;
}

bool ExpertSwapper::drain() {
  if (!usable_) return false;
  require_boundary();
  bool changed = false;
  for (;;) {
    bool busy = false;
    for (int32_t l = 0; l < placement_.plan().expert_layers; ++l) {
      complete_layer(l, changed);
      busy |= start_layer(l);
    }
    placement_.publish_slot_writes(compute_);
    if (!busy) {
      check(cudaStreamSynchronize(compute_), "expert swap slot maps");
      return changed;
    }
    check(cudaStreamSynchronize(copy_), "expert swap drain");
  }
}

}  // namespace fngine::model
