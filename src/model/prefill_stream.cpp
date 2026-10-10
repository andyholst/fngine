#include "model/prefill_stream.h"

#include "core/cpu_topology.h"
#include "core/device.h"
#include "core/model_dims.h"
#include "cpu/iq_cpu.h"
#include <cuda_bf16.h>

#include <sched.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fngine::model {
namespace {

constexpr int kLayers = dims::kLayers;
constexpr int kExperts = dims::kExperts;
constexpr int kTables = kLayers + 1;  // + MTP
constexpr int kHidden = dims::kHidden;
constexpr int kInter = dims::kExpertInter;
// FLOPs of one routed token through one expert: gate + up (2 x 640x2560) and down (2560x640).
constexpr double kExpertTokenFlops = 2.0 * 3.0 * double(kInter) * double(kHidden);
constexpr uint64_t kPcieProbeBytes = 128ull << 20;
constexpr int kCostExperts = 32;
// F_cpu is the slope between T = 1 and T = kCostTokens; 128 tokens make the GEMM part ~3x the
// T = 1 time, so timing noise moves the slope by a few percent instead of 2x at 32 tokens.
constexpr int kCostTokens = 128;

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

cpu::WeightFormat cpu_format(io::GgmlType t) { return static_cast<cpu::WeightFormat>(int32_t(t)); }

// Same BF16 activations as the CPU/GPU prompt paths; decode Q8 costs are not used for prefill.
std::vector<__nv_bfloat16> make_acts(int k, std::mt19937& rng) {
  std::vector<__nv_bfloat16> out(std::size_t(k) * kCostTokens);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : out) v = __float2bfloat16(nd(rng));
  return out;
}

}  // namespace

uint64_t PrefillStream::ring_bytes(const io::Pack& pack, bool enabled) {
  if (!enabled) return 0;
  uint64_t slot = 0;
  for (const io::ExpertLayout& el : pack.expert_layers()) slot = std::max(slot, el.blob_stride);
  return uint64_t(kRingBuffers) * kRingSlots * slot;
}

PrefillStream::PrefillStream(const io::Pack& pack, const ExpertPlacement& placement,
                             const ExpertBlobs& mtp, const PrefillStreamOptions& options)
    : pack_(&pack), placement_(&placement), mtp_(mtp) {
  tables_ = DeviceBuffer(std::size_t(kTables) * kExperts * sizeof(const uint8_t*));
  host_tables_.assign(std::size_t(kTables) * kExperts, nullptr);

  std::string off_reason;
  if (!options.enabled) off_reason = "disabled (--no-prefill-stream)";
  else if (placement.plan().cpu_experts == 0) off_reason = "off: no CPU-resident experts";

  if (off_reason.empty()) {
    for (const io::ExpertLayout& el : pack.expert_layers()) slot_bytes_ = std::max(slot_bytes_, el.blob_stride);
    ring_size_ = ring_bytes(pack, true);
    if (options.ring_memory) {
      if (options.ring_memory_bytes < ring_size_) throw std::invalid_argument("prefill stream loan too small");
      ring_ = static_cast<uint8_t*>(options.ring_memory);
    } else {
      ring_buf_ = DeviceBuffer(std::size_t(ring_size_));
      ring_ = static_cast<uint8_t*>(ring_buf_.p);
    }
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&copy_stream_, cudaStreamNonBlocking, least));
    for (int b = 0; b < kRingBuffers; ++b) {
      CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_[std::size_t(b)], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&buffer_free_[std::size_t(b)], cudaEventDisableTiming));
    }
    measure_cost_model(options.worker_cpus);
  } else {
    spdlog::info("prefill expert streaming {}", off_reason);
  }
  upload_tables();
}

PrefillStream::~PrefillStream() {
  if (copy_stream_) {
    cudaStreamSynchronize(copy_stream_);
    for (int b = 0; b < kRingBuffers; ++b) {
      cudaEventDestroy(copy_done_[std::size_t(b)]);
      cudaEventDestroy(buffer_free_[std::size_t(b)]);
    }
    cudaStreamDestroy(copy_stream_);
  }
}

void PrefillStream::measure_cost_model(const std::vector<int>& worker_cpus) {
  // B_pcie: a full 128 MiB transfer, including bounce staging when necessary. Never touch the loan.
  {
    const HostRegion& tier = placement_->ram_tier();
    const std::size_t n = std::size_t(kPcieProbeBytes);
    DeviceBuffer probe(n);
    std::unique_ptr<PinnedHostBuffer> small_tier;
    const void* source = tier.data();
    if (tier.size() < n) {
      small_tier = std::make_unique<PinnedHostBuffer>(n);
      std::memset(small_tier->data(), 0, n);
      source = small_tier->data();
    }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float best_ms = 0;
    for (int rep = 0; rep < 4; ++rep) {
      CUDA_CHECK(cudaEventRecord(e0, copy_stream_));
      placement_->enqueue_upload(probe.p, source, n, copy_stream_);
      CUDA_CHECK(cudaEventRecord(e1, copy_stream_));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      if (rep > 0 && (best_ms == 0 || ms < best_ms)) best_ms = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cost_.pcie_bytes_per_s = double(n) / (double(best_ms) * 1e-3);
  }

  // CPU: 32 real CPU-resident experts spread over the layers.
  struct Expert {
    const uint8_t* blob;
    const io::ExpertLayout* el;
  };
  std::vector<Expert> experts;
  const auto& layers = pack_->expert_layers();
  for (int i = 0; i < kCostExperts * 4 && int(experts.size()) < kCostExperts; ++i) {
    const int l = (i * 7) % kLayers;
    std::vector<int> cold;
    for (int e = 0; e < kExperts; ++e)
      if (placement_->host_blob(l, e)) cold.push_back(e);
    if (cold.empty()) continue;
    const int e = cold[std::size_t(i * 7) % cold.size()];
    experts.push_back({reinterpret_cast<const uint8_t*>(placement_->host_blob(l, e)), &layers[std::size_t(l)]});
  }
  std::mt19937 rng(1234);
  const auto acts_x = make_acts(kHidden, rng);
  const auto acts_h = make_acts(kInter, rng);
  std::array<const void*, kCostTokens> rows_x{}, rows_h{};
  for (int t = 0; t < kCostTokens; ++t) {
    rows_x[std::size_t(t)] = acts_x.data() + std::size_t(t) * kHidden;
    rows_h[std::size_t(t)] = acts_h.data() + std::size_t(t) * kInter;
  }

  // Tasks like the pool's: (expert, 64 gate+up rows) and (expert, 128 down rows).
  constexpr int kGuChunks = kInter / 64, kDownChunks = kHidden / 128;
  const int n_tasks = int(experts.size()) * (kGuChunks + kDownChunks);
  uint64_t total_bytes = 0;
  for (const Expert& x : experts) total_bytes += x.el->blob_bytes;

  const int threads = worker_cpus.empty() ? std::max(1u, std::thread::hardware_concurrency() / 2)
                                          : int(worker_cpus.size());
  auto run = [&](int T) {
    std::atomic<int> next{0};
    auto body = [&](int w) {
      if (!worker_cpus.empty()) pin_current_thread(worker_cpus[std::size_t(w)]);
      constexpr int tile_rows = 8;
      std::vector<uint16_t> decoded(std::size_t(tile_rows) * kHidden);
      std::vector<float> out(std::size_t(tile_rows) * kCostTokens);
      for (int task = next.fetch_add(1); task < n_tasks; task = next.fetch_add(1)) {
        const Expert& x = experts[std::size_t(task / (kGuChunks + kDownChunks))];
        const int c = task % (kGuChunks + kDownChunks);
        const io::ExpertLayout& el = *x.el;
        if (c < kGuChunks) {
          const cpu::WeightFormat f = cpu_format(el.gu_type);
          const int rb_rows = tile_rows;
          const std::size_t rb = cpu::weight_row_bytes(f, kHidden);
          for (int r = c * 64; r < c * 64 + 64; r += rb_rows) {
            cpu::decode_rows_bf16(f, x.blob + std::size_t(r) * rb, rb, rb_rows, kHidden, decoded.data());
            cpu::gemm_bf16(decoded.data(), rb_rows, kHidden, rows_x.data(), T, out.data(), kCostTokens);
            cpu::decode_rows_bf16(f, x.blob + el.gate_bytes + std::size_t(r) * rb, rb, rb_rows, kHidden,
                                  decoded.data());
            cpu::gemm_bf16(decoded.data(), rb_rows, kHidden, rows_x.data(), T, out.data(), kCostTokens);
          }
        } else {
          const cpu::WeightFormat f = cpu_format(el.down_type);
          const std::size_t rb = cpu::weight_row_bytes(f, kInter);
          const uint8_t* down = x.blob + el.gate_bytes + el.up_bytes;
          const int r0 = (c - kGuChunks) * 128;
          for (int r = r0; r < r0 + 128; r += tile_rows) {
            cpu::decode_rows_bf16(f, down + std::size_t(r) * rb, rb, tile_rows, kInter, decoded.data());
            cpu::gemm_bf16(decoded.data(), tile_rows, kInter, rows_h.data(), T, out.data(), kCostTokens);
          }
        }
      }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int w = 1; w < threads; ++w) pool.emplace_back(body, w);
    body(0);
    for (auto& t : pool) t.join();
    return seconds_since(t0);
  };
  // run() pins the calling thread as worker 0; restore its affinity afterwards.
  cpu_set_t saved;
  const bool have_saved = sched_getaffinity(0, sizeof(saved), &saved) == 0;
  run(1);  // warm-up
  double t1 = 1e30, tn = 1e30;
  for (int rep = 0; rep < 3; ++rep) t1 = std::min(t1, run(1));
  for (int rep = 0; rep < 3; ++rep) tn = std::min(tn, run(kCostTokens));
  if (have_saved) sched_setaffinity(0, sizeof(saved), &saved);

  cost_.cpu_bytes_per_s = double(total_bytes) / t1;
  const double dt = std::max(tn - t1, 1e-3 * t1);
  cost_.cpu_flops_per_s = double(kCostTokens - 1) * kExpertTokenFlops * double(experts.size()) / dt;
  spdlog::info("prefill cost model: B_pcie {:.1f} GB/s, B_cpu {:.1f} GB/s ({} experts, {} threads, "
               "T=1 {:.1f} ms), F_cpu {:.2f} TFLOP/s (T={} {:.1f} ms)",
               cost_.pcie_bytes_per_s / 1e9, cost_.cpu_bytes_per_s / 1e9, experts.size(), threads,
               t1 * 1e3, cost_.cpu_flops_per_s / 1e12, kCostTokens, tn * 1e3);
}

void PrefillStream::plan_for(int32_t T) {
  const PlacementPlan& pp = placement_->plan();
  const auto& layers = pack_->expert_layers();
  // Prompt routing frequency: the pack's prefill calibration profile. Placement scores rank by
  // decode routing and only stand in when the pack has no profile.
  const std::vector<uint64_t>& freq = pack_->has_profile() ? pack_->profile() : pp.score;
  double cpu_only = 0, split = 0;
  int32_t total_streamed = 0;
  uint64_t total_bytes = 0;
  std::ostringstream counts;
  for (int l = 0; l < kLayers; ++l) {
    LayerPlan& lp = plan_[std::size_t(l)];
    lp = LayerPlan{};
    const io::ExpertLayout& el = layers[std::size_t(l)];
    auto score = [&](int e) {
      const std::size_t i = std::size_t(l) * kExperts + std::size_t(e);
      return i < freq.size() ? freq[i] : uint64_t{0};
    };
    double layer_mass = 0;
    std::vector<int32_t> cold;
    for (int e = 0; e < kExperts; ++e) {
      layer_mass += double(score(e));
      if (!placement_->device_blob(l, e) && placement_->host_blob(l, e)) cold.push_back(e);
    }
    std::stable_sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) { return score(a) > score(b); });
    std::vector<double> cost(cold.size(), 0.0);
    if (streaming()) {
      for (std::size_t i = 0; i < cold.size(); ++i) {
        const double share = layer_mass > 0 ? double(score(cold[i])) / layer_mass : 1.0 / kExperts;
        const double tokens = share * double(T) * dims::kTopK;
        const double p_used = 1.0 - std::exp(-tokens);
        cost[i] = p_used * double(el.blob_bytes) / cost_.cpu_bytes_per_s +
                  tokens * kExpertTokenFlops / cost_.cpu_flops_per_s;
      }
    }
    double rest = std::accumulate(cost.begin(), cost.end(), 0.0);
    lp.cpu_only_s = rest;
    double pcie = 0;
    const std::size_t limit = std::min<std::size_t>(kRingSlots, cold.size());
    std::size_t k = 0;
    if (streaming()) {
      const double overlap = double(T) * kCopyOverlap;
      for (; k < limit; ++k) {
        const double nb = pcie + double(el.blob_bytes) / cost_.pcie_bytes_per_s;
        const double nr = rest - cost[k];
        if (nb > nr + overlap) break;
        pcie = nb;
        rest = nr;
      }
    }
    lp.cpu_rest_s = rest;
    lp.pcie_s = pcie;
    const int b = l % kRingBuffers;
    for (std::size_t i = 0; i < k; ++i) {
      const int32_t e = cold[i];
      lp.experts.push_back(e);
      lp.copies.push_back({placement_->host_blob(l, e),
                           ring_ + (std::size_t(b) * kRingSlots + i) * slot_bytes_, el.blob_bytes});
      lp.bytes += el.blob_bytes;
    }
    cpu_only += lp.cpu_only_s;
    split += std::max(lp.pcie_s, lp.cpu_rest_s);
    total_streamed += int32_t(k);
    total_bytes += lp.bytes;
    counts << (l ? "," : "") << k;
  }
  plan_tokens_ = T;
  if (streaming()) {
    spdlog::info("prefill stream plan T={}: {} experts ({:.2f} GiB) per chunk, per layer [{}]; "
                 "predicted CPU expert time per chunk {:.2f} s CPU-only vs {:.2f} s split",
                 T, total_streamed, double(total_bytes) / double(1ull << 30), counts.str(), cpu_only,
                 split);
  }
}

void PrefillStream::upload_tables() {
  for (int l = 0; l < kLayers; ++l) {
    const LayerPlan& lp = plan_[std::size_t(l)];
    const uint8_t** row = host_tables_.data() + std::size_t(l) * kExperts;
    for (int e = 0; e < kExperts; ++e) row[e] = placement_->device_blob(l, e);
    for (std::size_t i = 0; i < lp.experts.size(); ++i)
      row[lp.experts[i]] = static_cast<const uint8_t*>(lp.copies[i].dst);
  }
  const uint8_t** mrow = host_tables_.data() + std::size_t(kLayers) * kExperts;
  for (int e = 0; e < kExperts; ++e)
    mrow[e] = mtp_.base ? mtp_.base + std::size_t(e) * mtp_.blob_stride : nullptr;
  tables_.copy_from_host(host_tables_.data(), host_tables_.size() * sizeof(const uint8_t*));
}

void PrefillStream::begin_chunk(int32_t T, cudaStream_t main) {
  if (T != plan_tokens_) {
    CUDA_CHECK(cudaStreamSynchronize(main));  // the previous chunk's kernels read the tables
    plan_for(T);
    upload_tables();
  }
  issued_ = 0;
  prefetch(0);
}

void PrefillStream::prefetch(int32_t layer) {
  if (!streaming()) return;
  const int32_t upto = std::min<int32_t>(kLayers - 1, layer + kRingBuffers - 1);
  for (; issued_ <= upto; ++issued_) {
    const LayerPlan& lp = plan_[std::size_t(issued_)];
    if (lp.copies.empty()) continue;
    const std::size_t b = std::size_t(issued_ % kRingBuffers);
    // The buffer's previous user (an earlier layer, possibly of the previous chunk) must be done.
    if (buffer_used_[b]) CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, buffer_free_[b], 0));
    for (const Copy& c : lp.copies)
      placement_->enqueue_upload(c.dst, c.src, std::size_t(c.bytes), copy_stream_);
    ring_written_[b] = std::max(ring_written_[b], lp.copies.size());
    CUDA_CHECK(cudaEventRecord(copy_done_[b], copy_stream_));
  }
}

void PrefillStream::wait_ready(int32_t layer, cudaStream_t main) {
  if (!streaming() || plan_[std::size_t(layer)].copies.empty()) return;
  if (issued_ <= layer) prefetch(layer);
  CUDA_CHECK(cudaStreamWaitEvent(main, copy_done_[std::size_t(layer % kRingBuffers)], 0));
}

void PrefillStream::release(int32_t layer, cudaStream_t main) {
  if (!streaming() || plan_[std::size_t(layer)].copies.empty()) return;
  const std::size_t b = std::size_t(layer % kRingBuffers);
  CUDA_CHECK(cudaEventRecord(buffer_free_[b], main));
  buffer_used_[b] = true;
}

void PrefillStream::synchronize() {
  if (copy_stream_) CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
}

std::vector<DeviceRange> PrefillStream::take_ring_written() {
  std::vector<DeviceRange> out;
  for (int b = 0; b < kRingBuffers; ++b) {
    std::size_t& n = ring_written_[std::size_t(b)];
    if (n) out.push_back({ring_ + std::size_t(b) * kRingSlots * slot_bytes_, uint64_t(n) * slot_bytes_});
    n = 0;
  }
  return out;
}

const uint8_t* const* PrefillStream::table(int32_t layer) const noexcept {
  return static_cast<const uint8_t* const*>(tables_.p) + std::size_t(layer) * kExperts;
}

const uint8_t* const* PrefillStream::mtp_table() const noexcept { return table(kLayers); }

int32_t PrefillStream::streamed_experts() const noexcept {
  int32_t n = 0;
  for (const LayerPlan& lp : plan_) n += int32_t(lp.experts.size());
  return n;
}

uint64_t PrefillStream::streamed_bytes() const noexcept {
  uint64_t n = 0;
  for (const LayerPlan& lp : plan_) n += lp.bytes;
  return n;
}

std::string PrefillStream::summary() const {
  std::ostringstream os;
  if (!streaming()) {
    os << "prefill streaming off";
  } else {
    os << "prefill streaming ring " << kRingBuffers << "x" << kRingSlots << " ("
       << double(ring_size_) / double(1ull << 30) << " GiB), B_pcie "
       << cost_.pcie_bytes_per_s / 1e9 << " GB/s, B_cpu " << cost_.cpu_bytes_per_s / 1e9
       << " GB/s, F_cpu " << cost_.cpu_flops_per_s / 1e12 << " TFLOP/s";
  }
  return os.str();
}

}  // namespace fngine::model
