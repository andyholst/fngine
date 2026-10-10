// Exercise live map transitions with a captured MoE graph and the real CPU pool. Each topology's
// output is checked against a pack-backed FP64 oracle using that backend's activation quantization.
#include "cpu/cpu_test_util.h"
#include "cpu/expert_pool.h"
#include "io/direct_io.h"
#include "io/pack.h"
#include "kernels/moe/moe_test_common.h"
#include "model/adaptive_experts.h"
#include "model/placement.h"

#include <atomic>
#include <limits>
#include <set>

using namespace fngine;

namespace {

constexpr int kH = dims::kHidden;
constexpr int kI = dims::kExpertInter;
constexpr int kT = 2;

std::vector<double> quantized(const std::vector<float>& x, cpu::ActFormat format, bool gpu) {
  if (gpu) return test::moe::q8_dequant_row(x.data(), int(x.size()));
  const auto q = test::ref_quant_act(format, x.data(), int(x.size()));
  std::vector<float> deq(x.size());
  test::dequant_act(format, q.data(), int(x.size()), deq.data());
  return {deq.begin(), deq.end()};
}

std::vector<double> oracle(const io::ExpertLayout& layout, const std::vector<std::byte>& blob,
                           const __nv_bfloat16* x, bool gpu) {
  const auto gu = static_cast<cpu::WeightFormat>(layout.gu_type);
  const auto dn = static_cast<cpu::WeightFormat>(layout.down_type);
  std::vector<float> xf(kH), row(kH), h(kI);
  for (int d = 0; d < kH; ++d) xf[d] = test::bf16_to_float(x[d]);
  const auto a = quantized(xf, cpu::format_info(gu).act, gpu);
  const auto gu_row = cpu::weight_row_bytes(gu, kH);
  const auto dn_row = cpu::weight_row_bytes(dn, kI);
  for (int r = 0; r < kI; ++r) {
    double g = 0, u = 0;
    test::dequant_row(gu, blob.data() + std::size_t(r) * gu_row, kH, row.data());
    for (int d = 0; d < kH; ++d) g += row[d] * a[d];
    test::dequant_row(gu, blob.data() + layout.gate_bytes + std::size_t(r) * gu_row, kH, row.data());
    for (int d = 0; d < kH; ++d) u += row[d] * a[d];
    h[r] = float(test::moe::silu_d(g) * u);
  }
  const auto hq = quantized(h, cpu::ActFormat::Q8_0, gpu);
  std::vector<double> y(kH, 0.0);
  for (int r = 0; r < kH; ++r) {
    test::dequant_row(dn, blob.data() + layout.gate_bytes + layout.up_bytes + std::size_t(r) * dn_row,
                      kI, row.data());
    for (int d = 0; d < kI; ++d) y[r] += row[d] * hq[d];
  }
  return y;
}

void verify_maps(const model::ExpertPlacement& placement, bool transitional = false) {
  std::vector<int32_t> device(48 * 512);
  FNGINE_TEST_CUDA(cudaMemcpy(device.data(), placement.slot_of_device(), device.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
  FNGINE_CHECK(device == placement.plan().slot_of);
  std::set<int32_t> gpu;
  std::set<const std::byte*> cpu;
  for (int e = 0; e < 512; ++e) {
    const int slot = placement.slot_of(0, e);
    if (slot >= 0) {
      FNGINE_CHECK(placement.host_blob(0, e) == nullptr);
      FNGINE_CHECK(gpu.insert(slot).second);
    } else {
      FNGINE_CHECK(placement.host_blob(0, e) != nullptr);
      FNGINE_CHECK(cpu.insert(placement.host_blob(0, e)).second);
    }
  }
  FNGINE_CHECK(int(gpu.size()) == placement.plan().gpu_slots[0] + int(transitional));
  FNGINE_CHECK(int(cpu.size()) == placement.plan().cpu_slots[0] - int(transitional));
  if (transitional) FNGINE_CHECK(placement.spare_slot(0) == -1);
  else FNGINE_CHECK(placement.spare_slot(0) >= 0 && !gpu.contains(placement.spare_slot(0)));
}

}  // namespace

int run_swaps(bool register_ram) {
  const auto dir = test::env_or_skip("FNGINE_TEST_PACK");
  if (dir.empty() || test::skip_without_gpu()) return test::kSkip;
  const auto pack = io::Pack::open(dir);
  const auto& layout = pack.expert_layers()[0];
  cudaStream_t stream;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  model::PlacementOptions opt;
  opt.expert_layers = 1;
  opt.register_ram = register_ram;
  opt.bounce_buffer_bytes = 64 * 1024;  // force multi-chunk fallback transfers and staging reuse
  opt.vram_cap_bytes = 65 * layout.blob_stride;  // 64 hot experts and one spare, always a mixed tier
  // Persisted decode activity participates in the next placement without modifying the user's pack.
  opt.routing_stats = std::filesystem::temp_directory_path() /
                      ("fngine-swap-stats-" + std::to_string(::getpid()) + ".bin");
  std::vector<uint64_t> stored(48 * 512, 0);
  stored[0] = std::numeric_limits<uint64_t>::max();
  stored[511] = 12345;
  {
    std::ofstream out(opt.routing_stats, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(stored.data()), std::streamsize(stored.size() * sizeof(uint64_t)));
    out.close();
    FNGINE_CHECK(bool(out));
  }
  auto placement = model::ExpertPlacement::build(pack, stream, opt);
  std::filesystem::remove(opt.routing_stats);
  if (!register_ram) FNGINE_CHECK(!placement.ram_tier().registered());
  // The persisted decode count dominates the prefill calibration profile: expert 0 is resident.
  FNGINE_CHECK(placement.slot_of(0, 0) >= 0);
  FNGINE_CHECK(placement.plan().stats_total == std::numeric_limits<uint64_t>::max());
  FNGINE_CHECK(placement.plan().spares);
  FNGINE_CHECK(placement.plan().gpu_slots[0] == 64);
  const void* const* const stable_host_table = placement.host_slot_table();
  const int32_t* const stable_device_table = placement.slot_of_device();
  int b = -1, a = -1;
  for (int e = 0; e < 512; ++e) {
    if (placement.slot_of(0, e) < 0 && b < 0) b = e;
    if (placement.slot_of(0, e) >= 0 && a < 0) a = e;
  }
  FNGINE_CHECK(a >= 0 && b >= 0);
  const int old_spare = placement.spare_slot(0), old_a = placement.slot_of(0, a);
  const std::byte* const old_b_ram = placement.host_blob(0, b);

  const auto x = test::moe::make_x(kT, 123, 0.25);
  io::InputFile file(pack.file_path(io::PackFile::Experts));
  std::vector<double> expected[kT][2];  // token -> CPU/GPU oracle
  for (int t = 0; t < kT; ++t) {
    std::vector<std::byte> blob(layout.blob_bytes);
    file.read_exact(layout.blob_offset(t == 0 ? b : a), blob);
    for (int gpu = 0; gpu < 2; ++gpu) expected[t][gpu] = oracle(layout, blob, x.data() + t * kH, gpu);
  }

  test::moe::Mapped<DecodeExchange> exchange(sizeof(DecodeExchange));
  test::moe::Mapped<uint32_t> sequence(sizeof(uint32_t));
  cpu::PoolConfig cfg;
  cfg.worker_cpus = {-1, -1};
  cfg.host_slot_of = stable_host_table;
  cfg.decode = exchange.host;
  cfg.layers[0] = {static_cast<cpu::WeightFormat>(layout.gu_type),
                   static_cast<cpu::WeightFormat>(layout.down_type), 0,
                   std::size_t(layout.gate_bytes), std::size_t(layout.gate_bytes + layout.up_bytes)};
  cpu::ExpertPool pool(cfg);
  test::DeviceArray<__nv_bfloat16> dx(x.size());
  FNGINE_TEST_CUDA(cudaMemcpy(dx.ptr, x.data(), x.size() * sizeof(x[0]), cudaMemcpyHostToDevice));
  std::vector<float> logits(kT * dims::kRouterRows, -1000.0f);
  logits[b] = 1000.0f;
  logits[dims::kRouterRows + a] = 1000.0f;
  test::DeviceArray<float> dl(logits.size()), shared(kT * kH), inj(kT * 4), X(kT * 4 * kH), y(kT * kH);
  FNGINE_TEST_CUDA(cudaMemcpy(dl.ptr, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemset(shared.ptr, 0, kT * kH * sizeof(float)));
  FNGINE_TEST_CUDA(cudaMemset(inj.ptr, 0, kT * 4 * sizeof(float)));
  FNGINE_TEST_CUDA(cudaMemset(X.ptr, 0, kT * 4 * kH * sizeof(float)));
  test::DeviceArray<uint8_t> workspace(kernels::moe_decode_workspace_bytes());
  const auto ws = kernels::make_moe_decode_workspace(workspace.ptr);
  const auto gpu_layout = kernels::make_moe_expert_layout(placement.slot_base(0), layout.blob_stride,
      static_cast<kernels::MoeFormat>(layout.gu_type), static_cast<kernels::MoeFormat>(layout.down_type));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  FNGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  FNGINE_TEST_CUDA(kernels::moe_select_publish(dl.ptr, dx.ptr, stable_device_table, 0, sequence.dev,
                                               exchange.dev, kT, ws, stream));
  FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(gpu_layout, kT, ws, stream));
  FNGINE_TEST_CUDA(kernels::moe_hot_down(gpu_layout, kT, ws, stream));
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, shared.ptr, exchange.dev, 0, sequence.dev, inj.ptr, X.ptr, y.ptr,
                                       kT, kernels::kMoeExchangeTimeoutCycles, nullptr, 1.0f, stream));
  FNGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
  FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  uint32_t seq = 0;
  auto run = [&] {
    ++seq;
    *sequence.host = seq;
    for (int l = 1; l < dims::kLayers; ++l) {
      exchange.host->publish[l].tokens = kT;
      host_store_release(exchange.host->control.ready[l].value, seq);
    }
    pool.begin(seq, cpu::PoolMode::Decode);
    FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    pool.wait_idle();
    FNGINE_CHECK(!pool.failed() && host_load_acquire(exchange.host->control.error.value) == 0);
    const auto got = y.download();
    for (int t = 0; t < kT; ++t) {
      const auto& ref = expected[t][placement.slot_of(0, t == 0 ? b : a) >= 0];
      double err = 0, norm = 0;
      for (int d = 0; d < kH; ++d) {
        const double delta = got[t * kH + d] - ref[d];
        err += delta * delta;
        norm += ref[d] * ref[d];
      }
      FNGINE_CHECK(std::isfinite(err) && norm > 0 && std::sqrt(err / norm) <= 1e-3);
    }
    FNGINE_CHECK(placement.host_slot_table() == stable_host_table);
    FNGINE_CHECK(placement.slot_of_device() == stable_device_table);
  };

  model::SwapOptions options;
  options.decay = 1.0;  // the EMA is the cumulative routing count
  options.min_ratio = 1.5;
  options.prior = 1.0;  // EMA of never-routed placement residents
  std::vector<uint32_t> counts(48 * 512, 0);  // cumulative routing counters
  test::DeviceArray<uint32_t> device_counts(counts.size());
  model::ExpertSwapper swapper(placement, stream, device_counts.ptr, options);
  FNGINE_CHECK(swapper.usable());
  verify_maps(placement);
  run();
  swapper.queue(0, b, a);
  swapper.queue(0, b, a);  // stale duplicate must not consume the new spare or corrupt a map
  FNGINE_CHECK(!swapper.advance());  // upload only; b must remain readable by CPU workers
  FNGINE_CHECK(placement.host_blob(0, b) == old_b_ram && placement.slot_of(0, b) == -1);
  run();
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());
  FNGINE_CHECK(swapper.advance());  // publish b on GPU; retire its RAM before D2H starts
  FNGINE_CHECK(placement.slot_of(0, b) == old_spare && placement.slot_of(0, a) == old_a);
  run();  // real workers execute while the retired RAM is being repurposed
  verify_maps(placement, true);
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());
  FNGINE_CHECK(swapper.advance());  // publish downloaded a to CPU at a later boundary
  FNGINE_CHECK(placement.host_blob(0, a) == old_b_ram && placement.spare_slot(0) == old_a);
  FNGINE_CHECK(swapper.completed() == 1 && swapper.idle());
  run();
  verify_maps(placement);

  // Reversing a swap exercises reuse of both physical slots with the exact same captured graph.
  swapper.queue(0, a, b);
  FNGINE_CHECK(swapper.drain());
  run();
  verify_maps(placement);

  // Map mutation is rejected while computation is in flight, before either host table changes.
  std::atomic<bool> release{false};
  FNGINE_TEST_CUDA(cudaLaunchHostFunc(stream, [](void* p) {
    static_cast<std::atomic<bool>*>(p)->wait(false);
  }, &release));
  bool rejected = false;
  try { swapper.advance(); } catch (const std::logic_error&) { rejected = true; }
  release.store(true);
  release.notify_all();
  FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  FNGINE_CHECK(rejected);

  // Planning: a CPU expert replaces the coldest GPU expert (lowest id among equal EMAs) once its
  // EMA exceeds min_ratio times that expert's. One round = the kernel, then a boundary that queues
  // and starts at most one swap per layer.
  const auto plan_round = [&] {
    FNGINE_TEST_CUDA(cudaMemcpyAsync(device_counts.ptr, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice,
                                     stream));
    swapper.enqueue_plan(1);
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    swapper.boundary();
    return !swapper.idle();
  };
  std::vector<int> cold;
  int coldest = -1;
  for (int e = 0; e < 512; ++e) {
    if (placement.slot_of(0, e) < 0 && cold.size() < 3) cold.push_back(e);
    if (placement.slot_of(0, e) >= 0 && coldest < 0) coldest = e;
  }
  FNGINE_CHECK(cold.size() == 3 && coldest >= 0);
  const auto before = swapper.completed();
  counts[std::size_t(cold[0])] += 1;
  FNGINE_CHECK(!plan_round() && swapper.idle());  // 1 <= 1.5 * 1
  counts[std::size_t(cold[0])] += 1;
  FNGINE_CHECK(plan_round());  // 2 > 1.5 * 1
  swapper.drain();
  FNGINE_CHECK(swapper.completed() == before + 1);
  FNGINE_CHECK(placement.slot_of(0, cold[0]) >= 0 && placement.slot_of(0, coldest) < 0);
  run();
  verify_maps(placement);
  // One swap per layer and round, the hottest first; the EMA ranks the runner-up next round.
  counts[std::size_t(cold[1])] += 10;
  counts[std::size_t(cold[2])] += 20;
  FNGINE_CHECK(plan_round());
  swapper.drain();
  FNGINE_CHECK(swapper.completed() == before + 2);
  FNGINE_CHECK(placement.slot_of(0, cold[2]) >= 0 && placement.slot_of(0, cold[1]) < 0);
  FNGINE_CHECK(plan_round());
  swapper.drain();
  FNGINE_CHECK(swapper.completed() == before + 3 && placement.slot_of(0, cold[1]) >= 0);
  run();
  swapper.queue_random(8, 1234);
  swapper.drain();
  FNGINE_CHECK(swapper.completed() == before + 11);
  run();
  verify_maps(placement);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("PASS expert swap (%s RAM): map transitions, bounded planning and captured mixed-tier output\n",
              register_ram ? "registered" : "pageable");
  return 0;
}

int main() {
  const int registered = run_swaps(true);
  return registered == 0 ? run_swaps(false) : registered;
}
