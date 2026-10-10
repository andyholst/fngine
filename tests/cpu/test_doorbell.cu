// GPU<->CPU doorbell inside a captured CUDA graph: per layer a GPU kernel publishes one cold
// (token, expert) pair and releases ready[l]; the ExpertPool computes a synthetic IQ4_NL expert;
// a GPU combine kernel acquire-spins on done[l] and reads y. 10,032 publish/compute/ack cycles
// (209 graph replays x 48 layers); prints the median GPU-observed round trip.
#include "cpu/cpu_test_util.h"
#include "cpu/expert_pool.h"
#include "cpu/q8_cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fngine;
using namespace fngine::test;

namespace {

constexpr int kH = dims::kHidden;
constexpr int kI = dims::kExpertInter;
constexpr int kLayers = kExchangeLayers;
constexpr int kReplays = 209;

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void publish_kernel(DecodeExchange* ex, const volatile uint32_t* seq_cell, int layer,
                               const __nv_bfloat16* x_src, unsigned long long* t_pub) {
  const uint32_t seq = *seq_cell;
  MoeDecodePublish& p = ex->publish[layer];
  for (int i = threadIdx.x; i < kH; i += blockDim.x) p.x[0][i] = x_src[i];
  if (threadIdx.x == 0) {
    p.tokens = 1;
    p.n_cold = 1;
    p.cold_count[0] = 1;
    p.cold_expert[0][0] = 0;
    p.cold_weight[0][0] = 1.0f;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    t_pub[layer] = globaltimer();
    exchange_store_release_sys(&ex->control.ready[layer].value, seq);
  }
}

__global__ void combine_kernel(DecodeExchange* ex, const volatile uint32_t* seq_cell,
                               const volatile uint32_t* iter_cell, int layer,
                               const unsigned long long* t_pub, unsigned long long* rt,
                               float* checksum, uint32_t* error) {
  __shared__ int ok;
  __shared__ float part[256];
  const uint32_t seq = *seq_cell;
  if (threadIdx.x == 0) {
    ok = exchange_wait_sys(&ex->control.done[layer].value, seq, kExchangeTimeoutCycles);
    const unsigned long long t = globaltimer();
    if (!ok) atomicOr(error, kExchangeErrorMoeTimeout | uint32_t(layer + 1));
    else rt[size_t(*iter_cell) * kLayers + layer] = t - t_pub[layer];
  }
  __syncthreads();
  if (!ok) return;
  float s = 0;
  for (int i = threadIdx.x; i < kH; i += blockDim.x)
    s += exchange_load_relaxed_sys(&ex->result[layer].y[0][i]);
  part[threadIdx.x] = s;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) checksum[layer] = part[0];
}

// Synthetic expert blob: IQ4_NL gate | up | down from random weights.
std::vector<uint8_t> make_blob(Rng& rng, std::size_t* gate_bytes, std::size_t* down_bytes) {
  *gate_bytes = std::size_t(kI) * cpu::weight_row_bytes(cpu::WeightFormat::IQ4_NL, kH);
  *down_bytes = std::size_t(kH) * cpu::weight_row_bytes(cpu::WeightFormat::IQ4_NL, kI);
  std::vector<uint8_t> blob(2 * *gate_bytes + *down_bytes);
  std::vector<float> w(std::size_t(kI) * kH);
  for (auto& v : w) v = float(rng.normal() * 0.02);
  quantize_iq4_nl(w.data(), blob.data(), kI, kH, nullptr);
  for (auto& v : w) v = float(rng.normal() * 0.02);
  quantize_iq4_nl(w.data(), blob.data() + *gate_bytes, kI, kH, nullptr);
  for (auto& v : w) v = float(rng.normal() * 0.02);
  quantize_iq4_nl(w.data(), blob.data() + 2 * *gate_bytes, kH, kI, nullptr);
  return blob;
}

// y for one token through the public kernels, in the pool's order.
std::vector<float> reference_y(const uint8_t* blob, std::size_t gate_bytes, const uint16_t* x) {
  const std::size_t gu_rb = cpu::weight_row_bytes(cpu::WeightFormat::IQ4_NL, kH);
  const std::size_t dn_rb = cpu::weight_row_bytes(cpu::WeightFormat::IQ4_NL, kI);
  std::vector<block_q8_0> xq(kH / QK8_0), hq(kI / QK8_0);
  cpu::quantize_row_q8_0_bf16(x, xq.data(), kH);
  std::vector<float> h(kI);
  for (int j = 0; j < kI; ++j) {
    float g, u;
    cpu::cpu_dot(cpu::WeightFormat::IQ4_NL, blob + j * gu_rb, kH / QK4_NL, xq.data(), 1, 0, &g);
    cpu::cpu_dot(cpu::WeightFormat::IQ4_NL, blob + gate_bytes + j * gu_rb, kH / QK4_NL, xq.data(), 1, 0, &u);
    h[j] = g / (1.0f + std::exp(-g)) * u;
  }
  for (int c = 0; c < kI / 64; ++c) cpu::quantize_row_q8_0_f32(h.data() + 64 * c, hq.data() + 2 * c, 64);
  std::vector<float> y(kH);
  for (int n = 0; n < kH; ++n)
    cpu::cpu_dot(cpu::WeightFormat::IQ4_NL, blob + 2 * gate_bytes + n * dn_rb, kI / QK4_NL, hq.data(), 1, 0, &y[n]);
  return y;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(99);
  std::size_t gate_bytes = 0, down_bytes = 0;
  const std::vector<uint8_t> blob_v = make_blob(rng, &gate_bytes, &down_bytes);
  HostBuf blob(blob_v.size());
  std::memcpy(blob.p, blob_v.data(), blob_v.size());

  // Host-mapped exchange + step cells.
  DecodeExchange* ex = nullptr;
  uint32_t* cells = nullptr;  // [0] = seq, [1] = iteration
  FNGINE_TEST_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&ex), sizeof(DecodeExchange), cudaHostAllocMapped));
  FNGINE_TEST_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&cells), 64, cudaHostAllocMapped));
  std::memset(ex, 0, sizeof(DecodeExchange));
  std::memset(cells, 0, 64);
  DecodeExchange* ex_dev = nullptr;
  uint32_t* cells_dev = nullptr;
  FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ex_dev), ex, 0));
  FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&cells_dev), cells, 0));

  std::vector<uint16_t> x(kH);
  std::vector<__nv_bfloat16> xb(kH);
  for (int i = 0; i < kH; ++i) {
    x[i] = bf16_bits(float(rng.normal()));
    std::memcpy(&xb[i], &x[i], 2);
  }
  DeviceArray<__nv_bfloat16> x_dev(xb);
  DeviceArray<unsigned long long> t_pub(kLayers);
  DeviceArray<unsigned long long> rt(std::size_t(kReplays) * kLayers);
  DeviceArray<float> checksum(kLayers);
  DeviceArray<uint32_t> error(1);

  std::vector<const void*> slots(std::size_t(kLayers) * dims::kExperts, nullptr);
  cpu::PoolConfig cfg;
  for (int l = 0; l < kLayers; ++l) {
    cfg.layers[l] = cpu::ExpertLayerDesc{cpu::WeightFormat::IQ4_NL, cpu::WeightFormat::IQ4_NL, 0,
                                         gate_bytes, 2 * gate_bytes};
    slots[std::size_t(l) * dims::kExperts] = blob.p;
  }
  cfg.host_slot_of = slots.data();
  cfg.decode = ex;
  cpu::ExpertPool pool(cfg);

  cudaStream_t stream;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  FNGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < kLayers; ++l) {
    publish_kernel<<<1, 256, 0, stream>>>(ex_dev, cells_dev, l, x_dev.get(), t_pub.get());
    combine_kernel<<<1, 256, 0, stream>>>(ex_dev, cells_dev, cells_dev + 1, l, t_pub.get(),
                                          rt.get(), checksum.get(), error.get());
  }
  FNGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
  FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));

  for (int it = 0; it < kReplays; ++it) {
    const uint32_t seq = uint32_t(it + 1);
    pool.begin(seq, cpu::PoolMode::Decode);
    host_store_release(cells[0], seq);
    host_store_release(cells[1], uint32_t(it));
    FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  }
  pool.wait_idle();

  bool ok = true;
  const uint32_t err = error.download()[0];
  if (err != 0) {
    std::printf("FAIL GPU wait timed out (error 0x%08x)\n", err);
    ok = false;
  }
  if (pool.failed()) {
    std::printf("FAIL pool: %s\n", pool.failure().c_str());
    ok = false;
  }
  // Every layer computed the same expert on the same x: compare y with the direct kernels.
  const std::vector<float> want = reference_y(blob.p, gate_bytes, x.data());
  double max_rel = 0, want_sum = 0;
  for (int n = 0; n < kH; ++n) want_sum += want[n];
  for (int l = 0; l < kLayers; ++l)
    for (int n = 0; n < kH; ++n)
      max_rel = std::max(max_rel, std::fabs(double(ex->result[l].y[0][n]) - want[n]) /
                                      std::max(1e-6, std::fabs(double(want[n]))));
  const std::vector<float> cs = checksum.download();
  double max_cs = 0;
  for (int l = 0; l < kLayers; ++l)
    max_cs = std::max(max_cs, std::fabs(double(cs[l]) - want_sum) / std::max(1e-6, std::fabs(want_sum)));
  std::printf("%s y vs direct kernels: max rel %.3e; GPU checksum max rel %.3e\n",
              max_rel <= 1e-6 && max_cs <= 1e-4 ? "PASS" : "FAIL", max_rel, max_cs);
  ok &= max_rel <= 1e-6 && max_cs <= 1e-4;

  std::vector<unsigned long long> r = rt.download();
  {
    std::size_t slow = 0, slow_first = 0;
    for (std::size_t i = 0; i < r.size(); ++i) {
      if (r[i] > 1'000'000ull) {
        ++slow;
        if (i % kLayers == 0) ++slow_first;
      }
    }
    std::printf("doorbell: %zu cycles > 1 ms (%zu of them on the first layer of a replay)\n", slow,
                slow_first);
  }
  std::vector<double> us(r.begin(), r.end());
  for (auto& v : us) v /= 1e3;
  std::sort(us.begin(), us.end());
  const std::size_t n = us.size();
  const auto tm = pool.timings();
  double comp = 0;
  uint64_t steps = 0;
  for (const auto& t : tm) {
    comp += double(t.compute_ns);
    steps += t.steps;
  }
  std::printf("doorbell: %zu cycles, round trip (GPU publish -> done seen) median %.1f us, "
              "p10 %.1f us, p90 %.1f us, p99 %.1f us\n",
              n, us[n / 2], us[n / 10], us[n * 9 / 10], us[n * 99 / 100]);
  std::printf("doorbell: CPU side (ready seen -> done stored) mean %.1f us over %llu layer steps\n",
              comp / 1e3 / double(steps), (unsigned long long)steps);
  ok &= steps == uint64_t(n);

  FNGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
  FNGINE_TEST_CUDA(cudaGraphDestroy(graph));
  FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
  FNGINE_TEST_CUDA(cudaFreeHost(cells));
  FNGINE_TEST_CUDA(cudaFreeHost(ex));
  std::printf("%s cpu_doorbell\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
