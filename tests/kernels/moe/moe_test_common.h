#pragma once

// Shared helpers for the MoE kernel tests: router oracle, host-mapped buffers, timing.
#include "common/test_util.h"
#include "core/exchange.h"
#include "core/model_dims.h"
#include "kernels/moe/moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace fngine::test::moe {

using kernels::MoeFormat;
constexpr int kHidden = dims::kHidden;
constexpr int kInter = dims::kExpertInter;
constexpr int kTopK = dims::kTopK;
constexpr int kExperts = dims::kExperts;
constexpr int kRouter = dims::kRouterRows;

struct TopK {
  int id[kTopK];
  double w[kTopK];
};

// DeviceArray has no move assignment: swap the payload in (the temporary frees the old one).
template <class T>
inline void assign(DeviceArray<T>& dst, DeviceArray<T>&& src) {
  std::swap(dst.ptr, src.ptr);
  std::swap(dst.count, src.count);
}

// Prefill hot-expert sum (fixed point on the device) as FP32 [T][2560].
inline std::vector<float> download_y(const kernels::MoePrefillWorkspace& ws, int T) {
  std::vector<long long> q(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(q.data(), ws.y_fixed, q.size() * sizeof(long long),
                              cudaMemcpyDeviceToHost));
  std::vector<float> y(q.size());
  for (size_t i = 0; i < q.size(); ++i) y[i] = kernels::moe_y_fixed_to_float(q[i]);
  return y;
}

// Oracle: top-10 by (logit desc, id asc); weights renormalized softmax over the 10 (FP64).
inline TopK oracle_topk(const float* logits) {
  std::vector<int> ids(kExperts);
  for (int i = 0; i < kExperts; ++i) ids[i] = i;
  std::partial_sort(ids.begin(), ids.begin() + kTopK, ids.end(), [&](int a, int b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  });
  TopK r{};
  double sum = 0;
  for (int k = 0; k < kTopK; ++k) {
    r.id[k] = ids[k];
    r.w[k] = std::exp(double(logits[ids[k]]) - double(logits[ids[0]]));
    sum += r.w[k];
  }
  for (int k = 0; k < kTopK; ++k) r.w[k] /= sum;
  return r;
}

// Host-mapped (zero-copy) pinned buffer; `dev` is the device alias.
template <class T>
struct Mapped {
  T* host = nullptr;
  T* dev = nullptr;
  size_t bytes = 0;
  explicit Mapped(size_t n_bytes) : bytes(n_bytes) {
    FNGINE_TEST_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped));
    std::memset(static_cast<void*>(host), 0, bytes);
    FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev), host, 0));
  }
  ~Mapped() {
    if (host) cudaFreeHost(host);
  }
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;
};

// Device view of a host-mapped PrefillExchangeView (pointers translated to device aliases).
inline PrefillExchangeView device_view(const PrefillExchangeView& h, uint8_t* host_base,
                                       uint8_t* dev_base) {
  auto tr = [&](auto* p) {
    using P = decltype(p);
    return p ? reinterpret_cast<P>(dev_base + (reinterpret_cast<uint8_t*>(p) - host_base)) : p;
  };
  PrefillExchangeView d = h;
  d.header = tr(h.header);
  d.row_token = tr(h.row_token);
  d.cold_count = tr(h.cold_count);
  d.cold_expert = tr(h.cold_expert);
  d.cold_weight = tr(h.cold_weight);
  d.x = tr(h.x);
  d.y = tr(h.y);
  return d;
}

// Router logits: N(0,1), experts in `boost` get +boost_val, plus `extra` (token, expert, value).
inline std::vector<float> make_logits(int T, const std::vector<int>& boost, float boost_val,
                                      uint64_t seed) {
  Rng rng(seed);
  std::vector<float> l(size_t(T) * kRouter);
  for (auto& v : l) v = float(rng.normal());
  for (int t = 0; t < T; ++t)
    for (int e : boost) l[size_t(t) * kRouter + e] += boost_val;
  return l;
}

inline std::vector<__nv_bfloat16> make_x(int T, uint64_t seed, double scale = 1.0) {
  Rng rng(seed);
  std::vector<__nv_bfloat16> x(size_t(T) * kHidden);
  for (auto& v : x) v = float_to_bf16(float(rng.normal() * scale));
  return x;
}

template <class F>
inline void parallel_for(int n, F&& f) {
  const int threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w)
    pool.emplace_back([&, w] {
      for (int i = w; i < n; i += threads) f(i);
    });
  for (auto& th : pool) th.join();
}

// ggml quantize_q8_1 semantics as implemented by the kernels (FP32 scale, float division).
inline void quantize_q8_block(const float* v, int8_t* q, float* d_out) {
  float amax = 0.f;
  for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[j]));
  const float d = amax / 127.0f;
  *d_out = d;
  for (int j = 0; j < 32; ++j) q[j] = amax == 0.0f ? int8_t(0) : int8_t(std::roundf(v[j] / d));
}

// Dequantized q8 copy of a row of n values (n % 32 == 0).
inline std::vector<double> q8_dequant_row(const float* v, int n) {
  std::vector<double> out(n);
  for (int b = 0; b < n / 32; ++b) {
    int8_t q[32];
    float d;
    quantize_q8_block(v + 32 * b, q, &d);
    for (int j = 0; j < 32; ++j) out[32 * b + j] = double(q[j]) * double(d);
  }
  return out;
}

inline double silu_d(double x) { return x / (1.0 + std::exp(-x)); }

}  // namespace fngine::test::moe
