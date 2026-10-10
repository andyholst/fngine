#pragma once

// ExpertPool microbenchmark over real pack experts (used by cpu_iq_kernels). Depends only on the
// ExpertPool API so the same harness can time other pool implementations.
#include "cpu/cpu_test_util.h"
#include "cpu/expert_pool.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace fngine::test {

struct BenchLayer {
  int layer = 0;
  int n_experts = 0;
  PackExperts::Layer desc;
  std::vector<std::unique_ptr<HostBuf>> blobs;
  std::vector<const void*> slots;

  BenchLayer(const PackExperts& pack, int l, int n) : layer(l), n_experts(n), desc(pack.layer(l)) {
    slots.assign(std::size_t(kExchangeLayers) * dims::kExperts, nullptr);
    for (int e = 0; e < n; ++e) {
      blobs.push_back(std::make_unique<HostBuf>(desc.blob_bytes));
      pack.read_blob(l, e, blobs.back()->p);
      slots[std::size_t(l) * dims::kExperts + e] = blobs.back()->p;
    }
  }
};

struct BenchStats {
  double us_step = 0;        // pool compute time per step (ready seen -> done stored)
  double us_per_expert = 0;  // per distinct cold expert
  double us_per_pair = 0;    // per (token, expert) pair
  double tokens_per_expert = 0;
};

// Routing per step: uniform -> every row uses the same 10 experts (rotating through the
// layer's n_experts, so weights come from DRAM); skewed -> each row draws 10 distinct experts
// with Zipf(1) weights over n_experts (long-tailed token counts per expert).
inline BenchStats bench_pool(const BenchLayer& bl, const std::vector<int>& cpus,
                             cpu::PoolMode mode, int T, int iters, bool skewed) {
  const int l = bl.layer;
  const int K = dims::kTopK;
  HostExchange hx(std::max(T, 16));
  cpu::PoolConfig cfg;
  cfg.worker_cpus = cpus;
  cfg.layers[l] = cpu::ExpertLayerDesc{bl.desc.gu, bl.desc.down, 0, bl.desc.gate_bytes,
                                       bl.desc.gate_bytes + bl.desc.up_bytes};
  cfg.host_slot_of = bl.slots.data();
  cfg.decode = hx.dec;
  cfg.prefill = hx.pre;
  cpu::ExpertPool pool(cfg);

  Rng rng(1000 + l);
  std::vector<uint16_t> x(std::size_t(T) * dims::kHidden);
  for (auto& v : x) v = bf16_bits(float(rng.normal()));
  std::vector<double> cdf(std::size_t(bl.n_experts));
  double acc = 0;
  for (int e = 0; e < bl.n_experts; ++e) cdf[e] = (acc += 1.0 / double(e + 1));
  std::vector<int> route(std::size_t(T) * K);
  auto make_route = [&](int it) {
    for (int t = 0; t < T; ++t) {
      int* r = &route[std::size_t(t) * K];
      for (int j = 0; j < K; ++j) {
        if (!skewed) {
          r[j] = (it * K + j) % bl.n_experts;
          continue;
        }
        for (;;) {
          const double u = rng.uniform() * acc;
          const int e = int(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
          const int ee = (e + it * 7) % bl.n_experts;
          if (std::find(r, r + j, ee) == r + j) {
            r[j] = ee;
            break;
          }
        }
      }
    }
  };
  uint32_t seq = 0;
  auto step = [&](int it) {
    make_route(it);
    ++seq;
    pool.begin(seq, mode);
    if (mode == cpu::PoolMode::Decode) {
      MoeDecodePublish& p = hx.dec->publish[l];
      p.tokens = T;
      p.n_cold = T * K;
      for (int t = 0; t < T; ++t) {
        p.cold_count[t] = K;
        for (int j = 0; j < K; ++j) {
          p.cold_expert[t][j] = route[std::size_t(t) * K + j];
          p.cold_weight[t][j] = 0.1f;
        }
        std::memcpy(&p.x[t][0], x.data() + std::size_t(t) * dims::kHidden, dims::kHidden * 2);
      }
      hx.release_all(seq);
      hx.wait_done(l, seq);
      return;
    }
    hx.prefill_step(seq, [&](int layer) {
      PrefillExchangeView& p = hx.pre;
      p.header[0] = T;
      if (layer != l) {
        p.header[1] = 0;
        p.header[2] = 0;
        return;
      }
      p.header[1] = T;
      p.header[2] = T * K;
      for (int t = 0; t < T; ++t) {
        p.row_token[t] = t;
        p.cold_count[t] = K;
        for (int j = 0; j < K; ++j) {
          p.cold_expert[t * K + j] = route[std::size_t(t) * K + j];
          p.cold_weight[t * K + j] = 0.1f;
        }
      }
      std::memcpy(p.x, x.data(), std::size_t(T) * dims::kHidden * 2);
    });
  };
  for (int it = 0; it < 2; ++it) step(it);
  pool.reset_timings();
  for (int it = 0; it < iters; ++it) step(it + 2);
  pool.wait_idle();
  if (pool.failed()) {
    std::printf("FAIL pool: %s\n", pool.failure().c_str());
    std::exit(1);
  }
  const cpu::LayerTiming tm = pool.timings()[l];
  BenchStats s;
  s.us_step = double(tm.compute_ns) / 1e3 / double(tm.steps);
  s.us_per_expert = double(tm.compute_ns) / 1e3 / double(tm.experts);
  s.us_per_pair = double(tm.compute_ns) / 1e3 / double(tm.pairs);
  s.tokens_per_expert = double(tm.pairs) / double(tm.experts);
  return s;
}

}  // namespace fngine::test
