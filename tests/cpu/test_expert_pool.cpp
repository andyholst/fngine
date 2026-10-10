// ExpertPool on real pack experts vs an FP64 oracle: Q8 decode and BF16 prefill activations.
// Determinism across runs and worker counts; skipped layers untouched.
#include "cpu/cpu_test_util.h"
#include "cpu/expert_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

using namespace fngine;
using namespace fngine::test;

namespace {

constexpr int kH = dims::kHidden;
constexpr int kI = dims::kExpertInter;
constexpr int kK = dims::kTopK;

struct Row {
  std::vector<uint16_t> x;  // bf16 bits
  std::vector<int> experts;  // cold experts only
  std::vector<float> weights;
};

struct LayerData {
  int layer = 0;
  std::vector<int> universe;  // experts this layer routes to
  std::vector<bool> cold;     // per universe entry
  std::vector<std::unique_ptr<HostBuf>> blobs;
};

// Independent ggml weight dequantization and activation rounding for each backend.
std::vector<double> oracle(const PackExperts& pack, const LayerData& ld,
                           const std::vector<Row>& rows, cpu::PoolMode mode) {
  const PackExperts::Layer& L = pack.layer(ld.layer);
  const ActFormat gu_act = cpu::format_info(L.gu).act;
  const std::size_t gu_rb = cpu::weight_row_bytes(L.gu, kH);
  const std::size_t dn_rb = cpu::weight_row_bytes(L.down, kI);
  std::vector<std::vector<double>> a(rows.size(), std::vector<double>(kH));
  std::vector<float> tmp(kH);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (int i = 0; i < kH; ++i) tmp[i] = bf16_bits_to_float(rows[r].x[i]);
    if (mode == cpu::PoolMode::Decode) {
      const std::vector<uint8_t> q = ref_quant_act(gu_act, tmp.data(), kH);
      dequant_act(gu_act, q.data(), kH, tmp.data());
    }
    for (int i = 0; i < kH; ++i) a[r][i] = tmp[i];
  }
  std::vector<double> y(rows.size() * kH, 0.0);
  std::vector<float> wrow(kH);
  for (std::size_t u = 0; u < ld.universe.size(); ++u) {
    if (!ld.cold[u]) continue;
    const int e = ld.universe[u];
    std::vector<std::pair<std::size_t, float>> pairs;  // (row, weight)
    for (std::size_t r = 0; r < rows.size(); ++r)
      for (std::size_t j = 0; j < rows[r].experts.size(); ++j)
        if (rows[r].experts[j] == e) pairs.emplace_back(r, rows[r].weights[j]);
    if (pairs.empty()) continue;
    const uint8_t* blob = ld.blobs[u]->p;
    std::vector<std::vector<double>> g(pairs.size(), std::vector<double>(kI)), up = g;
    for (int j = 0; j < kI; ++j) {
      dequant_row(L.gu, blob + std::size_t(j) * gu_rb, kH, wrow.data());
      if (mode == cpu::PoolMode::Prefill)
        for (float& v : wrow) v = bf16_bits_to_float(bf16_bits(v));
      for (std::size_t p = 0; p < pairs.size(); ++p) {
        double s = 0;
        for (int i = 0; i < kH; ++i) s += double(wrow[i]) * a[pairs[p].first][i];
        g[p][j] = s;
      }
      dequant_row(L.gu, blob + L.gate_bytes + std::size_t(j) * gu_rb, kH, wrow.data());
      if (mode == cpu::PoolMode::Prefill)
        for (float& v : wrow) v = bf16_bits_to_float(bf16_bits(v));
      for (std::size_t p = 0; p < pairs.size(); ++p) {
        double s = 0;
        for (int i = 0; i < kH; ++i) s += double(wrow[i]) * a[pairs[p].first][i];
        up[p][j] = s;
      }
    }
    std::vector<std::vector<double>> hd(pairs.size(), std::vector<double>(kI));
    std::vector<float> hf(kI), hq(kI);
    for (std::size_t p = 0; p < pairs.size(); ++p) {
      for (int j = 0; j < kI; ++j) hf[j] = float(g[p][j] / (1.0 + std::exp(-g[p][j])) * up[p][j]);
      if (mode == cpu::PoolMode::Prefill) {
        for (int j = 0; j < kI; ++j) hd[p][j] = bf16_bits_to_float(bf16_bits(hf[j]));
      } else {
        const std::vector<uint8_t> q = ref_quant_act(ActFormat::Q8_0, hf.data(), kI);
        dequant_act(ActFormat::Q8_0, q.data(), kI, hq.data());
        for (int j = 0; j < kI; ++j) hd[p][j] = hq[j];
      }
    }
    std::vector<float> drow(kI);
    for (int n = 0; n < kH; ++n) {
      dequant_row(L.down, blob + L.gate_bytes + L.up_bytes + std::size_t(n) * dn_rb, kI, drow.data());
      if (mode == cpu::PoolMode::Prefill)
        for (float& v : drow) v = bf16_bits_to_float(bf16_bits(v));
      for (std::size_t p = 0; p < pairs.size(); ++p) {
        double s = 0;
        for (int j = 0; j < kI; ++j) s += double(drow[j]) * hd[p][j];
        y[pairs[p].first * kH + n] += double(pairs[p].second) * s;
      }
    }
  }
  return y;
}

std::vector<Row> make_rows(const LayerData& ld, int n, Rng& rng) {
  std::vector<Row> rows(n);
  for (Row& r : rows) {
    r.x.resize(kH);
    const double scale = 0.5 + 2.0 * rng.uniform();
    for (auto& v : r.x) v = bf16_bits(float(rng.normal() * scale));
    std::vector<int> idx(ld.universe.size());
    std::iota(idx.begin(), idx.end(), 0);
    for (int j = 0; j < kK; ++j) std::swap(idx[j], idx[j + rng.next() % (idx.size() - j)]);
    double total = 0;
    std::vector<double> w(kK);
    for (auto& v : w) total += (v = 0.05 + rng.uniform());
    for (int j = 0; j < kK; ++j) {
      if (!ld.cold[idx[j]]) continue;  // hot experts are computed by the GPU
      r.experts.push_back(ld.universe[idx[j]]);
      r.weights.push_back(float(w[j] / total));
    }
  }
  return rows;
}

void publish_decode(DecodeExchange* ex, int l, const std::vector<Row>& rows) {
  MoeDecodePublish& p = ex->publish[l];
  p.tokens = int(rows.size());
  int n_cold = 0;
  for (std::size_t t = 0; t < rows.size(); ++t) {
    p.cold_count[t] = int(rows[t].experts.size());
    for (std::size_t j = 0; j < rows[t].experts.size(); ++j) {
      p.cold_expert[t][j] = rows[t].experts[j];
      p.cold_weight[t][j] = rows[t].weights[j];
    }
    n_cold += p.cold_count[t];
    std::memcpy(&p.x[t][0], rows[t].x.data(), kH * 2);
  }
  p.n_cold = n_cold;
}

void publish_prefill(PrefillExchangeView& p, const std::vector<Row>& rows, int chunk_tokens) {
  int n_cold = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    p.row_token[i] = int(i) * 3 + 1;
    p.cold_count[i] = int(rows[i].experts.size());
    for (std::size_t j = 0; j < rows[i].experts.size(); ++j) {
      p.cold_expert[i * kK + j] = rows[i].experts[j];
      p.cold_weight[i * kK + j] = rows[i].weights[j];
    }
    n_cold += p.cold_count[i];
    std::memcpy(p.x + i * kH, rows[i].x.data(), kH * 2);
  }
  p.header[0] = chunk_tokens;
  p.header[1] = int(rows.size());
  p.header[2] = n_cold;
}

// Every row's ||y - ref||_2 / ||ref||_2 <= 1e-3. FP32 vs FP64 accumulation can cross
// one activation rounding boundary; max|err|/rms is printed for information.
bool check(const char* what, const float* got, const std::vector<double>& ref) {
  const std::size_t rows = ref.size() / kH;
  double worst = 0;
  std::size_t worst_row = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    double e2 = 0, r2 = 0;
    for (int n = 0; n < kH; ++n) {
      const double d = double(got[r * kH + n]) - ref[r * kH + n];
      e2 += d * d;
      r2 += ref[r * kH + n] * ref[r * kH + n];
    }
    const double rel = r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2);
    if (rel > worst) {
      worst = rel;
      worst_row = r;
    }
  }
  const ErrorStats st = compare(got, ref.data(), ref.size());
  const bool ok = worst <= 1e-3;
  std::printf("%s %s: max row L2 rel err %.3e (row %zu, tol 1e-3); max_abs/rms(ref) %.3e\n",
              ok ? "PASS" : "FAIL", what, worst, worst_row, st.max_rel_to_rms);
  return ok;
}

}  // namespace

int main() {
  const std::string pack_dir = env_or_skip("FNGINE_TEST_PACK");
  if (pack_dir.empty()) return kSkip;
  PackExperts pack;
  pack.open(pack_dir);
  Rng rng(2024);

  // Layer 0: IQ3_XXS gate/up + IQ4_NL down; layer 1: IQ2_S + Q2_0 down.
  const int test_layers[2] = {0, 1};
  std::vector<LayerData> lds(2);
  std::vector<const void*> slots(std::size_t(kExchangeLayers) * dims::kExperts, nullptr);
  cpu::PoolConfig cfg;
  for (int i = 0; i < 2; ++i) {
    LayerData& ld = lds[i];
    ld.layer = test_layers[i];
    const PackExperts::Layer& L = pack.layer(ld.layer);
    std::vector<int> all(dims::kExperts);
    std::iota(all.begin(), all.end(), 0);
    for (int j = 0; j < 20; ++j) std::swap(all[j], all[j + rng.next() % (all.size() - j)]);
    ld.universe.assign(all.begin(), all.begin() + 20);
    for (int e : ld.universe) {
      const bool cold = rng.uniform() < 0.6;
      ld.cold.push_back(cold);
      ld.blobs.push_back(std::make_unique<HostBuf>(L.blob_bytes));
      pack.read_blob(ld.layer, e, ld.blobs.back()->p);
      // Hot experts have no host blob: the pool must never ask for them.
      if (cold) slots[std::size_t(ld.layer) * dims::kExperts + e] = ld.blobs.back()->p;
    }
    cfg.layers[ld.layer] = cpu::ExpertLayerDesc{L.gu, L.down, 0, L.gate_bytes, L.gate_bytes + L.up_bytes};
  }
  constexpr int kPrefillRows = 40;
  HostExchange hx(kPrefillRows);
  cfg.host_slot_of = slots.data();
  cfg.decode = hx.dec;
  cfg.prefill = hx.pre;
  bool ok = true;
  uint32_t seq = 100;

  // ---------------------------------------------------------------- decode, T = 9
  std::vector<std::vector<Row>> drows = {make_rows(lds[0], 9, rng), make_rows(lds[1], 9, rng)};
  for (int i = 0; i < 2; ++i) publish_decode(hx.dec, lds[i].layer, drows[i]);
  std::vector<std::vector<float>> y_first(2);
  {
    cpu::ExpertPool pool(cfg);
    for (int rep = 0; rep < 2; ++rep) {
      ++seq;
      pool.begin(seq, cpu::PoolMode::Decode);
      hx.release_all(seq);
      for (const LayerData& ld : lds) hx.wait_done(ld.layer, seq);
      pool.wait_idle();
      for (int i = 0; i < 2; ++i) {
        const float* y = &hx.dec->result[lds[i].layer].y[0][0];
        if (rep == 0) {
          y_first[i].assign(y, y + 9 * kH);
        } else if (std::memcmp(y_first[i].data(), y, 9 * kH * sizeof(float)) != 0) {
          std::printf("FAIL decode L%d: second run differs\n", lds[i].layer);
          ok = false;
        }
      }
    }
    if (pool.failed()) {
      std::printf("FAIL pool reported: %s\n", pool.failure().c_str());
      ok = false;
    }
    const auto tm = pool.timings();
    for (const LayerData& ld : lds)
      std::printf("decode L%d: %llu cold pairs, %llu experts, compute %.1f us/step\n", ld.layer,
                  (unsigned long long)(tm[ld.layer].pairs / 2), (unsigned long long)(tm[ld.layer].experts / 2),
                  double(tm[ld.layer].compute_ns) / 2e3);
    for (int l = 0; l < kExchangeLayers; ++l) {
      if (l == lds[0].layer || l == lds[1].layer) continue;
      if (hx.dec->control.done[l].value != 0) {
        std::printf("FAIL layer %d without cold pairs got done written\n", l);
        ok = false;
      }
    }
  }
  for (int i = 0; i < 2; ++i) {
    const std::vector<double> ref = oracle(pack, lds[i], drows[i], cpu::PoolMode::Decode);
    char what[96];
    std::snprintf(what, sizeof(what), "decode T=9 L%d %s/%s vs FP64", lds[i].layer,
                  cpu::format_name(pack.layer(lds[i].layer).gu), cpu::format_name(pack.layer(lds[i].layer).down));
    ok &= check(what, y_first[i].data(), ref);
  }
  // Same step with 3 workers: bit-identical (fixed accumulation order).
  {
    cpu::PoolConfig c3 = cfg;
    c3.worker_cpus = {1, 2, 3};
    cpu::ExpertPool pool(c3);
    ++seq;
    pool.begin(seq, cpu::PoolMode::Decode);
    hx.release_all(seq);
    for (const LayerData& ld : lds) hx.wait_done(ld.layer, seq);
    for (int i = 0; i < 2; ++i) {
      const bool same = std::memcmp(y_first[i].data(), &hx.dec->result[lds[i].layer].y[0][0],
                                    9 * kH * sizeof(float)) == 0;
      std::printf("%s decode L%d: 3 workers bit-identical to 10 workers\n", same ? "PASS" : "FAIL", lds[i].layer);
      ok &= same;
    }
  }

  // ---------------------------------------------------------------- prefill, 40 compacted rows
  {
    cpu::ExpertPool pool(cfg);
    // Layers with no cold pairs in the decode exchange must not hold old publishes.
    for (const LayerData& ld : lds) hx.dec->publish[ld.layer].n_cold = 0;
    std::vector<std::vector<Row>> prows = {make_rows(lds[0], kPrefillRows, rng),
                                           make_rows(lds[1], kPrefillRows, rng)};
    ++seq;
    pool.begin(seq, cpu::PoolMode::Prefill);
    std::vector<std::vector<float>> y(2);
    // Played layer by layer; every layer (also those without cold pairs) must be acknowledged.
    hx.prefill_step(seq, [&](int l) {
      for (int i = 0; i < 2; ++i)  // y of a finished cold layer, before the region is reused
        if (l == lds[i].layer + 1) y[i].assign(hx.pre.y, hx.pre.y + std::size_t(kPrefillRows) * kH);
      for (int i = 0; i < 2; ++i) {
        if (l == lds[i].layer) {
          publish_prefill(hx.pre, prows[i], 3 * kPrefillRows + 7);
          return;
        }
      }
      hx.pre.header[1] = 0;
      hx.pre.header[2] = 0;
    });
    pool.wait_idle();
    if (pool.failed()) {
      std::printf("FAIL pool reported: %s\n", pool.failure().c_str());
      ok = false;
    }
    for (int i = 0; i < 2; ++i) {
      const std::vector<double> ref = oracle(pack, lds[i], prows[i], cpu::PoolMode::Prefill);
      char what[96];
      std::snprintf(what, sizeof(what), "prefill rows=%d L%d vs FP64", kPrefillRows, lds[i].layer);
      ok &= check(what, y[i].data(), ref);
    }
  }
  std::printf("%s cpu_expert_pool\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
