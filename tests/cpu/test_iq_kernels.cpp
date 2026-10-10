// CPU expert kernels vs an FP64 oracle over ggml reference dequantization.
//  * synthetic random blocks for every format (always), real pack blobs (FNGINE_TEST_PACK)
//  * T in {1, 3, 9, 64}: max|err| <= 1e-4 * rms(ref); T > 1 equals T = 1 per token (bit-exact)
//  * pointer-list variant equals the strided variant; the multi-row GEMM over decoded rows
//    equals the single-row path bit for bit (results independent of the tile shape)
//  * microbenchmark through the ExpertPool (pool_bench.h): us per expert at T = 1 (10 and 12
//    workers), T = 32, T = 128, and a long-tailed T = 512 prefill.
#include "cpu/cpu_test_util.h"
#include "cpu/pool_bench.h"
#include "cpu/q8_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace fngine;
using namespace fngine::test;

namespace {

constexpr int kMaxT = 64;
const int kTs[] = {1, 3, 9, 64};

struct Acts {
  ActFormat fmt;
  int k;
  std::size_t stride;
  std::vector<uint8_t> q;        // [kMaxT][stride]
  std::vector<double> deq;       // [kMaxT][k]
};

Acts make_acts(ActFormat fmt, int k, Rng& rng) {
  Acts a{fmt, k, cpu::act_row_bytes(fmt, k), {}, {}};
  a.q.resize(a.stride * kMaxT);
  a.deq.resize(std::size_t(k) * kMaxT);
  std::vector<uint16_t> xb(static_cast<std::size_t>(k));
  std::vector<float> tmp(static_cast<std::size_t>(k));
  for (int t = 0; t < kMaxT; ++t) {
    const double scale = 0.2 + 3.0 * rng.uniform();
    for (int i = 0; i < k; ++i) xb[i] = bf16_bits(float(rng.normal() * scale));
    uint8_t* dst = a.q.data() + a.stride * t;
    if (fmt == ActFormat::Q8_K) cpu::quantize_row_q8_K_bf16(xb.data(), reinterpret_cast<block_q8_K*>(dst), k);
    else cpu::quantize_row_q8_0_bf16(xb.data(), reinterpret_cast<block_q8_0*>(dst), k);
    dequant_act(fmt, dst, k, tmp.data());
    for (int i = 0; i < k; ++i) a.deq[std::size_t(t) * k + i] = tmp[i];
  }
  return a;
}

bool check_bf16(WeightFormat fmt, const uint8_t* rows, int nrows, int k, Rng& rng) {
  // Ragged row/token tiles and pointer permutation must preserve arithmetic, as CPU/GPU
  // placement can change the token group of an expert without changing its result.
  --nrows;
  const std::size_t rb = cpu::weight_row_bytes(fmt, k);
  std::vector<uint16_t> weights(std::size_t(nrows) * k), acts(std::size_t(kMaxT) * k);
  std::vector<const void*> ptrs(kMaxT);
  for (auto& v : acts) v = bf16_bits(float(rng.normal()));
  for (int t = 0; t < kMaxT; ++t) ptrs[t] = acts.data() + std::size_t(kMaxT - 1 - t) * k;
  cpu::decode_rows_bf16(fmt, rows, rb, nrows, k, weights.data());
  std::vector<float> deq(k), single(std::size_t(nrows) * kMaxT);
  std::vector<double> reference(single.size());
  bool ok = true;
  for (int r = 0; r < nrows; ++r) {
    dequant_row(fmt, rows + std::size_t(r) * rb, k, deq.data());
    for (int i = 0; i < k; ++i) ok &= weights[std::size_t(r) * k + i] == bf16_bits(deq[i]);
    for (int t = 0; t < kMaxT; ++t) {
      double sum = 0;
      const auto* x = static_cast<const uint16_t*>(ptrs[t]);
      for (int i = 0; i < k; ++i)
        sum += double(bf16_bits_to_float(bf16_bits(deq[i]))) * bf16_bits_to_float(x[i]);
      reference[std::size_t(r) * kMaxT + t] = sum;
      cpu::gemm_bf16(weights.data() + std::size_t(r) * k, 1, k, ptrs.data() + t,
                     1, single.data() + std::size_t(r) * kMaxT + t, 1);
    }
  }
  for (int T : kTs) {
    std::vector<float> got(std::size_t(nrows) * T);
    std::vector<double> want(got.size());
    cpu::gemm_bf16(weights.data(), nrows, k, ptrs.data(), T, got.data(), T);
    for (int r = 0; r < nrows; ++r)
      for (int t = 0; t < T; ++t) {
        ok &= got[std::size_t(r) * T + t] == single[std::size_t(r) * kMaxT + t];
        want[std::size_t(r) * T + t] = reference[std::size_t(r) * kMaxT + t];
      }
    char label[128];
    std::snprintf(label, sizeof(label), "BF16 %s K=%d T=%d vs FP64", cpu::format_name(fmt), k, T);
    ok &= report(label, compare(got.data(), want.data(), got.size()), 1e-4);
  }
  std::printf("%s BF16 %s K=%d: exact dequantization and batch-invariant arithmetic\n",
              ok ? "PASS" : "FAIL", cpu::format_name(fmt), k);
  return ok;
}

// Checks `nrows` rows (row r at rows + r*row_bytes) of format `fmt` with K = k.
bool check_rows(const char* label, WeightFormat fmt, const uint8_t* rows, int nrows, int k,
                Rng& rng) {
  const cpu::FormatInfo fi = cpu::format_info(fmt);
  const std::size_t row_bytes = cpu::weight_row_bytes(fmt, k);
  const int nblk = k / fi.block_elems;
  const Acts acts = make_acts(fi.act, k, rng);
  auto dot = [fmt](const void* row, int nb, const void* a, int T, std::size_t stride, float* out) {
    cpu::cpu_dot(fmt, row, nb, a, T, stride, out);
  };
  auto dotp = [fmt](const void* row, int nb, const void* const* a, int T, float* out) {
    cpu::cpu_dot_ptr(fmt, row, nb, a, T, out);
  };

  // Oracle [row][t] and the T = 1 results.
  std::vector<double> ref(std::size_t(nrows) * kMaxT);
  std::vector<float> one(std::size_t(nrows) * kMaxT);
  std::vector<float> w(static_cast<std::size_t>(k));
  for (int r = 0; r < nrows; ++r) {
    const uint8_t* row = rows + std::size_t(r) * row_bytes;
    dequant_row(fmt, row, k, w.data());
    for (int t = 0; t < kMaxT; ++t) {
      double s = 0;
      const double* a = acts.deq.data() + std::size_t(t) * k;
      for (int i = 0; i < k; ++i) s += double(w[i]) * a[i];
      ref[std::size_t(r) * kMaxT + t] = s;
      dot(row, nblk, acts.q.data() + acts.stride * t, 1, acts.stride, &one[std::size_t(r) * kMaxT + t]);
    }
  }
  bool ok = check_bf16(fmt, rows, nrows, k, rng);
  {
    // All rows decoded at once, every token in one GEMM: must equal the T = 1 single-row results.
    const std::size_t pb = cpu::packed_act_bytes(fi.act, k);
    const std::size_t db = cpu::decoded_row_bytes(fmt, k);
    HostBuf packed(pb * kMaxT, 64), dec(db * std::size_t(nrows), 64);
    std::vector<const void*> ptrs(static_cast<std::size_t>(kMaxT));
    for (int t = 0; t < kMaxT; ++t) {
      const uint8_t* src = acts.q.data() + acts.stride * t;
      if (fi.act == ActFormat::Q8_K) cpu::pack_act_q8_K(reinterpret_cast<const block_q8_K*>(src), k, packed.p + pb * t);
      else cpu::pack_act_q8_0(reinterpret_cast<const block_q8_0*>(src), k, packed.p + pb * t);
      ptrs[t] = packed.p + pb * t;
    }
    cpu::decode_rows(fmt, rows, row_bytes, nrows, k, dec.p);
    std::vector<float> all(std::size_t(nrows) * kMaxT);
    cpu::gemm_decoded(fmt, dec.p, nrows, k, ptrs.data(), kMaxT, all.data(), kMaxT);
    const bool same = std::memcmp(all.data(), one.data(), all.size() * sizeof(float)) == 0;
    std::printf("%s %s %s K=%d: %d-row x %d-token GEMM bit-identical to single-row T=1\n",
                same ? "PASS" : "FAIL", label, cpu::format_name(fmt), k, nrows, kMaxT);
    ok &= same;
  }
  for (int T : kTs) {
    std::vector<float> got(std::size_t(nrows) * T);
    std::vector<double> want(std::size_t(nrows) * T);
    double max_rel_one = 0;
    for (int r = 0; r < nrows; ++r) {
      const uint8_t* row = rows + std::size_t(r) * row_bytes;
      float* out = got.data() + std::size_t(r) * T;
      dot(row, nblk, acts.q.data(), T, acts.stride, out);
      for (int t = 0; t < T; ++t) {
        want[std::size_t(r) * T + t] = ref[std::size_t(r) * kMaxT + t];
        const float o1 = one[std::size_t(r) * kMaxT + t];
        const double rel = std::fabs(double(out[t]) - o1) / std::max(1e-30, std::fabs(double(o1)));
        max_rel_one = std::max(max_rel_one, rel);
      }
      // Pointer variant over the reversed token order.
      std::vector<const void*> ptrs(static_cast<std::size_t>(T));
      std::vector<float> outp(static_cast<std::size_t>(T));
      for (int t = 0; t < T; ++t) ptrs[t] = acts.q.data() + acts.stride * (T - 1 - t);
      dotp(row, nblk, ptrs.data(), T, outp.data());
      for (int t = 0; t < T; ++t) {
        if (outp[t] != out[T - 1 - t]) {
          std::printf("FAIL %s T=%d: pointer variant differs (row %d token %d)\n", label, T, r, t);
          ok = false;
          break;
        }
      }
    }
    const ErrorStats st = compare(got.data(), want.data(), got.size());
    char what[160];
    std::snprintf(what, sizeof(what), "%s %s K=%d rows=%d T=%d vs FP64", label,
                  cpu::format_name(fmt), k, nrows, T);
    ok &= report(what, st, 1e-4);
    const bool same = max_rel_one == 0.0;
    std::printf("%s %s %s K=%d T=%d vs T=1: max rel %.3e (bit-exact required)\n", same ? "PASS" : "FAIL",
                label, cpu::format_name(fmt), k, T, max_rel_one);
    ok &= same;
  }
  return ok;
}

}  // namespace

int main() {
  Rng rng(42);
  bool ok = true;

  // Synthetic blocks: every format, model row lengths plus an odd IQ4_NL block count.
  struct Syn { WeightFormat fmt; int k; };
  const Syn syn[] = {{WeightFormat::IQ2_S, 2560},  {WeightFormat::IQ3_XXS, 2560},
                     {WeightFormat::IQ3_S, 2560},  {WeightFormat::IQ4_XS, 2560},
                     {WeightFormat::IQ4_NL, 2560}, {WeightFormat::IQ4_NL, 640},
                     {WeightFormat::IQ4_NL, 160},  {WeightFormat::Q2_0, 640},
                     {WeightFormat::Q2_0, 2560}};
  for (const Syn& s : syn) {
    const int nrows = 8;
    const int nblk = s.k / cpu::format_info(s.fmt).block_elems;
    const std::vector<uint8_t> rows = random_blocks(s.fmt, nblk * nrows, rng);
    ok &= check_rows("synthetic", s.fmt, rows.data(), nrows, s.k, rng);
  }

  const std::string pack_dir = env_or_skip("FNGINE_TEST_PACK");
  if (pack_dir.empty()) return ok ? kSkip : 1;
  PackExperts pack;
  pack.open(pack_dir);

  // Real blobs: gate rows (K=2560) per gate/up format, down rows (K=640) per down format.
  struct Real { int layer; int which; };  // which: 0 gate, 1 up, 2 down
  const Real real[] = {{0, 0}, {2, 0}, {17, 0}, {47, 0}, {47, 1}, {-1, 0}, {0, 2}, {1, 2}};
  for (const Real& r : real) {
    const PackExperts::Layer& L = pack.layer(r.layer);
    HostBuf blob(L.blob_bytes);
    pack.read_blob(r.layer, 5, blob.p);
    const WeightFormat fmt = r.which == 2 ? L.down : L.gu;
    const int k = r.which == 2 ? dims::kExpertInter : dims::kHidden;
    const std::size_t off = r.which == 0 ? 0 : r.which == 1 ? L.gate_bytes : L.gate_bytes + L.up_bytes;
    char label[64];
    std::snprintf(label, sizeof(label), "real L%d.%s", r.layer, r.which == 0 ? "gate" : r.which == 1 ? "up" : "down");
    ok &= check_rows(label, fmt, blob.p + off + 100 * cpu::weight_row_bytes(fmt, k), 24, k, rng);
  }

  // Microbenchmark through the pool: 160 experts per layer (> L3), rotating.
  const std::vector<int> w10 = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};
  const std::vector<int> w12 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::printf("\nmicrobenchmark: us per expert (gate+up+down); T=1 decode with 10 and 12 workers,"
              " T=32/T=128 prefill (10 workers, every row on the same 10 experts),\n"
              "skewed = 512-row prefill, Zipf routing over 160 experts (us per token-expert pair)\n");
  const int bench_layers[] = {2, 0, 17, 47, 1};
  for (int l : bench_layers) {
    const BenchLayer bl(pack, l, 160);
    const BenchStats d10 = bench_pool(bl, w10, cpu::PoolMode::Decode, 1, 64, false);
    const BenchStats d12 = bench_pool(bl, w12, cpu::PoolMode::Decode, 1, 64, false);
    const BenchStats p32 = bench_pool(bl, w10, cpu::PoolMode::Prefill, 32, 16, false);
    const BenchStats p128 = bench_pool(bl, w10, cpu::PoolMode::Prefill, 128, 6, false);
    const BenchStats sk = bench_pool(bl, w10, cpu::PoolMode::Prefill, 512, 3, true);
    std::printf("BENCH L%-2d %-7s/%-6s T=1: %6.1f us (12w %6.1f)  T=32: %7.1f us  T=128: %8.1f us"
                "  skewed512: %5.2f us/pair (%.1f tok/expert, %.0f us/step)\n",
                l, cpu::format_name(bl.desc.gu), cpu::format_name(bl.desc.down), d10.us_per_expert,
                d12.us_per_expert, p32.us_per_expert, p128.us_per_expert, sk.us_per_pair,
                sk.tokens_per_expert, sk.us_step);
  }
  std::printf("%s cpu_iq_kernels\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
