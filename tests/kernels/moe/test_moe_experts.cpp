// MoE expert kernels on real pack blobs vs FP64 oracles (ggml-iq reference dequantization).
//   decode (T = 1, 3, 9): router top-k + publish, hot gate/up (h FP32) and down (partials) against
//     an oracle over the kernels' q8 activations: <= 1e-4 max|err|/rms per (token, expert) output.
//   prefill (T = 1, 3, 9, 64): route + grouped GEMMs against an oracle over BF16-rounded reference
//     weights and BF16 activations: h within BF16 rounding, y <= 1e-4 max|err|/rms per token.
// `--bench` additionally measures decode (T=1, 10 hot experts) and prefill (T=4096, 512 hot) on one
// layer per gate/up codebook format; `--bench-only` skips the checks.
// Needs FNGINE_TEST_PACK (skips otherwise).
#include "moe_test_common.h"

#include "ggml-iq/ggml-iq.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <string>

using namespace fngine;
using namespace fngine::test;
using namespace fngine::test::moe;
using fngine::kernels::MoeExpertLayout;

namespace {

constexpr double kTol = 1e-4;
constexpr int kLoaded = 12;

MoeFormat parse_format(const std::string& s) {
  static const std::map<std::string, MoeFormat> m = {
      {"IQ2_S", MoeFormat::IQ2_S},   {"IQ3_XXS", MoeFormat::IQ3_XXS}, {"IQ3_S", MoeFormat::IQ3_S},
      {"IQ4_XS", MoeFormat::IQ4_XS}, {"IQ4_NL", MoeFormat::IQ4_NL},   {"Q2_0", MoeFormat::Q2_0}};
  auto it = m.find(s);
  if (it == m.end()) {
    std::fprintf(stderr, "unknown expert format %s\n", s.c_str());
    std::exit(1);
  }
  return it->second;
}

struct Case {
  std::string name;
  int layer;  // exchange layer index (MTP uses 0 with no CPU tier semantics)
  bool mtp;
  MoeFormat gu, down;
  std::string file;
  int64_t offset, stride, blob_bytes;
};

void dequant_row(MoeFormat f, const uint8_t* src, float* dst, int k) {
  switch (f) {
    case MoeFormat::IQ2_S: dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s*>(src), dst, k); break;
    case MoeFormat::IQ3_XXS: dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs*>(src), dst, k); break;
    case MoeFormat::IQ3_S: dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s*>(src), dst, k); break;
    case MoeFormat::IQ4_XS: dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs*>(src), dst, k); break;
    case MoeFormat::IQ4_NL: dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl*>(src), dst, k); break;
    case MoeFormat::Q2_0: dequantize_row_q2_0(reinterpret_cast<const block_q2_0*>(src), dst, k); break;
  }
}

struct ExpertW {
  std::vector<float> gate, up, down;  // [640][2560], [640][2560], [2560][640]
};

struct Loaded {
  Case c;
  std::vector<int> experts;          // slot -> expert
  std::vector<int32_t> slot_of;      // expert -> slot / -1
  DeviceArray<uint8_t> slots;
  DeviceArray<const uint8_t*> blob_table;  // expert -> device blob / nullptr (prefill)
  std::vector<ExpertW> w;            // per slot (empty when oracle weights are not needed)
  MoeExpertLayout layout;
};

std::vector<uint8_t> read_blob(int fd, int64_t off, int64_t bytes) {
  std::vector<uint8_t> b(bytes);
  int64_t done = 0;
  while (done < bytes) {
    const ssize_t r = pread(fd, b.data() + done, bytes - done, off + done);
    if (r <= 0) {
      std::perror("pread");
      std::exit(1);
    }
    done += r;
  }
  return b;
}

void load(Loaded& L, const std::string& pack, const std::vector<int>& experts, bool oracle) {
  const Case& c = L.c;
  L.experts = experts;
  L.slot_of.assign(kExperts, -1);
  const std::string path = pack + "/" + c.file;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::perror(path.c_str());
    std::exit(1);
  }
  assign(L.slots, DeviceArray<uint8_t>(size_t(experts.size()) * c.stride));
  if (oracle) L.w.resize(experts.size());
  const int64_t gu_row = kernels::moe_row_bytes(c.gu, kHidden);
  const int64_t dn_row = kernels::moe_row_bytes(c.down, kInter);
  for (size_t s = 0; s < experts.size(); ++s) {
    L.slot_of[experts[s]] = int32_t(s);
    auto blob = read_blob(fd, c.offset + int64_t(experts[s]) * c.stride, c.blob_bytes);
    FNGINE_TEST_CUDA(cudaMemcpy(L.slots.ptr + s * c.stride, blob.data(), blob.size(),
                                cudaMemcpyHostToDevice));
    if (!oracle) continue;
    ExpertW& w = L.w[s];
    w.gate.resize(size_t(kInter) * kHidden);
    w.up.resize(size_t(kInter) * kHidden);
    w.down.resize(size_t(kHidden) * kInter);
    const uint8_t* g = blob.data();
    const uint8_t* u = g + kInter * gu_row;
    const uint8_t* d = u + kInter * gu_row;
    for (int r = 0; r < kInter; ++r) {
      dequant_row(c.gu, g + r * gu_row, &w.gate[size_t(r) * kHidden], kHidden);
      dequant_row(c.gu, u + r * gu_row, &w.up[size_t(r) * kHidden], kHidden);
    }
    for (int r = 0; r < kHidden; ++r) dequant_row(c.down, d + r * dn_row, &w.down[size_t(r) * kInter], kInter);
  }
  close(fd);
  L.layout = kernels::make_moe_expert_layout(L.slots.ptr, c.stride, c.gu, c.down);
  std::vector<const uint8_t*> table(kExperts, nullptr);
  for (int e = 0; e < kExperts; ++e)
    if (L.slot_of[e] >= 0) table[e] = L.slots.ptr + int64_t(L.slot_of[e]) * c.stride;
  assign(L.blob_table, DeviceArray<const uint8_t*>(table));
}

struct Worst {
  double ratio = 0;
  std::string where;
  void add(const ErrorStats& s, const std::string& w) {
    if (s.max_rel_to_rms > ratio || where.empty()) {
      ratio = std::max(ratio, s.max_rel_to_rms);
      where = w;
    }
  }
};

bool check(bool cond, const char* what, int& failures) {
  if (!cond) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
  return cond;
}

// ------------------------------------------------------------------------------- decode
int test_decode(Loaded& L, int T, uint64_t seed) {
  int failures = 0;
  const Case& c = L.c;
  const int layer = c.layer;
  auto logits = make_logits(T, L.experts, 4.0f, seed);
  if (c.mtp) {
    // MTP has no CPU tier: keep routing inside the loaded (GPU-resident) experts.
    for (int t = 0; t < T; ++t)
      for (int e = 0; e < kExperts; ++e)
        if (L.slot_of[e] < 0) logits[size_t(t) * kRouter + e] = -30.f;
  } else {
    // Force one cold pair for token 0 (an expert that is not GPU-resident).
    for (int e = 0; e < kExperts; ++e)
      if (L.slot_of[e] < 0) {
        logits[e] += 8.0f;
        break;
      }
  }
  auto x = make_x(T, seed + 1);
  DeviceArray<float> d_logits(logits);
  DeviceArray<__nv_bfloat16> d_x(x);
  DeviceArray<int32_t> d_slot(L.slot_of);
  DeviceArray<uint32_t> d_seq(std::vector<uint32_t>{7u});
  Mapped<DecodeExchange> ex(sizeof(DecodeExchange));
  DeviceArray<uint8_t> ws_mem(kernels::moe_decode_workspace_bytes());
  auto ws = kernels::make_moe_decode_workspace(ws_mem.ptr);

  FNGINE_TEST_CUDA(kernels::moe_select_publish(d_logits.ptr, d_x.ptr, d_slot.ptr, layer,
                                               d_seq.ptr, c.mtp ? nullptr : ex.dev, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(L.layout, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_down(L.layout, T, ws, 0));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());

  kernels::MoeDecodeWork work;
  FNGINE_TEST_CUDA(cudaMemcpy(&work, ws.work, sizeof(work), cudaMemcpyDeviceToHost));
  std::vector<int8_t> xq(size_t(T) * kHidden);
  std::vector<float> xd(size_t(T) * kernels::kMoeHiddenBlocks);
  std::vector<float> h(size_t(work.n_pairs) * kInter), part(size_t(work.n_pairs) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(xq.data(), ws.xq, xq.size(), cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(xd.data(), ws.xd, xd.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(h.data(), ws.h, h.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(part.data(), ws.partial, part.size() * 4, cudaMemcpyDeviceToHost));

  // Router.
  std::vector<TopK> ref(T);
  int ref_hot = 0, ref_cold = 0;
  double max_wdiff = 0;
  bool ids_ok = true;
  for (int t = 0; t < T; ++t) {
    ref[t] = oracle_topk(&logits[size_t(t) * kRouter]);
    for (int k = 0; k < kTopK; ++k) {
      ids_ok &= work.topk_expert[t][k] == ref[t].id[k];
      max_wdiff = std::max(max_wdiff, std::fabs(work.topk_weight[t][k] - ref[t].w[k]));
      const int slot = L.slot_of[ref[t].id[k]];
      (slot >= 0 ? ref_hot : ref_cold)++;
    }
  }
  check(ids_ok, "decode router top-k ids", failures);
  check(max_wdiff <= 1e-6, "decode router weights", failures);
  check(work.n_pairs == ref_hot && work.n_cold == ref_cold, "decode hot/cold counts", failures);

  // q8 activations bit-identical to the oracle quantizer.
  int q8_mismatch = 0;
  std::vector<std::vector<double>> xdeq(T);
  for (int t = 0; t < T; ++t) {
    std::vector<float> xf(kHidden);
    for (int i = 0; i < kHidden; ++i) xf[i] = bf16_to_float(x[size_t(t) * kHidden + i]);
    for (int b = 0; b < kHidden / 32; ++b) {
      int8_t q[32];
      float d;
      quantize_q8_block(&xf[32 * b], q, &d);
      if (d != xd[t * kernels::kMoeHiddenBlocks + b]) ++q8_mismatch;
      for (int j = 0; j < 32; ++j) q8_mismatch += q[j] != xq[size_t(t) * kHidden + 32 * b + j];
    }
    xdeq[t] = q8_dequant_row(xf.data(), kHidden);
  }
  check(q8_mismatch == 0, "decode q8 activations", failures);

  // Work list: items ascending, pair -> (token, expert, weight) consistent with the oracle.
  std::vector<int> pair_slot(work.n_pairs, -1);
  bool list_ok = work.n_items >= 1 && work.item_pair_begin[0] == 0 &&
                 work.item_pair_begin[work.n_items] == work.n_pairs;
  for (int i = 0; i < work.n_items; ++i) {
    if (i > 0) list_ok &= work.item_slot[i] > work.item_slot[i - 1];
    for (int p = work.item_pair_begin[i]; p < work.item_pair_begin[i + 1]; ++p) {
      pair_slot[p] = work.item_slot[i];
      if (p > work.item_pair_begin[i]) list_ok &= work.pair_token[p] > work.pair_token[p - 1];
    }
  }
  std::vector<int> pair_expert(work.n_pairs);
  std::vector<double> pair_w(work.n_pairs);
  for (int p = 0; p < work.n_pairs; ++p) {
    const int e = L.experts[pair_slot[p]];
    const int t = work.pair_token[p];
    int k = 0;
    while (k < kTopK && ref[t].id[k] != e) ++k;
    list_ok &= k < kTopK && std::fabs(work.pair_weight[p] - ref[t].w[k]) <= 1e-6;
    pair_expert[p] = e;
    pair_w[p] = k < kTopK ? ref[t].w[k] : 0;
  }
  for (int t = 0; t < T; ++t)
    for (int j = 1; j < work.token_pair_count[t]; ++j)
      list_ok &= pair_slot[work.token_pairs[t][j]] > pair_slot[work.token_pairs[t][j - 1]];
  check(list_ok, "decode work list", failures);

  // Exchange publish.
  if (!c.mtp) {
    const auto& pub = ex.host->publish[layer];
    bool pub_ok = pub.tokens == T && pub.n_cold == ref_cold &&
                  ex.host->control.ready[layer].value == 7u;
    for (int t = 0; t < T; ++t) {
      int nc = 0;
      for (int k = 0; k < kTopK; ++k) {
        if (L.slot_of[ref[t].id[k]] >= 0) continue;
        pub_ok &= pub.cold_expert[t][nc] == ref[t].id[k] &&
                  std::fabs(pub.cold_weight[t][nc] - ref[t].w[k]) <= 1e-6;
        ++nc;
      }
      pub_ok &= pub.cold_count[t] == nc;
      if (nc > 0) pub_ok &= std::memcmp(pub.x[t], &x[size_t(t) * kHidden], kHidden * 2) == 0;
    }
    check(pub_ok, "decode exchange publish", failures);
  }

  // Expert outputs.
  Worst gu_worst, dn_worst;
  std::vector<ErrorStats> gu_stats(work.n_pairs), dn_stats(work.n_pairs);
  parallel_for(work.n_pairs, [&](int p) {
    const ExpertW& w = L.w[pair_slot[p]];
    const int t = work.pair_token[p];
    std::vector<double> href(kInter);
    for (int r = 0; r < kInter; ++r) {
      double g = 0, u = 0;
      const float* gr = &w.gate[size_t(r) * kHidden];
      const float* ur = &w.up[size_t(r) * kHidden];
      for (int i = 0; i < kHidden; ++i) {
        g += double(gr[i]) * xdeq[t][i];
        u += double(ur[i]) * xdeq[t][i];
      }
      href[r] = silu_d(g) * u;
    }
    gu_stats[p] = compare(&h[size_t(p) * kInter], href.data(), kInter);
    const auto hq = q8_dequant_row(&h[size_t(p) * kInter], kInter);
    std::vector<double> yref(kHidden);
    for (int n = 0; n < kHidden; ++n) {
      double s = 0;
      const float* dr = &w.down[size_t(n) * kInter];
      for (int j = 0; j < kInter; ++j) s += double(dr[j]) * hq[j];
      yref[n] = pair_w[p] * s;
    }
    dn_stats[p] = compare(&part[size_t(p) * kHidden], yref.data(), kHidden);
  });
  for (int p = 0; p < work.n_pairs; ++p) {
    gu_worst.add(gu_stats[p], "t" + std::to_string(work.pair_token[p]) + "/e" + std::to_string(pair_expert[p]));
    dn_worst.add(dn_stats[p], "t" + std::to_string(work.pair_token[p]) + "/e" + std::to_string(pair_expert[p]));
  }
  const bool ok_gu = gu_worst.ratio <= kTol, ok_dn = dn_worst.ratio <= kTol;
  std::printf("%s decode %-22s T=%d pairs=%d cold=%d gate_up(h) worst=%.3e [%s] down worst=%.3e [%s]\n",
              ok_gu && ok_dn && failures == 0 ? "PASS" : "FAIL", c.name.c_str(), T, work.n_pairs,
              work.n_cold, gu_worst.ratio, gu_worst.where.c_str(), dn_worst.ratio,
              dn_worst.where.c_str());
  return failures + !ok_gu + !ok_dn;
}

// ------------------------------------------------------------------------------ prefill
int test_prefill(Loaded& L, int T, uint64_t seed) {
  int failures = 0;
  const Case& c = L.c;
  const int layer = c.layer;
  constexpr int kCap = 64;
  auto logits = make_logits(T, L.experts, c.mtp ? 4.0f : 2.0f, seed);
  auto x = make_x(T, seed + 1);
  DeviceArray<float> d_logits(logits);
  DeviceArray<__nv_bfloat16> d_x(x);
  DeviceArray<uint32_t> d_seq(std::vector<uint32_t>{11u});
  Mapped<ExchangeControl> control(sizeof(ExchangeControl));
  Mapped<uint8_t> view_mem(prefill_exchange_bytes(kCap));
  const PrefillExchangeView hview = make_prefill_exchange_view(view_mem.host, kCap);
  const PrefillExchangeView dview = device_view(hview, view_mem.host, view_mem.dev);
  DeviceArray<uint8_t> ws_mem(kernels::moe_prefill_workspace_bytes(kCap));
  auto ws = kernels::make_moe_prefill_workspace(ws_mem.ptr, kCap);

  // MTP: every expert is GPU resident; restrict routing to the loaded experts by making the
  // logits of the others very small.
  if (c.mtp) {
    for (int t = 0; t < T; ++t)
      for (int e = 0; e < kExperts; ++e)
        if (L.slot_of[e] < 0) logits[size_t(t) * kRouter + e] = -30.f;
    d_logits.upload(logits);
  }
  // Experts outside the loaded set have no blob (cold); for MTP use the loaded experts only.
  FNGINE_TEST_CUDA(kernels::moe_route(d_logits.ptr, d_x.ptr, L.blob_table.ptr, layer, d_seq.ptr,
                                      c.mtp ? nullptr : control.dev,
                                      c.mtp ? PrefillExchangeView{} : dview, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_grouped_gate_up(L.layout, L.blob_table.ptr, d_x.ptr, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_grouped_down(L.layout, L.blob_table.ptr, T, ws, 0));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());

  int32_t meta[16];
  FNGINE_TEST_CUDA(cudaMemcpy(meta, ws.meta, sizeof(meta), cudaMemcpyDeviceToHost));
  const int n_pairs = meta[1];
  std::vector<int32_t> topk(size_t(T) * kTopK), ptok(n_pairs), offs(kExperts + 1);
  std::vector<float> topw(size_t(T) * kTopK), pw(n_pairs), y(size_t(T) * kHidden);
  std::vector<__nv_bfloat16> hb(size_t(n_pairs) * kInter);
  FNGINE_TEST_CUDA(cudaMemcpy(topk.data(), ws.topk_expert, topk.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(topw.data(), ws.topk_weight, topw.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(ptok.data(), ws.pair_token, ptok.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(pw.data(), ws.pair_weight, pw.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(offs.data(), ws.expert_offset, offs.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(hb.data(), ws.h, hb.size() * 2, cudaMemcpyDeviceToHost));
  y = download_y(ws, T);

  std::vector<TopK> ref(T);
  int ref_hot = 0, ref_cold = 0, ref_rows = 0;
  bool ids_ok = true;
  double max_wdiff = 0;
  for (int t = 0; t < T; ++t) {
    ref[t] = oracle_topk(&logits[size_t(t) * kRouter]);
    int nc = 0;
    for (int k = 0; k < kTopK; ++k) {
      ids_ok &= topk[t * kTopK + k] == ref[t].id[k];
      max_wdiff = std::max(max_wdiff, std::fabs(topw[t * kTopK + k] - ref[t].w[k]));
      if (L.slot_of[ref[t].id[k]] >= 0) ++ref_hot;
      else ++nc;
    }
    ref_cold += nc;
    ref_rows += nc > 0;
  }
  check(ids_ok, "prefill router top-k ids", failures);
  check(max_wdiff <= 1e-6, "prefill router weights", failures);
  check(n_pairs == ref_hot && meta[2] == ref_cold && meta[3] == ref_rows, "prefill counts", failures);

  // Expert lists: pairs of expert e are tokens that route to e, each exactly once.
  std::vector<int> pair_expert(n_pairs, -1);
  std::vector<double> pair_wref(n_pairs, 0);
  bool lists_ok = offs[kExperts] == n_pairs;
  std::set<std::pair<int, int>> seen;
  for (int e = 0; e < kExperts && lists_ok; ++e) {
    for (int p = offs[e]; p < offs[e + 1]; ++p) {
      const int t = ptok[p];
      int k = 0;
      while (k < kTopK && ref[t].id[k] != e) ++k;
      lists_ok &= k < kTopK && L.slot_of[e] >= 0 && seen.insert({t, e}).second &&
                  std::fabs(pw[p] - ref[t].w[k]) <= 1e-6;
      pair_expert[p] = e;
      pair_wref[p] = k < kTopK ? ref[t].w[k] : 0;
    }
  }
  check(lists_ok, "prefill expert lists", failures);

  if (!c.mtp) {
    bool pub_ok = hview.header[0] == T && hview.header[1] == ref_rows && hview.header[2] == ref_cold &&
                  control.host->ready[layer].value == 11u;
    int row = 0;
    for (int t = 0; t < T && pub_ok; ++t) {
      int nc = 0;
      for (int k = 0; k < kTopK; ++k) nc += L.slot_of[ref[t].id[k]] < 0;
      if (nc == 0) continue;
      pub_ok &= hview.row_token[row] == t && hview.cold_count[row] == nc &&
                std::memcmp(hview.x + size_t(row) * kHidden, &x[size_t(t) * kHidden], kHidden * 2) == 0;
      int j = 0;
      for (int k = 0; k < kTopK; ++k) {
        if (L.slot_of[ref[t].id[k]] >= 0) continue;
        pub_ok &= hview.cold_expert[row * kTopK + j] == ref[t].id[k] &&
                  std::fabs(hview.cold_weight[row * kTopK + j] - ref[t].w[k]) <= 1e-6;
        ++j;
      }
      ++row;
    }
    check(pub_ok, "prefill exchange publish", failures);
  }

  // h per pair: oracle over BF16-rounded weights and BF16 x; GPU h is BF16, so require agreement
  // within BF16 rounding of the FP64 value (2^-8 relative) plus 1e-4 * rms.
  std::vector<double> h_excess(n_pairs, 0.0), h_rel(n_pairs, 0.0);
  parallel_for(n_pairs, [&](int p) {
    const ExpertW& w = L.w[L.slot_of[pair_expert[p]]];
    const int t = ptok[p];
    std::vector<double> xr(kHidden), href(kInter);
    for (int i = 0; i < kHidden; ++i) xr[i] = bf16_to_float(x[size_t(t) * kHidden + i]);
    double sq = 0;
    for (int r = 0; r < kInter; ++r) {
      double g = 0, u = 0;
      const float* gr = &w.gate[size_t(r) * kHidden];
      const float* ur = &w.up[size_t(r) * kHidden];
      for (int i = 0; i < kHidden; ++i) {
        g += double(round_bf16(gr[i])) * xr[i];
        u += double(round_bf16(ur[i])) * xr[i];
      }
      href[r] = silu_d(g) * u;
      sq += href[r] * href[r];
    }
    const double rms = std::sqrt(sq / kInter);
    for (int r = 0; r < kInter; ++r) {
      const double got = bf16_to_float(hb[size_t(p) * kInter + r]);
      const double err = std::fabs(got - href[r]);
      h_excess[p] = std::max(h_excess[p], (err - std::ldexp(std::fabs(href[r]), -8)) / rms);
      h_rel[p] = std::max(h_rel[p], err / rms);
    }
  });
  double h_worst_excess = 0, h_worst_rel = 0;
  for (int p = 0; p < n_pairs; ++p) {
    h_worst_excess = std::max(h_worst_excess, h_excess[p]);
    h_worst_rel = std::max(h_worst_rel, h_rel[p]);
  }
  const bool h_ok = h_worst_excess <= kTol;

  // y per token: sum over hot pairs of w * down(bf16 h_gpu) with BF16-rounded down weights.
  std::vector<std::vector<double>> yref(T, std::vector<double>(kHidden, 0.0));
  std::vector<std::vector<int>> pairs_of(T);
  for (int p = 0; p < n_pairs; ++p) pairs_of[ptok[p]].push_back(p);
  std::vector<ErrorStats> ystats(T);
  parallel_for(T, [&](int t) {
    for (int p : pairs_of[t]) {
      const ExpertW& w = L.w[L.slot_of[pair_expert[p]]];
      std::vector<double> hv(kInter);
      for (int j = 0; j < kInter; ++j) hv[j] = bf16_to_float(hb[size_t(p) * kInter + j]);
      for (int n = 0; n < kHidden; ++n) {
        double s = 0;
        const float* dr = &w.down[size_t(n) * kInter];
        for (int j = 0; j < kInter; ++j) s += double(round_bf16(dr[j])) * hv[j];
        yref[t][n] += pair_wref[p] * s;
      }
    }
    ystats[t] = compare(&y[size_t(t) * kHidden], yref[t].data(), kHidden);
  });
  Worst yw;
  for (int t = 0; t < T; ++t)
    if (!pairs_of[t].empty()) yw.add(ystats[t], "t" + std::to_string(t));
  const bool y_ok = yw.ratio <= kTol;
  std::printf("%s prefill %-22s T=%d pairs=%d cold=%d h: max err/rms=%.3e excess-over-bf16=%.3e  "
              "y worst=%.3e [%s]\n",
              h_ok && y_ok && failures == 0 ? "PASS" : "FAIL", c.name.c_str(), T, n_pairs, meta[2],
              h_worst_rel, h_worst_excess, yw.ratio, yw.where.c_str());
  return failures + !h_ok + !y_ok;
}

// --------------------------------------------------------------------------- determinism
// Two complete prefill runs (route, grouped GEMMs, combine) on the same inputs must give
// bit-identical results even though pair order within an expert and atomic order differ.
int test_prefill_determinism(Loaded& L, int T) {
  auto logits = make_logits(T, L.experts, 4.0f, 4242);
  for (int t = 0; t < T; ++t)
    for (int e = 0; e < kExperts; ++e)
      if (L.slot_of[e] < 0) logits[size_t(t) * kRouter + e] = -30.f;  // all pairs hot
  DeviceArray<float> d_logits(logits);
  DeviceArray<__nv_bfloat16> d_x(make_x(T, 4343));
  DeviceArray<float> d_ysh(std::vector<float>(size_t(T) * kHidden, 0.25f));
  DeviceArray<float> d_inj(std::vector<float>(size_t(T) * 4, 0.5f));
  DeviceArray<uint8_t> ws_mem(kernels::moe_prefill_workspace_bytes(T));
  auto ws = kernels::make_moe_prefill_workspace(ws_mem.ptr, T);
  std::vector<std::vector<long long>> yq(2);
  std::vector<std::vector<float>> X(2);
  for (int run = 0; run < 2; ++run) {
    DeviceArray<float> d_X(std::vector<float>(size_t(T) * 4 * kHidden, 1.0f));
    FNGINE_TEST_CUDA(kernels::moe_route(d_logits.ptr, d_x.ptr, L.blob_table.ptr, 0, nullptr,
                                        nullptr, PrefillExchangeView{}, T, ws, 0));
    FNGINE_TEST_CUDA(kernels::moe_grouped_gate_up(L.layout, L.blob_table.ptr, d_x.ptr, T, ws, 0));
    FNGINE_TEST_CUDA(kernels::moe_grouped_down(L.layout, L.blob_table.ptr, T, ws, 0));
    FNGINE_TEST_CUDA(kernels::moe_prefill_combine(ws, d_ysh.ptr, nullptr, PrefillExchangeView{}, 0,
                                                  nullptr, d_inj.ptr, d_X.ptr, nullptr, T,
                                                  kernels::kMoeExchangeTimeoutCycles, nullptr, 1.0f, 0));
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    yq[run].resize(size_t(T) * kHidden);
    FNGINE_TEST_CUDA(cudaMemcpy(yq[run].data(), ws.y_fixed, yq[run].size() * sizeof(long long),
                                cudaMemcpyDeviceToHost));
    X[run] = d_X.download();
  }
  const bool ok = yq[0] == yq[1] &&
                  std::memcmp(X[0].data(), X[1].data(), X[0].size() * sizeof(float)) == 0;
  std::printf("%s prefill determinism %s T=%d: two runs bit-identical (y_fixed and X)\n",
              ok ? "PASS" : "FAIL", L.c.name.c_str(), T);
  return ok ? 0 : 1;
}

// -------------------------------------------------------------------------------- bench
// Per-launch event timing; returns {min, median} ms. The minimum is the estimate that is least
// affected by other processes time-slicing the GPU.
template <class F>
std::pair<double, double> time_min_median(cudaStream_t s, int reps, F&& launch) {
  cudaEvent_t a, b;
  FNGINE_TEST_CUDA(cudaEventCreate(&a));
  FNGINE_TEST_CUDA(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();
  std::vector<double> ms;
  for (int i = 0; i < reps; ++i) {
    FNGINE_TEST_CUDA(cudaEventRecord(a, s));
    launch();
    FNGINE_TEST_CUDA(cudaEventRecord(b, s));
    FNGINE_TEST_CUDA(cudaEventSynchronize(b));
    float t = 0;
    FNGINE_TEST_CUDA(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  return {ms.front(), ms[ms.size() / 2]};
}

void bench(const Case& c, const std::string& pack) {
  cudaStream_t s;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  // Decode: 16 groups of 10 experts (working set >> L2), T = 1, all hot.
  {
    constexpr int kGroups = 16;
    std::vector<int> experts;
    for (int i = 0; i < kGroups * 10; ++i) experts.push_back((i * 3 + 1) % kExperts);
    Loaded L;
    L.c = c;
    load(L, pack, experts, false);
    DeviceArray<int32_t> d_slot(L.slot_of);
    DeviceArray<uint32_t> d_seq(std::vector<uint32_t>{1u});
    DeviceArray<__nv_bfloat16> d_x(make_x(1, 5));
    DeviceArray<float> d_ysh(std::vector<float>(kHidden, 0.f));
    DeviceArray<float> d_inj(std::vector<float>(4, 0.5f));
    DeviceArray<float> d_X(std::vector<float>(4 * kHidden, 0.f));
    std::vector<DeviceArray<float>> logits;
    std::vector<DeviceArray<uint8_t>> ws_mem;
    std::vector<kernels::MoeDecodeWorkspace> ws;
    for (int g = 0; g < kGroups; ++g) {
      std::vector<int> grp(experts.begin() + g * 10, experts.begin() + g * 10 + 10);
      logits.emplace_back(make_logits(1, grp, 20.f, 100 + g));
      ws_mem.emplace_back(kernels::moe_decode_workspace_bytes());
      ws.push_back(kernels::make_moe_decode_workspace(ws_mem.back().ptr));
    }
    // DeviceArray zero-fills on the legacy stream; finish that before using the non-blocking one.
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    for (int g = 0; g < kGroups; ++g)
      FNGINE_TEST_CUDA(kernels::moe_select_publish(logits[g].ptr, d_x.ptr, d_slot.ptr, 0,
                                                   d_seq.ptr, nullptr, 1, ws[g], s));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
    for (int g = 0; g < kGroups; ++g) {
      kernels::MoeDecodeWork w;
      FNGINE_TEST_CUDA(cudaMemcpy(&w, ws[g].work, sizeof(w), cudaMemcpyDeviceToHost));
      if (w.n_items != 10) {
        std::printf("bench: unexpected item count %d in group %d\n", w.n_items, g);
        std::exit(1);
      }
    }
    // mode: 0 = gate_up + down, 1 = select + gate_up + down + combine, 2 = gate_up, 3 = down.
    auto capture = [&](int mode) {
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      FNGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for (int g = 0; g < kGroups; ++g) {
        if (mode == 1)
          FNGINE_TEST_CUDA(kernels::moe_select_publish(logits[g].ptr, d_x.ptr, d_slot.ptr, 0,
                                                       d_seq.ptr, nullptr, 1, ws[g], s));
        if (mode != 3) FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(L.layout, 1, ws[g], s));
        if (mode != 2) FNGINE_TEST_CUDA(kernels::moe_hot_down(L.layout, 1, ws[g], s));
        if (mode == 1)
          FNGINE_TEST_CUDA(kernels::moe_combine(ws[g], d_ysh.ptr, nullptr, 0, d_seq.ptr, d_inj.ptr,
                                                d_X.ptr, nullptr, 1, kernels::kMoeExchangeTimeoutCycles, nullptr, 1.0f, s));
      }
      FNGINE_TEST_CUDA(cudaStreamEndCapture(s, &graph));
      FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
      cudaGraphDestroy(graph);
      return exec;
    };
    double us[4], us_med[4];
    for (int mode = 0; mode < 4; ++mode) {
      cudaGraphExec_t exec = capture(mode);
      const auto t = time_min_median(s, 300, [&] { FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s)); });
      us[mode] = t.first * 1e3 / kGroups;
      us_med[mode] = t.second * 1e3 / kGroups;
      cudaGraphExecDestroy(exec);
    }
    // Device-to-device copy of 256 MiB under the same conditions (read + write bytes).
    DeviceArray<uint8_t> c0(256u << 20), c1(256u << 20);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    auto t_copy = time_min_median(s, 20, [&] {
      FNGINE_TEST_CUDA(cudaMemcpyAsync(c1.ptr, c0.ptr, 256u << 20, cudaMemcpyDeviceToDevice, s));
    });
    const double bytes = 10.0 * double(c.blob_bytes);
    const double gu_bytes = 10.0 * 2 * kInter * double(kernels::moe_row_bytes(c.gu, kHidden));
    std::printf("BENCH decode %s T=1, 10 hot experts (%.2f MB), per step (min of 300 graph replays "
                "/ 16 steps):\n  gate_up+down %.1f us = %.0f GB/s (median %.1f us); gate_up alone "
                "%.1f us (%.0f GB/s), down alone %.1f us (%.0f GB/s); select+gate_up+down+combine "
                "%.1f us (median %.1f us); reference D2D copy %.0f GB/s (read+write)\n",
                c.name.c_str(), bytes / 1e6, us[0], bytes / (us[0] * 1e-6) / 1e9, us_med[0], us[2],
                gu_bytes / (us[2] * 1e-6) / 1e9, us[3], (bytes - gu_bytes) / (us[3] * 1e-6) / 1e9,
                us[1], us_med[1], 2.0 * (256u << 20) / (t_copy.first * 1e-3) / 1e9);
  }
  // Prefill: T = 4096, all 512 experts hot.
  {
    constexpr int T = 4096;
    std::vector<int> experts(kExperts);
    for (int e = 0; e < kExperts; ++e) experts[e] = e;
    Loaded L;
    L.c = c;
    load(L, pack, experts, false);
    DeviceArray<float> d_logits(make_logits(T, {}, 0.f, 77));
    DeviceArray<__nv_bfloat16> d_x(make_x(T, 78));
    DeviceArray<uint8_t> ws_mem(kernels::moe_prefill_workspace_bytes(T));
    auto ws = kernels::make_moe_prefill_workspace(ws_mem.ptr, T);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    FNGINE_TEST_CUDA(kernels::moe_route(d_logits.ptr, d_x.ptr, L.blob_table.ptr, 0, nullptr,
                                        nullptr, PrefillExchangeView{}, T, ws, s));
    auto t_gu = time_min_median(s, 15, [&] {
      FNGINE_TEST_CUDA(kernels::moe_grouped_gate_up(L.layout, L.blob_table.ptr, d_x.ptr, T, ws, s));
    });
    auto t_dn = time_min_median(s, 15, [&] {
      FNGINE_TEST_CUDA(kernels::moe_grouped_down(L.layout, L.blob_table.ptr, T, ws, s));
    });
    auto t_rt = time_min_median(s, 15, [&] {
      FNGINE_TEST_CUDA(kernels::moe_route(d_logits.ptr, d_x.ptr, L.blob_table.ptr, 0, nullptr,
                                          nullptr, PrefillExchangeView{}, T, ws, s));
    });
    const double ms_gu = t_gu.first, ms_dn = t_dn.first;
    int32_t meta[4];
    FNGINE_TEST_CUDA(cudaMemcpy(meta, ws.meta, sizeof(meta), cudaMemcpyDeviceToHost));
    const double pairs = meta[1];
    const double f_gu = 2.0 * pairs * 2 * kInter * kHidden, f_dn = 2.0 * pairs * kHidden * kInter;
    std::printf("BENCH prefill %s T=%d, 512 hot experts (%d tiles), min of 15: route %.3f ms, "
                "gate_up %.3f ms %.1f TFLOP/s (median %.3f), down %.3f ms %.1f TFLOP/s (median %.3f), "
                "gemm total %.3f ms %.1f TFLOP/s\n",
                c.name.c_str(), T, meta[0], t_rt.first, ms_gu, f_gu / (ms_gu * 1e-3) / 1e12,
                t_gu.second, ms_dn, f_dn / (ms_dn * 1e-3) / 1e12, t_dn.second, ms_gu + ms_dn,
                (f_gu + f_dn) / ((ms_gu + ms_dn) * 1e-3) / 1e12);
  }
  cudaStreamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
  if (skip_without_gpu()) return kSkip;
  const std::string pack = env_or_skip("FNGINE_TEST_PACK");
  if (pack.empty()) return kSkip;
  bool do_bench = false, bench_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--bench") do_bench = true;
    if (std::string(argv[i]) == "--bench-only") do_bench = bench_only = true;
  }
  nlohmann::json m;
  {
    std::ifstream f(pack + "/manifest.json");
    if (!f) {
      std::fprintf(stderr, "cannot open %s/manifest.json\n", pack.c_str());
      return 1;
    }
    f >> m;
  }
  const std::string experts_file = m["files"]["experts"].get<std::string>();
  std::vector<Case> cases;
  std::set<std::pair<int, int>> combos;
  const auto& layers = m["experts"]["layers"];
  for (int l = 0; l < int(layers.size()); ++l) {
    const auto& j = layers[l];
    const MoeFormat gu = parse_format(j["gu_type"].get<std::string>());
    const MoeFormat dn = parse_format(j["down_type"].get<std::string>());
    if (!combos.insert({int(gu), int(dn)}).second) continue;
    cases.push_back(Case{"L" + std::to_string(l) + " " + j["gu_type"].get<std::string>() + "/" +
                             j["down_type"].get<std::string>(),
                         l, false, gu, dn, experts_file, j["offset"].get<int64_t>(),
                         j["blob_stride"].get<int64_t>(), j["blob_bytes"].get<int64_t>()});
  }
  {
    const auto& j = m["mtp_experts"];
    cases.push_back(Case{"MTP " + j["gu_type"].get<std::string>() + "/" + j["down_type"].get<std::string>(),
                         0, true, parse_format(j["gu_type"].get<std::string>()),
                         parse_format(j["down_type"].get<std::string>()),
                         m["files"]["mtp_experts"].get<std::string>(), j["offset"].get<int64_t>(),
                         j["blob_stride"].get<int64_t>(), j["blob_bytes"].get<int64_t>()});
  }

  int failures = 0;
  if (!bench_only) {
    for (size_t ci = 0; ci < cases.size(); ++ci) {
      Loaded L;
      L.c = cases[ci];
      std::vector<int> experts;
      for (int i = 0; i < kLoaded; ++i) experts.push_back(int((37 * i + 11 * ci + 5) % kExperts));
      load(L, pack, experts, true);
      for (int T : {1, 3, 9}) failures += test_decode(L, T, 1000 * ci + T);
      for (int T : {1, 3, 9, 64}) failures += test_prefill(L, T, 2000 * ci + T);
      if (ci == 0) failures += test_prefill_determinism(L, 2048);
    }
  }
  if (do_bench) {
    // One layer per gate/up codebook format (IQ4_NL down where the pack has it).
    std::set<int> benched;
    for (const Case& c : cases)
      if (!c.mtp && c.down == MoeFormat::IQ4_NL && benched.insert(int(c.gu)).second) bench(c, pack);
  }
  if (failures) std::printf("%d failure(s)\n", failures);
  else std::printf("all MoE expert checks passed\n");
  return failures ? 1 : 0;
}
