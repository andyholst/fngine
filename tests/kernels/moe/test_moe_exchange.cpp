// GPU<->CPU MoE exchange protocol: a host thread plays the CPU expert pool (synthetic expert
// outputs) against moe_select_publish / moe_combine (inside a replayed CUDA graph) and against
// moe_route / moe_prefill_combine. Covers mixed hot/cold steps, steps without cold pairs (no wait,
// ready still released), and forced timeouts (error code, CPU contribution omitted).
// Synthetic IQ4_NL hot experts; no pack needed.
#include "moe_test_common.h"

#include <atomic>
#include <chrono>

using namespace fngine;
using namespace fngine::test;
using namespace fngine::test::moe;

namespace {

const std::vector<int> kHotExperts = {3, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 511};
constexpr long long kShortTimeout = 4'000'000;  // ~2 ms

double coef(int e) { return 0.01 * (e % 7 + 1); }

// Synthetic CPU expert output for one token: sum_j w_j * coef(e_j) * x.
void synth_y(const int32_t* experts, const float* weights, int n, const __nv_bfloat16* x, float* y) {
  for (int d = 0; d < kHidden; ++d) {
    double s = 0;
    for (int j = 0; j < n; ++j) s += double(weights[j]) * coef(experts[j]) * bf16_to_float(x[d]);
    y[d] = float(s);
  }
}

bool spin_until(const uint32_t& cell, uint32_t want, double seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  while (host_load_acquire(cell) != want) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds)
      return false;
  }
  return true;
}

struct Observed {
  bool saw_ready = false;
  int tokens = -1, n_cold = -1;
  std::vector<std::vector<int>> cold_expert;
};

void decode_worker(DecodeExchange* ex, int layer, uint32_t seq, Observed* obs) {
  if (!spin_until(ex->control.ready[layer].value, seq, 10.0)) return;
  obs->saw_ready = true;
  const MoeDecodePublish& pub = ex->publish[layer];
  obs->tokens = pub.tokens;
  obs->n_cold = pub.n_cold;
  obs->cold_expert.assign(pub.tokens, {});
  if (pub.n_cold == 0) return;  // GPU does not wait
  for (int t = 0; t < pub.tokens; ++t) {
    const int nc = pub.cold_count[t];
    obs->cold_expert[t].assign(pub.cold_expert[t], pub.cold_expert[t] + nc);
    if (nc > 0) synth_y(pub.cold_expert[t], pub.cold_weight[t], nc, pub.x[t], ex->result[layer].y[t]);
  }
  host_store_release(ex->control.done[layer].value, seq);
}

void prefill_worker(ExchangeControl* control, PrefillExchangeView v, int layer, uint32_t seq,
                    Observed* obs) {
  if (!spin_until(control->ready[layer].value, seq, 10.0)) return;
  obs->saw_ready = true;
  obs->tokens = v.header[0];
  obs->n_cold = v.header[2];
  const int rows = v.header[1];
  obs->cold_expert.assign(obs->tokens, {});
  for (int r = 0; r < rows; ++r) {
    const int t = v.row_token[r], nc = v.cold_count[r];
    obs->cold_expert[t].assign(v.cold_expert + r * kTopK, v.cold_expert + r * kTopK + nc);
    synth_y(v.cold_expert + r * kTopK, v.cold_weight + r * kTopK, nc, v.x + size_t(r) * kHidden,
            v.y + size_t(r) * kHidden);
  }
  host_store_release(control->done[layer].value, seq);  // every prefill chunk, also n_cold == 0
}

struct Fixture {
  std::vector<int32_t> slot_of = std::vector<int32_t>(kExperts, -1);
  DeviceArray<uint8_t> slots;
  kernels::MoeExpertLayout layout;
  DeviceArray<int32_t> d_slot;
  DeviceArray<const uint8_t*> blob_table;  // prefill: expert -> device blob / nullptr
  Fixture() {
    const int64_t blob = 3 * int64_t(kInter) * kernels::moe_row_bytes(MoeFormat::IQ4_NL, kHidden);
    std::vector<uint8_t> h(size_t(blob) * kHotExperts.size());
    Rng rng(42);
    for (auto& b : h) b = uint8_t(rng.next());
    for (size_t i = 0; i + 1 < h.size(); i += 18) {  // every IQ4_NL block: d = 0.01 (fp16 0x211F)
      h[i] = 0x1F;
      h[i + 1] = 0x21;
    }
    assign(slots, DeviceArray<uint8_t>(h));
    for (size_t s = 0; s < kHotExperts.size(); ++s) slot_of[kHotExperts[s]] = int32_t(s);
    assign(d_slot, DeviceArray<int32_t>(slot_of));
    layout = kernels::make_moe_expert_layout(slots.ptr, blob, MoeFormat::IQ4_NL, MoeFormat::IQ4_NL);
    std::vector<const uint8_t*> table(kExperts, nullptr);
    for (int e = 0; e < kExperts; ++e)
      if (slot_of[e] >= 0) table[e] = slots.ptr + int64_t(slot_of[e]) * blob;
    assign(blob_table, DeviceArray<const uint8_t*>(table));
  }
};

struct Inputs {
  int T;
  std::vector<float> logits, y_shared, inj, X0;
  std::vector<__nv_bfloat16> x;
  std::vector<TopK> ref;
  DeviceArray<float> d_logits, d_ysh, d_inj, d_X, d_y;
  DeviceArray<__nv_bfloat16> d_x;
  Inputs(int T_, float boost, uint64_t seed) : T(T_) {
    logits = make_logits(T, kHotExperts, boost, seed);
    x = make_x(T, seed + 1);
    Rng rng(seed + 2);
    y_shared.resize(size_t(T) * kHidden);
    for (auto& v : y_shared) v = float(rng.normal() * 0.1);
    inj.resize(size_t(T) * 4);
    for (auto& v : inj) v = float(rng.uniform(0.0, 2.0));
    X0.resize(size_t(T) * 4 * kHidden);
    for (auto& v : X0) v = float(rng.normal());
    for (int t = 0; t < T; ++t) ref.push_back(oracle_topk(&logits[size_t(t) * kRouter]));
    assign(d_logits, DeviceArray<float>(logits));
    assign(d_x, DeviceArray<__nv_bfloat16>(x));
    assign(d_ysh, DeviceArray<float>(y_shared));
    assign(d_inj, DeviceArray<float>(inj));
    assign(d_X, DeviceArray<float>(X0));
    assign(d_y, DeviceArray<float>(size_t(T) * kHidden));
  }
  // Oracle cold lists (rank order) and their synthetic CPU outputs.
  std::vector<std::vector<int>> cold_lists(const std::vector<int32_t>& slot_of) const {
    std::vector<std::vector<int>> c(T);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < kTopK; ++k)
        if (slot_of[ref[t].id[k]] < 0) c[t].push_back(ref[t].id[k]);
    return c;
  }
  std::vector<float> cpu_y(const std::vector<int32_t>& slot_of, const std::vector<float>& gpu_topk_w,
                           const std::vector<int32_t>& gpu_topk_id) const {
    std::vector<float> y(size_t(T) * kHidden, 0.f);
    for (int t = 0; t < T; ++t) {
      std::vector<int32_t> e;
      std::vector<float> w;
      for (int k = 0; k < kTopK; ++k) {
        const int id = gpu_topk_id[t * kTopK + k];
        if (slot_of[id] < 0) {
          e.push_back(id);
          w.push_back(gpu_topk_w[t * kTopK + k]);
        }
      }
      if (!e.empty()) synth_y(e.data(), w.data(), int(e.size()), &x[size_t(t) * kHidden], &y[size_t(t) * kHidden]);
    }
    return y;
  }
};

int expect(bool cond, const char* what) {
  std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
  return cond ? 0 : 1;
}

// Expected y for a decode step from the GPU's own hot partials (slot order) + shared + cpu.
std::vector<float> decode_expected_y(const kernels::MoeDecodeWorkspace& ws, const Inputs& in,
                                     const std::vector<float>& ycpu, bool use_cpu) {
  kernels::MoeDecodeWork work;
  FNGINE_TEST_CUDA(cudaMemcpy(&work, ws.work, sizeof(work), cudaMemcpyDeviceToHost));
  std::vector<float> part(size_t(kernels::kMoeDecodeMaxPairs) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(part.data(), ws.partial, part.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<float> y(size_t(in.T) * kHidden);
  for (int t = 0; t < in.T; ++t)
    for (int d = 0; d < kHidden; ++d) {
      float s = 0.f;
      for (int j = 0; j < work.token_pair_count[t]; ++j) s += part[size_t(work.token_pairs[t][j]) * kHidden + d];
      s += in.y_shared[size_t(t) * kHidden + d];
      if (use_cpu && work.token_cold_count[t] > 0) s += ycpu[size_t(t) * kHidden + d];
      y[size_t(t) * kHidden + d] = s;
    }
  return y;
}

void topk_from_decode(const kernels::MoeDecodeWorkspace& ws, int T, std::vector<int32_t>& ids,
                      std::vector<float>& w) {
  kernels::MoeDecodeWork work;
  FNGINE_TEST_CUDA(cudaMemcpy(&work, ws.work, sizeof(work), cudaMemcpyDeviceToHost));
  ids.resize(size_t(T) * kTopK);
  w.resize(size_t(T) * kTopK);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < kTopK; ++k) {
      ids[t * kTopK + k] = work.topk_expert[t][k];
      w[t * kTopK + k] = work.topk_weight[t][k];
    }
}

bool close(const std::vector<float>& got, const std::vector<float>& want, double tol) {
  std::vector<double> w(want.begin(), want.end());
  const ErrorStats s = compare(got.data(), w.data(), got.size());
  if (s.max_rel_to_rms > tol)
    std::printf("  mismatch: max_abs=%.3e rms=%.3e ratio=%.3e at %zu (got %.6g want %.6g)\n",
                s.max_abs, s.ref_rms, s.max_rel_to_rms, s.worst, got[s.worst], want[s.worst]);
  return s.max_rel_to_rms <= tol;
}

int test_decode_graph(Fixture& fx, Mapped<DecodeExchange>& ex, Mapped<uint32_t>& seq) {
  int fails = 0;
  constexpr int T = 3, layer = 5, kIters = 20;
  Inputs in(T, 3.0f, 1);
  const auto ref_cold = in.cold_lists(fx.slot_of);
  int n_cold = 0;
  for (auto& c : ref_cold) n_cold += int(c.size());
  DeviceArray<uint8_t> ws_mem(kernels::moe_decode_workspace_bytes());
  auto ws = kernels::make_moe_decode_workspace(ws_mem.ptr);
  cudaStream_t s;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  FNGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  FNGINE_TEST_CUDA(kernels::moe_select_publish(in.d_logits.ptr, in.d_x.ptr, fx.d_slot.ptr, layer,
                                               seq.dev, ex.dev, T, ws, s));
  FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(fx.layout, T, ws, s));
  FNGINE_TEST_CUDA(kernels::moe_hot_down(fx.layout, T, ws, s));
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, in.d_ysh.ptr, ex.dev, layer, seq.dev, in.d_inj.ptr,
                                        in.d_X.ptr, in.d_y.ptr, T, kernels::kMoeExchangeTimeoutCycles, nullptr, 1.0f, s));
  FNGINE_TEST_CUDA(cudaStreamEndCapture(s, &graph));
  FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));

  std::vector<float> X = in.X0;
  bool all_ok = true, obs_ok = true;
  for (int it = 1; it <= kIters; ++it) {
    const uint32_t sq = 100 + it;
    *seq.host = sq;
    Observed obs;
    std::thread worker(decode_worker, ex.host, layer, sq, &obs);
    FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
    worker.join();
    obs_ok &= obs.saw_ready && obs.tokens == T && obs.n_cold == n_cold;
    for (int t = 0; t < T && obs_ok; ++t) obs_ok &= obs.cold_expert[t] == ref_cold[t];
    std::vector<int32_t> ids;
    std::vector<float> w;
    topk_from_decode(ws, T, ids, w);
    const auto ycpu = in.cpu_y(fx.slot_of, w, ids);
    const auto want = decode_expected_y(ws, in, ycpu, true);
    const auto got = in.d_y.download();
    all_ok &= close(got, want, 1e-6);
    for (int t = 0; t < T; ++t)
      for (int st = 0; st < 4; ++st)
        for (int d = 0; d < kHidden; ++d)
          X[(size_t(t) * 4 + st) * kHidden + d] += in.inj[t * 4 + st] * want[size_t(t) * kHidden + d];
  }
  fails += expect(n_cold > 0 && obs_ok, "decode graph: worker saw every publish (T, n_cold, cold lists)");
  fails += expect(all_ok, "decode graph: y = hot (slot order) + shared + cpu, 20 replays");
  fails += expect(close(in.d_X.download(), X, 1e-5), "decode graph: GR write X += inj*y");
  fails += expect(ex.host->control.error.value == 0, "decode graph: no error flag");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
  return fails;
}

int test_decode_no_cold(Fixture& fx, Mapped<DecodeExchange>& ex, Mapped<uint32_t>& seq) {
  constexpr int T = 2, layer = 6;
  Inputs in(T, 20.0f, 2);
  DeviceArray<uint8_t> ws_mem(kernels::moe_decode_workspace_bytes());
  auto ws = kernels::make_moe_decode_workspace(ws_mem.ptr);
  *seq.host = 200;
  Observed obs;
  std::thread worker(decode_worker, ex.host, layer, 200u, &obs);
  FNGINE_TEST_CUDA(kernels::moe_select_publish(in.d_logits.ptr, in.d_x.ptr, fx.d_slot.ptr, layer,
                                               seq.dev, ex.dev, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(fx.layout, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_down(fx.layout, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, in.d_ysh.ptr, ex.dev, layer, seq.dev, in.d_inj.ptr,
                                        in.d_X.ptr, in.d_y.ptr, T, kShortTimeout, nullptr, 1.0f, 0));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());
  worker.join();
  int fails = 0;
  fails += expect(obs.saw_ready && obs.n_cold == 0 && obs.tokens == T,
                  "decode n_cold=0: ready released with n_cold=0");
  fails += expect(ex.host->control.error.value == 0 && ex.host->control.done[layer].value == 0,
                  "decode n_cold=0: combine did not wait (no timeout, no done needed)");
  const auto want = decode_expected_y(ws, in, {}, false);
  fails += expect(close(in.d_y.download(), want, 1e-6), "decode n_cold=0: y = hot + shared");
  return fails;
}

int test_decode_timeout(Fixture& fx, Mapped<DecodeExchange>& ex, Mapped<uint32_t>& seq) {
  constexpr int T = 1, layer = 7;
  Inputs in(T, 1.0f, 3);
  DeviceArray<uint8_t> ws_mem(kernels::moe_decode_workspace_bytes());
  auto ws = kernels::make_moe_decode_workspace(ws_mem.ptr);
  *seq.host = 300;
  FNGINE_TEST_CUDA(kernels::moe_select_publish(in.d_logits.ptr, in.d_x.ptr, fx.d_slot.ptr, layer,
                                               seq.dev, ex.dev, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_gate_up(fx.layout, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_hot_down(fx.layout, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, in.d_ysh.ptr, ex.dev, layer, seq.dev, in.d_inj.ptr,
                                        in.d_X.ptr, in.d_y.ptr, T, kShortTimeout, nullptr, 1.0f, 0));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());
  int fails = 0;
  const uint32_t err = ex.host->control.error.value;
  fails += expect(ex.host->publish[layer].n_cold > 0 && err == (kExchangeErrorMoeTimeout | uint32_t(layer + 1)),
                  "decode timeout: error = MoeTimeout | (layer+1)");
  const auto want = decode_expected_y(ws, in, {}, false);
  fails += expect(close(in.d_y.download(), want, 1e-6), "decode timeout: CPU contribution omitted");
  ex.host->control.error.value = 0;
  return fails;
}

int test_prefill(Fixture& fx, Mapped<DecodeExchange>& ex, Mapped<uint32_t>& seq, bool respond,
                 int T, int layer, float boost) {
  int fails = 0;
  Inputs in(T, boost, 10 + T);
  constexpr int kCap = 64;
  Mapped<uint8_t> view_mem(prefill_exchange_bytes(kCap));
  const PrefillExchangeView hv = make_prefill_exchange_view(view_mem.host, kCap);
  const PrefillExchangeView dv = device_view(hv, view_mem.host, view_mem.dev);
  DeviceArray<uint8_t> ws_mem(kernels::moe_prefill_workspace_bytes(kCap));
  auto ws = kernels::make_moe_prefill_workspace(ws_mem.ptr, kCap);
  const uint32_t sq = 400 + T;
  *seq.host = sq;
  Observed obs;
  std::thread worker;
  if (respond) worker = std::thread(prefill_worker, &ex.host->control, hv, layer, sq, &obs);
  ExchangeControl* dcontrol = &ex.dev->control;
  const long long timeout = respond ? kernels::kMoeExchangeTimeoutCycles : kShortTimeout;
  FNGINE_TEST_CUDA(kernels::moe_route(in.d_logits.ptr, in.d_x.ptr, fx.blob_table.ptr, layer,
                                      seq.dev, dcontrol, dv, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_grouped_gate_up(fx.layout, fx.blob_table.ptr, in.d_x.ptr, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_grouped_down(fx.layout, fx.blob_table.ptr, T, ws, 0));
  FNGINE_TEST_CUDA(kernels::moe_prefill_combine(ws, in.d_ysh.ptr, dcontrol, dv, layer, seq.dev,
                                                in.d_inj.ptr, in.d_X.ptr, in.d_y.ptr, T, timeout, nullptr, 1.0f, 0));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());
  if (respond) worker.join();

  std::vector<int32_t> ids(size_t(T) * kTopK), row_of(T);
  std::vector<float> w(size_t(T) * kTopK), yhot(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(ids.data(), ws.topk_expert, ids.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(w.data(), ws.topk_weight, w.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(row_of.data(), ws.row_of_token, row_of.size() * 4, cudaMemcpyDeviceToHost));
  yhot = download_y(ws, T);
  const auto ref_cold = in.cold_lists(fx.slot_of);
  int n_cold = 0, rows = 0;
  for (auto& c : ref_cold) {
    n_cold += int(c.size());
    rows += !c.empty();
  }
  const auto ycpu = in.cpu_y(fx.slot_of, w, ids);
  std::vector<float> want(size_t(T) * kHidden);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < kHidden; ++d) {
      float s = yhot[size_t(t) * kHidden + d] + in.y_shared[size_t(t) * kHidden + d];
      if (respond && row_of[t] >= 0) s += ycpu[size_t(t) * kHidden + d];
      want[size_t(t) * kHidden + d] = s;
    }
  std::vector<float> X = in.X0;
  for (int t = 0; t < T; ++t)
    for (int st = 0; st < 4; ++st)
      for (int d = 0; d < kHidden; ++d)
        X[(size_t(t) * 4 + st) * kHidden + d] += in.inj[t * 4 + st] * want[size_t(t) * kHidden + d];
  char name[160];
  if (respond) {
    bool obs_ok = obs.saw_ready && obs.tokens == T && obs.n_cold == n_cold && hv.header[1] == rows;
    for (int t = 0; t < T && obs_ok; ++t) obs_ok &= obs.cold_expert[t] == ref_cold[t];
    std::snprintf(name, sizeof(name), "prefill T=%d: worker saw compacted rows (%d rows, %d cold)", T, rows, n_cold);
    fails += expect(obs_ok, name);
    std::snprintf(name, sizeof(name), "prefill T=%d: y = hot + shared + cpu rows; X += inj*y", T);
    fails += expect(close(in.d_y.download(), want, 1e-6) && close(in.d_X.download(), X, 1e-5), name);
    fails += expect(ex.host->control.error.value == 0, "prefill: no error flag");
  } else {
    const uint32_t err = ex.host->control.error.value;
    std::snprintf(name, sizeof(name), "prefill T=%d timeout: error = MoeTimeout | (layer+1), CPU rows omitted", T);
    fails += expect(n_cold > 0 && err == (kExchangeErrorMoeTimeout | uint32_t(layer + 1)) &&
                        close(in.d_y.download(), want, 1e-6),
                    name);
    ex.host->control.error.value = 0;
  }
  return fails;
}

int test_sparse_slots(Mapped<DecodeExchange>& ex, Mapped<uint32_t>& seq) {
  const int ids[] = {4, 31, 67, 90, 122, 201, 303, 411, 477, 511};
  const int slots[] = {17003, -1, 512, 2, 4096, -1, 900, 17, -1, 43};
  std::vector<int32_t> map(kExperts, -1);
  std::vector<std::pair<int, int>> hot;
  for (int i = 0; i < kTopK; ++i) {
    map[ids[i]] = slots[i];
    if (slots[i] >= 0) hot.emplace_back(slots[i], ids[i]);
  }
  std::sort(hot.begin(), hot.end());
  DeviceArray<int32_t> dmap(map);
  DeviceArray<uint8_t> storage(kernels::moe_decode_workspace_bytes());
  const auto ws = kernels::make_moe_decode_workspace(storage.ptr);
  int failures = 0;
  for (int T : {1, 9, 16}) {
    std::vector<float> logits(size_t(T) * kRouter, -100.f);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < kTopK; ++i)
        logits[size_t(t) * kRouter + ids[i]] = float((i + t) % kTopK);
    DeviceArray<float> dlogits(logits);
    DeviceArray<__nv_bfloat16> dx(make_x(T, 731));
    *seq.host = 321 + T;
    FNGINE_TEST_CUDA(kernels::moe_select_publish(dlogits.ptr, dx.ptr, dmap.ptr, 0,
                                                seq.dev, ex.dev, T, ws, 0));
    kernels::MoeDecodeWork work;
    FNGINE_TEST_CUDA(cudaMemcpy(&work, ws.work, sizeof(work), cudaMemcpyDeviceToHost));
    bool ok = work.n_items == int(hot.size()) && work.n_pairs == int(hot.size()) * T &&
              work.n_cold == 3 * T;
    for (size_t item = 0; item < hot.size() && ok; ++item) {
      ok = work.item_slot[item] == hot[item].first && work.item_pair_begin[item] == int(item) * T;
      for (int t = 0; t < T && ok; ++t) {
        const int pair = int(item) * T + t;
        const auto ref = oracle_topk(logits.data() + size_t(t) * kRouter);
        int rank = 0;
        while (rank < kTopK && ref.id[rank] != hot[item].second) ++rank;
        ok = rank < kTopK && work.pair_token[pair] == t &&
             work.token_pair_count[t] == int(hot.size()) &&
             work.token_pairs[t][item] == pair &&
             std::abs(work.pair_weight[pair] - ref.w[rank]) < 1e-6f;
      }
    }
    ok = ok && work.item_pair_begin[hot.size()] == int(hot.size()) * T;
    failures += expect(ok, "sparse physical slots, spare512, deterministic hot-pair order");
  }
  return failures;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Fixture fx;
  Mapped<DecodeExchange> ex(sizeof(DecodeExchange));
  Mapped<uint32_t> seq(64);
  int fails = 0;
  fails += test_sparse_slots(ex, seq);
  fails += test_decode_graph(fx, ex, seq);
  fails += test_decode_no_cold(fx, ex, seq);
  fails += test_decode_timeout(fx, ex, seq);
  fails += test_prefill(fx, ex, seq, true, 64, 9, 1.5f);   // mixed hot/cold
  fails += test_prefill(fx, ex, seq, true, 32, 11, 20.f);  // n_cold == 0: done still awaited
  fails += test_prefill(fx, ex, seq, false, 16, 10, 1.0f);
  if (fails) std::printf("%d failure(s)\n", fails);
  else std::printf("all MoE exchange checks passed\n");
  return fails ? 1 : 0;
}
