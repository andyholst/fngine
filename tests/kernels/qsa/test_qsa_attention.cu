// qsa_attention vs FP64 oracles: BF16 cache (oracle over the stored bf16 K/V) and K8V4 cache
// (oracle over the dequantized, inverse-rotated cache, plus a report against the unquantized K/V).
#include "kernels/qsa/qsa.h"
#include "kernels/qsa/qsa_ref.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace fngine::test;
using namespace fngine::test::qsa_ref;
namespace fk = fngine::kernels;

namespace {

struct Case {
    const char* name;
    int tokens;
    int first_position;
};

struct Totals {
    double k8v4_vs_unquantized_worst = 0;
    double k8v4_vs_unquantized_worst_rms = 0;
    double k8v8_vs_unquantized_worst = 0;
    double k8v8_vs_unquantized_worst_rms = 0;
};

// Keys attended by a query at position p (HF order irrelevant for softmax).
std::vector<int> key_positions(int p, const int32_t* sel, int count) {
    const int nb = (p + 1) / 4;
    std::vector<int> keys;
    if (nb <= kTop) {
        for (int i = 0; i <= p; ++i) keys.push_back(i);
        return keys;
    }
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < 4; ++j) keys.push_back(sel[i] * 4 + j);
    for (int i = nb * 4; i <= p; ++i) keys.push_back(i);
    return keys;
}

// out[t][h][d] = softmax(q.k / 16) v * sigmoid(gate), FP64; K/V via callbacks returning rows.
template <class KRow, class VRow>
std::vector<double> oracle(int tokens, const std::vector<int32_t>& pos, const std::vector<int32_t>& sel,
                           const std::vector<int32_t>& cnt, const std::vector<float>& q,
                           const std::vector<float>& gate_src, KRow krow, VRow vrow) {
    std::vector<double> out(size_t(tokens) * kHeads * kD);
    parallel_for(tokens * kHeads, [&](int unit) {
        const int t = unit / kHeads, h = unit % kHeads, kv = h / kGroup;
        const auto keys = key_positions(pos[t], sel.data() + size_t(t) * kTop, cnt[t]);
        const float* qr = q.data() + (size_t(t) * kHeads + h) * kD;
        std::vector<double> logits(keys.size());
        double m = -1e300;
        for (size_t i = 0; i < keys.size(); ++i) {
            const double* kr = krow(keys[i], kv);
            double s = 0;
            for (int d = 0; d < kD; ++d) s += double(qr[d]) * kr[d];
            logits[i] = s / 16.0;
            m = std::max(m, logits[i]);
        }
        double l = 0;
        for (auto& x : logits) l += (x = std::exp(x - m));
        double acc[kD] = {};
        for (size_t i = 0; i < keys.size(); ++i) {
            const double* vr = vrow(keys[i], kv);
            for (int d = 0; d < kD; ++d) acc[d] += logits[i] * vr[d];
        }
        for (int d = 0; d < kD; ++d) {
            const double g = gate_src[size_t(t) * kQkvLd + h * 512 + 256 + d];
            out[(size_t(t) * kHeads + h) * kD + d] = acc[d] / l / (1.0 + std::exp(-g));
        }
    });
    return out;
}

bool run_case(const Case& c, uint64_t seed, Totals& totals) {
    Rng rng(seed);
    const int T = c.tokens, last = c.first_position + T - 1, n = last + 1;
    // Cache contents for positions 0..last (appended in one prefill-style call).
    auto k = random_bf16(rng, size_t(n) * kKvHeads * kD, 2.5);
    auto v = random_bf16(rng, size_t(n) * kKvHeads * kD, 1.0);
    auto q = random_bf16(rng, size_t(T) * kHeads * kD, 1.0);
    auto qkv = random_bf16(rng, size_t(T) * kQkvLd, 1.5);  // gates
    std::vector<int32_t> all_pos(n), pos(T);
    std::iota(all_pos.begin(), all_pos.end(), 0);
    std::iota(pos.begin(), pos.end(), c.first_position);
    // Random sorted selections of 512 distinct blocks for sparse queries.
    std::vector<int32_t> sel(size_t(T) * kTop, 0), cnt(T, 0);
    for (int t = 0; t < T; ++t) {
        const int nb = (pos[t] + 1) / 4;
        if (nb <= kTop) {
            cnt[t] = nb;
            std::iota(sel.begin() + size_t(t) * kTop, sel.begin() + size_t(t) * kTop + nb, 0);
            continue;
        }
        std::vector<int> all(nb);
        std::iota(all.begin(), all.end(), 0);
        for (int i = 0; i < kTop; ++i) std::swap(all[i], all[i + int(rng.next() % (nb - i))]);
        std::sort(all.begin(), all.begin() + kTop);
        std::copy(all.begin(), all.begin() + kTop, sel.begin() + size_t(t) * kTop);
        cnt[t] = kTop;
    }

    DeviceArray<__nv_bfloat16> d_k(k), d_v(v), d_q(q), d_qkv(qkv);
    DeviceArray<int32_t> d_all(all_pos), d_pos(pos), d_sel(sel), d_cnt(cnt);
    DeviceArray<__nv_bfloat16> d_out_bf16(size_t(T) * kHeads * kD), d_out_k8v4(size_t(T) * kHeads * kD);
    const size_t ws_bytes = fk::qsa_attention_workspace_bytes(T);
    DeviceArray<uint8_t> ws(std::max<size_t>(ws_bytes, 16));

    DeviceArray<uint8_t> mem_bf16(fk::qsa_kv_layer_bytes(fk::QsaKvFormat::BF16, n));
    DeviceArray<uint8_t> mem_k8v4(fk::qsa_kv_layer_bytes(fk::QsaKvFormat::K8V4, n));
    const auto cache_bf16 = fk::qsa_kv_cache_view(fk::QsaKvFormat::BF16, mem_bf16.get(), n);
    const auto cache_k8v4 = fk::qsa_kv_cache_view(fk::QsaKvFormat::K8V4, mem_k8v4.get(), n);
    fk::qsa_kv_append(cache_bf16, d_k.get(), d_v.get(), 512, d_all.get(), n, nullptr);
    fk::qsa_kv_append(cache_k8v4, d_k.get(), d_v.get(), 512, d_all.get(), n, nullptr);
    // Inactive split storage may be uninitialized or left over from a longer context.
    // Poison it with NaNs so an accidental read cannot silently contribute zero.
    FNGINE_TEST_CUDA(cudaMemset(ws.get(), 0xff, ws_bytes));
    fk::qsa_attention(cache_bf16, d_q.get(), d_qkv.get(), d_pos.get(), d_sel.get(), d_cnt.get(), T,
                      ws.get(), ws_bytes, d_out_bf16.get(), nullptr);
    FNGINE_TEST_CUDA(cudaMemset(ws.get(), 0xff, ws_bytes));
    fk::qsa_attention(cache_k8v4, d_q.get(), d_qkv.get(), d_pos.get(), d_sel.get(), d_cnt.get(), T,
                      ws.get(), ws_bytes, d_out_k8v4.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto got_bf16 = to_float(d_out_bf16.download());
    const auto got_k8v4 = to_float(d_out_k8v4.download());
    FNGINE_CHECK(std::all_of(got_bf16.begin(), got_bf16.end(), [](float v) { return std::isfinite(v); }));
    FNGINE_CHECK(std::all_of(got_k8v4.begin(), got_k8v4.end(), [](float v) { return std::isfinite(v); }));

    // Host views of K/V.
    const auto kf = to_float(k), vf = to_float(v), qf = to_float(q), gf = to_float(qkv);
    std::vector<double> kd(kf.begin(), kf.end()), vd(vf.begin(), vf.end());
    auto k_orig = [&](int p, int h) { return kd.data() + (size_t(p) * kKvHeads + h) * kD; };
    auto v_orig = [&](int p, int h) { return vd.data() + (size_t(p) * kKvHeads + h) * kD; };

    K8v4Host host;
    const size_t ht = size_t(cache_k8v4.capacity_tokens) * kKvHeads;
    host.capacity = cache_k8v4.capacity_tokens;
    host.k.resize(ht * 256);
    host.k_scale.resize(ht);
    host.v.resize(ht * 128);
    host.v_scale.resize(ht * 16);
    FNGINE_TEST_CUDA(cudaMemcpy(host.k.data(), cache_k8v4.k, host.k.size(), cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.k_scale.data(), cache_k8v4.k_scale, ht * 2, cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.v.data(), cache_k8v4.v, host.v.size(), cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.v_scale.data(), cache_k8v4.v_scale, host.v_scale.size(), cudaMemcpyDeviceToHost));
    std::vector<double> kq(size_t(n) * kKvHeads * kD), vq(kq.size());
    parallel_for(n * kKvHeads, [&](int u) {
        host.k_values(u / kKvHeads, u % kKvHeads, kq.data() + size_t(u) * kD);
        host.v_values(u / kKvHeads, u % kKvHeads, vq.data() + size_t(u) * kD);
    });
    auto k_deq = [&](int p, int h) { return kq.data() + (size_t(p) * kKvHeads + h) * kD; };
    auto v_deq = [&](int p, int h) { return vq.data() + (size_t(p) * kKvHeads + h) * kD; };

    const auto ref_orig = oracle(T, pos, sel, cnt, qf, gf, k_orig, v_orig);
    const auto ref_deq  = oracle(T, pos, sel, cnt, qf, gf, k_deq, v_deq);

    char name[160];
    bool ok = true;
    std::snprintf(name, sizeof(name), "attention bf16-cache %s (T=%d, pos %d..%d, splits %d)", c.name, T,
                  c.first_position, last, fk::qsa_attention_splits(T));
    ok &= report_bf16_output(name, got_bf16, ref_orig);
    std::snprintf(name, sizeof(name), "attention k8v4 vs dequantized-cache oracle %s", c.name);
    ok &= report_bf16_output(name, got_k8v4, ref_deq);
    // Against the unquantized K/V the K8V4 codec error dominates (4-bit V: the cache round trip
    // alone has ~10% rms relative error). Bound the rms error loosely and report the maximum.
    ErrorStats s = compare(got_k8v4.data(), ref_orig.data(), ref_orig.size());
    double rms_err = 0;
    for (size_t i = 0; i < ref_orig.size(); ++i) rms_err += (got_k8v4[i] - ref_orig[i]) * (got_k8v4[i] - ref_orig[i]);
    rms_err = std::sqrt(rms_err / ref_orig.size()) / s.ref_rms;
    const bool loose_ok = rms_err <= 0.15 && s.max_rel_to_rms <= 2.5;
    std::printf("%s attention k8v4 vs unquantized K/V %s: rms_err/rms=%.3e (tol 0.15), max_abs/rms=%.3e (tol 2.5)\n",
                loose_ok ? "PASS" : "FAIL", c.name, rms_err, s.max_rel_to_rms);
    totals.k8v4_vs_unquantized_worst = std::max(totals.k8v4_vs_unquantized_worst, s.max_rel_to_rms);
    totals.k8v4_vs_unquantized_worst_rms = std::max(totals.k8v4_vs_unquantized_worst_rms, rms_err);

    // ---- K8V8: same inputs; oracle over its dequantized cache, plus the unquantized K/V.
    DeviceArray<uint8_t> mem_k8v8(fk::qsa_kv_layer_bytes(fk::QsaKvFormat::K8V8, n));
    const auto cache_k8v8 = fk::qsa_kv_cache_view(fk::QsaKvFormat::K8V8, mem_k8v8.get(), n);
    DeviceArray<__nv_bfloat16> d_out_k8v8(size_t(T) * kHeads * kD);
    fk::qsa_kv_append(cache_k8v8, d_k.get(), d_v.get(), 512, d_all.get(), n, nullptr);
    FNGINE_TEST_CUDA(cudaMemset(ws.get(), 0xff, ws_bytes));
    fk::qsa_attention(cache_k8v8, d_q.get(), d_qkv.get(), d_pos.get(), d_sel.get(), d_cnt.get(), T,
                      ws.get(), ws_bytes, d_out_k8v8.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto got_k8v8  = to_float(d_out_k8v8.download());
    FNGINE_CHECK(std::all_of(got_k8v8.begin(), got_k8v8.end(), [](float v) { return std::isfinite(v); }));
    const auto k8v8_bytes = mem_k8v8.download();
    K8v8Host host8;
    host8.load(k8v8_bytes.data(), cache_k8v8.capacity_tokens);
    parallel_for(n * kKvHeads, [&](int u) {
        host8.k_values(u / kKvHeads, u % kKvHeads, kq.data() + size_t(u) * kD);
        host8.v_values(u / kKvHeads, u % kKvHeads, vq.data() + size_t(u) * kD);
    });
    const auto ref_deq8 = oracle(T, pos, sel, cnt, qf, gf, k_deq, v_deq);
    std::snprintf(name, sizeof(name), "attention k8v8 vs dequantized-cache oracle %s", c.name);
    ok &= report_bf16_output(name, got_k8v8, ref_deq8);
    ErrorStats s8 = compare(got_k8v8.data(), ref_orig.data(), ref_orig.size());
    double rms_err8 = 0;
    for (size_t i = 0; i < ref_orig.size(); ++i) rms_err8 += (got_k8v8[i] - ref_orig[i]) * (got_k8v8[i] - ref_orig[i]);
    rms_err8 = std::sqrt(rms_err8 / ref_orig.size()) / s8.ref_rms;
    // Share of that error caused by the E4M3 K alone (FP64: quantized K, exact V).
    const auto ref_kq = oracle(T, pos, sel, cnt, qf, gf, k_deq, v_orig);
    double k_only = 0;
    for (size_t i = 0; i < ref_orig.size(); ++i) k_only += (ref_kq[i] - ref_orig[i]) * (ref_kq[i] - ref_orig[i]);
    k_only = std::sqrt(k_only / ref_orig.size()) / s8.ref_rms;
    const bool loose8_ok = rms_err8 <= 0.1 && s8.max_rel_to_rms <= 1.5;
    std::printf("%s attention k8v8 vs unquantized K/V %s: rms_err/rms=%.3e (tol 0.1), max_abs/rms=%.3e (tol 1.5); "
                "E4M3 K alone accounts for rms %.3e\n",
                loose8_ok ? "PASS" : "FAIL", c.name, rms_err8, s8.max_rel_to_rms, k_only);
    totals.k8v8_vs_unquantized_worst = std::max(totals.k8v8_vs_unquantized_worst, s8.max_rel_to_rms);
    totals.k8v8_vs_unquantized_worst_rms = std::max(totals.k8v8_vs_unquantized_worst_rms, rms_err8);
    return ok && loose_ok && loose8_ok;
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    const Case cases[] = {
        {"decode first token", 1, 0},
        {"verify crossing 32-key chunk", 4, 30},
        {"decode dense", 1, 100},
        {"decode dense max keys", 1, 2050},
        {"verify crossing dense->sparse", 9, 2045},
        {"decode sparse", 1, 5999},
        {"verify sparse T=3", 3, 4000},
        {"verify sparse T=9", 9, 5990},
        {"verify sparse T=16", 16, 3000},
        {"prefill dense", 256, 0},
        {"prefill sparse", 256, 5800},
    };
    bool ok = true;
    Totals totals;
    uint64_t seed = 500;
    for (const auto& c : cases) ok &= run_case(c, seed++, totals);
    std::printf("k8v4 vs unquantized K/V over all cases: worst rms_err/rms %.3e, worst max_abs/rms %.3e\n",
                totals.k8v4_vs_unquantized_worst_rms, totals.k8v4_vs_unquantized_worst);
    std::printf("k8v8 vs unquantized K/V over all cases: worst rms_err/rms %.3e, worst max_abs/rms %.3e\n",
                totals.k8v8_vs_unquantized_worst_rms, totals.k8v8_vs_unquantized_worst);
    return ok ? 0 : 1;
}
