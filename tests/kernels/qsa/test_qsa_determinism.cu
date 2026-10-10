// Whole QSA layer path at long prefill chunks: qsa_qk_norm_rope -> qsa_kv_append ->
// indexer_prepare -> indexer_score_topk -> qsa_attention, for one 2048-token chunk at positions
// 0..2047 (dense) and two 4096-token chunks at 0..4095 then 4096..8191 (sparse). The sequence runs
// twice on freshly poisoned state; every output must be bit-identical between runs, and sampled
// attention outputs must match an FP64 oracle over the cache (BF16) / dequantized cache (K8V4,
// K8V8).
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

const std::vector<__nv_bfloat16>* g_qkv = nullptr;  // full-sequence qkv (gate source)

struct Chunk {
    int first, tokens;
};

struct RunOutput {
    std::vector<std::vector<uint8_t>> blobs;  // per chunk: attention out, q_idx, selected, counts
    std::vector<uint8_t> cache, indexer;
    std::vector<std::vector<__nv_bfloat16>> out;
    std::vector<std::vector<int32_t>> sel, cnt;
    std::vector<std::vector<__nv_bfloat16>> q;
};

template <class T>
std::vector<uint8_t> bytes_of(const DeviceArray<T>& a) {
    std::vector<uint8_t> b(a.count * sizeof(T));
    FNGINE_TEST_CUDA(cudaMemcpy(b.data(), a.get(), b.size(), cudaMemcpyDeviceToHost));
    return b;
}

struct Inputs {
    int total;
    std::vector<__nv_bfloat16> qkv, idx;
    std::vector<float> qn, kn, iqn, ikn;
};

RunOutput run(fk::QsaKvFormat format, const Inputs& in, const std::vector<Chunk>& chunks) {
    const int capacity = 8192;
    DeviceArray<uint8_t> kv(fk::qsa_kv_layer_bytes(format, capacity));
    DeviceArray<uint8_t> idx_state(fk::indexer_state_bytes(capacity));
    FNGINE_TEST_CUDA(cudaMemset(kv.get(), 0xff, kv.count));          // poison: NaN bf16 / codes
    FNGINE_TEST_CUDA(cudaMemset(idx_state.get(), 0xff, idx_state.count));
    const auto cache = fk::qsa_kv_cache_view(format, kv.get(), capacity);
    auto* ring       = reinterpret_cast<__nv_bfloat16*>(idx_state.get());
    auto* blocks     = ring + 16 * kIdxDim;
    DeviceArray<float> qn(in.qn), kn(in.kn), iqn(in.iqn), ikn(in.ikn);
    const int max_blocks = capacity / 4;
    RunOutput r;
    for (const auto& c : chunks) {
        const int T = c.tokens;
        std::vector<__nv_bfloat16> qkv(in.qkv.begin() + size_t(c.first) * kQkvLd,
                                       in.qkv.begin() + size_t(c.first + T) * kQkvLd);
        std::vector<__nv_bfloat16> idx(in.idx.begin() + size_t(c.first) * kIdxLd,
                                       in.idx.begin() + size_t(c.first + T) * kIdxLd);
        std::vector<int32_t> pos(T);
        std::iota(pos.begin(), pos.end(), c.first);
        DeviceArray<__nv_bfloat16> d_qkv(qkv), d_idx(idx);
        DeviceArray<int32_t> d_pos(pos);
        DeviceArray<__nv_bfloat16> q(size_t(T) * kHeads * kD), k(size_t(T) * kKvHeads * kD);
        DeviceArray<__nv_bfloat16> q_idx(size_t(T) * kIdxHeads * kIdxDim), out(size_t(T) * kHeads * kD);
        DeviceArray<int32_t> sel(size_t(T) * kTop), cnt(T);
        FNGINE_TEST_CUDA(cudaMemset(q.get(), 0xff, q.count * 2));
        FNGINE_TEST_CUDA(cudaMemset(out.get(), 0xff, out.count * 2));
        FNGINE_TEST_CUDA(cudaMemset(sel.get(), 0xff, sel.count * 4));
        const size_t score_bytes = fk::indexer_score_workspace_bytes(T, max_blocks);
        const size_t attn_bytes  = fk::qsa_attention_workspace_bytes(T);
        DeviceArray<uint8_t> score_ws(std::max<size_t>(score_bytes, 16)), attn_ws(std::max<size_t>(attn_bytes, 16));
        FNGINE_TEST_CUDA(cudaMemset(score_ws.get(), 0xff, score_ws.count));
        fk::qsa_qk_norm_rope(d_qkv.get(), qn.get(), kn.get(), d_pos.get(), T, q.get(), k.get(), nullptr);
        fk::qsa_kv_append(cache, k.get(), d_qkv.get() + 12800, kQkvLd, d_pos.get(), T, nullptr);
        fk::indexer_prepare(d_idx.get(), d_pos.get(), T, iqn.get(), ikn.get(), ring, blocks, q_idx.get(), nullptr);
        fk::indexer_score_topk(q_idx.get(), blocks, d_pos.get(), T, max_blocks, score_ws.get(), score_bytes,
                               sel.get(), cnt.get(), nullptr);
        fk::qsa_attention(cache, q.get(), d_qkv.get(), d_pos.get(), sel.get(), cnt.get(), T, attn_ws.get(),
                          attn_bytes, out.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        r.blobs.push_back(bytes_of(out));
        r.blobs.push_back(bytes_of(q_idx));
        r.blobs.push_back(bytes_of(sel));
        r.blobs.push_back(bytes_of(cnt));
        r.out.push_back(out.download());
        r.q.push_back(q.download());
        auto s = sel.download();
        auto n = cnt.download();
        r.sel.push_back(s);
        r.cnt.push_back(n);
    }
    r.cache   = bytes_of(kv);
    r.indexer = bytes_of(idx_state);
    return r;
}

// FP64 oracle for sampled tokens of one chunk over the K/V the cache holds.
bool check_accuracy(const char* what, fk::QsaKvFormat format, const RunOutput& r, size_t chunk_index,
                    const Chunk& c) {
    const int capacity = 8192;
    std::vector<double> kv_k(size_t(capacity) * kKvHeads * kD), kv_v(kv_k.size());
    if (format == fk::QsaKvFormat::BF16) {
        const auto* kb = reinterpret_cast<const __nv_bfloat16*>(r.cache.data());
        const auto* vb = kb + size_t(capacity) * kKvHeads * kD;
        for (int p = 0; p < c.first + c.tokens; ++p)
            for (int h = 0; h < kKvHeads; ++h)
                for (int d = 0; d < kD; ++d) {
                    const size_t src = (size_t(p >> 6) * 2 + h) * 64 * 256 + size_t(p & 63) * 256 + d;
                    kv_k[(size_t(p) * kKvHeads + h) * kD + d] = bf16_to_float(kb[src]);
                    kv_v[(size_t(p) * kKvHeads + h) * kD + d] = bf16_to_float(vb[src]);
                }
    } else if (format == fk::QsaKvFormat::K8V8) {
        K8v8Host host;
        host.load(r.cache.data(), capacity);
        parallel_for((c.first + c.tokens) * kKvHeads, [&](int u) {
            host.k_values(u / kKvHeads, u % kKvHeads, kv_k.data() + size_t(u) * kD);
            host.v_values(u / kKvHeads, u % kKvHeads, kv_v.data() + size_t(u) * kD);
        });
    } else {
        K8v4Host host;
        const size_t ht = size_t(capacity) * kKvHeads;
        const uint8_t* base = r.cache.data();
        host.k.assign(base, base + ht * 256);
        host.k_scale.resize(ht);
        std::memcpy(host.k_scale.data(), base + ht * 256, ht * 2);
        host.v.assign(base + ht * 258, base + ht * 386);
        host.v_scale.assign(base + ht * 386, base + ht * 402);
        parallel_for((c.first + c.tokens) * kKvHeads, [&](int u) {
            host.k_values(u / kKvHeads, u % kKvHeads, kv_k.data() + size_t(u) * kD);
            host.v_values(u / kKvHeads, u % kKvHeads, kv_v.data() + size_t(u) * kD);
        });
    }
    const auto& q   = r.q[chunk_index];
    const auto& out = r.out[chunk_index];
    const auto& sel = r.sel[chunk_index];
    const auto& cnt = r.cnt[chunk_index];
    std::vector<int> tokens;
    for (int t = 0; t < c.tokens; t += 61) tokens.push_back(t);
    tokens.push_back(c.tokens - 1);
    std::vector<double> ref(tokens.size() * kHeads * kD);
    std::vector<float> got(ref.size());
    parallel_for(int(tokens.size()) * kHeads, [&](int unit) {
        const int ti = unit / kHeads, h = unit % kHeads, t = tokens[ti], kv = h / kGroup;
        const int p = c.first + t, nb = (p + 1) / 4;
        std::vector<int> keys;
        if (nb <= kTop) {
            for (int i = 0; i <= p; ++i) keys.push_back(i);
        } else {
            for (int i = 0; i < cnt[t]; ++i)
                for (int j = 0; j < 4; ++j) keys.push_back(sel[size_t(t) * kTop + i] * 4 + j);
            for (int i = nb * 4; i <= p; ++i) keys.push_back(i);
        }
        std::vector<double> logit(keys.size());
        double m = -1e300;
        for (size_t i = 0; i < keys.size(); ++i) {
            double s = 0;
            for (int d = 0; d < kD; ++d)
                s += double(bf16_to_float(q[(size_t(t) * kHeads + h) * kD + d])) * kv_k[(size_t(keys[i]) * kKvHeads + kv) * kD + d];
            logit[i] = s / 16.0;
            m = std::max(m, logit[i]);
        }
        double l = 0;
        for (auto& x : logit) l += (x = std::exp(x - m));
        for (int d = 0; d < kD; ++d) {
            double acc = 0;
            for (size_t i = 0; i < keys.size(); ++i) acc += logit[i] * kv_v[(size_t(keys[i]) * kKvHeads + kv) * kD + d];
            const double g = bf16_to_float((*g_qkv)[size_t(p) * kQkvLd + h * 512 + 256 + d]);
            ref[(size_t(ti) * kHeads + h) * kD + d] = acc / l / (1.0 + std::exp(-g));
            got[(size_t(ti) * kHeads + h) * kD + d] = bf16_to_float(out[(size_t(t) * kHeads + h) * kD + d]);
        }
    });
    return report_bf16_output(what, got, ref);
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    Rng rng(4242);
    Inputs in;
    in.total = 8192;
    in.qkv   = random_bf16(rng, size_t(in.total) * kQkvLd, 1.5);
    in.idx   = random_bf16(rng, size_t(in.total) * kIdxLd, 1.0);
    for (auto* w : {&in.qn, &in.kn}) {
        w->resize(kD);
        for (auto& x : *w) x = float(rng.uniform(0.5, 1.5));
    }
    for (auto* w : {&in.iqn, &in.ikn}) {
        w->resize(kIdxDim);
        for (auto& x : *w) x = float(rng.uniform(0.5, 1.5));
    }
    g_qkv = &in.qkv;
    const std::vector<std::pair<const char*, std::vector<Chunk>>> scenarios = {
        {"dense prefill T=2048 @0", {{0, 2048}}},
        {"sparse prefill T=4096 @0 then T=4096 @4096", {{0, 4096}, {4096, 4096}}},
    };
    bool ok = true;
    for (auto format : {fk::QsaKvFormat::BF16, fk::QsaKvFormat::K8V4, fk::QsaKvFormat::K8V8}) {
        const char* fname = format == fk::QsaKvFormat::BF16   ? "bf16"
                            : format == fk::QsaKvFormat::K8V4 ? "k8v4"
                                                              : "k8v8";
        for (const auto& [name, chunks] : scenarios) {
            const RunOutput a = run(format, in, chunks);
            const RunOutput b = run(format, in, chunks);
            size_t differ = 0;
            for (size_t i = 0; i < a.blobs.size(); ++i) differ += a.blobs[i] != b.blobs[i];
            differ += a.cache != b.cache;
            differ += a.indexer != b.indexer;
            const bool same = differ == 0;
            std::printf("%s determinism %s %s: %zu of %zu buffers differ between runs\n",
                        same ? "PASS" : "FAIL", fname, name, differ, a.blobs.size() + 2);
            ok &= same;
            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                char what[160];
                std::snprintf(what, sizeof(what), "attention %s %s chunk %zu (sampled tokens)", fname, name, ci);
                ok &= check_accuracy(what, format, a, ci, chunks[ci]);
            }
        }
    }
    return ok ? 0 : 1;
}
