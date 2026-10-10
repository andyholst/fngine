// QSA indexer: indexer_prepare vs an FP64 oracle, multi-batch / rollback ring correctness (bit
// exact vs a single pass), and indexer_score_topk selection vs an FP64 oracle.
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

struct IndexerState {
    int capacity;
    DeviceArray<__nv_bfloat16> state;  // ring | block keys
    explicit IndexerState(int cap) : capacity(cap), state(fk::indexer_state_bytes(cap) / 2) {
        FNGINE_TEST_CUDA(cudaMemset(state.get(), 0xff, fk::indexer_state_bytes(cap)));  // NaN garbage
    }
    __nv_bfloat16* ring() const { return state.get(); }
    __nv_bfloat16* blocks() const { return state.get() + 16 * kIdxDim; }
};

// Runs one batch of `tokens` positions starting at p0 using rows [p0, p0+tokens) of `idx`
// (the full-sequence input, or garbage rows when emulating rejected drafts).
void run_batch(const IndexerState& st, const std::vector<__nv_bfloat16>& idx, int p0, int tokens,
               const DeviceArray<float>& qn, const DeviceArray<float>& kn,
               std::vector<__nv_bfloat16>* q_out = nullptr) {
    std::vector<__nv_bfloat16> rows(idx.begin() + size_t(p0) * kIdxLd,
                                    idx.begin() + size_t(p0 + tokens) * kIdxLd);
    std::vector<int32_t> pos(tokens);
    std::iota(pos.begin(), pos.end(), p0);
    DeviceArray<__nv_bfloat16> d_idx(rows), d_q(size_t(tokens) * kIdxHeads * kIdxDim);
    DeviceArray<int32_t> d_pos(pos);
    fk::indexer_prepare(d_idx.get(), d_pos.get(), tokens, qn.get(), kn.get(), st.ring(), st.blocks(),
                        d_q.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    if (q_out) {
        auto q = d_q.download();
        q_out->insert(q_out->end(), q.begin(), q.end());
    }
}

std::vector<uint16_t> download_bits(const __nv_bfloat16* p, size_t n) {
    std::vector<uint16_t> h(n);
    FNGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return h;
}

bool test_prepare() {
    const int n = 3001;  // positions 0..3000
    Rng rng(21);
    auto idx = random_bf16(rng, size_t(n) * kIdxLd, 2.0);
    std::vector<float> q_norm(kIdxDim), k_norm(kIdxDim);
    for (auto& w : q_norm) w = static_cast<float>(rng.uniform(0.3, 1.8));
    for (auto& w : k_norm) w = static_cast<float>(rng.uniform(0.3, 1.8));
    DeviceArray<float> qn(q_norm), kn(k_norm);
    const auto x = to_float(idx);
    const int blocks = n / 4;
    bool ok = true;

    // --- Single pass (multi-CTA path) vs FP64 oracle.
    IndexerState single(4096);
    std::vector<__nv_bfloat16> q_single;
    run_batch(single, idx, 0, n, qn, kn, &q_single);
    const auto keys_single = download_bits(single.blocks(), size_t(blocks) * kIdxDim);
    {
        std::vector<float> got;
        std::vector<double> ref, tol;
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < kIdxHeads; ++h) {
                const RowRef r = norm_rope_row(x.data() + size_t(t) * kIdxLd + h * kIdxDim,
                                               q_norm.data(), kIdxDim, t);
                for (int d = 0; d < kIdxDim; ++d) {
                    got.push_back(bf16_to_float(q_single[(size_t(t) * kIdxHeads + h) * kIdxDim + d]));
                }
                ref.insert(ref.end(), r.out.begin(), r.out.end());
                tol.insert(tol.end(), r.tol.begin(), r.tol.end());
            }
        }
        ok &= report_budget("indexer_prepare queries (T=3001)", got, ref, tol);
    }
    {
        std::vector<float> got;
        std::vector<double> ref, tol;
        for (int b = 0; b < blocks; ++b) {
            // Mean in FP32 exactly as the kernel ((a+b)+c)+d, * 0.25, BF16.
            float mean[kIdxDim];
            for (int d = 0; d < kIdxDim; ++d) {
                float s = 0.0f;
                for (int j = 0; j < 4; ++j) s = s + x[size_t(4 * b + j) * kIdxLd + 512 + d];
                mean[d] = round_bf16(s * 0.25f);
            }
            const RowRef r = norm_rope_row(mean, k_norm.data(), kIdxDim, 4 * b);
            for (int d = 0; d < kIdxDim; ++d) {
                uint16_t bits = keys_single[size_t(b) * kIdxDim + d];
                __nv_bfloat16 v;
                std::memcpy(&v, &bits, 2);
                got.push_back(bf16_to_float(v));
            }
            ref.insert(ref.end(), r.out.begin(), r.out.end());
            tol.insert(tol.end(), r.tol.begin(), r.tol.end());
        }
        ok &= report_budget("indexer_prepare block keys (750 blocks)", got, ref, tol);
    }

    // --- Multi-batch: mixed batch sizes (single-CTA and multi-CTA paths) == single pass, bit exact.
    {
        IndexerState multi(4096);
        const int sizes[] = {5, 3, 7, 1, 12, 40, 16, 2, 9, 17, 4, 300, 1, 11};
        int p = 0, k = 0;
        std::vector<__nv_bfloat16> q_multi;
        while (p < n) {
            const int t = std::min(sizes[k++ % 14], n - p);
            run_batch(multi, idx, p, t, qn, kn, &q_multi);
            p += t;
        }
        const auto keys_multi = download_bits(multi.blocks(), size_t(blocks) * kIdxDim);
        const auto ring       = download_bits(multi.ring(), 16 * kIdxDim);
        size_t ring_bad = 0;
        for (int pos = n - 16; pos < n; ++pos) {
            for (int d = 0; d < kIdxDim; ++d) {
                uint16_t want;
                std::memcpy(&want, &idx[size_t(pos) * kIdxLd + 512 + d], 2);
                ring_bad += ring[(pos % 16) * kIdxDim + d] != want;
            }
        }
        const bool keys_equal = keys_multi == keys_single;
        const bool q_equal = std::memcmp(q_multi.data(), q_single.data(), q_single.size() * 2) == 0;
        const bool pass = keys_equal && q_equal && ring_bad == 0;
        std::printf("%s indexer multi-batch (sizes 5,3,7,1,12,40,16,2,9,17,4,300,1,11 ...): block keys "
                    "%s, queries %s, final ring %zu bad values\n",
                    pass ? "PASS" : "FAIL", keys_equal ? "bit-exact" : "DIFFER",
                    q_equal ? "bit-exact" : "DIFFER", ring_bad);
        ok &= pass;
    }

    // --- Speculative rollback: verify batches of width w <= 12 whose rejected tail carries garbage
    // inputs; the next batch restarts after the committed prefix. Must equal the single pass.
    {
        IndexerState spec(4096);
        Rng r2(77);
        auto garbage = random_bf16(r2, size_t(n + 16) * kIdxLd, 5.0);
        int p = 0;
        run_batch(spec, idx, 0, 37, qn, kn);
        p = 37;
        int rounds = 0;
        while (p < n) {
            const int width  = std::min(n - p, 1 + int(r2.next() % 12));
            const int commit = 1 + int(r2.next() % width);
            // Rows of the batch: committed prefix from the real sequence, rejected tail garbage.
            std::vector<__nv_bfloat16> mixed(garbage);
            std::copy(idx.begin() + size_t(p) * kIdxLd, idx.begin() + size_t(p + commit) * kIdxLd,
                      mixed.begin() + size_t(p) * kIdxLd);
            run_batch(spec, mixed, p, width, qn, kn);
            p += commit;
            ++rounds;
        }
        const auto keys_spec = download_bits(spec.blocks(), size_t(blocks) * kIdxDim);
        const bool pass = keys_spec == keys_single;
        std::printf("%s indexer rollback (%d verify rounds, widths 1..12, random commits): block keys %s\n",
                    pass ? "PASS" : "FAIL", rounds, pass ? "bit-exact vs single pass" : "DIFFER");
        ok &= pass;
    }
    return ok;
}

// ------------------------------------------------------------------------------------------------
struct TopkCase {
    const char* name;
    int tokens;
    int first_position;
    int max_blocks;
    bool all_equal_keys = false;
    // Large scores exactly at the select kernel's 512 sample positions: the sampled threshold then
    // admits fewer than 512 candidates and the kernel must take its full radix-select fallback.
    bool adversarial_samples = false;
};

bool run_topk(const TopkCase& c, uint64_t seed) {
    Rng rng(seed);
    const int last = c.first_position + c.tokens - 1;
    const int nb_last = (last + 1) / 4;
    std::vector<__nv_bfloat16> keys(size_t(c.max_blocks) * kIdxDim);
    if (c.all_equal_keys) {
        auto row = random_bf16(rng, kIdxDim, 1.0);
        for (int b = 0; b < c.max_blocks; ++b) std::copy(row.begin(), row.end(), keys.begin() + size_t(b) * kIdxDim);
    } else {
        keys = random_bf16(rng, keys.size(), 1.0);
        // Exact duplicates create score ties that must resolve to the lower block id.
        for (int i = 0; i < 64 && nb_last > 2; ++i) {
            const int a = int(rng.next() % nb_last), b = int(rng.next() % nb_last);
            std::copy(keys.begin() + size_t(a) * kIdxDim, keys.begin() + size_t(a + 1) * kIdxDim,
                      keys.begin() + size_t(b) * kIdxDim);
        }
    }
    auto q = random_bf16(rng, size_t(c.tokens) * kIdxHeads * kIdxDim, 1.0);
    std::vector<int32_t> pos(c.tokens);
    std::iota(pos.begin(), pos.end(), c.first_position);
    if (c.adversarial_samples) {
        const int nb = (pos[0] + 1) / 4;
        std::vector<double> dir(kIdxDim, 0.0);
        double norm = 0;
        for (int d = 0; d < kIdxDim; ++d) {
            for (int h = 0; h < kIdxHeads; ++h) dir[d] += bf16_to_float(q[size_t(h) * kIdxDim + d]);
            norm += dir[d] * dir[d];
        }
        for (int i = 0; i < 512; ++i) {
            const int b = int((int64_t(i) * nb) / 512);
            for (int d = 0; d < kIdxDim; ++d) {
                keys[size_t(b) * kIdxDim + d] = float_to_bf16(float(8.0 * dir[d] / std::sqrt(norm) + 0.01 * rng.normal()));
            }
        }
    }

    DeviceArray<__nv_bfloat16> d_keys(keys), d_q(q);
    DeviceArray<int32_t> d_pos(pos), d_sel(size_t(c.tokens) * kTop), d_cnt(c.tokens);
    const size_t ws_bytes = fk::indexer_score_workspace_bytes(c.tokens, c.max_blocks);
    DeviceArray<uint8_t> ws(std::max<size_t>(ws_bytes, 16));
    fk::indexer_score_topk(d_q.get(), d_keys.get(), d_pos.get(), c.tokens, c.max_blocks, ws.get(),
                           ws_bytes, d_sel.get(), d_cnt.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto sel = d_sel.download();
    const auto cnt = d_cnt.download();
    const auto kf  = to_float(keys);
    const auto qf  = to_float(q);

    std::vector<int> failures(c.tokens, 0), boundary(c.tokens, 0);
    parallel_for(c.tokens, [&](int t) {
        const int nb = (pos[t] + 1) / 4;
        const int want = std::min(nb, kTop);
        if (cnt[t] != want) { failures[t] = 1; return; }
        std::vector<int> got(sel.begin() + size_t(t) * kTop, sel.begin() + size_t(t) * kTop + want);
        for (int i = 0; i < want; ++i) {
            if (got[i] < 0 || got[i] >= nb || (i > 0 && got[i] <= got[i - 1])) { failures[t] = 2; return; }
        }
        if (nb <= kTop) return;  // dense: 0..nb-1 implied by ascending + distinct + in range
        std::vector<double> score(nb);
        for (int b = 0; b < nb; ++b) {
            double s = 0;
            for (int h = 0; h < kIdxHeads; ++h) {
                double dot = 0;
                for (int d = 0; d < kIdxDim; ++d) {
                    dot += double(qf[(size_t(t) * kIdxHeads + h) * kIdxDim + d]) * kf[size_t(b) * kIdxDim + d];
                }
                s += std::max(dot, 0.0);
            }
            score[b] = s / std::sqrt(128.0);
        }
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](int a, int b) {
            return score[a] != score[b] ? score[a] > score[b] : a < b;
        });
        const double threshold = score[order[kTop - 1]];
        std::vector<char> in_ref(nb, 0), in_got(nb, 0);
        for (int i = 0; i < kTop; ++i) in_ref[order[i]] = 1;
        for (int b : got) in_got[b] = 1;
        for (int b = 0; b < nb; ++b) {
            if (in_ref[b] == in_got[b]) continue;
            // Allowed only for near-ties at the selection boundary.
            if (std::fabs(score[b] - threshold) <= 1e-5 * std::fabs(threshold)) {
                ++boundary[t];
            } else {
                failures[t] = 3;
                return;
            }
        }
    });
    int bad = 0, near_ties = 0;
    for (int t = 0; t < c.tokens; ++t) {
        bad += failures[t] != 0;
        near_ties += boundary[t];
        if (failures[t] && bad <= 3) std::printf("  token %d (pos %d): failure kind %d count %d\n", t, pos[t], failures[t], cnt[t]);
    }
    if (c.all_equal_keys) {
        // Every score ties: the selection must be exactly blocks 0..511 for sparse queries.
        for (int t = 0; t < c.tokens; ++t) {
            if ((pos[t] + 1) / 4 <= kTop) continue;
            for (int i = 0; i < kTop; ++i) bad += sel[size_t(t) * kTop + i] != i;
        }
    }
    const bool ok = bad == 0;
    std::printf("%s indexer_score_topk %s: T=%d pos %d..%d max_blocks=%d, %d boundary near-tie swaps\n",
                ok ? "PASS" : "FAIL", c.name, c.tokens, c.first_position, last, c.max_blocks, near_ties);
    return ok;
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    bool ok = test_prepare();
    const TopkCase cases[] = {
        {"decode dense edge", 1, 2050, 1024},
        {"decode first sparse", 1, 2051, 1024},
        {"verify crossing dense->sparse", 9, 2045, 1024},
        {"decode T=3", 3, 6000, 2048},
        {"decode T=9 @90K (shared-memory keys)", 9, 89990, 98304 / 4},
        {"decode T=3 @100K (global keys)", 3, 99990, 131072 / 4},
        {"decode T=1 @210K (global keys)", 1, 210000, 262144 / 4},
        {"decode native final block", 1, 262143, 262144 / 4},
        {"verify native final positions", 9, 262135, 262144 / 4},
        {"short decode in native capacity", 1, 6000, 262144 / 4},
        {"native final block ties", 1, 262143, 262144 / 4, true},
        {"prefill T=600 (3 query chunks)", 600, 5400, 1536},
        {"prefill T=256 all-equal keys (ties -> lower id)", 256, 5000, 1536, true},
        {"decode all-equal keys @100K (candidate overflow fallback)", 1, 99999, 131072 / 4, true},
        {"decode adversarial samples @100K (sparse-candidate fallback)", 1, 99999, 131072 / 4, false, true},
    };
    uint64_t seed = 100;
    for (const auto& c : cases) ok &= run_topk(c, seed++);
    return ok ? 0 : 1;
}
