// kernels_head: embedding gather, FP32 argmax (ties -> lower id), sampling (greedy, penalties,
// chi-square distribution check, determinism per (seed, position)), speculative acceptance
// (greedy and rejection sampling vs a CPU reference) and the MTP round transition.
#include "kernels/head/head.h"

#include "common/test_util.h"
#include "core/model_dims.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace fngine;
using namespace fngine::test;
namespace k = fngine::kernels;

namespace {

constexpr int V = dims::kVocab;
bool g_ok = true;

void expect(bool cond, const char* what) {
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    g_ok = g_ok && cond;
}

// ---------------------------------------------------------------- CPU reference

uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// Reference counter RNG (same keying as the device sampler).
float uniform_ref(uint64_t seed, int position, int purpose) {
    uint64_t key = splitmix(seed ^ (uint64_t(uint32_t(position)) * 0xD1B54A32D192ED03ull));
    key          = splitmix(key ^ (uint64_t(uint32_t(purpose)) << 21));
    return float(uint32_t(key >> 40)) * (1.0f / 16777216.0f);
}

struct Dist {
    std::vector<int> idx;
    std::vector<double> prob;
};

float adjusted(float raw, int v, const k::SamplingConfig& c, const std::vector<int32_t>& counts,
               const int32_t* overlay, int overlay_len) {
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) return raw;
    int cnt = counts.empty() ? 0 : counts[v];
    for (int j = 0; j < overlay_len; ++j) cnt += overlay[j] == v;
    float x = raw;
    if (cnt > 0) x -= c.presence_penalty;
    if (c.frequency_penalty != 0.0f) x -= c.frequency_penalty * float(cnt);
    return x;
}

std::vector<std::pair<float, int>> sorted_candidates(const float* logits, int domain, const k::SamplingConfig& c,
                                                     const std::vector<int32_t>& counts, const int32_t* overlay,
                                                     int overlay_len, int cap) {
    std::vector<std::pair<float, int>> all(domain);
    for (int v = 0; v < domain; ++v) all[v] = {adjusted(logits[v], v, c, counts, overlay, overlay_len), v};
    const auto better = [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    std::partial_sort(all.begin(), all.begin() + cap, all.end(), better);
    all.resize(cap);
    return all;
}

int argmax_ref(const float* logits, int domain, const k::SamplingConfig& c, const std::vector<int32_t>& counts,
               const int32_t* overlay = nullptr, int overlay_len = 0) {
    return sorted_candidates(logits, domain, c, counts, overlay, overlay_len, 1)[0].second;
}

Dist dist_ref(const float* logits, int domain, const k::SamplingConfig& c, const std::vector<int32_t>& counts,
              const int32_t* overlay = nullptr, int overlay_len = 0) {
    int cap = 20;
    if (c.top_k > 0 && c.top_k < cap) cap = c.top_k;
    cap             = std::min(cap, domain);
    const auto cand = sorted_candidates(logits, domain, c, counts, overlay, overlay_len, cap);
    const double inv_temp = 1.0 / c.temperature;
    std::vector<double> w(cap);
    double sum = 0;
    for (int j = 0; j < cap; ++j) sum += (w[j] = std::exp((double(cand[j].first) - cand[0].first) * inv_temp));
    const double min_p = c.min_p > 0 ? c.min_p * w[0] : -1.0;
    double cum         = 0;
    int support        = 0;
    for (int j = 0; j < cap; ++j) {
        if (min_p >= 0 && w[j] < min_p) break;
        cum += w[j];
        support = j + 1;
        if (c.top_p < 1.0f && cum >= c.top_p * sum) break;
    }
    support = std::max(support, 1);
    Dist d;
    double ssum = 0;
    for (int j = 0; j < support; ++j) ssum += w[j];
    for (int j = 0; j < support; ++j) {
        d.idx.push_back(cand[j].second);
        d.prob.push_back(w[j] / ssum);
    }
    return d;
}

// Inverse-CDF pick (optionally excluding one id). `margin` receives the distance of the goal
// from the nearest CDF boundary (small values mean float/double evaluation may disagree).
int pick_ref(const Dist& d, int exclude, float u, double* margin) {
    double mass = 0;
    for (size_t j = 0; j < d.idx.size(); ++j)
        if (d.idx[j] != exclude) mass += d.prob[j];
    const double goal = u * mass;
    double acc = 0, m = 1;
    int picked = -1;
    for (size_t j = 0; j < d.idx.size(); ++j) {
        if (d.idx[j] == exclude) continue;
        acc += d.prob[j];
        m = std::min(m, std::fabs(goal - acc));
        picked = d.idx[j];
        if (goal < acc) break;
    }
    if (margin) *margin = std::min(*margin, m);
    return picked;
}

struct AcceptResult {
    int a = 0, terminal = 0;
    double margin = 1; // distance of the closest random decision from its threshold
};

AcceptResult accept_ref(const std::vector<float>& logits, const std::vector<int32_t>& drafts, int extent,
                        int domain, const k::SamplingConfig& c, const std::vector<int32_t>& counts, int L) {
    AcceptResult r;
    for (int i = 0; i <= extent; ++i) {
        const float* col = logits.data() + static_cast<size_t>(i) * V;
        if (!(c.temperature > 0)) {
            const int sel = argmax_ref(col, domain, c, counts, drafts.data(), i);
            if (i < extent && sel == drafts[i]) {
                r.a = i + 1;
                continue;
            }
            r.terminal = sel;
            return r;
        }
        const Dist d = dist_ref(col, domain, c, counts, drafts.data(), i);
        if (i < extent) {
            double pd = 0;
            for (size_t j = 0; j < d.idx.size(); ++j)
                if (d.idx[j] == drafts[i]) pd = d.prob[j];
            const float u = uniform_ref(c.seed, L + i + 1, k::kSamplePurposeSpeculativeAccept);
            r.margin      = std::min(r.margin, std::fabs(double(u) - pd));
            if (u < pd) {
                r.a = i + 1;
                continue;
            }
            r.terminal = pick_ref(d, drafts[i], uniform_ref(c.seed, L + i + 1, k::kSamplePurposeSpeculativeCorrection),
                                  &r.margin);
            return r;
        }
        r.terminal = pick_ref(d, -1, uniform_ref(c.seed, L + extent + 1, k::kSamplePurposeSpeculativeBonus), &r.margin);
    }
    return r;
}

std::vector<float> random_logits(Rng& rng, int cols, double scale) {
    std::vector<float> v(static_cast<size_t>(cols) * V);
    for (auto& x : v) x = static_cast<float>(rng.normal() * scale);
    return v;
}

// ---------------------------------------------------------------- embed_gather

void test_embed_gather(Rng& rng) {
    const int rows = 4096, T = 5;
    __nv_bfloat16* h_table = nullptr;
    FNGINE_TEST_CUDA(cudaHostAlloc(&h_table, static_cast<size_t>(rows) * dims::kHidden * 2, cudaHostAllocMapped));
    for (size_t i = 0; i < static_cast<size_t>(rows) * dims::kHidden; ++i)
        h_table[i] = float_to_bf16(static_cast<float>(rng.normal()));
    __nv_bfloat16* d_table = nullptr;
    FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_table), h_table, 0));
    const std::vector<int32_t> tokens = {7, 4095, 0, 1234, 7};
    DeviceArray<int32_t> d_tokens(tokens);
    DeviceArray<float> X(static_cast<size_t>(T) * dims::kStreamHidden);
    DeviceArray<__nv_bfloat16> emb(static_cast<size_t>(T) * dims::kHidden);
    k::embed_gather(d_table, d_tokens.get(), T, X.get(), emb.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto x = X.download();
    const auto e = emb.download();
    int bad      = 0;
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < dims::kHidden; ++d) {
            const __nv_bfloat16 ref = h_table[static_cast<size_t>(tokens[t]) * dims::kHidden + d];
            bad += std::memcmp(&ref, &e[t * dims::kHidden + d], 2) != 0;
            for (int s = 0; s < 4; ++s) bad += x[(t * 4 + s) * dims::kHidden + d] != bf16_to_float(ref);
        }
    expect(bad == 0, "embed_gather host-mapped table -> 4 streams + bf16 copy (bit-exact)");
    cudaFreeHost(h_table);
}

// ---------------------------------------------------------------- argmax

void test_argmax(Rng& rng) {
    for (int T : {9, 12}) {
        auto logits = random_logits(rng, T, 1.0);
        auto at     = [&](int t, int v) -> float& { return logits[static_cast<size_t>(t) * V + v]; };
        at(0, 1000) = at(0, 200000) = 50.0f; // tie across tiles
        at(1, 5) = at(1, 4) = 50.0f;          // adjacent tie
        at(2, 4096) = at(2, 4095) = 50.0f;     // tie across the 512x8 tile boundary
        for (int v = 0; v < V; ++v) at(3, v) = 1.0f; // all equal -> 0
        at(4, V - 1) = 60.0f;                 // last id wins
        at(5, 248100) = 70.0f;                // outside a 248077 domain
        at(5, 248076) = 60.0f;
        at(6, 17) = 0.0f;                     // -0 / +0 tie with everything else negative
        for (int v = 0; v < V; ++v) at(6, v) = -1.0f - std::fabs(at(6, v));
        at(6, 900) = -0.0f;
        at(6, 901) = 0.0f;
        for (int domain : {V, 248077}) {
            DeviceArray<float> d(logits);
            DeviceArray<int32_t> out(T);
            k::argmax(d.get(), T, domain, out.get(), nullptr);
            FNGINE_TEST_CUDA(cudaDeviceSynchronize());
            const auto got = out.download();
            int bad        = 0;
            for (int t = 0; t < T; ++t) {
                int best = 0;
                for (int v = 1; v < domain; ++v)
                    if (at(t, v) > at(t, best)) best = v;
                bad += got[t] != best;
                if (got[t] != best) std::printf("  argmax col %d: got %d want %d\n", t, got[t], best);
            }
            char what[96];
            std::snprintf(what, sizeof(what), "argmax T=%d domain=%d (ties -> lower id)", T, domain);
            expect(bad == 0, what);
        }
    }
}

// ---------------------------------------------------------------- sampling

struct Sampler {
    int domain;
    DeviceArray<uint8_t> ws;
    DeviceArray<k::SamplingConfig> cfg;
    explicit Sampler(int domain_) : domain(domain_), ws(k::sampling_workspace_bytes(domain_, 16)), cfg(1) {}
    std::vector<int32_t> run(const DeviceArray<float>& logits, int T, const k::SamplingConfig& c,
                             const std::vector<int32_t>& positions, int purpose) {
        cfg.upload({c});
        DeviceArray<int32_t> pos(positions), out(T);
        k::sample(logits.get(), T, domain, cfg.get(), pos.get(), purpose, out.get(), ws.get(), ws.count, nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        return out.download();
    }
};

void test_sampling_greedy(Rng& rng) {
    const int T       = 3;
    const auto logits = random_logits(rng, T, 2.0);
    DeviceArray<float> d(logits);
    Sampler s(V);
    std::vector<int32_t> counts(V, 0);
    DeviceArray<int32_t> d_counts(counts);
    k::SamplingConfig c;
    c.temperature  = 0.0f;
    c.token_counts = d_counts.get();
    const auto got = s.run(d, T, c, {10, 11, 12}, k::kSamplePurposeDecode);
    int bad        = 0;
    for (int t = 0; t < T; ++t) bad += got[t] != argmax_ref(logits.data() + static_cast<size_t>(t) * V, V, c, {});
    expect(bad == 0, "sample greedy == argmax");
    const auto after = d_counts.download();
    int inc          = 0;
    for (int t = 0; t < T; ++t) inc += after[got[t]];
    expect(std::accumulate(after.begin(), after.end(), 0) == T && inc >= T, "sample greedy increments token_counts");

    // Penalties: the raw winner of column 0 has occurred, so presence/frequency move the pick.
    std::vector<float> pl(logits.begin(), logits.begin() + V);
    const int raw_best = argmax_ref(pl.data(), V, k::SamplingConfig{}, {});
    std::vector<int32_t> pc(V, 0);
    pc[raw_best] = 2;
    for (int i = 0; i < 64; ++i) pc[rng.next() % V] += 1;
    DeviceArray<float> dp(pl);
    DeviceArray<int32_t> dpc(pc);
    k::SamplingConfig cp;
    cp.temperature       = 0.0f;
    cp.presence_penalty  = 1.5f;
    cp.frequency_penalty = 1.0f;
    cp.token_counts      = dpc.get();
    const auto gp        = s.run(dp, 1, cp, {3}, k::kSamplePurposeDecode);
    const int want       = argmax_ref(pl.data(), V, cp, pc);
    expect(gp[0] == want && want != raw_best, "sample greedy with penalties == penalized argmax");
}

// Chi-square goodness of fit of 16 x rounds draws (distinct positions) against the reference
// truncated distribution; also checks nothing outside the support is ever drawn.
void chi_square_case(const char* what, Rng& rng, const k::SamplingConfig& c) {
    const int cols = 16, rounds = 1000, slice = 40;
    std::vector<float> row(V, -30.0f);
    std::vector<int> slice_ids;
    for (int i = 0; i < slice; ++i) {
        const int id = 977 + i * 6151; // spread over many partial tiles and groups
        slice_ids.push_back(id);
        row[id] = static_cast<float>(rng.normal() * 1.5);
    }
    std::vector<float> logits(static_cast<size_t>(cols) * V);
    for (int t = 0; t < cols; ++t) std::copy(row.begin(), row.end(), logits.begin() + static_cast<size_t>(t) * V);
    DeviceArray<float> d(logits);
    Sampler s(V);
    const Dist ref = dist_ref(row.data(), V, c, {});
    std::vector<long> hist(V, 0);
    std::vector<int32_t> pos(cols);
    for (int r = 0; r < rounds; ++r) {
        for (int t = 0; t < cols; ++t) pos[t] = r * cols + t;
        for (int id : s.run(d, cols, c, pos, k::kSamplePurposeDecode)) ++hist[id];
    }
    const double n = double(cols) * rounds;
    long outside   = n;
    for (int id : ref.idx) outside -= hist[id];
    // Merge bins with expected count < 5.
    double chi = 0, merged_e = 0, merged_o = 0;
    int bins   = 0;
    for (size_t j = 0; j < ref.idx.size(); ++j) {
        const double e = ref.prob[j] * n, o = double(hist[ref.idx[j]]);
        if (e < 5) {
            merged_e += e;
            merged_o += o;
            continue;
        }
        chi += (o - e) * (o - e) / e;
        ++bins;
    }
    if (merged_e > 0) {
        chi += (merged_o - merged_e) * (merged_o - merged_e) / std::max(merged_e, 1e-9);
        ++bins;
    }
    const int df       = std::max(bins - 1, 1);
    const double z     = 4.75; // p ~ 1e-6 (Wilson-Hilferty)
    const double crit  = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3.0);
    const bool ok      = outside == 0 && chi <= crit;
    std::printf("%s sampling chi-square %s: support=%zu chi2=%.2f df=%d crit=%.2f outside=%ld\n",
                ok ? "PASS" : "FAIL", what, ref.idx.size(), chi, df, crit, outside);
    g_ok = g_ok && ok;
}

void test_sampling_distribution(Rng& rng) {
    k::SamplingConfig c;
    c.seed        = 0x1234;
    c.temperature = 1.0f;
    c.top_k       = 20;
    chi_square_case("T=1 top_k=20", rng, c);
    c.temperature = 0.7f;
    c.top_k       = 8;
    c.top_p       = 0.9f;
    chi_square_case("T=0.7 top_k=8 top_p=0.9", rng, c);
    c.temperature = 1.3f;
    c.top_k       = 20;
    c.top_p       = 0.95f;
    c.min_p       = 0.05f;
    chi_square_case("T=1.3 top_p=0.95 min_p=0.05", rng, c);
}

void test_sampling_determinism_and_penalties(Rng& rng) {
    const int T       = 8;
    const auto logits = random_logits(rng, T, 3.0);
    DeviceArray<float> d(logits);
    Sampler s(V);
    k::SamplingConfig c;
    c.temperature = 0.9f;
    c.top_k       = 20;
    c.top_p       = 0.95f;
    c.seed        = 42;
    std::vector<int32_t> pos = {100, 101, 102, 103, 104, 105, 106, 107};
    const auto a = s.run(d, T, c, pos, k::kSamplePurposeDecode);
    const auto b = s.run(d, T, c, pos, k::kSamplePurposeDecode);
    expect(a == b, "sample deterministic per (seed, position)");
    // Each column must equal the reference draw keyed by its own position.
    int bad = 0, ambiguous = 0;
    for (int t = 0; t < T; ++t) {
        const Dist dist = dist_ref(logits.data() + static_cast<size_t>(t) * V, V, c, {});
        double margin   = 1;
        const int want  = pick_ref(dist, -1, uniform_ref(c.seed, pos[t], k::kSamplePurposeDecode), &margin);
        if (margin < 1e-4) {
            ++ambiguous;
            continue;
        }
        bad += want != a[t];
    }
    expect(bad == 0, "sample draws == reference (seed, position, purpose) keyed draws");
    int differ = 0;
    for (int seed = 1; seed <= 8; ++seed) {
        c.seed = seed;
        differ += s.run(d, T, c, pos, k::kSamplePurposeDecode) != a;
    }
    expect(differ >= 6, "different seeds give different draws");

    // Penalized stochastic draws vs the reference.
    std::vector<int32_t> counts(V, 0);
    for (int t = 0; t < T; ++t) {
        const auto top = sorted_candidates(logits.data() + static_cast<size_t>(t) * V, V, k::SamplingConfig{}, {},
                                           nullptr, 0, 6);
        for (const auto& p : top) counts[p.second] += 1 + int(rng.next() % 3);
    }
    DeviceArray<int32_t> d_counts(counts);
    k::SamplingConfig cp;
    cp.temperature       = 0.8f;
    cp.top_k             = 10;
    cp.top_p             = 0.9f;
    cp.min_p             = 0.02f;
    cp.presence_penalty  = 0.7f;
    cp.frequency_penalty = 0.4f;
    cp.seed              = 99;
    cp.token_counts      = d_counts.get();
    bad = ambiguous = 0;
    for (int t = 0; t < T; ++t) {
        // One column per launch so each draw sees the counts its predecessors committed.
        std::vector<float> col(logits.begin() + static_cast<size_t>(t) * V, logits.begin() + static_cast<size_t>(t + 1) * V);
        DeviceArray<float> dc(col);
        const Dist dist = dist_ref(col.data(), V, cp, counts);
        double margin   = 1;
        const int want  = pick_ref(dist, -1, uniform_ref(cp.seed, 500 + t, k::kSamplePurposeDecode), &margin);
        const int got   = s.run(dc, 1, cp, {500 + t}, k::kSamplePurposeDecode)[0];
        if (margin < 1e-4) ++ambiguous;
        else bad += want != got;
        counts[got] += 1;
    }
    expect(bad == 0 && d_counts.download() == counts, "sample with penalties == reference and counts updated");
}

// ---------------------------------------------------------------- speculative

struct SpecState {
    DeviceArray<int32_t> target, drafts, extent, length, anchor, licensed, licensed_count, accepted;
    DeviceArray<float> logits;
    DeviceArray<k::SamplingConfig> cfg;
    DeviceArray<uint8_t> ws;
    explicit SpecState(int kk)
        : target(kk + 1), drafts(kk), extent(1), length(1), anchor(1), licensed(kk + 1), licensed_count(1),
          accepted(1), logits(static_cast<size_t>(kk + 1) * V), cfg(1),
          ws(k::speculative_accept_workspace_bytes(V, 15)) {}
};

// Runs speculative_accept and checks every output against the CPU reference.
bool run_accept_case(const char* what, const std::vector<float>& logits, const std::vector<int32_t>& drafts,
                     int extent, k::SamplingConfig c, std::vector<int32_t> counts, int L, bool expect_skip_ok = false) {
    const int kk = static_cast<int>(drafts.size());
    SpecState st(kk);
    st.logits.upload(logits);
    st.drafts.upload(drafts);
    st.extent.upload({extent});
    st.length.upload({L});
    st.anchor.upload({-7});
    st.licensed.upload(std::vector<int32_t>(kk + 1, -1));
    const bool use_counts = !counts.empty();
    DeviceArray<int32_t> d_counts(use_counts ? counts : std::vector<int32_t>(1, 0));
    if (use_counts) c.token_counts = d_counts.get();
    st.cfg.upload({c});
    k::argmax(st.logits.get(), kk + 1, V, st.target.get(), nullptr);
    k::speculative_accept(st.target.get(), st.logits.get(), st.drafts.get(), st.extent.get(), kk, V, st.cfg.get(),
                          st.length.get(), st.anchor.get(), st.licensed.get(), st.licensed_count.get(),
                          st.accepted.get(), st.ws.get(), st.ws.count, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());

    const AcceptResult ref = accept_ref(logits, drafts, std::min(std::max(extent, 0), kk), V, c, counts, L);
    if (ref.margin < 1e-4) {
        std::printf("skip %s: random decision within 1e-4 of its threshold\n", what);
        return expect_skip_ok;
    }
    std::vector<int32_t> want_tokens(kk + 1, 0);
    for (int i = 0; i < ref.a; ++i) want_tokens[i] = drafts[i];
    want_tokens[ref.a] = ref.terminal;
    bool ok = st.accepted.download()[0] == ref.a && st.licensed_count.download()[0] == ref.a + 1 &&
              st.anchor.download()[0] == ref.terminal && st.length.download()[0] == L + ref.a + 1 &&
              st.licensed.download() == want_tokens;
    if (use_counts) {
        for (int i = 0; i <= ref.a; ++i) counts[want_tokens[i]] += 1;
        ok = ok && d_counts.download() == counts;
    }
    std::printf("%s %s: accepted %d/%d terminal %d (got accepted %d terminal %d)\n", ok ? "PASS" : "FAIL", what,
                ref.a, extent, ref.terminal, st.accepted.download()[0], st.anchor.download()[0]);
    g_ok = g_ok && ok;
    return true;
}

void test_speculative(Rng& rng) {
    // prepare_verify_inputs
    {
        DeviceArray<int32_t> anchor(std::vector<int32_t>{11}), drafts(std::vector<int32_t>{21, 22, 23, 24}),
            base(std::vector<int32_t>{100}), extent(std::vector<int32_t>{2}), ids(5), pos(5);
        k::speculative_prepare_verify_inputs(anchor.get(), drafts.get(), base.get(), extent.get(), 4, ids.get(),
                                             pos.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        expect(ids.download() == std::vector<int32_t>{11, 21, 22, 11, 11} &&
                   pos.download() == std::vector<int32_t>{100, 101, 102, 102, 102},
               "speculative_prepare_verify_inputs");
    }

    const int kk  = 4;
    auto logits   = random_logits(rng, kk + 1, 2.0);
    auto col_best = [&](int i) { return argmax_ref(logits.data() + static_cast<size_t>(i) * V, V, {}, {}); };
    std::vector<int32_t> best(kk + 1);
    for (int i = 0; i <= kk; ++i) best[i] = col_best(i);
    const std::vector<int32_t> zero_counts(V, 0);

    // Greedy (raw argmax path).
    k::SamplingConfig g;
    g.temperature = 0.0f;
    run_accept_case("greedy all accept", logits, {best[0], best[1], best[2], best[3]}, kk, g, zero_counts, 50);
    run_accept_case("greedy first reject", logits, {best[0] + 1, best[1], best[2], best[3]}, kk, g, zero_counts, 50);
    run_accept_case("greedy middle reject", logits, {best[0], best[1], best[2] + 1, best[3]}, kk, g, zero_counts, 50);
    run_accept_case("greedy extent 2 all accept", logits, {best[0], best[1], best[2], best[3]}, 2, g, {}, 50);
    run_accept_case("greedy extent 0", logits, {best[0], best[1], best[2], best[3]}, 0, g, {}, 50);

    // Greedy with penalties: column 1's raw winner equals draft 0, so the overlay must demote it.
    {
        auto pl = logits;
        auto at = [&](int i, int v) -> float& { return pl[static_cast<size_t>(i) * V + v]; };
        at(0, 777) = 40.0f;
        at(1, 777) = 40.0f;
        at(1, 888) = 39.5f;
        at(2, 999) = 40.0f;
        k::SamplingConfig gp;
        gp.temperature      = 0.0f;
        gp.presence_penalty = 2.0f;
        run_accept_case("greedy penalties overlay", pl, {777, 888, 999, 5}, kk, gp, zero_counts, 7);
        run_accept_case("greedy penalties overlay rejects repeat", pl, {777, 777, 999, 5}, kk, gp, zero_counts, 7);
    }

    // Sampling: drafts with p ~ 1 accept, drafts outside the support reject.
    k::SamplingConfig sc;
    sc.temperature = 1.0f;
    sc.top_k       = 20;
    sc.seed        = 2024;
    auto craft = [&](const std::vector<bool>& accept) {
        auto pl = logits;
        std::vector<int32_t> drafts(kk);
        for (int i = 0; i < kk; ++i) {
            drafts[i]                                     = 1000 + 37 * i;
            pl[static_cast<size_t>(i) * V + drafts[i]] = accept[i] ? 40.0f : -50.0f;
        }
        return std::make_pair(pl, drafts);
    };
    {
        auto [pl, dr] = craft({true, true, true, true});
        run_accept_case("sampling all accept (+bonus)", pl, dr, kk, sc, zero_counts, 300);
    }
    {
        auto [pl, dr] = craft({false, true, true, true});
        run_accept_case("sampling first reject (residual)", pl, dr, kk, sc, zero_counts, 300);
    }
    {
        auto [pl, dr] = craft({true, true, false, true});
        run_accept_case("sampling middle reject (residual)", pl, dr, kk, sc, zero_counts, 300);
    }

    // Random rounds: drafts among each column's top candidates, mixed filters/penalties.
    int ran = 0;
    for (int trial = 0; trial < 24; ++trial) {
        const int kt = 1 + static_cast<int>(rng.next() % 8);
        auto lg      = random_logits(rng, kt + 1, 2.5);
        // Peak each column's leader so drafts equal to it are usually accepted.
        for (int i = 0; i <= kt; ++i) lg[static_cast<size_t>(i) * V + rng.next() % V] = 16.0f;
        std::vector<int32_t> dr(kt);
        for (int i = 0; i < kt; ++i) {
            const auto top = sorted_candidates(lg.data() + static_cast<size_t>(i) * V, V, {}, {}, nullptr, 0, 4);
            dr[i]          = top[rng.next() % 3 == 0 ? rng.next() % 4 : 0].second;
        }
        k::SamplingConfig c;
        c.temperature = static_cast<float>(0.5 + rng.uniform());
        c.top_k       = 1 + static_cast<int>(rng.next() % 20);
        c.top_p       = trial % 3 == 0 ? 0.8f : 1.0f;
        c.min_p       = trial % 4 == 0 ? 0.05f : 0.0f;
        c.seed        = rng.next();
        std::vector<int32_t> counts;
        if (trial % 2 == 0) {
            counts.assign(V, 0);
            for (int i = 0; i < kt; ++i) counts[dr[i]] += static_cast<int>(rng.next() % 2);
            c.presence_penalty  = 0.5f;
            c.frequency_penalty = 0.25f;
        }
        if (trial % 5 == 0) c.temperature = 0.0f; // greedy, possibly penalized
        char what[96];
        std::snprintf(what, sizeof(what), "random round %d (k=%d temp=%.2f top_k=%d)", trial, kt, c.temperature,
                      c.top_k);
        ran += run_accept_case(what, lg, dr, static_cast<int>(rng.next() % (kt + 1)), c, counts,
                               static_cast<int>(rng.next() % 5000), false);
    }
    expect(ran >= 20, "random speculative rounds compared");

    // Capturability: argmax + accept recorded into a CUDA graph.
    {
        SpecState st(kk);
        st.logits.upload(logits);
        st.drafts.upload({best[0], best[1], best[2] + 1, best[3]});
        st.extent.upload({kk});
        st.length.upload({10});
        st.cfg.upload({sc});
        cudaStream_t stream;
        FNGINE_TEST_CUDA(cudaStreamCreate(&stream));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        FNGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        k::argmax(st.logits.get(), kk + 1, V, st.target.get(), stream);
        k::speculative_accept(st.target.get(), st.logits.get(), st.drafts.get(), st.extent.get(), kk, V,
                              st.cfg.get(), st.length.get(), st.anchor.get(), st.licensed.get(),
                              st.licensed_count.get(), st.accepted.get(), st.ws.get(), st.ws.count, stream);
        FNGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
        FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
        FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
        FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
        const AcceptResult r1 = accept_ref(logits, {best[0], best[1], best[2] + 1, best[3]}, kk, V, sc, {}, 10);
        const AcceptResult r2 =
            accept_ref(logits, {best[0], best[1], best[2] + 1, best[3]}, kk, V, sc, {}, 10 + r1.a + 1);
        expect(st.length.download()[0] == 10 + r1.a + 1 + r2.a + 1 && st.anchor.download()[0] == r2.terminal,
               "argmax + speculative_accept graph replay (length advances, RNG keyed by length)");
        FNGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
        FNGINE_TEST_CUDA(cudaGraphDestroy(graph));
        FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
    }
}

// ---------------------------------------------------------------- mtp round

void test_mtp_round() {
    struct Case {
        int k, accepted, frontier, budget, licensed, ctx;
    };
    const Case cases[] = {{3, 2, 100, 50, 3, 1000}, {3, 3, 100, 4, 4, 1000}, {3, 0, 998, 50, 1, 1000},
                          {1, 1, 10, 100, 2, 1000}, {8, 5, 40, 9, 6, 1000}};
    int bad = 0;
    for (const Case& c : cases) {
        std::vector<int32_t> verify(c.k + 1);
        for (int j = 0; j <= c.k; ++j) verify[j] = 500 + j;
        const int steps = c.k > 1 ? c.k - 1 : 1;
        DeviceArray<int32_t> d_verify(verify), anchor(std::vector<int32_t>{77}), acc(std::vector<int32_t>{c.accepted}),
            front(std::vector<int32_t>{c.frontier}), budget(std::vector<int32_t>{c.budget}),
            lic(std::vector<int32_t>{c.licensed}), align(c.k + 1), next(1), arp(steps), arv(steps);
        k::mtp_prepare_next_round(d_verify.get(), c.k, anchor.get(), acc.get(), front.get(), budget.get(), lic.get(),
                                  c.ctx, align.get(), next.get(), arp.get(), arv.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        std::vector<int32_t> want_align(c.k + 1), want_p(steps), want_v(steps);
        for (int j = 0; j <= c.k; ++j) want_align[j] = j < c.accepted ? verify[j + 1] : 77;
        const int remaining = c.budget - c.licensed;
        int want_next = std::min(remaining > 1 ? remaining - 1 : 0, c.ctx - c.frontier - 1);
        want_next     = std::clamp(want_next, 0, c.k);
        for (int s = 0; s < steps; ++s) {
            want_p[s] = c.frontier + s;
            want_v[s] = s + 1 < want_next;
        }
        bad += align.download() != want_align || next.download()[0] != want_next || arp.download() != want_p ||
               arv.download() != want_v;
    }
    expect(bad == 0, "mtp_prepare_next_round (alignment ids, next extent, AR steps)");
}

} // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    Rng rng(0x4ead5eedull);
    test_embed_gather(rng);
    test_argmax(rng);
    test_sampling_greedy(rng);
    test_sampling_distribution(rng);
    test_sampling_determinism_and_penalties(rng);
    test_speculative(rng);
    test_mtp_round();
    std::printf("%s\n", g_ok ? "kernels_head: OK" : "kernels_head: FAILED");
    return g_ok ? 0 : 1;
}
