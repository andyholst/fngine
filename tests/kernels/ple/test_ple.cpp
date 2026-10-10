// kernels_ple: IQ4_NL row dequantization (bit-exact vs ggml), the PLE mixer vs an FP64 oracle
// (decode/verify and prefill), chunked prefill == one chunk, and record+fold == sequential
// commits (bit-exact state).
#include "kernels/ple/ple.h"

#include "common/test_util.h"
#include "core/exchange.h"
#include "core/model_dims.h"

#include "ggml-iq.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>

using namespace fngine;
using namespace fngine::test;
namespace k = fngine::kernels;

namespace {

constexpr int H  = dims::kHidden;       // 2560
constexpr int C  = dims::kStreamHidden; // 10240
constexpr int S9 = dims::kPleConvState; // 9
constexpr int kTokenRaw = dims::kPleHeads * dims::kPleRowBytes;

bool g_ok = true;
void expect(bool cond, const char* what) {
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    g_ok = g_ok && cond;
}

std::vector<__nv_bfloat16> random_bf16(Rng& rng, size_t n, double scale) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = float_to_bf16(static_cast<float>(rng.normal() * scale));
    return v;
}
std::vector<float> random_f32(Rng& rng, size_t n, double mean, double scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = static_cast<float>(mean + rng.normal() * scale);
    return v;
}
bool same_bits(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

// ---------------------------------------------------------------- dequant

// Random IQ4_NL rows with finite fp16 scales.
std::vector<uint8_t> random_rows(Rng& rng, int T) {
    std::vector<uint8_t> raw(static_cast<size_t>(T) * kTokenRaw);
    for (int r = 0; r < T * dims::kPleHeads; ++r) {
        for (int b = 0; b < 5; ++b) {
            uint8_t* blk          = raw.data() + r * dims::kPleRowBytes + b * 18;
            const ggml_half d     = GGML_FP32_TO_FP16(static_cast<float>(rng.normal() * 0.01));
            blk[0]                = static_cast<uint8_t>(d & 0xff);
            blk[1]                = static_cast<uint8_t>(d >> 8);
            for (int j = 0; j < 16; ++j) blk[2 + j] = static_cast<uint8_t>(rng.next());
        }
    }
    return raw;
}

std::vector<__nv_bfloat16> ggml_dequant(const std::vector<uint8_t>& raw, int T) {
    std::vector<__nv_bfloat16> out(static_cast<size_t>(T) * H);
    std::vector<float> row(dims::kPleHeadDim);
    for (int r = 0; r < T * dims::kPleHeads; ++r) {
        dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl*>(raw.data() + r * dims::kPleRowBytes),
                              row.data(), dims::kPleHeadDim);
        for (int i = 0; i < dims::kPleHeadDim; ++i) out[r * dims::kPleHeadDim + i] = float_to_bf16(row[i]);
    }
    return out;
}

void test_dequant_device(Rng& rng) {
    const int T = 7;
    const auto raw = random_rows(rng, T);
    DeviceArray<uint8_t> d_raw(raw);
    DeviceArray<__nv_bfloat16> d_e(static_cast<size_t>(T) * H);
    k::ple_dequant(d_raw.get(), T, d_e.get(), {}, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto got = d_e.download();
    const auto ref = ggml_dequant(raw, T);
    expect(same_bits(got.data(), ref.data(), ref.size() * 2), "ple_dequant device rows bit-exact");
}

void test_dequant_mapped_wait(Rng& rng) {
    const int T = 3;
    const auto raw = random_rows(rng, T);
    uint8_t* h_rows   = nullptr;
    uint32_t* h_cells = nullptr; // [0]=ready, [16]=expected, [32]=error
    FNGINE_TEST_CUDA(cudaHostAlloc(&h_rows, raw.size(), cudaHostAllocMapped));
    FNGINE_TEST_CUDA(cudaHostAlloc(&h_cells, 256, cudaHostAllocMapped));
    std::memset(h_cells, 0, 256);
    h_cells[0]  = 4;
    h_cells[16] = 5;
    uint8_t* d_rows   = nullptr;
    uint32_t* d_cells = nullptr;
    FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_rows), h_rows, 0));
    FNGINE_TEST_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_cells), h_cells, 0));
    DeviceArray<__nv_bfloat16> d_e(static_cast<size_t>(T) * H);

    cudaStream_t stream;
    FNGINE_TEST_CUDA(cudaStreamCreate(&stream));
    k::ple_dequant(d_rows, T, d_e.get(), {d_cells, d_cells + 16, d_cells + 32}, stream);
    // The host stages the rows only after launch, then publishes the sequence number.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    std::memcpy(h_rows, raw.data(), raw.size());
    host_store_release(h_cells[0], 5u);
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    expect(waiting, "ple_dequant waits for ple_ready");
    expect(h_cells[32] == 0, "ple_dequant no error after publish");
    const auto got = d_e.download();
    const auto ref = ggml_dequant(raw, T);
    expect(same_bits(got.data(), ref.data(), ref.size() * 2), "ple_dequant host-mapped rows bit-exact");
    FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
    cudaFreeHost(h_rows);
    cudaFreeHost(h_cells);
}

// Real rows of the built pack's ngram.bin (row r at 4096 + (r/45)*4096 + (r%45)*90).
void test_dequant_pack(Rng& rng) {
    const char* pack = std::getenv("FNGINE_TEST_PACK");
    if (!pack || !*pack) {
        std::printf("skip ple_dequant pack rows: FNGINE_TEST_PACK is not set\n");
        return;
    }
    std::ifstream f(std::string(pack) + "/ngram.bin", std::ios::binary);
    if (!f) {
        std::printf("FAIL cannot open %s/ngram.bin\n", pack);
        g_ok = false;
        return;
    }
    const int T = 4;
    std::vector<uint8_t> raw(static_cast<size_t>(T) * kTokenRaw);
    for (int r = 0; r < T * dims::kPleHeads; ++r) {
        const uint64_t row = rng.next() % 320001536ull;
        f.seekg(static_cast<std::streamoff>(4096 + (row / 45) * 4096 + (row % 45) * 90));
        f.read(reinterpret_cast<char*>(raw.data() + r * dims::kPleRowBytes), dims::kPleRowBytes);
    }
    DeviceArray<uint8_t> d_raw(raw);
    DeviceArray<__nv_bfloat16> d_e(static_cast<size_t>(T) * H);
    k::ple_dequant(d_raw.get(), T, d_e.get(), {}, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto got = d_e.download();
    const auto ref = ggml_dequant(raw, T);
    expect(same_bits(got.data(), ref.data(), ref.size() * 2), "ple_dequant pack rows bit-exact");
}

// ---------------------------------------------------------------- mixer oracle

struct PleWeightsHost {
    std::vector<float> nk, nq, nc, conv; // conv [10240][4]
};

struct PleInputs {
    int T = 0;
    std::vector<float> X;               // [T][10240]
    std::vector<__nv_bfloat16> K, V;    // [T][10240], [T][2560]
};

PleInputs random_inputs(Rng& rng, int T) {
    PleInputs in;
    in.T = T;
    in.X.resize(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < 4; ++s) {
            const double scale = 0.3 + 2.0 * rng.uniform();
            for (int d = 0; d < H; ++d) in.X[(t * 4 + s) * H + d] = static_cast<float>(rng.normal() * scale);
        }
    in.K = random_bf16(rng, static_cast<size_t>(T) * C, 1.0);
    in.V = random_bf16(rng, static_cast<size_t>(T) * H, 1.0);
    return in;
}

double bf16_ulp(double v) {
    const double a = std::fabs(v);
    if (a < 1e-30) return 1e-38;
    int e;
    std::frexp(a, &e); // a = m * 2^e, m in [0.5,1)
    return std::ldexp(1.0, e - 8);
}
// True when the exact value lies within `tol` of a BF16 rounding midpoint, so FP32 evaluation
// may legitimately round to the other neighbour.
bool near_midpoint(double y, double tol) {
    const double b   = bf16_to_float(float_to_bf16(static_cast<float>(y)));
    const double ulp = bf16_ulp(std::max(std::fabs(y), std::fabs(b)));
    return std::fabs(std::fabs(y - b) - ulp / 2) <= tol;
}
bool near_midpoint(double y) { return near_midpoint(y, 3e-5 * std::fabs(y)); }
double rbf(double y) { return bf16_to_float(float_to_bf16(static_cast<float>(y))); }

// sigmoid(sign(g) * sqrt(max(|g|, 1e-6))) with g = dot / sqrt(2560): monotone in dot.
double gate_of(double dot) {
    double g = dot / std::sqrt(2560.0);
    g        = g == 0 ? 0 : std::copysign(std::sqrt(std::max(std::fabs(g), 1e-6)), g);
    return 1.0 / (1.0 + std::exp(-g));
}

struct PleRef {
    std::vector<double> out;     // [T][10240]
    std::vector<double> allow;   // extra per-element allowance from rounding ambiguity
    std::vector<double> c;       // bf16-valued [T][10240]
    std::vector<double> c_allow; // allowed |c_device - c| (rounding flips of GV / c)
    std::vector<double> state;   // last 9 of (state0 || c), bf16-valued
    double worst_gate_rel = 0;   // largest relative gate uncertainty (ill-conditioned dot ~ 0)
};

PleRef oracle(const PleInputs& in, const PleWeightsHost& w, const std::vector<__nv_bfloat16>& state0) {
    const int T = in.T;
    PleRef r;
    r.out.assign(static_cast<size_t>(T) * C, 0.0);
    r.allow.assign(static_cast<size_t>(T) * C, 0.0);
    r.c.assign(static_cast<size_t>(T) * C, 0.0);
    r.c_allow.assign(static_cast<size_t>(T) * C, 0.0);
    std::vector<double> gv(static_cast<size_t>(T) * C);
    std::vector<double> gv_allow(static_cast<size_t>(T) * C, 0.0); // gate scale + rounding flip
    std::vector<double> gv_flip(static_cast<size_t>(T) * C, 0.0);  // rounding flip only
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < 4; ++s) {
            const size_t g = static_cast<size_t>(t) * C + s * H;
            double sk = 0, sx = 0;
            for (int d = 0; d < H; ++d) {
                const double kv = bf16_to_float(in.K[g + d]);
                sk += kv * kv;
                sx += double(in.X[g + d]) * in.X[g + d];
            }
            const double rk = 1.0 / std::sqrt(sk / H + 1e-6), rx = 1.0 / std::sqrt(sx / H + 1e-6);
            // dot uncertainty of the FP32 kernel: accumulation (~log2(2560) ulps of sum|kn*qn|)
            // plus possible one-ulp BF16 flips of Kn/Qn values sitting on a rounding midpoint.
            double dot = 0, dot_unc = 0;
            for (int d = 0; d < H; ++d) {
                const double ky = bf16_to_float(in.K[g + d]) * rk * w.nk[s * H + d];
                const double qy = double(in.X[g + d]) * rx * w.nq[s * H + d];
                const double kn = rbf(ky), qn = rbf(qy);
                dot += kn * qn;
                dot_unc += 2e-6 * std::fabs(kn * qn);
                if (near_midpoint(ky, 2e-6 * std::fabs(ky))) dot_unc += bf16_ulp(kn) * std::fabs(qn);
                if (near_midpoint(qy, 2e-6 * std::fabs(qy))) dot_unc += bf16_ulp(qn) * std::fabs(kn);
            }
            // The sign-sqrt makes the gate ill-conditioned near dot = 0; bound the gate's range.
            const double sig     = gate_of(dot);
            const double sig_unc = std::max(std::fabs(gate_of(dot + dot_unc) - sig),
                                            std::fabs(gate_of(dot - dot_unc) - sig)) +
                                   1e-6 * sig;
            r.worst_gate_rel     = std::max(r.worst_gate_rel, sig_unc / sig);
            double sg            = 0;
            for (int d = 0; d < H; ++d) {
                const double v   = bf16_to_float(in.V[static_cast<size_t>(t) * H + d]);
                const double y   = sig * v;
                const double unc = std::fabs(v) * sig_unc;
                gv[g + d]        = rbf(y);
                gv_flip[g + d]   = near_midpoint(y, unc + 3e-5 * std::fabs(y)) ? bf16_ulp(y) : 0.0;
                gv_allow[g + d]  = unc + gv_flip[g + d];
                sg += gv[g + d] * gv[g + d];
            }
            // c = GV / rms(GV) * norm is invariant to the gate scale, so gate uncertainty only
            // shows up through GV rounding flips (scaled into c units) and c's own rounding.
            const double rc = 1.0 / std::sqrt(sg / H + 1e-6);
            for (int d = 0; d < H; ++d) {
                const double y   = gv[g + d] * rc * w.nc[s * H + d];
                const double rel = 1.01 * gv_flip[g + d] * rc * std::fabs(w.nc[s * H + d]);
                r.c[g + d]       = rbf(y);
                r.c_allow[g + d] = rel + (rel > 0 || near_midpoint(y) || sig_unc > 1e-5 * sig
                                              ? bf16_ulp(std::fabs(y) + rel)
                                              : 0.0);
            }
        }
    }
    auto cval = [&](int p, int ch) -> double {
        return p >= 0 ? r.c[static_cast<size_t>(p) * C + ch] : bf16_to_float(state0[(S9 + p) * C + ch]);
    };
    for (int t = 0; t < T; ++t)
        for (int ch = 0; ch < C; ++ch) {
            double acc = 0, allow = 0;
            for (int j = 0; j < 4; ++j) {
                const int p = t - S9 + 3 * j;
                acc += double(w.conv[ch * 4 + j]) * cval(p, ch);
                if (p >= 0)
                    allow += 1.2 * std::fabs(w.conv[ch * 4 + j]) * r.c_allow[static_cast<size_t>(p) * C + ch];
            }
            const size_t i = static_cast<size_t>(t) * C + ch;
            allow += gv_allow[i];
            r.out[i]   = gv[i] + acc / (1.0 + std::exp(-acc));
            r.allow[i] = allow;
        }
    r.state.resize(static_cast<size_t>(S9) * C);
    for (int i = 0; i < S9; ++i)
        for (int ch = 0; ch < C; ++ch) r.state[static_cast<size_t>(i) * C + ch] = cval(T - S9 + i, ch);
    return r;
}

struct DeviceWeights {
    DeviceArray<float> nk, nq, nc, conv;
    explicit DeviceWeights(const PleWeightsHost& w) : nk(w.nk), nq(w.nq), nc(w.nc), conv(w.conv) {}
    k::PleMixWeights view() const { return {nk.get(), nq.get(), nc.get(), conv.get()}; }
};

struct MixRun {
    std::vector<float> X;
    std::vector<__nv_bfloat16> c, state;
};

// Runs ple_mix over the tokens in `chunks` consecutive pieces on the given state.
MixRun run_mix(const PleInputs& in, const DeviceWeights& w, const std::vector<__nv_bfloat16>& state0,
               const std::vector<int>& chunks, bool commit) {
    DeviceArray<float> X(in.X);
    DeviceArray<__nv_bfloat16> K(in.K), V(in.V), state(state0), c(static_cast<size_t>(in.T) * C);
    DeviceArray<uint8_t> ws(k::ple_mix_workspace_bytes(in.T));
    int t0 = 0;
    for (int n : chunks) {
        k::ple_mix(X.get() + static_cast<size_t>(t0) * C, K.get() + static_cast<size_t>(t0) * C,
                   V.get() + static_cast<size_t>(t0) * H, w.view(), state.get(),
                   c.get() + static_cast<size_t>(t0) * C, n, commit, ws.get(), ws.count, nullptr);
        t0 += n;
    }
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    return {X.download(), c.download(), state.download()};
}

void check_against_oracle(const char* what, const PleInputs& in, const PleRef& ref, const MixRun& got) {
    const size_t n = ref.out.size();
    double sq      = 0;
    for (double v : ref.out) sq += v * v;
    const double rms = std::sqrt(sq / n);
    double worst = 0, raw = 0, max_allow = 0, mean_err = 0;
    int beyond_tol = 0;
    for (size_t i = 0; i < n; ++i) {
        const double err   = std::fabs(double(got.X[i]) - (double(in.X[i]) + ref.out[i]));
        const double ratio = (err - ref.allow[i]) / rms;
        raw                = std::max(raw, err / rms);
        max_allow          = std::max(max_allow, ref.allow[i] / rms);
        mean_err += err / rms / n;
        beyond_tol += err > 2e-3 * rms; // only explainable by an allowed rounding flip
        worst = std::max(worst, ratio);
    }
    const bool ok = worst <= 2e-3;
    std::printf("%s %s X delta: max (|err|-allow)/rms(ref)=%.3e (tol 2e-3); raw max|err|/rms=%.3e "
                "mean=%.1e, %d elements above tol (rounding flips), max allow/rms=%.2e, worst gate rel. "
                "uncertainty=%.1e\n",
                ok ? "PASS" : "FAIL", what, worst, raw, mean_err, beyond_tol, max_allow, ref.worst_gate_rel);
    g_ok = g_ok && ok;

    int c_bad = 0, c_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        const double g = bf16_to_float(got.c[i]);
        if (g == ref.c[i]) continue;
        ++c_diff;
        if (std::fabs(g - ref.c[i]) > ref.c_allow[i]) {
            if (c_bad < 5)
                std::printf("  c[%zu] t=%zu s=%zu d=%zu got %.6g ref %.6g allow %.3g\n", i, i / C, (i % C) / H,
                            i % H, g, ref.c[i], ref.c_allow[i]);
            ++c_bad;
        }
    }
    std::printf("%s %s c record: %d differ from the oracle's rounding, %d beyond the rounding-flip allowance\n",
                c_bad ? "FAIL" : "PASS", what, c_diff, c_bad);
    g_ok = g_ok && c_bad == 0;
}

PleWeightsHost random_weights(Rng& rng) {
    PleWeightsHost w;
    w.nk   = random_f32(rng, C, 1.0, 0.2);
    w.nq   = random_f32(rng, C, 1.0, 0.2);
    w.nc   = random_f32(rng, C, 1.0, 0.2);
    w.conv = random_f32(rng, static_cast<size_t>(C) * 4, 0.0, 0.5);
    return w;
}

void test_mix(Rng& rng) {
    const PleWeightsHost wh = random_weights(rng);
    const DeviceWeights w(wh);
    const auto state0 = random_bf16(rng, static_cast<size_t>(S9) * C, 1.0);

    // Decode/verify: T = 9, state read only, c recorded.
    {
        const PleInputs in = random_inputs(rng, 9);
        const PleRef ref   = oracle(in, wh, state0);
        const MixRun got   = run_mix(in, w, state0, {9}, false);
        check_against_oracle("ple_mix decode T=9", in, ref, got);
        expect(same_bits(got.state.data(), state0.data(), state0.size() * 2), "ple_mix decode leaves state unchanged");
    }
    // Decode T = 1.
    {
        const PleInputs in = random_inputs(rng, 1);
        const PleRef ref   = oracle(in, wh, state0);
        check_against_oracle("ple_mix decode T=1", in, ref, run_mix(in, w, state0, {1}, false));
    }
    // Prefill: T = 201 in one chunk, state committed; then the same in 37/64/100 chunks.
    const PleInputs in = random_inputs(rng, 201);
    const PleRef ref   = oracle(in, wh, state0);
    const MixRun one   = run_mix(in, w, state0, {201}, true);
    check_against_oracle("ple_mix prefill T=201", in, ref, one);
    {
        int bad = 0;
        for (size_t i = 0; i < ref.state.size(); ++i) {
            // State holds the committed c values; for the last 9 tokens compare with the run's c.
            const size_t src = static_cast<size_t>(201 - S9) * C + i;
            bad += !same_bits(&one.state[i], &one.c[src], 2);
        }
        expect(bad == 0, "ple_mix prefill state == last 9 c");
    }
    const MixRun chunked = run_mix(in, w, state0, {37, 64, 100}, true);
    expect(same_bits(chunked.X.data(), one.X.data(), one.X.size() * 4) &&
               same_bits(chunked.c.data(), one.c.data(), one.c.size() * 2) &&
               same_bits(chunked.state.data(), one.state.data(), one.state.size() * 2),
           "ple_mix prefill chunks 37/64/100 == one chunk (bit-exact)");
    const MixRun tiny = run_mix(in, w, state0, {3, 5, 1, 192}, true);
    expect(same_bits(tiny.X.data(), one.X.data(), one.X.size() * 4) &&
               same_bits(tiny.state.data(), one.state.data(), one.state.size() * 2),
           "ple_mix prefill chunks 3/5/1/192 (shorter than the state) == one chunk");
}

// Verify T = 9 with record, then fold n accepted columns; compare with n sequential T = 1 commits.
void test_record_fold(Rng& rng) {
    const PleWeightsHost wh = random_weights(rng);
    const DeviceWeights w(wh);
    const auto state0  = random_bf16(rng, static_cast<size_t>(S9) * C, 1.0);
    const PleInputs in = random_inputs(rng, 9);

    DeviceArray<float> X(in.X);
    DeviceArray<__nv_bfloat16> K(in.K), V(in.V), state(state0), c(static_cast<size_t>(9) * C);
    DeviceArray<uint8_t> ws(k::ple_mix_workspace_bytes(9));
    DeviceArray<int32_t> n_commit(1);

    // Capture verify + fold as a graph; n_commit is read from device memory at replay.
    cudaStream_t stream;
    FNGINE_TEST_CUDA(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    FNGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    k::ple_mix(X.get(), K.get(), V.get(), w.view(), state.get(), c.get(), 9, false, ws.get(), ws.count, stream);
    k::ple_fold(state.get(), c.get(), n_commit.get(), 9, stream);
    FNGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
    FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));

    for (int n : {0, 1, 5, 9}) {
        state.upload(state0);
        X.upload(in.X);
        n_commit.upload(std::vector<int32_t>{n});
        FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
        FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
        const auto folded = state.download();
        const auto Xv     = X.download();

        // Sequential plain decode: n steps of T = 1 with state commit.
        DeviceArray<float> X1(in.X);
        DeviceArray<__nv_bfloat16> s1(state0), c1(C);
        DeviceArray<uint8_t> ws1(k::ple_mix_workspace_bytes(1));
        for (int t = 0; t < n; ++t) {
            k::ple_mix(X1.get() + static_cast<size_t>(t) * C, K.get() + static_cast<size_t>(t) * C,
                       V.get() + static_cast<size_t>(t) * H, w.view(), s1.get(), c1.get(), 1, true, ws1.get(),
                       ws1.count, nullptr);
        }
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        const auto seq_state = s1.download();
        const auto seq_X     = X1.download();
        char what[128];
        std::snprintf(what, sizeof(what), "record+fold n=%d state == sequential T=1 commits (bit-exact)", n);
        expect(same_bits(folded.data(), seq_state.data(), folded.size() * 2), what);
        std::snprintf(what, sizeof(what), "record n=%d committed columns X == sequential (bit-exact)", n);
        expect(same_bits(Xv.data(), seq_X.data(), static_cast<size_t>(n) * C * 4), what);
    }
    FNGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
    FNGINE_TEST_CUDA(cudaGraphDestroy(graph));
    FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
}

} // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    Rng rng(0x91e5ull);
    test_dequant_device(rng);
    test_dequant_mapped_wait(rng);
    test_dequant_pack(rng);
    test_mix(rng);
    test_record_fold(rng);
    std::printf("%s\n", g_ok ? "kernels_ple: OK" : "kernels_ple: FAILED");
    return g_ok ? 0 : 1;
}
