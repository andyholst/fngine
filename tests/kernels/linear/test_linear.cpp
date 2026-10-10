// kernels_linear: every registered FP8 (N, K, epilogue) on the A16 and A8 paths and every BF16
// entry at T in {1, 3, 9, 16, 64, 128, 300}, against FP64 oracles over the stored (dequantized)
// weights and the same BF16 inputs. A8 is compared with an oracle that applies the same per-token
// E4M3 activation quantization; its error against the unquantized oracle is reported. With
// FNGINE_TEST_PACK set, real pack tensors are also checked (and the C++ port of the pack's FP8
// rule must reproduce the pack's codes and scales bit-exactly from dense_bf16.bin).
//
// Pass rule: max_i (|got_i - ref_i| - a_i) <= 2e-3 * rms(ref), where a_i = 2^-8 |ref_i| for BF16
// outputs (the RNE rounding of the output itself) and 0 for FP32 outputs; BF16 outputs must also
// equal bf16_rne(ref) for >= 99% of elements; a repeated call must be bitwise identical; the
// SiluQuarter pad columns 320..383 must be untouched.

#include "kernels/linear/linear.h"
#include "kernels/linear/linear_ref.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace fngine;
using namespace fngine::kernels;
using namespace fngine::test;
using namespace fngine::test::linear_ref;

namespace {

constexpr double kTol = 2e-3;
const int kTokenCases[] = {1, 3, 9, 16, 64, 128, 300};

struct Shape {
    int n, k;
    LinearEpilogue kind;
};

const Shape kFp8Shapes[] = {
    {16384, 2560, LinearEpilogue::StoreBf16},  {13312, 2560, LinearEpilogue::StoreBf16},
    {2560, 6144, LinearEpilogue::GrWrite},      {1280, 2560, LinearEpilogue::SwiGlu},
    {2560, 640, LinearEpilogue::SharedScale},   {320, 10240, LinearEpilogue::SiluQuarter},
    {10240, 384, LinearEpilogue::GrMix},        {10240, 2560, LinearEpilogue::StoreBf16},
    {2560, 2560, LinearEpilogue::StoreBf16},    {248320, 2560, LinearEpilogue::StoreF32},
};

const Shape kBf16OnlyShapes[] = {
    {513, 2560, LinearEpilogue::StoreF32},
    {96, 2560, LinearEpilogue::StoreF32},
    {640, 2560, LinearEpilogue::StoreBf16},
};

bool bf16_output(LinearEpilogue kind) {
    return kind == LinearEpilogue::StoreBf16 || kind == LinearEpilogue::SwiGlu ||
           kind == LinearEpilogue::SiluQuarter || kind == LinearEpilogue::GrMix;
}

// Number of output units (elements per token) of an epilogue.
int units_of(const Shape& s) {
    switch (s.kind) {
    case LinearEpilogue::SwiGlu: return s.n / 2;
    case LinearEpilogue::GrMix: return s.n / 4;
    default: return s.n;
    }
}

double sigmoid_d(double v) { return 1.0 / (1.0 + std::exp(-v)); }
double silu_d(double v) { return v * sigmoid_d(v); }

// Host copies of one weight: BF16 source, FP8 codes + scales (dequantized on the fly).
struct HostWeight {
    Shape shape;
    std::vector<uint16_t> bf16;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
};

HostWeight make_weight(const Shape& s, uint64_t seed) {
    HostWeight w{s, std::vector<uint16_t>(size_t(s.n) * s.k), std::vector<uint8_t>(size_t(s.n) * s.k),
                 std::vector<uint16_t>(size_t(s.n))};
    const double sigma = 1.0 / std::sqrt(double(s.k));
    parallel_for(s.n, [&](int64_t r) {
        Rng rng(seed * 0x9E3779B97F4A7C15ull + uint64_t(r));
        const double row_scale = std::exp(rng.uniform(-1.0, 1.0)) * sigma;
        uint16_t* row          = w.bf16.data() + r * s.k;
        for (int j = 0; j < s.k; ++j) row[j] = bf16_bits(float_to_bf16(float(rng.normal() * row_scale)));
        if (r % 997 == 5) std::fill(row, row + s.k, uint16_t(0));  // zero rows: scale 0
        quantize_fp8_rows(row, 1, s.k, w.codes.data() + r * s.k, w.scales.data() + r);
    });
    return w;
}

std::vector<uint16_t> random_bf16(size_t count, uint64_t seed, double sigma) {
    std::vector<uint16_t> v(count);
    parallel_for(int64_t((count + 4095) / 4096), [&](int64_t block) {
        Rng rng(seed + uint64_t(block) * 7919);
        const size_t end = std::min(count, size_t(block + 1) * 4096);
        for (size_t i = size_t(block) * 4096; i < end; ++i)
            v[i] = bf16_bits(float_to_bf16(float(rng.normal() * sigma)));
    });
    return v;
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) {
        FNGINE_TEST_CUDA(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T) + 256));
    }
    Dev(const std::vector<T>& h) : Dev(h.size()) { upload(h); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    Dev& operator=(Dev&& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
    void upload(const std::vector<T>& h) {
        FNGINE_TEST_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(n);
        FNGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

// Epilogue-side inputs shared by both weight formats for one (shape, T).
struct CaseInputs {
    int tokens;
    std::vector<uint16_t> x;        // [T][K]
    std::vector<float> x_stream;    // GrWrite: X [T][4][N] before the write
    std::vector<float> inj;         // GrWrite [T][4]
    std::vector<float> logits;      // SharedScale [T][513]
    std::vector<uint16_t> xn;       // GrMix [T][N]
};

CaseInputs make_inputs(const Shape& s, int tokens, uint64_t seed) {
    CaseInputs in;
    in.tokens = tokens;
    in.x      = random_bf16(size_t(tokens) * s.k, seed, 1.0);
    Rng rng(seed ^ 0xABCDEFull);
    if (s.kind == LinearEpilogue::GrWrite) {
        in.x_stream.resize(size_t(tokens) * 4 * s.n);
        for (auto& v : in.x_stream) v = float(rng.normal());
        in.inj.resize(size_t(tokens) * 4);
        for (auto& v : in.inj) v = float(rng.uniform(0.2, 2.0));
    } else if (s.kind == LinearEpilogue::SharedScale) {
        in.logits.resize(size_t(tokens) * 513);
        for (auto& v : in.logits) v = float(rng.normal() * 2.0);
    } else if (s.kind == LinearEpilogue::GrMix) {
        in.xn = random_bf16(size_t(tokens) * s.n, seed ^ 0x55ull, 1.0);
    }
    return in;
}

// Units checked by the oracle: all when cheap, otherwise a deterministic sample incl. the ends.
std::vector<int> sample_units(const Shape& s, int tokens) {
    const int units       = units_of(s);
    const int rows_per    = s.kind == LinearEpilogue::SwiGlu ? 2 : (s.kind == LinearEpilogue::GrMix ? 4 : 1);
    const double budget   = 6e8;  // multiply-adds
    const int max_units   = int(std::max(64.0, budget / (double(rows_per) * s.k * tokens)));
    std::vector<int> out;
    if (units <= max_units) {
        for (int u = 0; u < units; ++u) out.push_back(u);
        return out;
    }
    Rng rng(uint64_t(units) * 31 + uint64_t(tokens));
    out.push_back(0);
    out.push_back(units - 1);
    while (int(out.size()) < max_units) out.push_back(int(rng.next() % uint64_t(units)));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

enum class Oracle { Fp8, Fp8A8, Bf16 };

// FP64 dot product of weight row r with token t under the oracle's weight/activation model.
struct OracleData {
    const HostWeight* w;
    const CaseInputs* in;
    Oracle mode;
    std::vector<uint8_t> act_codes;  // A8: [T][K]
    std::vector<float> act_scales;   // A8: [T]

    double dot(int r, int t) const {
        const int k = w->shape.k;
        double acc  = 0.0;
        if (mode == Oracle::Bf16) {
            const uint16_t* wr = w->bf16.data() + size_t(r) * k;
            const uint16_t* xr = in->x.data() + size_t(t) * k;
            for (int j = 0; j < k; ++j) acc += double(bf16_bits_to_float(wr[j])) * bf16_bits_to_float(xr[j]);
            return acc;
        }
        const uint8_t* wr = w->codes.data() + size_t(r) * k;
        const double scale = bf16_bits_to_float(w->scales[r]);
        if (mode == Oracle::Fp8) {
            const uint16_t* xr = in->x.data() + size_t(t) * k;
            for (int j = 0; j < k; ++j) acc += double(e4m3_to_float(wr[j])) * bf16_bits_to_float(xr[j]);
            return acc * scale;
        }
        const uint8_t* xr = act_codes.data() + size_t(t) * k;
        for (int j = 0; j < k; ++j) acc += double(e4m3_to_float(wr[j])) * e4m3_to_float(xr[j]);
        return acc * double(act_scales[t]) * scale;
    }
};

// Reference values for the sampled units: ref[t][i] for unit units[i].
std::vector<double> oracle_values(const OracleData& o, const std::vector<int>& units) {
    const Shape& s    = o.w->shape;
    const int tokens  = o.in->tokens;
    const size_t m    = units.size();
    std::vector<double> ref(size_t(tokens) * m);
    parallel_for(int64_t(m), [&](int64_t i) {
        const int u = units[size_t(i)];
        for (int t = 0; t < tokens; ++t) {
            double v = 0.0;
            switch (s.kind) {
            case LinearEpilogue::StoreBf16:
            case LinearEpilogue::StoreF32: v = o.dot(u, t); break;
            case LinearEpilogue::GrWrite: v = o.dot(u, t); break;  // per-stream scaling below
            case LinearEpilogue::SwiGlu: v = silu_d(o.dot(u, t)) * o.dot(u + s.n / 2, t); break;
            case LinearEpilogue::SharedScale: v = sigmoid_d(o.in->logits[size_t(t) * 513 + 512]) * o.dot(u, t); break;
            case LinearEpilogue::SiluQuarter: v = silu_d(o.dot(u, t) / 4.0); break;
            case LinearEpilogue::GrMix: {
                const int dims = s.n / 4;
                double sum     = 0.0;
                for (int st = 0; st < 4; ++st) {
                    sum += sigmoid_d(o.dot(4 * u + st, t)) *
                           bf16_bits_to_float(o.in->xn[size_t(t) * s.n + size_t(st) * dims + u]);
                }
                v = 0.25 * sum;
                break;
            }
            }
            ref[size_t(t) * m + size_t(i)] = v;
        }
    });
    return ref;
}

struct Outcome {
    double ratio = 0;   // max adjusted error / rms(ref)
    double match = 1;   // BF16 outputs: fraction equal to bf16_rne(ref)
    double rms_ratio = 0;  // rms(got - ref) / rms(ref)
    bool ok      = true;
};

// BF16 outputs must also round like the oracle almost everywhere (only near-ties may differ).
constexpr double kMinRoundingMatch = 0.99;

// Compares the GPU result with `ref` (GrWrite: ref is acc, compared per stream as the X delta).
Outcome compare_case(const Shape& s, const CaseInputs& in, const std::vector<int>& units,
                     const std::vector<double>& ref, const void* got_host, bool check_tail) {
    const int tokens = in.tokens;
    const size_t m   = units.size();
    double sq = 0, worst = 0;
    size_t count = 0;
    const bool bf16 = bf16_output(s.kind);
    std::vector<std::pair<double, double>> pairs;  // (got, ref)
    pairs.reserve(size_t(tokens) * m * (s.kind == LinearEpilogue::GrWrite ? 4 : 1));
    for (int t = 0; t < tokens; ++t) {
        for (size_t i = 0; i < m; ++i) {
            const int u = units[i];
            const double r = ref[size_t(t) * m + i];
            switch (s.kind) {
            case LinearEpilogue::StoreBf16:
                pairs.emplace_back(bf16_bits_to_float(static_cast<const uint16_t*>(got_host)[size_t(t) * s.n + u]), r);
                break;
            case LinearEpilogue::StoreF32:
            case LinearEpilogue::SharedScale:
                pairs.emplace_back(static_cast<const float*>(got_host)[size_t(t) * s.n + u], r);
                break;
            case LinearEpilogue::GrWrite:
                for (int st = 0; st < 4; ++st) {
                    const size_t idx = (size_t(t) * 4 + st) * s.n + u;
                    const double delta = double(static_cast<const float*>(got_host)[idx]) - in.x_stream[idx];
                    pairs.emplace_back(delta, double(in.inj[size_t(t) * 4 + st]) * r);
                }
                break;
            case LinearEpilogue::SwiGlu:
                pairs.emplace_back(bf16_bits_to_float(static_cast<const uint16_t*>(got_host)[size_t(t) * (s.n / 2) + u]), r);
                break;
            case LinearEpilogue::SiluQuarter:
                pairs.emplace_back(bf16_bits_to_float(static_cast<const uint16_t*>(got_host)[size_t(t) * 384 + u]), r);
                break;
            case LinearEpilogue::GrMix:
                pairs.emplace_back(bf16_bits_to_float(static_cast<const uint16_t*>(got_host)[size_t(t) * (s.n / 4) + u]), r);
                break;
            }
        }
    }
    for (const auto& [g, r] : pairs) {
        sq += r * r;
        ++count;
    }
    const double rms = std::sqrt(sq / double(std::max<size_t>(count, 1)));
    size_t matches = 0;
    double err_sq  = 0;
    for (const auto& [g, r] : pairs) {
        const double allowance = bf16 ? std::ldexp(std::fabs(r), -8) * 1.001 : 0.0;
        worst = std::max(worst, std::fabs(g - r) - allowance);
        err_sq += (g - r) * (g - r);
        if (bf16 && float(g) == round_bf16(float(r))) ++matches;
    }
    Outcome out;
    out.ratio     = (rms > 0 ? worst / rms : worst);
    out.rms_ratio = std::sqrt(err_sq / double(std::max<size_t>(count, 1))) / (rms > 0 ? rms : 1.0);
    out.match     = bf16 ? double(matches) / double(std::max<size_t>(count, 1)) : 1.0;
    out.ok        = std::isfinite(out.ratio) && out.ratio <= kTol && out.match >= kMinRoundingMatch;
    if (check_tail && s.kind == LinearEpilogue::SiluQuarter) {
        const auto* h = static_cast<const uint16_t*>(got_host);
        for (int t = 0; t < tokens; ++t)
            for (int c = 320; c < 384; ++c)
                if (h[size_t(t) * 384 + c] != 0x4049) out.ok = false;
    }
    return out;
}

// Device-side state of one case: inputs + output buffer, re-initialized per run.
struct DeviceCase {
    Dev<uint16_t> x;
    Dev<float> inj, logits;
    Dev<uint16_t> xn;
    Dev<uint8_t> out;
    size_t out_bytes = 0;

    DeviceCase(const Shape& s, const CaseInputs& in) : x(in.x) {
        const int t = in.tokens;
        switch (s.kind) {
        case LinearEpilogue::StoreBf16: out_bytes = size_t(t) * s.n * 2; break;
        case LinearEpilogue::StoreF32:
        case LinearEpilogue::SharedScale: out_bytes = size_t(t) * s.n * 4; break;
        case LinearEpilogue::GrWrite: out_bytes = size_t(t) * 4 * s.n * 4; break;
        case LinearEpilogue::SwiGlu: out_bytes = size_t(t) * (s.n / 2) * 2; break;
        case LinearEpilogue::SiluQuarter: out_bytes = size_t(t) * 384 * 2; break;
        case LinearEpilogue::GrMix: out_bytes = size_t(t) * (s.n / 4) * 2; break;
        }
        out = Dev<uint8_t>(out_bytes);
        if (!in.inj.empty()) inj = Dev<float>(in.inj);
        if (!in.logits.empty()) logits = Dev<float>(in.logits);
        if (!in.xn.empty()) xn = Dev<uint16_t>(in.xn);
    }

    void reset(const Shape& s, const CaseInputs& in) {
        if (s.kind == LinearEpilogue::GrWrite) {
            FNGINE_TEST_CUDA(cudaMemcpy(out.p, in.x_stream.data(), out_bytes, cudaMemcpyHostToDevice));
        } else if (s.kind == LinearEpilogue::SiluQuarter) {
            std::vector<uint16_t> fill(out_bytes / 2, 0x4049);
            FNGINE_TEST_CUDA(cudaMemcpy(out.p, fill.data(), out_bytes, cudaMemcpyHostToDevice));
        } else {
            FNGINE_TEST_CUDA(cudaMemset(out.p, 0xFF, out_bytes));  // NaN pattern: catches holes
        }
    }

    LinearEpilogueParams params(const Shape& s) {
        switch (s.kind) {
        case LinearEpilogue::StoreBf16: return linear_store_bf16(reinterpret_cast<__nv_bfloat16*>(out.p));
        case LinearEpilogue::StoreF32: return linear_store_f32(reinterpret_cast<float*>(out.p));
        case LinearEpilogue::GrWrite: return linear_gr_write(reinterpret_cast<float*>(out.p), inj.p);
        case LinearEpilogue::SwiGlu: return linear_swiglu(reinterpret_cast<__nv_bfloat16*>(out.p));
        case LinearEpilogue::SharedScale: return linear_shared_scale(reinterpret_cast<float*>(out.p), logits.p + 512, 513);
        case LinearEpilogue::SiluQuarter: return linear_silu_quarter(reinterpret_cast<__nv_bfloat16*>(out.p));
        case LinearEpilogue::GrMix:
            return linear_gr_mix(reinterpret_cast<__nv_bfloat16*>(out.p), reinterpret_cast<const __nv_bfloat16*>(xn.p));
        }
        return {};
    }

    std::vector<uint8_t> result() const { return out.download(); }
};

struct DeviceWeight {
    Dev<uint8_t> codes;
    Dev<uint16_t> scales;
    Dev<uint16_t> bf16;
};

int g_failures = 0;
int g_cases    = 0;

void record(const char* label, const Shape& s, int tokens, const char* path, const Outcome& o,
            const char* extra = "") {
    ++g_cases;
    if (!o.ok) ++g_failures;
    std::printf("%s %-5s N=%-6d K=%-5d %-11s T=%-3d %-6s ratio=%.3e match=%.4f%s\n",
                o.ok ? "PASS" : "FAIL", label, s.n, s.k, linear_epilogue_name(s.kind), tokens, path,
                o.ratio, o.match, extra);
}

// Runs `launch` twice from the same initial output state; returns the first result and whether
// the second is bitwise identical (every path is specified deterministic).
template <class Launch>
std::pair<std::vector<uint8_t>, bool> run_twice(DeviceCase& dc, const Shape& s,
                                                const CaseInputs& in, Launch&& launch) {
    dc.reset(s, in);
    launch();
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    auto first = dc.result();
    dc.reset(s, in);
    launch();
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const bool same = dc.result() == first;
    return {std::move(first), same};
}

Outcome with_determinism(Outcome o, bool identical) {
    o.ok = o.ok && identical;
    if (!identical) std::printf("FAIL repeated call differs bitwise\n");
    return o;
}

void run_fp8_cases(const Shape& s, const HostWeight& w, const DeviceWeight& dw, const char* label,
                   const int* token_cases, size_t token_case_count, uint64_t seed) {
    const Fp8RowWeight weight{dw.codes.p, reinterpret_cast<const __nv_bfloat16*>(dw.scales.p), s.n, s.k};
    const bool has_a8 = linear_fp8_has_a8(s.n, s.k);
    for (size_t ci = 0; ci < token_case_count; ++ci) {
        const int tokens = token_cases[ci];
        const CaseInputs in = make_inputs(s, tokens, seed + uint64_t(tokens));
        const auto units    = sample_units(s, tokens);
        DeviceCase dc(s, in);
        const size_t ws_bytes = linear_fp8_workspace_bytes(s.n, s.k, tokens);
        Dev<uint8_t> ws(ws_bytes);
        const auto launch = [&](Fp8LinearPath path) {
            return [&, path] {
                linear_fp8(reinterpret_cast<const __nv_bfloat16*>(dc.x.p), tokens, weight,
                           dc.params(s), ws.p, ws_bytes, nullptr, path);
            };
        };

        OracleData plain{&w, &in, Oracle::Fp8, {}, {}};
        const auto ref = oracle_values(plain, units);

        // A16.
        {
            const auto [got, same] = run_twice(dc, s, in, launch(Fp8LinearPath::A16));
            record(label, s, tokens, "A16",
                   with_determinism(compare_case(s, in, units, ref, got.data(), true), same));
        }
        if (!has_a8) continue;

        // A8 against the activation-quantized oracle; error against the plain oracle reported.
        OracleData quant{&w, &in, Oracle::Fp8A8, std::vector<uint8_t>(size_t(tokens) * s.k),
                         std::vector<float>(size_t(tokens))};
        for (int t = 0; t < tokens; ++t)
            quant.act_scales[size_t(t)] = quantize_token_e4m3(in.x.data() + size_t(t) * s.k, s.k,
                                                              quant.act_codes.data() + size_t(t) * s.k);
        const auto ref_q       = oracle_values(quant, units);
        const auto [got, same] = run_twice(dc, s, in, launch(Fp8LinearPath::A8));
        const Outcome q        = compare_case(s, in, units, ref_q, got.data(), true);
        const Outcome loose    = compare_case(s, in, units, ref, got.data(), false);
        char extra[96];
        std::snprintf(extra, sizeof(extra), "  (A8 vs unquantized oracle: max %.3e rms %.3e)",
                      loose.ratio, loose.rms_ratio);
        record(label, s, tokens, "A8", with_determinism(q, same), extra);
    }
}

void run_bf16_cases(const Shape& s, const HostWeight& w, const DeviceWeight& dw, CublasLtContext& ctx,
                    const char* label, const int* token_cases, size_t token_case_count, uint64_t seed) {
    const Bf16Weight weight{reinterpret_cast<const __nv_bfloat16*>(dw.bf16.p), s.n, s.k};
    for (size_t ci = 0; ci < token_case_count; ++ci) {
        const int tokens    = token_cases[ci];
        const CaseInputs in = make_inputs(s, tokens, seed + 17 * uint64_t(tokens));
        const auto units    = sample_units(s, tokens);
        DeviceCase dc(s, in);
        const size_t ws_bytes = linear_bf16_workspace_bytes(s.n, s.k, s.kind, tokens);
        Dev<uint8_t> ws(ws_bytes);
        OracleData plain{&w, &in, Oracle::Bf16, {}, {}};
        const auto ref         = oracle_values(plain, units);
        const auto [got, same] = run_twice(dc, s, in, [&] {
            linear_bf16(reinterpret_cast<const __nv_bfloat16*>(dc.x.p), tokens, weight,
                        dc.params(s), &ctx, ws.p, ws_bytes, nullptr);
        });
        record(label, s, tokens, tokens <= kLinearBf16GemvMaxTokens ? "gemv" : "cublas",
               with_determinism(compare_case(s, in, units, ref, got.data(), true), same));
    }
}

DeviceWeight upload_weight(const HostWeight& w, bool fp8, bool bf16) {
    DeviceWeight dw;
    if (fp8) {
        dw.codes  = Dev<uint8_t>(w.codes);
        dw.scales = Dev<uint16_t>(w.scales);
    }
    if (bf16) dw.bf16 = Dev<uint16_t>(w.bf16);
    return dw;
}

// Registry contract: unregistered shapes / mismatched epilogues are rejected.
void check_registry() {
    bool ok = true;
    const auto expect_throw = [&](auto&& fn, const char* what) {
        try {
            fn();
            std::printf("FAIL registry: %s did not throw\n", what);
            ok = false;
        } catch (const std::invalid_argument&) {
        }
    };
    alignas(256) static unsigned char dummy[1024];
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(dummy);
    expect_throw([&] { linear_fp8(x, 1, Fp8RowWeight{dummy, x, 4096, 2560}, linear_store_bf16(reinterpret_cast<__nv_bfloat16*>(dummy)), nullptr, 0, nullptr); },
                 "FP8 unregistered shape");
    expect_throw([&] { linear_fp8(x, 1, Fp8RowWeight{dummy, x, 16384, 2560}, linear_store_f32(reinterpret_cast<float*>(dummy)), nullptr, 0, nullptr); },
                 "FP8 wrong epilogue");
    expect_throw([&] { linear_fp8(x, 1, Fp8RowWeight{dummy, x, 248320, 2560}, linear_store_f32(reinterpret_cast<float*>(dummy)), nullptr, 0, nullptr, Fp8LinearPath::A8); },
                 "FP8 lm_head A8");
    expect_throw([&] { linear_bf16(x, 1, Bf16Weight{x, 513, 2560}, linear_store_bf16(reinterpret_cast<__nv_bfloat16*>(dummy)), nullptr, nullptr, 0, nullptr); },
                 "BF16 wrong epilogue");
    expect_throw([&] { linear_bf16(x, 1, Bf16Weight{x, 100, 2560}, linear_store_f32(reinterpret_cast<float*>(dummy)), nullptr, nullptr, 0, nullptr); },
                 "BF16 unregistered shape");
    ++g_cases;
    if (!ok) ++g_failures;
    std::printf("%s registry rejects unregistered (N,K,epilogue)\n", ok ? "PASS" : "FAIL");
}

// ------------------------------------------------------------------------------------------
// Real pack tensors.

std::vector<uint8_t> read_file_range(const std::string& path, uint64_t offset, uint64_t bytes) {
    std::vector<uint8_t> out(bytes);
    const int fd = ::open(path.c_str(), O_RDONLY);
    FNGINE_CHECK(fd >= 0);
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t r = ::pread(fd, out.data() + done, bytes - done, off_t(offset + done));
        FNGINE_CHECK(r > 0);
        done += uint64_t(r);
    }
    ::close(fd);
    return out;
}

void run_pack(const std::string& pack, CublasLtContext& ctx) {
    std::ifstream f(pack + "/manifest.json");
    FNGINE_CHECK(f.good());
    const auto manifest = nlohmann::json::parse(f);
    const struct { const char* name; LinearEpilogue kind; } tensors[] = {
        {"L0.gdn.qkvz", LinearEpilogue::StoreBf16},
        {"L0.attn_gr.down", LinearEpilogue::SiluQuarter},
        {"L0.attn_gr.up", LinearEpilogue::GrMix},
        {"L0.gdn.out", LinearEpilogue::GrWrite},
    };
    const int pack_tokens[] = {1, 9, 64};
    for (const auto& t : tensors) {
        const auto& fp8 = manifest["tensors"][t.name];
        const auto& b16 = manifest["tensors_bf16"][t.name];
        const Shape s{fp8["shape"][0].get<int>(), fp8["shape"][1].get<int>(), t.kind};
        HostWeight w{s, {}, {}, {}};
        w.codes = read_file_range(pack + "/dense.bin", fp8["offset"].get<uint64_t>(), fp8["bytes"].get<uint64_t>());
        const auto scale_bytes = read_file_range(pack + "/dense.bin", fp8["scale_offset"].get<uint64_t>(), fp8["scale_bytes"].get<uint64_t>());
        w.scales.resize(size_t(s.n));
        std::memcpy(w.scales.data(), scale_bytes.data(), size_t(s.n) * 2);
        const auto bf16_bytes = read_file_range(pack + "/dense_bf16.bin", b16["offset"].get<uint64_t>(), b16["bytes"].get<uint64_t>());
        w.bf16.resize(size_t(s.n) * s.k);
        std::memcpy(w.bf16.data(), bf16_bytes.data(), bf16_bytes.size());

        // The C++ port of the pack's FP8 rule reproduces the pack bit-exactly.
        std::vector<uint8_t> codes(w.codes.size());
        std::vector<uint16_t> scales(w.scales.size());
        parallel_for(s.n, [&](int64_t r) {
            quantize_fp8_rows(w.bf16.data() + r * s.k, 1, s.k, codes.data() + r * s.k, scales.data() + r);
        });
        const bool same = codes == w.codes && scales == w.scales;
        ++g_cases;
        if (!same) ++g_failures;
        std::printf("%s pack  %s: C++ FP8-row rule reproduces dense.bin from dense_bf16.bin\n",
                    same ? "PASS" : "FAIL", t.name);

        run_fp8_cases(s, w, upload_weight(w, true, false), "pack", pack_tokens,
                      std::size(pack_tokens), 0x7777);
        run_bf16_cases(s, w, upload_weight(w, false, true), ctx, "pack", pack_tokens,
                       std::size(pack_tokens), 0x8888);
    }
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    CublasLtContext ctx;
    check_registry();

    uint64_t seed = 1;
    for (const Shape& s : kFp8Shapes) {
        const HostWeight w   = make_weight(s, seed++);
        // One device copy at a time keeps lm_head under ~1.6 GiB of device memory.
        run_fp8_cases(s, w, upload_weight(w, true, false), "fp8", kTokenCases,
                      std::size(kTokenCases), seed * 1000);
        run_bf16_cases(s, w, upload_weight(w, false, true), ctx, "bf16", kTokenCases,
                       std::size(kTokenCases), seed * 2000);
    }
    for (const Shape& s : kBf16OnlyShapes) {
        const HostWeight w   = make_weight(s, seed++);
        const DeviceWeight dw = upload_weight(w, false, true);
        run_bf16_cases(s, w, dw, ctx, "bf16", kTokenCases, std::size(kTokenCases), seed * 2000);
    }

    if (const char* pack = std::getenv("FNGINE_TEST_PACK"); pack && *pack) {
        run_pack(pack, ctx);
    } else {
        std::printf("note: FNGINE_TEST_PACK not set; real pack tensors skipped\n");
    }

    std::printf("%s: %d/%d cases passed\n", g_failures ? "FAIL" : "PASS", g_cases - g_failures, g_cases);
    return g_failures ? 1 : 0;
}
