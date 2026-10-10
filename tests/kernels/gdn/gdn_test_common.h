// FP64 Gated DeltaNet oracle modified for fngine from cinference:tests/ops/gdn_ref.h @ 383e5db.
// See NOTICE. Plus shared helpers for the GDN kernel tests.
#pragma once

#include "common/test_util.h"
#include "kernels/gdn/gdn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace fngine::test::gdn {

using namespace fngine::kernels::gdn_dims;

// ---------------------------------------------------------------------------------------------
// FP64 recurrence oracle (one layer). q/k [T][16][128], v [T][48][128] hold the exact values of
// the BF16 inputs; g/beta [T][48]; state [48][128 v][128 k]. Q/K are L2-normalized in FP64
// (eps 1e-6) and the readout is scaled by 1/sqrt(128).
struct RefInputs {
    std::int64_t tokens = 0;
    std::vector<float> q, k, v, g, beta, state;
};

struct RefResult {
    std::vector<double> out;          // [T][48][128]
    std::vector<double> final_state;  // [48][128][128]
};

inline RefResult reference(const RefInputs& in) {
    constexpr std::int64_t S = kHeadDim, Hqk = kQkHeads, Hv = kValueHeads;
    const std::int64_t T     = in.tokens;
    if (in.q.size() != size_t(T * Hqk * S) || in.k.size() != in.q.size() ||
        in.v.size() != size_t(T * Hv * S) || in.g.size() != size_t(T * Hv) ||
        in.beta.size() != in.g.size() || in.state.size() != size_t(Hv * S * S)) {
        throw std::invalid_argument("gdn reference: input size does not match geometry");
    }
    std::vector<double> qn(in.q.size()), kn(in.k.size());
    for (std::int64_t r = 0; r < T * Hqk; ++r) {
        double qs = 0, ks = 0;
        for (std::int64_t d = 0; d < S; ++d) {
            const double qv = in.q[r * S + d], kv = in.k[r * S + d];
            qs += qv * qv;
            ks += kv * kv;
        }
        const double qi = 1.0 / std::sqrt(qs + 1e-6), ki = 1.0 / std::sqrt(ks + 1e-6);
        for (std::int64_t d = 0; d < S; ++d) {
            qn[r * S + d] = in.q[r * S + d] * qi;
            kn[r * S + d] = in.k[r * S + d] * ki;
        }
    }
    const double scale = 1.0 / std::sqrt(double(S));
    RefResult result;
    result.out.resize(in.v.size());
    result.final_state.resize(in.state.size());
    std::vector<double> state(S * S), delta(S);
    for (std::int64_t h = 0; h < Hv; ++h) {
        for (std::int64_t i = 0; i < S * S; ++i) state[i] = in.state[h * S * S + i];
        const std::int64_t qh = h / (Hv / Hqk);
        for (std::int64_t t = 0; t < T; ++t) {
            const double* kq    = &kn[(t * Hqk + qh) * S];
            const double* qq    = &qn[(t * Hqk + qh) * S];
            const double alpha  = std::exp(double(in.g[t * Hv + h]));
            const double beta   = in.beta[t * Hv + h];
            for (std::int64_t row = 0; row < S; ++row) {
                double dot = 0;
                for (std::int64_t col = 0; col < S; ++col) dot += state[row * S + col] * kq[col];
                delta[row] = beta * (double(in.v[(t * Hv + h) * S + row]) - alpha * dot);
            }
            for (std::int64_t row = 0; row < S; ++row)
                for (std::int64_t col = 0; col < S; ++col)
                    state[row * S + col] = alpha * state[row * S + col] + delta[row] * kq[col];
            for (std::int64_t row = 0; row < S; ++row) {
                double dot = 0;
                for (std::int64_t col = 0; col < S; ++col) dot += state[row * S + col] * qq[col];
                result.out[(t * Hv + h) * S + row] = scale * dot;
            }
        }
        for (std::int64_t i = 0; i < S * S; ++i) result.final_state[h * S * S + i] = state[i];
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// Comparison of BF16 outputs. A BF16 result cannot be closer to the FP64 reference than half an
// ulp of itself, so the tolerance applies to the error of the unrounded value:
// err_i = max(0, |got - ref| - halfulp_bf16(got) - extra_i).
inline double bf16_half_ulp(double v) {
    if (v == 0.0 || !std::isfinite(v)) return 0.0;
    int e = 0;
    std::frexp(std::fabs(v), &e);  // |v| in [2^(e-1), 2^e): bf16 ulp 2^(e-8)
    return std::ldexp(1.0, e - 9);
}

inline ErrorStats compare_bf16(const std::vector<float>& got, const std::vector<double>& ref,
                               const std::vector<double>* extra = nullptr) {
    ErrorStats s;
    double sq = 0;
    for (double r : ref) sq += r * r;
    s.ref_rms = std::sqrt(sq / double(ref.empty() ? 1 : ref.size()));
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs(double(got[i]) - ref[i]) - bf16_half_ulp(got[i]);
        if (extra) d -= (*extra)[i];
        if (d < 0) d = 0;
        if (!(d <= s.max_abs)) {
            s.max_abs = d;
            s.worst   = i;
        }
    }
    s.max_rel_to_rms = s.ref_rms > 0 ? s.max_abs / s.ref_rms : s.max_abs;
    return s;
}

inline std::vector<float> to_float(const std::vector<__nv_bfloat16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = bf16_to_float(v[i]);
    return out;
}

inline std::vector<__nv_bfloat16> random_bf16(Rng& rng, size_t n, double lo, double hi) {
    std::vector<__nv_bfloat16> out(n);
    for (auto& x : out) x = float_to_bf16(float(rng.uniform(lo, hi)));
    return out;
}

inline std::vector<float> random_f32(Rng& rng, size_t n, double lo, double hi) {
    std::vector<float> out(n);
    for (auto& x : out) x = float(rng.uniform(lo, hi));
    return out;
}

// Splits conv_out rows [T][10240] into the oracle's contiguous q/k/v.
inline void split_qkv(const std::vector<float>& conv_out, std::int64_t T, RefInputs& in) {
    in.tokens = T;
    in.q.assign(size_t(T) * kKeyDim, 0);
    in.k.assign(size_t(T) * kKeyDim, 0);
    in.v.assign(size_t(T) * kValueDim, 0);
    for (std::int64_t t = 0; t < T; ++t) {
        const float* row = &conv_out[size_t(t) * kConvChannels];
        for (int i = 0; i < kKeyDim; ++i) {
            in.q[t * kKeyDim + i] = row[i];
            in.k[t * kKeyDim + i] = row[kKeyDim + i];
        }
        for (int i = 0; i < kValueDim; ++i) in.v[t * kValueDim + i] = row[2 * kKeyDim + i];
    }
}

// Number of 32-bit words that differ between two device buffers (bitwise comparison, synchronizes
// the legacy stream). Defined in gdn_test_device.cu.
unsigned long long device_mismatches(const void* a, const void* b, size_t bytes);

}  // namespace fngine::test::gdn
