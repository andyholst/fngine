#pragma once

// FP64 oracles and comparison helpers for the QSA kernel tests.
#include "common/test_util.h"
#include "kernels/qsa/rope_freq.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace fngine::test::qsa_ref {

inline constexpr int kHeads = 24, kKvHeads = 2, kGroup = 12, kD = 256, kQkvLd = 13312;
inline constexpr int kIdxDim = 128, kIdxHeads = 4, kIdxLd = 640, kTop = 512;

// Spacing of bf16 values at |x| (normal range).
inline double ulp_bf16(double x) {
    x = std::fabs(x);
    if (x < 1e-30) return std::ldexp(1.0, -133);
    int e;
    std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
    return std::ldexp(1.0, e - 8);
}

inline float bits_to_float(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline void rope_cos_sin(int position, int frequency, double* c, double* s) {
    const float inv   = bits_to_float(fngine::kernels::kQsaRopeInvFreqBits[frequency]);
    const float angle = static_cast<float>(position) * inv;  // one float rounding, like HF
    *c                = std::cos(static_cast<double>(angle));
    *s                = std::sin(static_cast<double>(angle));
}

// RMSNorm (eps 1e-6, multiplier w) -> bf16 -> NeoX RoPE on dims [0, 64) at `position`.
// out = FP64 rotation of the oracle's bf16-rounded normalized values; tol = rounding budget of
// the kernel's two BF16 roundings (a normalized value may round the other way: 1 ulp of it,
// times its rotation coefficient; plus half an ulp for the final rounding).
struct RowRef {
    std::vector<double> out, tol;
};
inline RowRef norm_rope_row(const float* x, const float* w, int width, int position) {
    double sq = 0;
    for (int i = 0; i < width; ++i) sq += double(x[i]) * x[i];
    const double inv = 1.0 / std::sqrt(sq / width + 1e-6);
    std::vector<double> n(width);
    for (int i = 0; i < width; ++i) n[i] = round_bf16(static_cast<float>(x[i] * inv * w[i]));
    RowRef r;
    r.out.resize(width);
    r.tol.resize(width);
    for (int i = 0; i < width; ++i) {
        if (i < 64) {
            const int j = i & 31;
            double c, s;
            rope_cos_sin(position, j, &c, &s);
            const int partner = i < 32 ? i + 32 : i - 32;
            r.out[i] = i < 32 ? n[i] * c - n[partner] * s : n[i] * c + n[partner] * s;
            r.tol[i] = 0.5 * ulp_bf16(r.out[i]) + ulp_bf16(n[i]) * std::fabs(c) +
                       ulp_bf16(n[partner]) * std::fabs(s);
        } else {
            r.out[i] = n[i];
            r.tol[i] = ulp_bf16(n[i]);
        }
    }
    return r;
}

// Error beyond a per-element rounding budget: max(0, |got - ref| - tol) relative to rms(ref).
inline bool report_budget(const char* what, const std::vector<float>& got,
                          const std::vector<double>& ref, const std::vector<double>& tol,
                          double tol_rel_rms = 2e-3) {
    ErrorStats plain = compare(got.data(), ref.data(), ref.size());
    double worst     = 0;
    size_t worst_i   = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double excess = std::fabs(double(got[i]) - ref[i]) - tol[i];
        if (!(excess <= worst)) {
            worst   = excess;
            worst_i = i;
        }
    }
    const double ratio = plain.ref_rms > 0 ? worst / plain.ref_rms : worst;
    const bool ok      = ratio <= tol_rel_rms;
    std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e max_abs/rms=%.3e; beyond bf16 rounding "
                "budget=%.3e (tol %.1e) worst=%zu\n",
                ok ? "PASS" : "FAIL", what, plain.max_abs, plain.ref_rms, plain.max_rel_to_rms,
                ratio, tol_rel_rms, worst_i);
    return ok;
}

// BF16 output compared with an unrounded FP64 reference: the unavoidable output rounding
// (half an ulp of the stored value) is excluded; everything else counts.
inline bool report_bf16_output(const char* what, const std::vector<float>& got,
                               const std::vector<double>& ref, double tol_rel_rms = 2e-3) {
    std::vector<double> tol(ref.size());
    for (size_t i = 0; i < ref.size(); ++i) tol[i] = 0.5 * ulp_bf16(got[i]);
    return report_budget(what, got, ref, tol, tol_rel_rms);
}

// Normalized Sylvester H256 / 16 (symmetric, orthogonal) in FP64.
inline void hadamard256(const double* in, double* out) {
    for (int i = 0; i < 256; ++i) {
        double s = 0;
        for (int j = 0; j < 256; ++j) s += (__builtin_popcount(i & j) & 1) ? -in[j] : in[j];
        out[i] = s / 16.0;
    }
}

// E4M3FN round-to-nearest-even with saturation to +-448 (NaN not produced for finite input).
inline uint8_t e4m3_encode(double x) {
    const uint8_t sign = x < 0 ? 0x80 : 0;
    const double a     = std::fabs(x);
    int best           = 0;
    double best_err    = a;
    for (int code = 1; code <= 126; ++code) {
        const double v   = e4m3_to_float(static_cast<uint8_t>(code));
        const double err = std::fabs(v - a);
        if (err < best_err || (err == best_err && (code & 1) == 0 && (best & 1) == 1)) {
            best     = code;
            best_err = err;
        }
    }
    if (a >= 448.0) best = 126;
    return static_cast<uint8_t>(sign | best);
}

inline constexpr double kE2m1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
inline uint8_t e2m1_encode(double x) {
    const uint8_t sign = x < 0 ? 8 : 0;
    const double a     = std::min(std::fabs(x), 6.0);
    int best           = 0;
    double best_err    = a;
    for (int code = 1; code < 8; ++code) {
        const double err = std::fabs(kE2m1[code] - a);
        if (err < best_err || (err == best_err && (code & 1) == 0 && (best & 1) == 1)) {
            best     = code;
            best_err = err;
        }
    }
    return static_cast<uint8_t>(sign | best);
}
inline double e2m1_decode(uint8_t code) {
    return (code & 8 ? -1.0 : 1.0) * kE2m1[code & 7];
}

inline double half_to_double(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, 2);
    return static_cast<double>(__half2float(h));
}
inline uint16_t half_bits(double x) {
    const __half h = __float2half_rn(static_cast<float>(x));
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

// Host copy of a K8V4 cache layer (identity paging, layout as in qsa.h).
struct K8v4Host {
    int capacity = 0;
    std::vector<uint8_t> k, v, v_scale;
    std::vector<uint16_t> k_scale;
    size_t code_k(int pos, int head, int d) const {
        return (size_t(pos >> 6) * 2 + head) * 64 * 256 + size_t(pos & 63) * 256 + d;
    }
    size_t scale_k(int pos, int head) const { return (size_t(pos >> 6) * 2 + head) * 64 + (pos & 63); }
    size_t code_v(int pos, int head, int d) const {
        return (size_t(pos >> 6) * 2 + head) * 64 * 128 + size_t(pos & 63) * 128 + d / 2;
    }
    size_t scale_v(int pos, int head, int group) const {
        return (size_t(pos >> 6) * 2 + head) * 64 * 16 + size_t(pos & 63) * 16 + group;
    }
    // Stored (rotated) values.
    void rotated_k(int pos, int head, double* out) const {
        const double s = half_to_double(k_scale[scale_k(pos, head)]);
        for (int d = 0; d < 256; ++d) out[d] = e4m3_to_float(k[code_k(pos, head, d)]) * s;
    }
    void rotated_v(int pos, int head, double* out) const {
        for (int d = 0; d < 256; ++d) {
            const uint8_t byte = v[code_v(pos, head, d)];
            const uint8_t code = d & 1 ? byte >> 4 : byte & 15;
            out[d] = e2m1_decode(code) * e4m3_to_float(v_scale[scale_v(pos, head, d / 16)]);
        }
    }
    // Values in the original basis (H is its own inverse).
    void k_values(int pos, int head, double* out) const {
        double r[256];
        rotated_k(pos, head, r);
        hadamard256(r, out);
    }
    void v_values(int pos, int head, double* out) const {
        double r[256];
        rotated_v(pos, head, r);
        hadamard256(r, out);
    }
};

// Host copy of a K8V8 cache layer: K and V both E4M3 codes [pages][2][64][256] with an FP16
// scale per (token, head).
struct K8v8Host {
    int capacity = 0;
    std::vector<uint8_t> k, v;
    std::vector<uint16_t> k_scale, v_scale;
    size_t code(int pos, int head, int d) const {
        return (size_t(pos >> 6) * 2 + head) * 64 * 256 + size_t(pos & 63) * 256 + d;
    }
    size_t scale(int pos, int head) const { return (size_t(pos >> 6) * 2 + head) * 64 + (pos & 63); }
    // Planes laid out as qsa_kv_cache_view carves them from one allocation of `bytes`.
    void load(const uint8_t* bytes, int capacity_tokens) {
        capacity        = capacity_tokens;
        const size_t ht = size_t(capacity_tokens) * 2;
        k.assign(bytes, bytes + ht * 256);
        k_scale.resize(ht);
        std::memcpy(k_scale.data(), bytes + ht * 256, ht * 2);
        v.assign(bytes + ht * 258, bytes + ht * 514);
        v_scale.resize(ht);
        std::memcpy(v_scale.data(), bytes + ht * 514, ht * 2);
    }
    void rotated(const std::vector<uint8_t>& codes, const std::vector<uint16_t>& scales, int pos,
                 int head, double* out) const {
        const double s = half_to_double(scales[scale(pos, head)]);
        for (int d = 0; d < 256; ++d) out[d] = e4m3_to_float(codes[code(pos, head, d)]) * s;
    }
    void k_values(int pos, int head, double* out) const {
        double r[256];
        rotated(k, k_scale, pos, head, r);
        hadamard256(r, out);
    }
    void v_values(int pos, int head, double* out) const {
        double r[256];
        rotated(v, v_scale, pos, head, r);
        hadamard256(r, out);
    }
};

inline void parallel_for(int n, const std::function<void(int)>& fn) {
    const int threads = std::max(1, std::min<int>(n, int(std::thread::hardware_concurrency())));
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) {
        pool.emplace_back([&, w] {
            for (int i = w; i < n; i += threads) fn(i);
        });
    }
    for (auto& t : pool) t.join();
}

inline std::vector<__nv_bfloat16> random_bf16(Rng& rng, size_t n, double scale) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = float_to_bf16(static_cast<float>(rng.normal() * scale));
    return v;
}

inline std::vector<float> to_float(const std::vector<__nv_bfloat16>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = bf16_to_float(v[i]);
    return f;
}

}  // namespace fngine::test::qsa_ref
