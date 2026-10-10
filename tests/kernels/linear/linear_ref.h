#pragma once

// Host references for the linear tests: the pack's FP8-row quantization rule (a C++ port of
// cinference tools/convert/quantization/fp8_row.py::quantize_bf16_rows), the A8 per-token
// activation quantization, and a small parallel-for.

#include "common/test_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace fngine::test::linear_ref {

inline uint16_t bf16_bits(__nv_bfloat16 v) {
    uint16_t b;
    std::memcpy(&b, &v, 2);
    return b;
}
inline __nv_bfloat16 bf16_from_bits(uint16_t b) {
    __nv_bfloat16 v;
    std::memcpy(&v, &b, 2);
    return v;
}
inline float bf16_bits_to_float(uint16_t b) { return bf16_to_float(bf16_from_bits(b)); }

// Exact BF16 word of a nonnegative binary32 value, ties to even (fp8_row.py::_bf16_rne_words).
inline uint16_t bf16_rne_word(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const uint64_t rounded = (uint64_t(bits) + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    return uint16_t(rounded);
}

// Positive E4M3FN values of codes 0..0x7E in increasing order.
inline const std::array<float, 0x7F>& e4m3_positive_values() {
    static const std::array<float, 0x7F> table = [] {
        std::array<float, 0x7F> t{};
        for (int c = 0; c < 0x7F; ++c) t[c] = e4m3_to_float(uint8_t(c));
        return t;
    }();
    return table;
}

// Round a finite binary32 value with |v| <= 448 to E4M3FN, ties to even
// (fp8_row.py::_round_e4m3fn_rne).
inline uint8_t e4m3_rne(float v) {
    const auto& table = e4m3_positive_values();
    const float magnitude = std::fabs(v);
    int upper = int(std::lower_bound(table.begin(), table.end(), magnitude) - table.begin());
    upper     = std::min(upper, 0x7E);
    const int lower = std::max(upper - 1, 0);
    const float lower_distance = magnitude - table[lower];
    const float upper_distance = table[upper] - magnitude;
    const bool choose_upper =
        (upper_distance < lower_distance) || (upper_distance == lower_distance && (upper & 1) == 0);
    uint8_t word = uint8_t(choose_upper ? upper : lower);
    if (std::signbit(v)) word |= 0x80;
    return word;
}

// E4M3 saturating RNE conversion of a binary32 value (cvt.rn.satfinite.e4m3x2.f32).
inline uint8_t e4m3_satfinite(float v) {
    const float clipped = std::min(448.0f, std::max(-448.0f, v));
    return e4m3_rne(clipped);
}

// Quantizes BF16 rows [n][k] to E4M3 codes and BF16 row scales with the pack rule.
inline void quantize_fp8_rows(const uint16_t* weight, int64_t n, int64_t k, uint8_t* codes,
                              uint16_t* scales) {
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t* row = weight + r * k;
        float max_abs       = 0.0f;
        for (int64_t j = 0; j < k; ++j) max_abs = std::max(max_abs, std::fabs(bf16_bits_to_float(row[j])));
        uint8_t* out = codes + r * k;
        if (max_abs == 0.0f) {
            std::memset(out, 0, size_t(k));
            scales[r] = 0;
            continue;
        }
        const float raw_scale = float(double(max_abs) / 448.0);
        uint16_t word         = bf16_rne_word(raw_scale);
        if (word == 0) word = 0x0001;
        scales[r]             = word;
        const float scale32   = bf16_bits_to_float(word);
        const float reciprocal = float(1.0 / double(scale32));
        for (int64_t j = 0; j < k; ++j) {
            float normalized = float(double(bf16_bits_to_float(row[j])) * double(reciprocal));
            normalized       = std::min(448.0f, std::max(-448.0f, normalized));
            out[j]           = e4m3_rne(normalized);
        }
    }
}

// A8 activation quantization of one BF16 token row (fp8_a8_quantize.cuh semantics).
inline float quantize_token_e4m3(const uint16_t* x, int64_t k, uint8_t* codes) {
    float maximum = 0.0f;
    for (int64_t j = 0; j < k; ++j) maximum = std::max(maximum, std::fabs(bf16_bits_to_float(x[j])));
    const float scale   = maximum > 0.0f ? maximum / 448.0f : 0.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int64_t j = 0; j < k; ++j) codes[j] = e4m3_satfinite(bf16_bits_to_float(x[j]) * inverse);
    return scale;
}

// Runs fn(i) for i in [0, count) on all hardware threads.
template <class Fn>
void parallel_for(int64_t count, Fn&& fn) {
    const int threads = int(std::max(1u, std::min(32u, std::thread::hardware_concurrency())));
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) {
        pool.emplace_back([&, w] {
            for (int64_t i = w; i < count; i += threads) fn(i);
        });
    }
    for (auto& t : pool) t.join();
}

}  // namespace fngine::test::linear_ref
