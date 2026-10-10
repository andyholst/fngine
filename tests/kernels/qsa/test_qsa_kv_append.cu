// qsa_kv_append round trip. BF16: exact copy. K8V4: every stored code/scale equals the FP64
// oracle's Hadamard + quantization (up to rounding-boundary ties of the FP32 transform), and the
// dequantized, inverse-rotated cache reproduces k/v within the codec's quantization error.
#include "kernels/qsa/qsa.h"
#include "kernels/qsa/qsa_ref.h"

#include <cstdio>
#include <vector>

using namespace fngine::test;
using namespace fngine::test::qsa_ref;
namespace fk = fngine::kernels;

struct Data {
    int tokens;
    std::vector<int32_t> positions;
    std::vector<__nv_bfloat16> k;    // [T][2][256]
    std::vector<__nv_bfloat16> qkv;  // [T][13312], v at cols 12800..
};

static Data make_data(int tokens, int first_position, uint64_t seed) {
    Rng rng(seed);
    Data d;
    d.tokens = tokens;
    d.positions.resize(tokens);
    for (int t = 0; t < tokens; ++t) d.positions[t] = first_position + t;
    d.k   = random_bf16(rng, size_t(tokens) * kKvHeads * kD, 1.5);
    d.qkv = random_bf16(rng, size_t(tokens) * kQkvLd, 3.0);
    // A few outlier rows/channels and an all-zero row exercise the scale paths.
    if (tokens > 2) {
        for (int d0 = 0; d0 < kD; ++d0) d.k[size_t(1) * kKvHeads * kD + d0] = float_to_bf16(0.0f);
        d.qkv[size_t(2) * kQkvLd + 12800 + 7] = float_to_bf16(300.0f);
    }
    return d;
}

static void append(const fk::QsaKvCache& cache, const Data& d) {
    DeviceArray<__nv_bfloat16> k(d.k), qkv(d.qkv);
    DeviceArray<int32_t> pos(d.positions);
    fk::qsa_kv_append(cache, k.get(), qkv.get() + 12800, kQkvLd, pos.get(), d.tokens, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
}

static float v_value(const Data& d, int t, int head, int dim) {
    return bf16_to_float(d.qkv[size_t(t) * kQkvLd + 12800 + head * kD + dim]);
}
static float k_value(const Data& d, int t, int head, int dim) {
    return bf16_to_float(d.k[(size_t(t) * kKvHeads + head) * kD + dim]);
}

static bool test_bf16(const std::vector<Data>& batches, int capacity) {
    const size_t bytes = fk::qsa_kv_layer_bytes(fk::QsaKvFormat::BF16, capacity);
    DeviceArray<uint8_t> mem(bytes);
    const auto cache = fk::qsa_kv_cache_view(fk::QsaKvFormat::BF16, mem.get(), capacity);
    for (const auto& b : batches) append(cache, b);
    std::vector<__nv_bfloat16> hk(size_t(cache.capacity_tokens) * kKvHeads * kD), hv(hk.size());
    FNGINE_TEST_CUDA(cudaMemcpy(hk.data(), cache.k, hk.size() * 2, cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(hv.data(), cache.v, hv.size() * 2, cudaMemcpyDeviceToHost));
    size_t mismatches = 0, checked = 0;
    for (const auto& b : batches) {
        for (int t = 0; t < b.tokens; ++t) {
            const int p = b.positions[t];
            for (int h = 0; h < kKvHeads; ++h) {
                for (int dim = 0; dim < kD; ++dim) {
                    const size_t idx = (size_t(p >> 6) * 2 + h) * 64 * 256 + size_t(p & 63) * 256 + dim;
                    mismatches += bf16_to_float(hk[idx]) != k_value(b, t, h, dim);
                    mismatches += bf16_to_float(hv[idx]) != v_value(b, t, h, dim);
                    checked += 2;
                }
            }
        }
    }
    const bool ok = mismatches == 0;
    std::printf("%s bf16 append: %zu/%zu mismatches (exact copy required), %zu B/token/layer\n",
                ok ? "PASS" : "FAIL", mismatches, checked, fk::qsa_kv_bytes_per_token(fk::QsaKvFormat::BF16));
    return ok;
}

// Checks one row-256 E4M3 codec row (K of K8V4/K8V8, V of K8V8) against the FP64 oracle: scale =
// fp16(absmax / 448) of the exactly rotated row, codes = E4M3 RNE of x / scale. Differences are
// allowed only where the kernel's FP32 transform sits on a rounding tie.
struct CodecStats {
    size_t code_checked = 0, code_mismatch = 0, code_unexplained = 0;
    size_t scale_checked = 0, scale_mismatch = 0, scale_unexplained = 0;
};

template <class CodeAt>
static void check_fp8_row(const double* rot, CodeAt code_at, uint16_t got_scale, CodecStats& st) {
    double absmax = 0;
    for (int d0 = 0; d0 < kD; ++d0) absmax = std::max(absmax, std::fabs(rot[d0]));
    ++st.scale_checked;
    if (absmax == 0) {
        if (got_scale != 0) ++st.scale_mismatch, ++st.scale_unexplained;
    } else {
        const double raw_scale = std::min(65504.0, std::max(std::ldexp(1.0, -24), absmax / 448.0));
        const uint16_t want = half_bits(raw_scale);
        if (want != got_scale) {
            ++st.scale_mismatch;
            const double a = half_to_double(want), g = half_to_double(got_scale);
            if (std::fabs(a - g) > 1.01 * std::fabs(a) * std::ldexp(1.0, -10) ||
                std::fabs(raw_scale - 0.5 * (a + g)) > 1e-6 * raw_scale) {
                ++st.scale_unexplained;
            }
        }
    }
    const double s = half_to_double(got_scale);
    for (int d0 = 0; d0 < kD; ++d0) {
        const uint8_t got  = code_at(d0);
        const uint8_t want = s == 0 ? 0 : e4m3_encode(rot[d0] / s);
        ++st.code_checked;
        if (got != want) {
            ++st.code_mismatch;
            const double x   = s == 0 ? 0 : rot[d0] / s;
            const double mid = 0.5 * (e4m3_to_float(got) + e4m3_to_float(want));
            if (std::fabs(std::fabs(x) - std::fabs(mid)) > 1e-5 * std::max(1.0, std::fabs(x))) ++st.code_unexplained;
        }
    }
}

static bool test_k8v4(const std::vector<Data>& batches, int capacity) {
    const size_t bytes = fk::qsa_kv_layer_bytes(fk::QsaKvFormat::K8V4, capacity);
    DeviceArray<uint8_t> mem(bytes);
    const auto cache = fk::qsa_kv_cache_view(fk::QsaKvFormat::K8V4, mem.get(), capacity);
    for (const auto& b : batches) append(cache, b);
    K8v4Host host;
    host.capacity = cache.capacity_tokens;
    const size_t ht = size_t(cache.capacity_tokens) * kKvHeads;
    host.k.resize(ht * 256);
    host.k_scale.resize(ht);
    host.v.resize(ht * 128);
    host.v_scale.resize(ht * 16);
    FNGINE_TEST_CUDA(cudaMemcpy(host.k.data(), cache.k, host.k.size(), cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.k_scale.data(), cache.k_scale, ht * 2, cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.v.data(), cache.v, host.v.size(), cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaMemcpy(host.v_scale.data(), cache.v_scale, host.v_scale.size(), cudaMemcpyDeviceToHost));

    CodecStats k_stats;
    size_t code_checked = 0, code_mismatch = 0, code_unexplained = 0;
    size_t scale_checked = 0, scale_mismatch = 0, scale_unexplained = 0;
    double k_err2 = 0, k_ref2 = 0, v_err2 = 0, v_ref2 = 0;
    for (const auto& b : batches) {
        for (int t = 0; t < b.tokens; ++t) {
            const int p = b.positions[t];
            for (int h = 0; h < kKvHeads; ++h) {
                double raw[256], rot[256], back[256];
                // ---- K: row E4M3 with FP16 scale.
                for (int d0 = 0; d0 < kD; ++d0) raw[d0] = k_value(b, t, h, d0);
                hadamard256(raw, rot);
                check_fp8_row(rot, [&](int d0) { return host.k[host.code_k(p, h, d0)]; },
                              host.k_scale[host.scale_k(p, h)], k_stats);
                host.k_values(p, h, back);
                for (int d0 = 0; d0 < kD; ++d0) {
                    k_err2 += (back[d0] - raw[d0]) * (back[d0] - raw[d0]);
                    k_ref2 += raw[d0] * raw[d0];
                }
                // ---- V: group-16 E2M1 with E4M3 scales.
                for (int d0 = 0; d0 < kD; ++d0) raw[d0] = v_value(b, t, h, d0);
                hadamard256(raw, rot);
                for (int g = 0; g < 16; ++g) {
                    double gmax = 0;
                    for (int i = 0; i < 16; ++i) gmax = std::max(gmax, std::fabs(rot[g * 16 + i]));
                    const uint8_t got_scale_v = host.v_scale[host.scale_v(p, h, g)];
                    uint8_t want_scale_v = 0;
                    if (gmax > 0) want_scale_v = e4m3_encode(std::min(448.0, std::max(std::ldexp(1.0, -9), gmax / 6.0)));
                    ++scale_checked;
                    if (got_scale_v != want_scale_v) {
                        ++scale_mismatch;
                        const double raw_s = gmax / 6.0;
                        const double mid = 0.5 * (e4m3_to_float(got_scale_v) + e4m3_to_float(want_scale_v));
                        if (std::fabs(raw_s - mid) > 1e-5 * raw_s) ++scale_unexplained;
                    }
                    const double vs = e4m3_to_float(got_scale_v);
                    for (int i = 0; i < 16; ++i) {
                        const int d0 = g * 16 + i;
                        const uint8_t byte = host.v[host.code_v(p, h, d0)];
                        const uint8_t got = d0 & 1 ? byte >> 4 : byte & 15;
                        const double x = vs == 0 ? 0 : rot[d0] / vs;
                        uint8_t want = vs == 0 ? 0 : e2m1_encode(x);
                        ++code_checked;
                        // +-0 are the same value.
                        if (got != want && !((got & 7) == 0 && (want & 7) == 0)) {
                            ++code_mismatch;
                            const double mid = 0.5 * (std::fabs(e2m1_decode(got)) + std::fabs(e2m1_decode(want)));
                            if (std::fabs(std::fabs(x) - mid) > 1e-5 * std::max(1.0, std::fabs(x))) ++code_unexplained;
                        }
                    }
                }
                host.v_values(p, h, back);
                for (int d0 = 0; d0 < kD; ++d0) {
                    v_err2 += (back[d0] - raw[d0]) * (back[d0] - raw[d0]);
                    v_ref2 += raw[d0] * raw[d0];
                }
            }
        }
    }
    const double k_rel = std::sqrt(k_err2 / k_ref2), v_rel = std::sqrt(v_err2 / v_ref2);
    code_checked += k_stats.code_checked;
    code_mismatch += k_stats.code_mismatch;
    code_unexplained += k_stats.code_unexplained;
    scale_checked += k_stats.scale_checked;
    scale_mismatch += k_stats.scale_mismatch;
    scale_unexplained += k_stats.scale_unexplained;
    const bool ok = code_unexplained == 0 && scale_unexplained == 0 &&
                    code_mismatch * 1000 <= code_checked && k_rel < 0.05 && v_rel < 0.15;
    std::printf("%s k8v4 append: codes %zu checked, %zu differ from FP64 oracle (%zu not at a "
                "rounding tie); scales %zu checked, %zu differ (%zu not at a tie); round-trip "
                "rms rel err K=%.4f V=%.4f; %zu B/token/layer\n",
                ok ? "PASS" : "FAIL", code_checked, code_mismatch, code_unexplained, scale_checked,
                scale_mismatch, scale_unexplained, k_rel, v_rel,
                fk::qsa_kv_bytes_per_token(fk::QsaKvFormat::K8V4));
    return ok;
}

// FP64 round trip of one rotated row through an E4M3 codec with an FP16 scale per `group`
// elements (group 256 = the K8V8 per-(token, head) scale); returns sum of squared errors.
static double fp8_group_roundtrip_err2(const double* rot, int group) {
    double err2 = 0;
    for (int g0 = 0; g0 < kD; g0 += group) {
        double absmax = 0;
        for (int i = 0; i < group; ++i) absmax = std::max(absmax, std::fabs(rot[g0 + i]));
        if (absmax == 0) continue;
        const double s = half_to_double(half_bits(std::min(65504.0, std::max(std::ldexp(1.0, -24), absmax / 448.0))));
        for (int i = 0; i < group; ++i) {
            const double q = e4m3_to_float(e4m3_encode(rot[g0 + i] / s)) * s;
            err2 += (q - rot[g0 + i]) * (q - rot[g0 + i]);
        }
    }
    return err2;
}

static bool test_k8v8(const std::vector<Data>& batches, int capacity) {
    const size_t bytes = fk::qsa_kv_layer_bytes(fk::QsaKvFormat::K8V8, capacity);
    DeviceArray<uint8_t> mem(bytes);
    const auto cache = fk::qsa_kv_cache_view(fk::QsaKvFormat::K8V8, mem.get(), capacity);
    for (const auto& b : batches) append(cache, b);
    const auto raw_bytes = mem.download();
    K8v8Host host;
    host.load(raw_bytes.data(), cache.capacity_tokens);
    CodecStats k_stats, v_stats;
    double k_err2 = 0, k_ref2 = 0, v_err2 = 0, v_ref2 = 0, v_group32_err2 = 0, v_row_err2 = 0;
    for (const auto& b : batches) {
        for (int t = 0; t < b.tokens; ++t) {
            const int p = b.positions[t];
            for (int h = 0; h < kKvHeads; ++h) {
                double raw[256], rot[256], back[256];
                for (int d0 = 0; d0 < kD; ++d0) raw[d0] = k_value(b, t, h, d0);
                hadamard256(raw, rot);
                check_fp8_row(rot, [&](int d0) { return host.k[host.code(p, h, d0)]; },
                              host.k_scale[host.scale(p, h)], k_stats);
                host.k_values(p, h, back);
                for (int d0 = 0; d0 < kD; ++d0) {
                    k_err2 += (back[d0] - raw[d0]) * (back[d0] - raw[d0]);
                    k_ref2 += raw[d0] * raw[d0];
                }
                for (int d0 = 0; d0 < kD; ++d0) raw[d0] = v_value(b, t, h, d0);
                hadamard256(raw, rot);
                check_fp8_row(rot, [&](int d0) { return host.v[host.code(p, h, d0)]; },
                              host.v_scale[host.scale(p, h)], v_stats);
                host.v_values(p, h, back);
                for (int d0 = 0; d0 < kD; ++d0) {
                    v_err2 += (back[d0] - raw[d0]) * (back[d0] - raw[d0]);
                    v_ref2 += raw[d0] * raw[d0];
                }
                // Scale-granularity comparison (H is orthonormal: rotated-domain error == value error).
                v_row_err2 += fp8_group_roundtrip_err2(rot, 256);
                v_group32_err2 += fp8_group_roundtrip_err2(rot, 32);
            }
        }
    }
    const double k_rel = std::sqrt(k_err2 / k_ref2), v_rel = std::sqrt(v_err2 / v_ref2);
    const size_t checked = k_stats.code_checked + v_stats.code_checked;
    const size_t mismatch = k_stats.code_mismatch + v_stats.code_mismatch;
    const bool ok = k_stats.code_unexplained + v_stats.code_unexplained == 0 &&
                    k_stats.scale_unexplained + v_stats.scale_unexplained == 0 &&
                    mismatch * 1000 <= checked && k_rel < 0.05 && v_rel < 0.05;
    std::printf("%s k8v8 append: codes %zu checked, %zu differ from FP64 oracle (%zu not at a "
                "rounding tie); scales %zu checked, %zu differ (%zu not at a tie); round-trip rms "
                "rel err K=%.4f V=%.4f; %zu B/token/layer\n",
                ok ? "PASS" : "FAIL", checked, mismatch,
                k_stats.code_unexplained + v_stats.code_unexplained,
                k_stats.scale_checked + v_stats.scale_checked,
                k_stats.scale_mismatch + v_stats.scale_mismatch,
                k_stats.scale_unexplained + v_stats.scale_unexplained, k_rel, v_rel,
                fk::qsa_kv_bytes_per_token(fk::QsaKvFormat::K8V8));
    std::printf("     V scale granularity (FP64 codec model): per-row FP16 scale rms rel err %.4f, "
                "per-32-group FP16 scales %.4f (+14 B/token/head)\n",
                std::sqrt(v_row_err2 / v_ref2), std::sqrt(v_group32_err2 / v_ref2));
    return ok;
}

int main() {
    if (skip_without_gpu()) return kSkip;
    const int capacity = 8192;
    // Prefill chunk crossing page boundaries, then decode/verify-sized appends.
    std::vector<Data> batches;
    batches.push_back(make_data(300, 4000, 11));
    batches.push_back(make_data(1, 4300, 12));
    batches.push_back(make_data(9, 4301, 13));
    batches.push_back(make_data(3, 63, 14));
    bool ok = true;
    ok &= test_bf16(batches, capacity);
    ok &= test_k8v4(batches, capacity);
    ok &= test_k8v8(batches, capacity);
    return ok ? 0 : 1;
}
