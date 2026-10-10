// kernels_gr: gr_norm (grouped RMSNorm of the four residual streams + fused inject GEMV) against
// an FP64 oracle at T in {1, 3, 9, 16, 64, 300}, with and without inject; gr_norm_down (the same
// norm fused with the FP8 GR down projection, decode widths T <= 16) against the same oracle.
//   Xn: max_i (|got - ref| - 2^-8 |ref|) <= 2e-3 * rms(ref)  (ref unrounded; BF16 output rounding)
//   inj: computed by the oracle from the kernel's own BF16 Xn (the contract's rounded input);
//        max |got - ref| <= 2e-3 * rms(ref).
//   r (gr_norm_down): bf16(silu(scale * sum_k e4m3(W[j][k]) * Xn[k] / 4)) from the kernel's own
//        Xn; max (|got - ref| - 2^-8 |ref|) <= 2e-3 * rms(ref); columns 320..383 untouched.
//   gr_norm_down batch invariance: token 0 of a T-token launch is bit-identical to a T=1 launch.

#include "kernels/gr/gr.h"
#include "kernels/linear/linear_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fngine;
using namespace fngine::test;
using namespace fngine::test::linear_ref;

namespace {

constexpr int kHidden = 2560;
constexpr int kWidth  = 4 * kHidden;
constexpr int kDown   = 320;
constexpr int kRLd    = 384;
constexpr double kTol = 2e-3;

template <class T>
T* dev_alloc(size_t n) {
    T* p = nullptr;
    FNGINE_TEST_CUDA(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

template <class T>
T* dev_copy(const std::vector<T>& v) {
    T* p = dev_alloc<T>(v.size());
    FNGINE_TEST_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

template <class T>
std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    FNGINE_TEST_CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

struct Inputs {
    std::vector<float> x, hc;
    std::vector<uint16_t> inj_w;
};

Inputs make_inputs(int tokens, Rng& rng) {
    Inputs in;
    const double stream_sigma[4] = {0.3, 2.0, 15.0, 120.0};
    in.x.resize(size_t(tokens) * kWidth);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < 4; ++s)
            for (int d = 0; d < kHidden; ++d)
                in.x[(size_t(t) * 4 + s) * kHidden + d] = float(rng.normal() * stream_sigma[(s + t) % 4]);
    in.hc.resize(kWidth);
    for (auto& v : in.hc) v = float(1.0 + 0.2 * rng.normal());
    in.inj_w.resize(size_t(4) * kWidth);
    for (auto& v : in.inj_w) v = bf16_bits(float_to_bf16(float(rng.normal() * 0.02)));
    return in;
}

// Xn / inj oracle checks; returns ok and prints the ratios into `msg`.
bool check_norm(const Inputs& in, int tokens, bool inject, const std::vector<uint16_t>& xn,
                const std::vector<float>& inj, char* msg, size_t msg_len) {
    double sq = 0, worst = 0;
    size_t matches = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < 4; ++s) {
            const float* row = in.x.data() + (size_t(t) * 4 + s) * kHidden;
            double ss = 0;
            for (int d = 0; d < kHidden; ++d) ss += double(row[d]) * row[d];
            const double rs = 1.0 / std::sqrt(ss / kHidden + 1e-6);
            for (int d = 0; d < kHidden; ++d) {
                const double ref = double(row[d]) * rs * in.hc[size_t(s) * kHidden + d];
                const double got = bf16_bits_to_float(xn[size_t(t) * kWidth + size_t(s) * kHidden + d]);
                sq += ref * ref;
                worst = std::max(worst, std::fabs(got - ref) - std::ldexp(std::fabs(ref), -8) * 1.001);
                if (float(got) == round_bf16(float(ref))) ++matches;
            }
        }
    }
    const double xn_ratio = worst / std::sqrt(sq / (double(tokens) * kWidth));
    const double xn_match = double(matches) / (double(tokens) * kWidth);
    bool ok               = std::isfinite(xn_ratio) && xn_ratio <= kTol && xn_match >= 0.99;

    double inj_ratio = 0;
    if (inject) {
        double isq = 0, iworst = 0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < 4; ++r) {
                double dot = 0;
                for (int j = 0; j < kWidth; ++j)
                    dot += double(bf16_bits_to_float(in.inj_w[size_t(r) * kWidth + j])) *
                           bf16_bits_to_float(xn[size_t(t) * kWidth + j]);
                const double ref = 2.0 / (1.0 + std::exp(-dot / 4.0));
                isq += ref * ref;
                iworst = std::max(iworst, std::fabs(double(inj[size_t(t) * 4 + r]) - ref));
            }
        }
        inj_ratio = iworst / std::sqrt(isq / (double(tokens) * 4));
        ok        = ok && std::isfinite(inj_ratio) && inj_ratio <= kTol;
    } else {
        for (float v : inj) ok = ok && v == 1234.5f;  // inj untouched without inject
    }
    std::snprintf(msg, msg_len, "Xn ratio=%.3e match=%.5f  inj ratio=%.3e", xn_ratio, xn_match, inj_ratio);
    return ok;
}

bool run_case(int tokens, bool inject, uint64_t seed) {
    Rng rng(seed);
    const Inputs in = make_inputs(tokens, rng);
    float* dx     = dev_copy(in.x);
    float* dhc    = dev_copy(in.hc);
    uint16_t* dw  = dev_copy(in.inj_w);
    uint16_t* dxn = dev_alloc<uint16_t>(size_t(tokens) * kWidth);
    float* dinj   = dev_copy(std::vector<float>(size_t(tokens) * 4, 1234.5f));

    kernels::gr_norm(dx, tokens, dhc, inject ? reinterpret_cast<const __nv_bfloat16*>(dw) : nullptr,
                     reinterpret_cast<__nv_bfloat16*>(dxn), inject ? dinj : nullptr, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto xn  = download(dxn, size_t(tokens) * kWidth);
    const auto inj = download(dinj, size_t(tokens) * 4);
    cudaFree(dx);
    cudaFree(dhc);
    cudaFree(dw);
    cudaFree(dxn);
    cudaFree(dinj);

    char msg[160];
    const bool ok = check_norm(in, tokens, inject, xn, inj, msg, sizeof msg);
    std::printf("%s gr_norm T=%-3d inject=%d  %s\n", ok ? "PASS" : "FAIL", tokens, int(inject), msg);
    return ok;
}

struct FusedOut {
    std::vector<uint16_t> xn, r;
    std::vector<float> inj;
};

FusedOut run_fused(const Inputs& in, int tokens, bool inject, const std::vector<uint8_t>& codes,
                   const std::vector<uint16_t>& scales) {
    float* dx     = dev_copy(std::vector<float>(in.x.begin(), in.x.begin() + size_t(tokens) * kWidth));
    float* dhc    = dev_copy(in.hc);
    uint16_t* dw  = dev_copy(in.inj_w);
    uint8_t* dc   = dev_copy(codes);
    uint16_t* ds  = dev_copy(scales);
    uint16_t* dxn = dev_alloc<uint16_t>(size_t(tokens) * kWidth);
    float* dinj   = dev_copy(std::vector<float>(size_t(tokens) * 4, 1234.5f));
    uint16_t* dr  = dev_copy(std::vector<uint16_t>(size_t(tokens) * kRLd, 0x7777));
    kernels::gr_norm_down(dx, tokens, dhc, inject ? reinterpret_cast<const __nv_bfloat16*>(dw) : nullptr,
                          dc, reinterpret_cast<const __nv_bfloat16*>(ds),
                          reinterpret_cast<__nv_bfloat16*>(dxn), inject ? dinj : nullptr,
                          reinterpret_cast<__nv_bfloat16*>(dr), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    FusedOut out{download(dxn, size_t(tokens) * kWidth), download(dr, size_t(tokens) * kRLd),
                 download(dinj, size_t(tokens) * 4)};
    cudaFree(dx);
    cudaFree(dhc);
    cudaFree(dw);
    cudaFree(dc);
    cudaFree(ds);
    cudaFree(dxn);
    cudaFree(dinj);
    cudaFree(dr);
    return out;
}

bool run_fused_case(int tokens, bool inject, uint64_t seed) {
    Rng rng(seed);
    const Inputs in = make_inputs(tokens, rng);
    std::vector<uint8_t> codes(size_t(kDown) * kWidth);
    for (auto& c : codes) c = e4m3_rne(float(rng.normal()));
    std::vector<uint16_t> scales(kDown);
    for (auto& v : scales) v = bf16_bits(float_to_bf16(float(0.004 * (1.0 + 0.5 * rng.uniform()))));

    const FusedOut out = run_fused(in, tokens, inject, codes, scales);
    char msg[160];
    bool ok = check_norm(in, tokens, inject, out.xn, out.inj, msg, sizeof msg);

    // r oracle from the kernel's own Xn.
    double sq = 0, worst = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < kDown; ++j) {
            double dot = 0;
            for (int k = 0; k < kWidth; ++k)
                dot += double(e4m3_to_float(codes[size_t(j) * kWidth + k])) *
                       bf16_bits_to_float(out.xn[size_t(t) * kWidth + k]);
            const double v   = dot * bf16_bits_to_float(scales[j]) / 4.0;
            const double ref = v / (1.0 + std::exp(-v));
            const double got = bf16_bits_to_float(out.r[size_t(t) * kRLd + j]);
            sq += ref * ref;
            worst = std::max(worst, std::fabs(got - ref) - std::ldexp(std::fabs(ref), -8) * 1.001);
        }
        for (int j = kDown; j < kRLd; ++j) ok = ok && out.r[size_t(t) * kRLd + j] == 0x7777;
    }
    const double r_ratio = worst / std::sqrt(sq / (double(tokens) * kDown));
    ok = ok && std::isfinite(r_ratio) && r_ratio <= kTol;

    // Batch invariance: token 0 alone reproduces token 0 of the T-token launch bit for bit.
    bool invariant = true;
    if (tokens > 1) {
        const FusedOut one = run_fused(in, 1, inject, codes, scales);
        invariant = std::memcmp(one.xn.data(), out.xn.data(), size_t(kWidth) * 2) == 0 &&
                    std::memcmp(one.r.data(), out.r.data(), size_t(kRLd) * 2) == 0 &&
                    std::memcmp(one.inj.data(), out.inj.data(), 4 * sizeof(float)) == 0;
        ok = ok && invariant;
    }
    std::printf("%s gr_norm_down T=%-3d inject=%d  %s  r ratio=%.3e  token-0 invariant=%d\n",
                ok ? "PASS" : "FAIL", tokens, int(inject), msg, r_ratio, int(invariant));
    return ok;
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    int failures = 0, cases = 0;
    const int token_cases[] = {1, 3, 9, 16, 64, 300};
    for (int tokens : token_cases) {
        for (bool inject : {true, false}) {
            ++cases;
            if (!run_case(tokens, inject, uint64_t(tokens) * 2 + inject)) ++failures;
        }
    }
    const int fused_cases[] = {1, 2, 3, 4, 7, 16};
    for (int tokens : fused_cases) {
        for (bool inject : {true, false}) {
            ++cases;
            if (!run_fused_case(tokens, inject, 1000 + uint64_t(tokens) * 2 + inject)) ++failures;
        }
    }
    std::printf("%s: %d/%d cases passed\n", failures ? "FAIL" : "PASS", cases - failures, cases);
    return failures ? 1 : 0;
}
