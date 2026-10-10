// kernels_mtp: whole-row RMSNorm to BF16 (2560 from BF16, 10240 from FP32) vs an FP64 oracle,
// and the stream merge X = bf16(h + e) (bit-exact).
#include "kernels/mtp/mtp.h"

#include "common/test_util.h"
#include "core/model_dims.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fngine;
using namespace fngine::test;
namespace k = fngine::kernels;

namespace {

bool g_ok = true;

double bf16_ulp(double v) {
    const double a = std::fabs(v);
    if (a < 1e-30) return 1e-38;
    int e;
    std::frexp(a, &e);
    return std::ldexp(1.0, e - 8);
}

// Every output must be a BF16 neighbour of the exact value, at most half an ulp away (plus a
// hair for FP32 evaluation noise at rounding midpoints).
template <class In>
void check_norm(const char* what, const std::vector<In>& x, int T, int n, const std::vector<float>& mult,
                const std::vector<__nv_bfloat16>& got, double (*to_double)(In)) {
    int bad = 0, inexact = 0;
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        double sq = 0;
        for (int i = 0; i < n; ++i) sq += to_double(x[t * n + i]) * to_double(x[t * n + i]);
        const double r = 1.0 / std::sqrt(sq / n + 1e-6);
        for (int i = 0; i < n; ++i) {
            const double ref  = to_double(x[t * n + i]) * r * mult[i];
            const double g    = bf16_to_float(got[t * n + i]);
            const double ulp  = bf16_ulp(std::max(std::fabs(ref), std::fabs(g)));
            const double frac = std::fabs(g - ref) / ulp;
            worst             = std::max(worst, frac);
            inexact += g != bf16_to_float(float_to_bf16(static_cast<float>(ref)));
            bad += frac > 0.5 + 1e-3;
        }
    }
    std::printf("%s %s: worst |err| = %.4f bf16 ulp, %d not nearest (tol: 0.5 ulp + 1e-3)\n",
                bad ? "FAIL" : "PASS", what, worst, inexact);
    g_ok = g_ok && bad == 0;
}

double from_bf16(__nv_bfloat16 v) { return bf16_to_float(v); }
double from_f32(float v) { return v; }

} // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    Rng rng(0x3a7cull);
    const int T = 5;

    // Embedding norm: BF16 [T][2560].
    {
        const int n = dims::kHidden;
        std::vector<__nv_bfloat16> x(static_cast<size_t>(T) * n);
        std::vector<float> mult(n);
        for (int t = 0; t < T; ++t) {
            const double scale = 0.01 + 3.0 * rng.uniform();
            for (int i = 0; i < n; ++i) x[t * n + i] = float_to_bf16(static_cast<float>(rng.normal() * scale));
        }
        for (auto& m : mult) m = static_cast<float>(1.0 + 0.3 * rng.normal());
        DeviceArray<__nv_bfloat16> dx(x), dout(x.size());
        DeviceArray<float> dm(mult);
        k::rmsnorm_bf16(dx.get(), T, n, dm.get(), dout.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        check_norm("rmsnorm_bf16 2560 (bf16 in)", x, T, n, mult, dout.download(), from_bf16);
    }
    // Hidden norm: one norm over the 10240 pre-mixer stream values per token.
    {
        const int n = dims::kStreamHidden;
        std::vector<float> x(static_cast<size_t>(T) * n);
        std::vector<float> mult(n);
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < 4; ++s) {
                const double scale = 0.1 + 5.0 * rng.uniform(); // streams differ in scale
                for (int d = 0; d < dims::kHidden; ++d)
                    x[t * n + s * dims::kHidden + d] = static_cast<float>(rng.normal() * scale);
            }
        for (auto& m : mult) m = static_cast<float>(1.0 + 0.3 * rng.normal());
        DeviceArray<float> dx(x), dm(mult);
        DeviceArray<__nv_bfloat16> dout(x.size());
        k::rmsnorm_bf16(dx.get(), T, n, dm.get(), dout.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        check_norm("rmsnorm_bf16 10240 (fp32 in)", x, T, n, mult, dout.download(), from_f32);
    }
    // Merge.
    {
        std::vector<__nv_bfloat16> h(static_cast<size_t>(T) * dims::kStreamHidden), e(static_cast<size_t>(T) * dims::kHidden);
        for (auto& v : h) v = float_to_bf16(static_cast<float>(rng.normal()));
        for (auto& v : e) v = float_to_bf16(static_cast<float>(rng.normal() * 0.5));
        DeviceArray<__nv_bfloat16> dh(h), de(e);
        DeviceArray<float> dx(h.size());
        k::mtp_merge_streams(dh.get(), de.get(), T, dx.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        const auto got = dx.download();
        int bad        = 0;
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < 4; ++s)
                for (int d = 0; d < dims::kHidden; ++d) {
                    const size_t i  = (static_cast<size_t>(t) * 4 + s) * dims::kHidden + d;
                    const float ref = round_bf16(bf16_to_float(h[i]) + bf16_to_float(e[t * dims::kHidden + d]));
                    bad += std::memcmp(&ref, &got[i], 4) != 0;
                }
        std::printf("%s mtp_merge_streams bit-exact (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);
        g_ok = g_ok && bad == 0;
    }
    std::printf("%s\n", g_ok ? "kernels_mtp: OK" : "kernels_mtp: FAILED");
    return g_ok ? 0 : 1;
}
