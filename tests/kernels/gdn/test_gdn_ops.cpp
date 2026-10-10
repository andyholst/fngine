// GDN conv (decode, record, prefill), gating and gated output norm vs FP64 oracles.
#include "kernels/gdn/gdn_test_common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fngine;
using namespace fngine::kernels;
using namespace fngine::test;
using namespace fngine::test::gdn;

namespace {

constexpr double kTol = 2e-3;

double silu(double x) { return x / (1.0 + std::exp(-x)); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

bool same_bits(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}

// Oracle conv over T tokens: x rows (qkvz, ld 16384), history [3][10240], weight [10240][4].
std::vector<double> conv_reference(const std::vector<__nv_bfloat16>& qkvz,
                                   const std::vector<__nv_bfloat16>& history,
                                   const std::vector<float>& w, int T) {
    std::vector<double> out(size_t(T) * kConvChannels);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < kConvChannels; ++c) {
            double acc = 0;
            for (int k = 0; k < 4; ++k) {
                const int pos = t - 3 + k;
                const float x = pos < 0 ? bf16_to_float(history[size_t(pos + 3) * kConvChannels + c])
                                        : bf16_to_float(qkvz[size_t(pos) * kQkvzLd + c]);
                acc += double(w[size_t(c) * 4 + k]) * x;
            }
            out[size_t(t) * kConvChannels + c] = silu(acc);
        }
    return out;
}

// tail3(history ‖ x[0:T]).
std::vector<__nv_bfloat16> conv_tail(const std::vector<__nv_bfloat16>& qkvz,
                                     const std::vector<__nv_bfloat16>& history, int T) {
    std::vector<__nv_bfloat16> out(kConvLayerElems);
    for (int s = 0; s < 3; ++s)
        for (int c = 0; c < kConvChannels; ++c) {
            const int pos = T - 3 + s;
            out[size_t(s) * kConvChannels + c] =
                pos < 0 ? history[size_t(pos + 3) * kConvChannels + c] : qkvz[size_t(pos) * kQkvzLd + c];
        }
    return out;
}

struct ConvData {
    std::vector<__nv_bfloat16> qkvz, history;
    std::vector<float> w;
};

ConvData make_conv(int T, std::uint64_t seed) {
    Rng rng(seed);
    return {random_bf16(rng, size_t(T) * kQkvzLd, -2, 2), random_bf16(rng, kConvLayerElems, -2, 2),
            random_f32(rng, size_t(kConvChannels) * 4, -0.6, 0.6)};
}

bool test_conv_prefill(int T, bool in_place) {
    const ConvData d = make_conv(T, 10 + T);
    DeviceArray<__nv_bfloat16> qkvz(d.qkvz), hist(d.history), hist_out(kConvLayerElems);
    DeviceArray<float> w(d.w);
    DeviceArray<__nv_bfloat16> out(size_t(T) * kConvChannels);
    __nv_bfloat16* state_out = in_place ? hist.get() : hist_out.get();
    gdn_conv_prefill(qkvz.get(), w.get(), hist.get(), state_out, out.get(), T, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    char label[64];
    std::snprintf(label, sizeof label, "conv prefill T=%d%s", T, in_place ? " (in place)" : "");
    bool ok = report(label, compare_bf16(to_float(out.download()), conv_reference(d.qkvz, d.history, d.w, T)),
                     kTol);
    std::vector<__nv_bfloat16> state(kConvLayerElems);
    FNGINE_TEST_CUDA(cudaMemcpy(state.data(), state_out, state.size() * 2, cudaMemcpyDeviceToHost));
    const bool tail_ok = same_bits(state, conv_tail(d.qkvz, d.history, T));
    std::printf("%s %s history tail exact\n", tail_ok ? "PASS" : "FAIL", label);
    return ok && tail_ok;
}

// Decode (in place on slot 1 of a 2-slot pool, layer 7) and record (T=9 from slot 1, layer 30).
bool test_conv_decode_record() {
    bool ok           = true;
    const size_t bytes = gdn_state_pool_bytes(2);
    DeviceArray<unsigned char> mem(bytes);
    const GdnStatePool pool = gdn_state_pool_bind(mem.get(), bytes, 2);
    const size_t rbytes     = gdn_records_bytes();
    DeviceArray<unsigned char> rmem(rbytes);
    const GdnRecords records = gdn_records_bind(rmem.get(), rbytes);
    DeviceArray<GdnStepControl> control(std::vector<GdnStepControl>{{1, 0, 0, 0}});

    {
        const ConvData d = make_conv(1, 77);
        FNGINE_TEST_CUDA(cudaMemcpy(gdn_conv_state(pool, 1, 7), d.history.data(), kConvLayerElems * 2,
                                    cudaMemcpyHostToDevice));
        DeviceArray<__nv_bfloat16> qkvz(d.qkvz), out(kConvChannels);
        DeviceArray<float> w(d.w);
        gdn_conv_decode(qkvz.get(), w.get(), pool, 7, control.get(), out.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        ok &= report("conv decode T=1",
                     compare_bf16(to_float(out.download()), conv_reference(d.qkvz, d.history, d.w, 1)), kTol);
        std::vector<__nv_bfloat16> state(kConvLayerElems);
        FNGINE_TEST_CUDA(cudaMemcpy(state.data(), gdn_conv_state(pool, 1, 7), state.size() * 2,
                                    cudaMemcpyDeviceToHost));
        const bool shift_ok = same_bits(state, conv_tail(d.qkvz, d.history, 1));
        std::printf("%s conv decode history shift exact\n", shift_ok ? "PASS" : "FAIL");
        ok &= shift_ok;
    }
    {
        constexpr int T  = 9;
        const ConvData d = make_conv(T, 78);
        FNGINE_TEST_CUDA(cudaMemcpy(gdn_conv_state(pool, 1, 30), d.history.data(), kConvLayerElems * 2,
                                    cudaMemcpyHostToDevice));
        DeviceArray<__nv_bfloat16> qkvz(d.qkvz), out(size_t(T) * kConvChannels);
        DeviceArray<float> w(d.w);
        gdn_conv_record(qkvz.get(), w.get(), pool, 30, control.get(), records, T, out.get(), nullptr);
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        ok &= report("conv record T=9",
                     compare_bf16(to_float(out.download()), conv_reference(d.qkvz, d.history, d.w, T)), kTol);
        std::vector<__nv_bfloat16> state(kConvLayerElems), rec(size_t(T) * kConvChannels), want(rec.size());
        FNGINE_TEST_CUDA(cudaMemcpy(state.data(), gdn_conv_state(pool, 1, 30), state.size() * 2,
                                    cudaMemcpyDeviceToHost));
        FNGINE_TEST_CUDA(cudaMemcpy(rec.data(), records.conv + size_t(30) * kMaxRecordWidth * kConvChannels,
                                    rec.size() * 2, cudaMemcpyDeviceToHost));
        for (int t = 0; t < T; ++t)
            std::memcpy(&want[size_t(t) * kConvChannels], &d.qkvz[size_t(t) * kQkvzLd], kConvChannels * 2);
        const bool untouched = same_bits(state, d.history);
        const bool rec_ok    = same_bits(rec, want);
        std::printf("%s conv record leaves history untouched; %s conv record = raw inputs\n",
                    untouched ? "PASS" : "FAIL", rec_ok ? "PASS" : "FAIL");
        ok &= untouched && rec_ok;
    }
    return ok;
}

bool test_gating(int T) {
    Rng rng(31);
    const auto ab = random_f32(rng, size_t(T) * kAbLd, -30, 30);  // covers the softplus x > 20 branch
    const auto a_log = random_f32(rng, kValueHeads, -3, 2);
    const auto dt    = random_f32(rng, kValueHeads, -2, 2);
    DeviceArray<float> d_ab(ab), d_alog(a_log), d_dt(dt), g(size_t(T) * kValueHeads),
        beta(size_t(T) * kValueHeads);
    gdn_gating(d_ab.get(), d_alog.get(), d_dt.get(), g.get(), beta.get(), T, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<double> g_ref(size_t(T) * kValueHeads), b_ref(g_ref.size());
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < kValueHeads; ++h) {
            const double x = double(ab[size_t(t) * kAbLd + h]) + dt[h];
            g_ref[size_t(t) * kValueHeads + h] = -std::exp(double(a_log[h])) * std::log1p(std::exp(x));
            b_ref[size_t(t) * kValueHeads + h] = sigmoid(ab[size_t(t) * kAbLd + kValueHeads + h]);
        }
    const auto gh = g.download(), bh = beta.download();
    // Elementwise FP32 results: tolerance relative to each element's own magnitude.
    double worst_g = 0, worst_b = 0;
    for (size_t i = 0; i < g_ref.size(); ++i) {
        worst_g = std::max(worst_g, std::fabs(gh[i] - g_ref[i]) / std::max(std::fabs(g_ref[i]), 1e-30));
        worst_b = std::max(worst_b, std::fabs(bh[i] - b_ref[i]) / std::max(std::fabs(b_ref[i]), 1e-30));
    }
    bool ok = report("gating g", compare(gh.data(), g_ref.data(), gh.size()), kTol);
    ok &= report("gating beta", compare(bh.data(), b_ref.data(), bh.size()), kTol);
    const bool rel_ok = worst_g < 1e-5 && worst_b < 1e-5;
    std::printf("%s gating max elementwise relative error: g %.2e beta %.2e (tol 1e-5)\n",
                rel_ok ? "PASS" : "FAIL", worst_g, worst_b);
    return ok && rel_ok;
}

bool test_out_norm(int T) {
    Rng rng(41);
    std::vector<__nv_bfloat16> core(size_t(T) * kValueDim);
    for (int r = 0; r < T * kValueHeads; ++r) {  // per-head magnitudes from 1e-3 to 10
        const double scale = std::pow(10.0, rng.uniform(-3, 1));
        for (int d = 0; d < kHeadDim; ++d)
            core[size_t(r) * kHeadDim + d] = float_to_bf16(float(scale * rng.normal()));
    }
    const auto qkvz = random_bf16(rng, size_t(T) * kQkvzLd, -6, 6);
    const auto w    = random_f32(rng, kHeadDim, -1.5, 1.5);
    DeviceArray<__nv_bfloat16> d_core(core), d_qkvz(qkvz), out(size_t(T) * kValueDim);
    DeviceArray<float> d_w(w);
    gdn_out_norm(d_core.get(), d_qkvz.get() + kZOffset, kQkvzLd, d_w.get(), out.get(), T, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<double> ref(core.size()), extra(core.size());
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < kValueHeads; ++h) {
            const size_t base = size_t(t) * kValueDim + size_t(h) * kHeadDim;
            double ss         = 0;
            for (int d = 0; d < kHeadDim; ++d) ss += double(bf16_to_float(core[base + d])) * bf16_to_float(core[base + d]);
            const double inv = 1.0 / std::sqrt(ss / kHeadDim + 1e-6);
            for (int d = 0; d < kHeadDim; ++d) {
                const double n    = bf16_to_float(core[base + d]) * inv;
                const double gate = w[d] * sigmoid(bf16_to_float(qkvz[size_t(t) * kQkvzLd + kZOffset + h * kHeadDim + d]));
                ref[base + d]     = n * gate;
                extra[base + d]   = bf16_half_ulp(n) * std::fabs(gate);  // n is rounded to BF16
            }
        }
    return report("out_norm", compare_bf16(to_float(out.download()), ref, &extra), kTol);
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    bool ok = true;
    ok &= test_conv_prefill(1, false);
    ok &= test_conv_prefill(2, true);
    ok &= test_conv_prefill(37, false);
    ok &= test_conv_prefill(300, true);
    ok &= test_conv_decode_record();
    ok &= test_gating(37);
    ok &= test_out_norm(37);
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
