// GDN recurrence accuracy (decode T=1 steps, prefill: recurrent-only, chunked, chunked + tail)
// against the FP64 oracle from a nonzero initial state, plus timing of the decode and prefill paths.
#include "kernels/gdn/gdn_test_common.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace fngine;
using namespace fngine::kernels;
using namespace fngine::test;
using namespace fngine::test::gdn;

namespace {

constexpr double kTol = 2e-3;

struct Inputs {
    std::int64_t T = 0;
    std::vector<__nv_bfloat16> conv_out;  // [T][10240]
    std::vector<float> g, beta;           // [T][48]
    std::vector<float> state;             // [48][128][128]
};

Inputs make_inputs(std::int64_t T, std::uint64_t seed) {
    Rng rng(seed);
    Inputs in;
    in.T        = T;
    in.conv_out = random_bf16(rng, size_t(T) * kConvChannels, -1.0, 1.0);
    for (std::int64_t t = 0; t < T; ++t)  // v columns in [-0.5, 0.5)
        for (int i = 2 * kKeyDim; i < kConvChannels; ++i) {
            auto& x = in.conv_out[size_t(t) * kConvChannels + i];
            x       = float_to_bf16(0.5f * bf16_to_float(x));
        }
    in.g     = random_f32(rng, size_t(T) * kValueHeads, -0.10, -0.005);
    in.beta  = random_f32(rng, size_t(T) * kValueHeads, 0.05, 0.95);
    in.state = random_f32(rng, kRecurrentLayerElems, -0.1, 0.1);
    return in;
}

RefResult run_reference(const Inputs& in) {
    RefInputs r;
    split_qkv(to_float(in.conv_out), in.T, r);
    r.g     = in.g;
    r.beta  = in.beta;
    r.state = in.state;
    return reference(r);
}

bool check(const char* what, const std::vector<__nv_bfloat16>& core, const std::vector<float>& state,
           const RefResult& ref) {
    char label[128];
    std::snprintf(label, sizeof label, "%s output", what);
    bool ok = report(label, compare_bf16(to_float(core), ref.out), kTol);
    std::snprintf(label, sizeof label, "%s final state", what);
    ok &= report(label, compare(state.data(), ref.final_state.data(), state.size()), kTol);
    return ok;
}

// T sequential gdn_recurrent_decode steps on one layer of a 1-slot pool.
bool test_decode_steps(std::int64_t T, std::int32_t layer) {
    const Inputs in = make_inputs(T, 100 + T);
    const size_t pool_bytes = gdn_state_pool_bytes(1);
    DeviceArray<unsigned char> pool_mem(pool_bytes);
    const GdnStatePool pool = gdn_state_pool_bind(pool_mem.get(), pool_bytes, 1);
    FNGINE_TEST_CUDA(cudaMemcpy(gdn_recurrent_state(pool, 0, layer), in.state.data(),
                                in.state.size() * 4, cudaMemcpyHostToDevice));
    DeviceArray<GdnStepControl> control(std::vector<GdnStepControl>{{0, 0, 0, 0}});
    DeviceArray<__nv_bfloat16> conv_out(in.conv_out);
    DeviceArray<float> g(in.g), beta(in.beta);
    DeviceArray<__nv_bfloat16> core(size_t(T) * kValueDim);
    for (std::int64_t t = 0; t < T; ++t) {
        gdn_recurrent_decode(conv_out.get() + t * kConvChannels, g.get() + t * kValueHeads,
                             beta.get() + t * kValueHeads, pool, layer, control.get(),
                             core.get() + t * kValueDim, nullptr);
    }
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> state(kRecurrentLayerElems);
    FNGINE_TEST_CUDA(cudaMemcpy(state.data(), gdn_recurrent_state(pool, 0, layer),
                                state.size() * 4, cudaMemcpyDeviceToHost));
    char label[64];
    std::snprintf(label, sizeof label, "decode x%lld (layer %d)", (long long)T, layer);
    return check(label, core.download(), state, run_reference(in));
}

bool test_prefill(std::int64_t T, bool in_place) {
    const Inputs in = make_inputs(T, 200 + T);
    DeviceArray<__nv_bfloat16> conv_out(in.conv_out);
    DeviceArray<float> g(in.g), beta(in.beta), state_in(in.state), state_out(kRecurrentLayerElems);
    DeviceArray<__nv_bfloat16> core(size_t(T) * kValueDim);
    const size_t ws_bytes = gdn_prefill_workspace_bytes(std::int32_t(T));
    DeviceArray<unsigned char> ws(ws_bytes ? ws_bytes : 1);
    float* out_state = in_place ? state_in.get() : state_out.get();
    gdn_prefill(conv_out.get(), g.get(), beta.get(), state_in.get(), out_state, core.get(),
                std::int32_t(T), ws.get(), ws_bytes, nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> state(kRecurrentLayerElems);
    FNGINE_TEST_CUDA(cudaMemcpy(state.data(), out_state, state.size() * 4, cudaMemcpyDeviceToHost));
    char label[64];
    std::snprintf(label, sizeof label, "prefill T=%lld (%s, %s)", (long long)T,
                  T % 64 == 0 ? "chunked" : (T < 64 ? "recurrent" : "chunked+tail"),
                  in_place ? "in place" : "in->out");
    return check(label, core.download(), state, run_reference(in));
}

template <class F>
double time_ms(F&& f, int reps) {
    cudaEvent_t a, b;
    FNGINE_TEST_CUDA(cudaEventCreate(&a));
    FNGINE_TEST_CUDA(cudaEventCreate(&b));
    f();
    FNGINE_TEST_CUDA(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    FNGINE_TEST_CUDA(cudaEventRecord(b));
    FNGINE_TEST_CUDA(cudaEventSynchronize(b));
    float ms = 0;
    FNGINE_TEST_CUDA(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / reps;
}

void timing() {
    // Decode: one T=1 step for each of the 36 layers (distinct 3 MiB states), captured as a graph.
    {
        const size_t pool_bytes = gdn_state_pool_bytes(1);
        DeviceArray<unsigned char> pool_mem(pool_bytes);
        const GdnStatePool pool = gdn_state_pool_bind(pool_mem.get(), pool_bytes, 1);
        DeviceArray<GdnStepControl> control(std::vector<GdnStepControl>{{0, 0, 0, 0}});
        Rng rng(7);
        DeviceArray<__nv_bfloat16> conv_out(random_bf16(rng, kConvChannels, -1, 1));
        DeviceArray<float> g(random_f32(rng, kValueHeads, -0.1, -0.01));
        DeviceArray<float> beta(random_f32(rng, kValueHeads, 0.1, 0.9));
        DeviceArray<__nv_bfloat16> core(kValueDim);
        cudaStream_t s;
        FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        FNGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < kLayers; ++l)
            gdn_recurrent_decode(conv_out.get(), g.get(), beta.get(), pool, l, control.get(),
                                 core.get(), s);
        FNGINE_TEST_CUDA(cudaStreamEndCapture(s, &graph));
        FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        const double ms = time_ms([&] { FNGINE_TEST_CUDA(cudaGraphLaunch(exec, nullptr)); }, 50);
        std::printf("timing: recurrent decode T=1, 36 layers (graph): %.1f us total, %.2f us/layer, "
                    "%.0f GB/s state traffic\n",
                    ms * 1e3, ms * 1e3 / kLayers,
                    2.0 * kLayers * kRecurrentLayerElems * 4 / (ms * 1e-3) / 1e9);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
    }
    // Prefill: one layer, T = 1024 and 4096. Kernels longer than a few ms are time-sliced when
    // another process keeps the GPU busy, so the shorter run is the better per-token estimate.
    for (const std::int32_t T : {1024, 4096}) {
        Rng rng(9);
        DeviceArray<__nv_bfloat16> qkvz(random_bf16(rng, size_t(T) * kQkvzLd, -1, 1));
        DeviceArray<float> conv_w(random_f32(rng, size_t(kConvChannels) * 4, -0.5, 0.5));
        DeviceArray<float> ab(random_f32(rng, size_t(T) * kAbLd, -2, 2));
        DeviceArray<float> a_log(random_f32(rng, kValueHeads, -1, 1));
        DeviceArray<float> dt_bias(random_f32(rng, kValueHeads, -1, 1));
        DeviceArray<float> norm_w(random_f32(rng, kHeadDim, 0.5, 1.5));
        DeviceArray<__nv_bfloat16> conv_state(kConvLayerElems), conv_out(size_t(T) * kConvChannels);
        DeviceArray<__nv_bfloat16> core(size_t(T) * kValueDim), out(size_t(T) * kValueDim);
        DeviceArray<float> g(size_t(T) * kValueHeads), beta(size_t(T) * kValueHeads);
        DeviceArray<float> state(kRecurrentLayerElems);
        const size_t ws_bytes = gdn_prefill_workspace_bytes(T);
        DeviceArray<unsigned char> ws(ws_bytes);
        gdn_conv_prefill(qkvz.get(), conv_w.get(), conv_state.get(), conv_state.get(), conv_out.get(),
                         T, nullptr);
        gdn_gating(ab.get(), a_log.get(), dt_bias.get(), g.get(), beta.get(), T, nullptr);
        const double core_ms = time_ms(
            [&] {
                gdn_prefill(conv_out.get(), g.get(), beta.get(), state.get(), state.get(), core.get(),
                            T, ws.get(), ws_bytes, nullptr);
            },
            10);
        const double layer_ms = time_ms(
            [&] {
                gdn_conv_prefill(qkvz.get(), conv_w.get(), conv_state.get(), conv_state.get(),
                                 conv_out.get(), T, nullptr);
                gdn_gating(ab.get(), a_log.get(), dt_bias.get(), g.get(), beta.get(), T, nullptr);
                gdn_prefill(conv_out.get(), g.get(), beta.get(), state.get(), state.get(), core.get(),
                            T, ws.get(), ws_bytes, nullptr);
                gdn_out_norm(core.get(), qkvz.get() + kZOffset, kQkvzLd, norm_w.get(), out.get(), T,
                             nullptr);
            },
            10);
        std::printf("timing: chunked prefill T=%d, one layer: gdn_prefill %.3f ms (%.0f tok/s); "
                    "conv+gating+prefill+out_norm %.3f ms (%.0f tok/s); workspace %.1f MiB\n",
                    T, core_ms, T / (core_ms * 1e-3), layer_ms, T / (layer_ms * 1e-3),
                    ws_bytes / 1048576.0);
    }
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    bool ok = true;
    ok &= test_decode_steps(1, 0);
    ok &= test_decode_steps(6, 35);
    ok &= test_prefill(1, false);
    ok &= test_prefill(37, false);
    ok &= test_prefill(64, false);
    ok &= test_prefill(192, true);
    ok &= test_prefill(256 + 37, false);
    ok &= test_prefill(256 + 37, true);
    ok &= test_prefill(1024 + 17, false);  // 16 chunks: error growth over a longer sequence
    timing();
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
