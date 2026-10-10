// Timing for the QSA kernels (not a ctest): decode T=1 attention at 2051 and 100K context,
// indexer score/top-k, the whole per-layer QSA path, and a 4096-token prefill chunk at 32K.
// Usage: qsa_bench [--iters N]
#include "kernels/qsa/qsa.h"
#include "kernels/qsa/qsa_ref.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

using namespace fngine::test;
using namespace fngine::test::qsa_ref;
namespace fk = fngine::kernels;

namespace {

int g_iters = 200;

// Average microseconds per call of `body`, replayed from a CUDA graph (graph=true) or launched
// eagerly.
double time_us(cudaStream_t stream, const std::function<void()>& body, bool graph) {
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        cudaGraph_t g;
        FNGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        body();
        FNGINE_TEST_CUDA(cudaStreamEndCapture(stream, &g));
        FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, g, 0));
        FNGINE_TEST_CUDA(cudaGraphDestroy(g));
    }
    auto run = [&] {
        if (graph) FNGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
        else body();
    };
    for (int i = 0; i < 10; ++i) run();
    cudaEvent_t a, b;
    FNGINE_TEST_CUDA(cudaEventCreate(&a));
    FNGINE_TEST_CUDA(cudaEventCreate(&b));
    FNGINE_TEST_CUDA(cudaEventRecord(a, stream));
    for (int i = 0; i < g_iters; ++i) run();
    FNGINE_TEST_CUDA(cudaEventRecord(b, stream));
    FNGINE_TEST_CUDA(cudaEventSynchronize(b));
    float ms = 0;
    FNGINE_TEST_CUDA(cudaEventElapsedTime(&ms, a, b));
    FNGINE_TEST_CUDA(cudaEventDestroy(a));
    FNGINE_TEST_CUDA(cudaEventDestroy(b));
    if (exec) FNGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
    return 1000.0 * ms / g_iters;
}

// One QSA layer's state and inputs at a given context capacity.
struct Layer {
    int capacity;
    fk::QsaKvCache cache;
    DeviceArray<uint8_t> kv;
    DeviceArray<__nv_bfloat16> idx_state;
    DeviceArray<float> qn, kn, iqn, ikn;
    Layer(fk::QsaKvFormat format, int cap, Rng& rng)
        : capacity(cap), kv(fk::qsa_kv_layer_bytes(format, cap)),
          idx_state(fk::indexer_state_bytes(cap) / 2), qn(std::vector<float>(256, 1.0f)),
          kn(std::vector<float>(256, 1.0f)), iqn(std::vector<float>(128, 1.0f)),
          ikn(std::vector<float>(128, 1.0f)) {
        cache = fk::qsa_kv_cache_view(format, kv.get(), cap);
        // Fill the whole cache and every block key with random data, 4096 positions at a time.
        const int chunk = 4096;
        auto k   = random_bf16(rng, size_t(chunk) * 512, 1.0);
        auto qkv = random_bf16(rng, size_t(chunk) * kQkvLd, 1.0);
        DeviceArray<__nv_bfloat16> dk(k), dqkv(qkv);
        auto keys = random_bf16(rng, size_t(cap / 4) * 128, 1.0);
        FNGINE_TEST_CUDA(cudaMemcpy(idx_state.get() + 16 * 128, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice));
        for (int p0 = 0; p0 < cap; p0 += chunk) {
            const int n = std::min(chunk, cap - p0);
            std::vector<int32_t> pos(n);
            std::iota(pos.begin(), pos.end(), p0);
            DeviceArray<int32_t> dpos(pos);
            fk::qsa_kv_append(cache, dk.get(), dqkv.get() + 12800, kQkvLd, dpos.get(), n, nullptr);
        }
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    }
    __nv_bfloat16* ring() const { return idx_state.get(); }
    __nv_bfloat16* blocks() const { return idx_state.get() + 16 * 128; }
};

// Step inputs and outputs for T tokens at positions p0..p0+T-1.
struct Step {
    int tokens, max_blocks;
    DeviceArray<__nv_bfloat16> qkv, idx, q, k, q_idx, out;
    DeviceArray<int32_t> pos, sel, cnt;
    DeviceArray<uint8_t> score_ws, attn_ws;
    size_t score_bytes, attn_bytes;
    Step(int T, int p0, int max_blocks_, Rng& rng)
        : tokens(T), max_blocks(max_blocks_), qkv(random_bf16(rng, size_t(T) * kQkvLd, 1.0)),
          idx(random_bf16(rng, size_t(T) * 640, 1.0)), q(size_t(T) * 24 * 256), k(size_t(T) * 512),
          q_idx(size_t(T) * 512), out(size_t(T) * 6144), pos(std::vector<int32_t>(T)),
          sel(size_t(T) * 512), cnt(T),
          score_ws(std::max<size_t>(16, fk::indexer_score_workspace_bytes(T, max_blocks_))),
          attn_ws(std::max<size_t>(16, fk::qsa_attention_workspace_bytes(T))),
          score_bytes(fk::indexer_score_workspace_bytes(T, max_blocks_)),
          attn_bytes(fk::qsa_attention_workspace_bytes(T)) {
        std::vector<int32_t> p(T);
        std::iota(p.begin(), p.end(), p0);
        pos.upload(p);
    }
    void norm_rope(const Layer& l, cudaStream_t s) {
        fk::qsa_qk_norm_rope(qkv.get(), l.qn.get(), l.kn.get(), pos.get(), tokens, q.get(), k.get(), s);
    }
    void append(const Layer& l, cudaStream_t s) {
        fk::qsa_kv_append(l.cache, k.get(), qkv.get() + 12800, kQkvLd, pos.get(), tokens, s);
    }
    void prepare(const Layer& l, cudaStream_t s) {
        fk::indexer_prepare(idx.get(), pos.get(), tokens, l.iqn.get(), l.ikn.get(), l.ring(), l.blocks(), q_idx.get(), s);
    }
    void topk(const Layer& l, cudaStream_t s) {
        fk::indexer_score_topk(q_idx.get(), l.blocks(), pos.get(), tokens, max_blocks, score_ws.get(),
                               score_bytes, sel.get(), cnt.get(), s);
    }
    void attention(const Layer& l, cudaStream_t s) {
        fk::qsa_attention(l.cache, q.get(), qkv.get(), pos.get(), sel.get(), cnt.get(), tokens, attn_ws.get(),
                          attn_bytes, out.get(), s);
    }
    void layer(const Layer& l, cudaStream_t s) {
        norm_rope(l, s);
        append(l, s);
        prepare(l, s);
        topk(l, s);
        attention(l, s);
    }
};

const char* name(fk::QsaKvFormat f) {
    return f == fk::QsaKvFormat::K8V4 ? "k8v4" : f == fk::QsaKvFormat::K8V8 ? "k8v8" : "bf16";
}

}  // namespace

int main(int argc, char** argv) {
    if (skip_without_gpu()) return kSkip;
    std::string only_format = "all", only_case = "all";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0) g_iters = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--format") == 0) only_format = argv[i + 1];  // k8v4|k8v8|bf16|all
        if (std::strcmp(argv[i], "--case") == 0) only_case = argv[i + 1];      // decode|verify|prefill|all
    }
    auto want_case = [&](const char* c) { return only_case == "all" || only_case == c; };
    cudaStream_t s;
    FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Rng rng(9);
    for (auto format : {fk::QsaKvFormat::K8V4, fk::QsaKvFormat::K8V8, fk::QsaKvFormat::BF16}) {
        if (only_format != "all" && only_format != name(format)) continue;
        // ---- Decode T=1.
        for (int ctx : {2051, 100000}) {
            if (!want_case("decode")) break;
            const int capacity = ((ctx + 63) / 64) * 64;
            Layer layer(format, capacity, rng);
            const int max_blocks = 131072 / 4;  // a graph captured for --ctx 131072
            Step step(1, ctx - 1, max_blocks, rng);
            step.norm_rope(layer, s);
            step.prepare(layer, s);
            step.topk(layer, s);  // selection for the attention-only timing
            FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
            const double attn_graph = time_us(s, [&] { step.attention(layer, s); }, true);
            const double attn_eager = time_us(s, [&] { step.attention(layer, s); }, false);
            const double topk_graph = time_us(s, [&] { step.topk(layer, s); }, true);
            const double layer_graph = time_us(s, [&] { step.layer(layer, s); }, true);
            std::printf("decode T=1 %s ctx=%6d: attention %.2f us (graph) / %.2f us (eager), "
                        "score+topk %.2f us, full QSA path (norm/rope, append, indexer, top-k, "
                        "attention) %.2f us\n",
                        name(format), ctx, attn_graph, attn_eager, topk_graph, layer_graph);
        }
        // ---- Verify widths at 100K.
        if (want_case("verify")) {
            const int ctx = 100000;
            Layer layer(format, ((ctx + 63) / 64) * 64, rng);
            for (int T : {3, 9}) {
                Step step(T, ctx - T, 131072 / 4, rng);
                step.norm_rope(layer, s);
                step.prepare(layer, s);
                step.topk(layer, s);
                const double attn = time_us(s, [&] { step.attention(layer, s); }, true);
                const double full = time_us(s, [&] { step.layer(layer, s); }, true);
                std::printf("verify T=%d %s ctx=%d: attention %.2f us, full QSA path %.2f us\n", T,
                            name(format), ctx, attn, full);
            }
        }
        // ---- Prefill: one 4096-token chunk ending at 32K.
        if (want_case("prefill")) {
            const int ctx = 32768, T = 4096;
            Layer layer(format, ctx, rng);
            Step step(T, ctx - T, ctx / 4, rng);
            step.norm_rope(layer, s);
            step.prepare(layer, s);
            step.topk(layer, s);
            const int saved = g_iters;
            g_iters = 10;
            const double attn = time_us(s, [&] { step.attention(layer, s); }, false);
            const double topk = time_us(s, [&] { step.topk(layer, s); }, false);
            const double full = time_us(s, [&] { step.layer(layer, s); }, false);
            g_iters = saved;
            std::printf("prefill T=%d %s positions %d..%d: attention %.3f ms (%.0f tok/s), "
                        "score+topk %.3f ms, full QSA path %.3f ms (%.0f tok/s)\n",
                        T, name(format), ctx - T, ctx - 1, attn / 1000, T / (attn * 1e-6),
                        topk / 1000, full / 1000, T / (full * 1e-6));
        }
    }
    FNGINE_TEST_CUDA(cudaStreamDestroy(s));
    return 0;
}
