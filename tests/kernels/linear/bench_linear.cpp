// linear_bench: times every registered FP8 shape on the A16 and A8 paths over a token sweep.
// Weights rotate over enough copies to defeat the 96 MiB L2 (>= 512 MiB per rotation), so the
// T = 1 numbers are DRAM-bound effective bandwidths (codes + scales bytes / time).
//
//   linear_bench [--tokens 1,2,4] [--shape N,K] [--bf16] [--l2]
// --l2 keeps one weight copy, so after warmup the weights are L2-resident (the decode case after
// an L2 prefetch of the next layer's dense weights).
// Prints one line per (shape, T, path): microseconds, effective GB/s and TFLOP/s.

#include "kernels/linear/linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace fngine::kernels;

namespace {

#define BENCH_CUDA(expr)                                                                    \
    do {                                                                                    \
        cudaError_t e_ = (expr);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(e_));          \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

struct Shape {
    int n, k;
    LinearEpilogue kind;
};

const Shape kShapes[] = {
    {16384, 2560, LinearEpilogue::StoreBf16},  {13312, 2560, LinearEpilogue::StoreBf16},
    {2560, 6144, LinearEpilogue::GrWrite},      {1280, 2560, LinearEpilogue::SwiGlu},
    {2560, 640, LinearEpilogue::SharedScale},   {320, 10240, LinearEpilogue::SiluQuarter},
    {10240, 384, LinearEpilogue::GrMix},        {10240, 2560, LinearEpilogue::StoreBf16},
    {2560, 2560, LinearEpilogue::StoreBf16},    {248320, 2560, LinearEpilogue::StoreF32},
    {513, 2560, LinearEpilogue::StoreF32},      {96, 2560, LinearEpilogue::StoreF32},
    {640, 2560, LinearEpilogue::StoreBf16},
};

void* dalloc(size_t bytes) {
    void* p = nullptr;
    BENCH_CUDA(cudaMalloc(&p, std::max<size_t>(bytes, 256)));
    BENCH_CUDA(cudaMemset(p, 0, std::max<size_t>(bytes, 256)));
    return p;
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        const size_t next = str.find(',', pos);
        out.push_back(std::atoi(str.substr(pos, next - pos).c_str()));
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> tokens_list = {1, 2, 3, 4, 6, 9, 16, 32, 64, 256, 1024, 4096};
    int only_n = 0, only_k = 0;
    bool bf16 = false, l2 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_list = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) {
            const auto v = parse_list(argv[++i]);
            only_n = v.at(0);
            only_k = v.at(1);
        } else if (!std::strcmp(argv[i], "--bf16")) bf16 = true;
        else if (!std::strcmp(argv[i], "--l2")) l2 = true;
    }
    const int max_tokens = *std::max_element(tokens_list.begin(), tokens_list.end());
    cudaStream_t stream;
    BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, stop;
    BENCH_CUDA(cudaEventCreate(&start));
    BENCH_CUDA(cudaEventCreate(&stop));
    CublasLtContext cublas;

    for (const Shape& s : kShapes) {
        if (only_n && (s.n != only_n || s.k != only_k)) continue;
        const bool fp8 = linear_fp8_registered(s.n, s.k, s.kind);
        if (!bf16 && !fp8) continue;
        const size_t weight_bytes = bf16 ? size_t(s.n) * s.k * 2 : size_t(s.n) * s.k + size_t(s.n) * 2;
        const int copies = l2 ? 1 : int(std::clamp<size_t>((size_t(512) << 20) / weight_bytes + 1, 1, 32));
        std::vector<void*> codes(copies), scales(copies);
        for (int c = 0; c < copies; ++c) {
            codes[c]  = dalloc(bf16 ? size_t(s.n) * s.k * 2 : size_t(s.n) * s.k);
            BENCH_CUDA(cudaMemset(codes[c], bf16 ? 0x3C : 0x38, bf16 ? size_t(s.n) * s.k * 2 : size_t(s.n) * s.k));
            scales[c] = dalloc(size_t(s.n) * 2);
        }
        void* x      = dalloc(size_t(max_tokens) * s.k * 2);
        const size_t out_per_token = s.kind == LinearEpilogue::GrWrite ? size_t(s.n) * 16
                                                                       : size_t(std::max(s.n, 384)) * 4;
        void* out = dalloc(size_t(max_tokens) * out_per_token);
        // inj [T][4] / gate logits [T][513] / Xn [T][N] (GrMix).
        const size_t aux_per_token =
            s.kind == LinearEpilogue::GrMix ? size_t(s.n) * 2 : size_t(513) * 4;
        void* aux = dalloc(size_t(max_tokens) * aux_per_token);
        size_t ws_bytes = 0;
        if (bf16) ws_bytes = linear_bf16_workspace_bytes(s.n, s.k, s.kind, max_tokens);
        else ws_bytes = linear_fp8_workspace_bytes(s.n, s.k, max_tokens);
        void* ws = dalloc(ws_bytes);

        for (int tokens : tokens_list) {
            LinearEpilogueParams ep{s.kind, out, static_cast<float*>(aux), static_cast<float*>(aux) + 512, 513,
                                    static_cast<__nv_bfloat16*>(aux)};
            const Fp8LinearPath paths[] = {Fp8LinearPath::A16, Fp8LinearPath::A8};
            for (Fp8LinearPath path : paths) {
                if (bf16 && path == Fp8LinearPath::A8) continue;
                if (!bf16 && path == Fp8LinearPath::A8 && !linear_fp8_has_a8(s.n, s.k)) continue;
                auto call = [&](int c) {
                    if (bf16) {
                        linear_bf16(static_cast<__nv_bfloat16*>(x), tokens,
                                    Bf16Weight{static_cast<__nv_bfloat16*>(codes[c]), s.n, s.k}, ep, &cublas,
                                    ws, ws_bytes, stream);
                    } else {
                        linear_fp8(static_cast<__nv_bfloat16*>(x), tokens,
                                   Fp8RowWeight{static_cast<uint8_t*>(codes[c]),
                                                static_cast<__nv_bfloat16*>(scales[c]), s.n, s.k},
                                   ep, ws, ws_bytes, stream, path);
                    }
                };
                for (int c = 0; c < copies; ++c) call(c);
                BENCH_CUDA(cudaStreamSynchronize(stream));
                const double flops_per_call = 2.0 * s.n * s.k * tokens;
                const int iters = int(std::clamp(2e11 / std::max(flops_per_call, 1.0) + copies, double(copies), 400.0));
                // Back-to-back average (throughput of a stream of calls).
                float loop_ms = 1e30f;
                for (int rep = 0; rep < 3; ++rep) {
                    BENCH_CUDA(cudaEventRecord(start, stream));
                    for (int i = 0; i < iters; ++i) call(i % copies);
                    BENCH_CUDA(cudaEventRecord(stop, stream));
                    BENCH_CUDA(cudaEventSynchronize(stop));
                    float ms = 0;
                    BENCH_CUDA(cudaEventElapsedTime(&ms, start, stop));
                    loop_ms = std::min(loop_ms, ms / float(iters));
                }
                // Minimum single-call time: filters GPU time slices taken by other processes.
                constexpr int kSamples = 40;
                std::vector<cudaEvent_t> marks(kSamples + 1);
                for (auto& e : marks) BENCH_CUDA(cudaEventCreate(&e));
                float call_ms = 1e30f;
                for (int i = 0; i < kSamples; ++i) {
                    BENCH_CUDA(cudaEventRecord(marks[i], stream));
                    call((iters + i) % copies);  // continue the rotation: least recently used copy
                    BENCH_CUDA(cudaEventRecord(marks[i + 1], stream));
                    BENCH_CUDA(cudaEventSynchronize(marks[i + 1]));
                    float ms = 0;
                    BENCH_CUDA(cudaEventElapsedTime(&ms, marks[i], marks[i + 1]));
                    call_ms = std::min(call_ms, ms);
                }
                for (auto& e : marks) cudaEventDestroy(e);
                const double us = call_ms * 1e3;
                std::printf("N=%-6d K=%-5d %-11s T=%-4d %-6s min %9.2f us %8.1f GB/s %7.1f TFLOP/s | loop %9.2f us\n",
                            s.n, s.k, linear_epilogue_name(s.kind), tokens,
                            bf16 ? "bf16" : (path == Fp8LinearPath::A16 ? "A16" : "A8"), us,
                            double(weight_bytes) / (us * 1e3), flops_per_call / (us * 1e6), loop_ms * 1e3);
                std::fflush(stdout);
            }
        }
        for (int c = 0; c < copies; ++c) {
            cudaFree(codes[c]);
            cudaFree(scales[c]);
        }
        cudaFree(x);
        cudaFree(out);
        cudaFree(aux);
        cudaFree(ws);
    }
    return 0;
}
