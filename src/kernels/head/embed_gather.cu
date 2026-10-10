// Token embedding gather from the (host-mapped) BF16 table into the 4 residual streams.

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/head/head.h"

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

inline constexpr int kEmbedChunks = dims::kHidden / 8; // 320 x 16-byte chunks per row

__launch_bounds__(kEmbedChunks) __global__
    void embed_gather_kernel(const uint4* table, const std::int32_t* tokens, float* X,
                             uint4* emb) {
    const int t     = static_cast<int>(blockIdx.x);
    const int chunk = static_cast<int>(threadIdx.x);
    const std::int64_t row = tokens[t];
    const uint4 packed     = table[row * kEmbedChunks + chunk];
    if (emb != nullptr) { emb[static_cast<std::int64_t>(t) * kEmbedChunks + chunk] = packed; }

    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
    float4 lo, hi;
    float2 f  = __bfloat1622float2(pairs[0]);
    lo.x      = f.x;
    lo.y      = f.y;
    f         = __bfloat1622float2(pairs[1]);
    lo.z      = f.x;
    lo.w      = f.y;
    f         = __bfloat1622float2(pairs[2]);
    hi.x      = f.x;
    hi.y      = f.y;
    f         = __bfloat1622float2(pairs[3]);
    hi.z      = f.x;
    hi.w      = f.y;
#pragma unroll
    for (int s = 0; s < dims::kStreams; ++s) {
        float4* dst = reinterpret_cast<float4*>(
            X + (static_cast<std::int64_t>(t) * dims::kStreams + s) * dims::kHidden + chunk * 8);
        dst[0] = lo;
        dst[1] = hi;
    }
}

} // namespace

void embed_gather(const __nv_bfloat16* table, const std::int32_t* tokens, std::int32_t T,
                  float* X, __nv_bfloat16* emb, cudaStream_t stream) {
    if (T == 0) { return; }
    if (reinterpret_cast<std::uintptr_t>(table) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(X) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(emb) % 16 != 0) {
        throw std::invalid_argument("embed_gather: table, X and emb must be 16-byte aligned");
    }
    embed_gather_kernel<<<static_cast<unsigned int>(T), kEmbedChunks, 0, stream>>>(
        reinterpret_cast<const uint4*>(table), tokens, X, reinterpret_cast<uint4*>(emb));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
