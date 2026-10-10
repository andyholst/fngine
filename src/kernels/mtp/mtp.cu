// MTP glue kernels: whole-row RMSNorm to BF16 and the stream merge h + e.

#include "kernels/mtp/mtp.h"

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/common/math.h"

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

inline constexpr int kNormThreads = 256;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&out)[8]) {
    const uint4 packed          = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void load8(const float* p, float (&out)[8]) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w;
    out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// One CTA per row; deterministic reduction (fixed per-thread order, shuffle tree, warps in order).
template <class In>
__launch_bounds__(kNormThreads) __global__
    void rmsnorm_bf16_kernel(const In* x, std::int32_t n_cols, const float* mult,
                             __nv_bfloat16* out) {
    __shared__ float red[kNormThreads / 32];
    const In* row         = x + static_cast<std::int64_t>(blockIdx.x) * n_cols;
    __nv_bfloat16* orow   = out + static_cast<std::int64_t>(blockIdx.x) * n_cols;
    const int chunks      = n_cols / 8;
    float sum             = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += kNormThreads) {
        float v[8];
        load8(row + c * 8, v);
#pragma unroll
        for (int e = 0; e < 8; ++e) { sum += v[e] * v[e]; }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = sum; }
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kNormThreads / 32; ++w) { total += red[w]; }
    const float r = 1.0f / sqrtf(total / static_cast<float>(n_cols) + dims::kRmsEps);

    for (int c = threadIdx.x; c < chunks; c += kNormThreads) {
        float v[8];
        load8(row + c * 8, v);
        const float4 m0 = reinterpret_cast<const float4*>(mult + c * 8)[0];
        const float4 m1 = reinterpret_cast<const float4*>(mult + c * 8)[1];
        const float m[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { o[e] = __float2bfloat16_rn(v[e] * r * m[e]); }
        *reinterpret_cast<uint4*>(orow + c * 8) = *reinterpret_cast<const uint4*>(o);
    }
}

inline constexpr int kMergeThreads = 256;
inline constexpr int kMergeChunks  = dims::kStreamHidden / 8; // 1280 per token

__global__ void mtp_merge_streams_kernel(const __nv_bfloat16* h, const __nv_bfloat16* e,
                                         std::int32_t T, float* X) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::int64_t>(T) * kMergeChunks) { return; }
    const std::int64_t t = idx / kMergeChunks;
    const int ch0        = static_cast<int>(idx - t * kMergeChunks) * 8;
    const int d0         = ch0 % dims::kHidden;
    float hv[8], ev[8];
    load8(h + t * dims::kStreamHidden + ch0, hv);
    load8(e + t * dims::kHidden + d0, ev);
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { o[i] = __bfloat162float(__float2bfloat16_rn(hv[i] + ev[i])); }
    float4* dst = reinterpret_cast<float4*>(X + t * dims::kStreamHidden + ch0);
    dst[0]      = make_float4(o[0], o[1], o[2], o[3]);
    dst[1]      = make_float4(o[4], o[5], o[6], o[7]);
}

template <class In>
void rmsnorm_launch(const In* x, std::int32_t T, std::int32_t n_cols, const float* mult,
                    __nv_bfloat16* out, cudaStream_t stream) {
    if (T == 0) { return; }
    if (n_cols <= 0 || n_cols % 8 != 0) {
        throw std::invalid_argument("rmsnorm_bf16: n_cols must be a positive multiple of 8");
    }
    if (reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(out) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(mult) % 16 != 0) {
        throw std::invalid_argument("rmsnorm_bf16: x, mult and out must be 16-byte aligned");
    }
    rmsnorm_bf16_kernel<In><<<static_cast<unsigned int>(T), kNormThreads, 0, stream>>>(
        x, n_cols, mult, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rmsnorm_bf16(const __nv_bfloat16* x, std::int32_t T, std::int32_t n_cols, const float* mult,
                  __nv_bfloat16* out, cudaStream_t stream) {
    rmsnorm_launch(x, T, n_cols, mult, out, stream);
}

void rmsnorm_bf16(const float* x, std::int32_t T, std::int32_t n_cols, const float* mult,
                  __nv_bfloat16* out, cudaStream_t stream) {
    rmsnorm_launch(x, T, n_cols, mult, out, stream);
}

void mtp_merge_streams(const __nv_bfloat16* h, const __nv_bfloat16* e, std::int32_t T,
                       float* X_mtp, cudaStream_t stream) {
    if (T == 0) { return; }
    const std::int64_t items = static_cast<std::int64_t>(T) * kMergeChunks;
    mtp_merge_streams_kernel<<<static_cast<unsigned int>(div_up<std::int64_t>(items, kMergeThreads)),
                               kMergeThreads, 0, stream>>>(h, e, T, X_mtp);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
