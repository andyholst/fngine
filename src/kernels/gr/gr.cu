#include "kernels/gr/gr.h"

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"

#include <cooperative_groups.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

constexpr int kHidden  = dims::kHidden;   // 2560
constexpr int kStreams = dims::kStreams;  // 4
constexpr int kWidth   = kHidden * kStreams;
constexpr int kVector  = 8;                     // dims per thread per stream
constexpr int kThreads = kHidden / kVector;     // 320
constexpr int kWarps   = kThreads / 32;         // 10
static_assert(kHidden % (kVector * 32) == 0);

// Sums four per-thread values over the CTA; every thread receives the totals.
__device__ __forceinline__ float4 block_sum4(float4 v, float4* scratch) {
    v.x = warp_sum(v.x);
    v.y = warp_sum(v.y);
    v.z = warp_sum(v.z);
    v.w = warp_sum(v.w);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float4 total = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const float4 s = scratch[w];
        total.x += s.x;
        total.y += s.y;
        total.z += s.z;
        total.w += s.w;
    }
    __syncthreads();
    return total;
}

template <bool Inject>
__global__ __launch_bounds__(kThreads) void gr_norm_kernel(const float* __restrict__ x,
                                                           const float* __restrict__ hc_norm,
                                                           const __nv_bfloat16* __restrict__ inject,
                                                           __nv_bfloat16* __restrict__ xn,
                                                           float* __restrict__ inj) {
    __shared__ float4 scratch[kWarps];
    const int token = static_cast<int>(blockIdx.x);
    const int d0    = static_cast<int>(threadIdx.x) * kVector;
    const float* xt = x + static_cast<std::int64_t>(token) * kWidth;

    float v[kStreams][kVector];
    float ss[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const float4 a = load_vec<float4>(xt + s * kHidden + d0);
        const float4 b = load_vec<float4>(xt + s * kHidden + d0 + 4);
        v[s][0] = a.x, v[s][1] = a.y, v[s][2] = a.z, v[s][3] = a.w;
        v[s][4] = b.x, v[s][5] = b.y, v[s][6] = b.z, v[s][7] = b.w;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kVector; ++i) acc = fmaf(v[s][i], v[s][i], acc);
        ss[s] = acc;
    }
    const float4 sums = block_sum4(make_float4(ss[0], ss[1], ss[2], ss[3]), scratch);
    const float rs[kStreams] = {rsqrtf(sums.x / kHidden + dims::kRmsEps),
                                rsqrtf(sums.y / kHidden + dims::kRmsEps),
                                rsqrtf(sums.z / kHidden + dims::kRmsEps),
                                rsqrtf(sums.w / kHidden + dims::kRmsEps)};

    __nv_bfloat16* xnt = xn + static_cast<std::int64_t>(token) * kWidth;
    float rounded[kStreams][kVector];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const float4 ha = load_vec<float4>(hc_norm + s * kHidden + d0);
        const float4 hb = load_vec<float4>(hc_norm + s * kHidden + d0 + 4);
        const float h[kVector] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
        std::uint32_t packed[kVector / 2];
#pragma unroll
        for (int i = 0; i < kVector; i += 2) {
            const __nv_bfloat16 lo = __float2bfloat16_rn(v[s][i] * rs[s] * h[i]);
            const __nv_bfloat16 hi = __float2bfloat16_rn(v[s][i + 1] * rs[s] * h[i + 1]);
            rounded[s][i]          = __bfloat162float(lo);
            rounded[s][i + 1]      = __bfloat162float(hi);
            packed[i / 2] = static_cast<std::uint32_t>(__bfloat16_as_ushort(lo)) |
                            (static_cast<std::uint32_t>(__bfloat16_as_ushort(hi)) << 16);
        }
        store_vec(xnt + s * kHidden + d0, make_uint4(packed[0], packed[1], packed[2], packed[3]));
    }

    if constexpr (Inject) {
        float dot[kStreams] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kStreams; ++r) {
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const uint4 w = load_vec<uint4>(inject + static_cast<std::int64_t>(r) * kWidth +
                                                s * kHidden + d0);
                const float2 w0 = bf16x2_bits_to_float2(w.x);
                const float2 w1 = bf16x2_bits_to_float2(w.y);
                const float2 w2 = bf16x2_bits_to_float2(w.z);
                const float2 w3 = bf16x2_bits_to_float2(w.w);
                const float wf[kVector] = {w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
#pragma unroll
                for (int i = 0; i < kVector; ++i) dot[r] = fmaf(wf[i], rounded[s][i], dot[r]);
            }
        }
        const float4 total = block_sum4(make_float4(dot[0], dot[1], dot[2], dot[3]), scratch);
        if (threadIdx.x == 0) {
            store_vec(inj + static_cast<std::int64_t>(token) * kStreams,
                      make_float4(2.f * sigmoid(total.x * 0.25f), 2.f * sigmoid(total.y * 0.25f),
                                  2.f * sigmoid(total.z * 0.25f), 2.f * sigmoid(total.w * 0.25f)));
        }
    }
}

// ------------------------------------------------------------------ fused norm + GR down
namespace cg = cooperative_groups;

constexpr int kDownRows      = 320;
constexpr int kRLd           = 384;
constexpr int kFuseWarps     = 10;                  // K sub-slices of 256 inside a stream
constexpr int kFuseThreads   = kFuseWarps * 32;     // 320
constexpr int kFuseValues    = 8;                   // per lane: one 8-byte FP8 code pack
constexpr int kFuseSlice     = 32 * kFuseValues;    // 256 values per warp
constexpr int kFuseRows      = 8;                   // down rows per cluster
constexpr int kFuseClusters  = kDownRows / kFuseRows;  // 40
constexpr int kFuseChunk     = 4;                   // max tokens per register chunk
constexpr int kFuseMaxTokens = kGrNormDownMaxTokens;
static_assert(kFuseWarps * kFuseSlice == kHidden);

__device__ __forceinline__ uint2 load_codes_cg(const std::uint8_t* p) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// E4M3 x8 (little-endian bytes) -> FP32, exact.
__device__ __forceinline__ void decode_e4m3x8(uint2 q, float (&w)[kFuseValues]) {
    const std::uint32_t words[2] = {q.x, q.y};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = static_cast<__nv_fp8x2_storage_t>(words[i] & 0xffffu);
        hi.__x = static_cast<__nv_fp8x2_storage_t>(words[i] >> 16);
        const float2 a = static_cast<float2>(lo), b = static_cast<float2>(hi);
        w[4 * i] = a.x, w[4 * i + 1] = a.y, w[4 * i + 2] = b.x, w[4 * i + 3] = b.y;
    }
}

__device__ __forceinline__ void load_f32x8(const float* p, float (&v)[kFuseValues]) {
    const float4 a = load_vec<float4>(p), b = load_vec<float4>(p + 4);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w;
    v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}

// Lane dot of eight weights with eight packed BF16 activations.
__device__ __forceinline__ float dot8(const float (&w)[kFuseValues], const uint4& xb) {
    const std::uint32_t words[4] = {xb.x, xb.y, xb.z, xb.w};
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = bf16x2_bits_to_float2(words[i]);
        a = fmaf(w[2 * i], f.x, a);
        a = fmaf(w[2 * i + 1], f.y, a);
    }
    return a;
}

// Cluster rank s = stream s. Warp w / lane l own the eight values at s*2560 + w*256 + l*8 of every
// row; every lane keeps the eight rows' code packs in registers for all tokens.
template <bool Inject, int Chunk>
__global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(kFuseThreads)
gr_norm_down_kernel(const float* __restrict__ x, int tokens, const float* __restrict__ hc_norm,
                    const __nv_bfloat16* __restrict__ inject, const std::uint8_t* __restrict__ codes,
                    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ xn,
                    float* __restrict__ inj, __nv_bfloat16* __restrict__ r) {
    // Dispatch uses these specializations only at exactly Chunk tokens.
    if constexpr (Chunk < kFuseChunk) tokens = Chunk;
    cg::cluster_group cluster = cg::this_cluster();
    const int s      = static_cast<int>(cluster.block_rank());
    const int group  = static_cast<int>(blockIdx.x) / 4;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int d0     = warp * kFuseSlice + lane * kFuseValues;  // offset inside stream s
    const int k0     = s * kHidden + d0;                         // offset inside a 10240 row
    const bool owner = group == 0;  // cluster 0 also writes Xn and reduces inj
    constexpr int kRowsAll = kFuseRows + 4;
    // The one/two-token specializations never need the full 16-token scratch.
    constexpr int kScratchTokens = Chunk < kFuseChunk ? Chunk : kFuseMaxTokens;

    __shared__ float s_ss[kFuseWarps][kScratchTokens];
    __shared__ float s_part[kFuseWarps][kRowsAll][Chunk];
    __shared__ float s_cta[4][kRowsAll][kScratchTokens];  // rank 0 only: stream partials

    // Every CTA writes its partials into rank 0's shared memory, which therefore must be running:
    // arrive now, wait right before the first remote store (normally long satisfied by then).
    pdl_trigger();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

    // Weights first: they do not depend on X (and are requested before pdl_wait()), as are the
    // inject rows (cluster 0) and the row scales of the epilogue (rank 0, thread i < 8 * tokens
    // finishes row i / tokens).
    uint2 wq[kFuseRows];
#pragma unroll
    for (int j = 0; j < kFuseRows; ++j)
        wq[j] = load_codes_cg(codes + static_cast<std::int64_t>(group * kFuseRows + j) * kWidth + k0);
    float h[kFuseValues];
    load_f32x8(hc_norm + k0, h);
    uint4 iw[4] = {};
    if (Inject && owner) {
#pragma unroll
        for (int q = 0; q < 4; ++q) iw[q] = load_vec<uint4>(inject + static_cast<std::int64_t>(q) * kWidth + k0);
    }
    static_assert(kFuseRows * kFuseMaxTokens <= kFuseThreads, "one epilogue item per thread");
    const float row_scale = __bfloat162float(
        scales[group * kFuseRows + min(static_cast<int>(threadIdx.x) / tokens, kFuseRows - 1)]);
    pdl_wait();

    // Pass 1: per-token sum of squares of stream s (lane chain, warp butterfly, warps in order).
    // The first chunk's activations stay in registers for pass 2.
    float xv[Chunk][kFuseValues];
    const float* const xs = pdl_coherent(x);
    auto x_at = [&](int t) {
        return xs + (static_cast<std::int64_t>(t) * kStreams + s) * kHidden + d0;
    };
    auto sum_squares = [&](int t, const float (&v)[kFuseValues]) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < kFuseValues; ++i) ss = fmaf(v[i], v[i], ss);
        ss = warp_sum(ss);
        if (lane == 0) s_ss[warp][t] = ss;
    };
#pragma unroll
    for (int u = 0; u < Chunk; ++u) {
        if (u < tokens) load_f32x8(x_at(u), xv[u]);
    }
#pragma unroll
    for (int u = 0; u < Chunk; ++u) {
        if (u < tokens) sum_squares(u, xv[u]);
    }
    for (int t = Chunk; t < tokens; ++t) {
        float v[kFuseValues];
        load_f32x8(x_at(t), v);
        sum_squares(t, v);
    }
    __syncthreads();

    // Pass 2 in chunks of up to four tokens: BF16-rounded Xn (packed), eight row partials, plus
    // the four inject rows in cluster 0.
    for (int t0 = 0; t0 < tokens; t0 += Chunk) {
        const int nt = min(Chunk, tokens - t0);
        uint4 xb[Chunk];
#pragma unroll
        for (int u = 0; u < Chunk; ++u) {
            xb[u] = make_uint4(0, 0, 0, 0);
            if (u >= nt) continue;
            const int t = t0 + u;
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < kFuseWarps; ++w) sum += s_ss[w][t];
            const float rs = rsqrtf(sum / kHidden + dims::kRmsEps);
            float v[kFuseValues];
            if (t0 == 0) {
#pragma unroll
                for (int i = 0; i < kFuseValues; ++i) v[i] = xv[u][i];
            } else {
                load_f32x8(x_at(t), v);
            }
            std::uint32_t packed[kFuseValues / 2];
#pragma unroll
            for (int i = 0; i < kFuseValues; i += 2) {
                const __nv_bfloat16 lo = __float2bfloat16_rn(v[i] * rs * h[i]);
                const __nv_bfloat16 hi = __float2bfloat16_rn(v[i + 1] * rs * h[i + 1]);
                packed[i / 2] = static_cast<std::uint32_t>(__bfloat16_as_ushort(lo)) |
                                (static_cast<std::uint32_t>(__bfloat16_as_ushort(hi)) << 16);
            }
            xb[u] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            if (owner) store_vec(xn + static_cast<std::int64_t>(t) * kWidth + k0, xb[u]);
        }
#pragma unroll
        for (int j = 0; j < kFuseRows; ++j) {
            float w[kFuseValues];
            decode_e4m3x8(wq[j], w);
            float a[Chunk];
#pragma unroll
            for (int u = 0; u < Chunk; ++u) a[u] = dot8(w, xb[u]);
#pragma unroll
            for (int u = 0; u < Chunk; ++u) a[u] = warp_sum(a[u]);
            if (lane == 0) {
#pragma unroll
                for (int u = 0; u < Chunk; ++u) s_part[warp][j][u] = a[u];
            }
        }
        if (Inject && owner) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 iwq = iw[q];
                const std::uint32_t words[4] = {iwq.x, iwq.y, iwq.z, iwq.w};
                float w[kFuseValues];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = bf16x2_bits_to_float2(words[i]);
                    w[2 * i] = f.x, w[2 * i + 1] = f.y;
                }
                float a[Chunk];
#pragma unroll
                for (int u = 0; u < Chunk; ++u) a[u] = warp_sum(dot8(w, xb[u]));
                if (lane == 0) {
#pragma unroll
                    for (int u = 0; u < Chunk; ++u) s_part[warp][kFuseRows + q][u] = a[u];
                }
            }
        }
        __syncthreads();
        const int rows_here = (Inject && owner) ? kRowsAll : kFuseRows;
        // First chunk: complete the early arrive (all threads, uniform) before remote stores.
        if (t0 == 0) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
        if (static_cast<int>(threadIdx.x) < rows_here * Chunk) {
            const int j = static_cast<int>(threadIdx.x) / Chunk;
            const int u = static_cast<int>(threadIdx.x) % Chunk;
            if (u < nt) {
                float v = 0.f;
#pragma unroll
                for (int w = 0; w < kFuseWarps; ++w) v += s_part[w][j][u];
                *cluster.map_shared_rank(&s_cta[s][j][t0 + u], 0) = v;
            }
        }
        __syncthreads();
    }

    // Rank 0 sums the four stream partials (streams in order) and runs the epilogues; the other
    // CTAs may exit once their stores are released.
    cluster.sync();
    if (s != 0) return;
    for (int i = static_cast<int>(threadIdx.x); i < kFuseRows * tokens; i += kFuseThreads) {
        const int j = i / tokens, t = i % tokens;
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) v += s_cta[q][j][t];
        const int row = group * kFuseRows + j;
        v *= row_scale;
        r[static_cast<std::int64_t>(t) * kRLd + row] = __float2bfloat16_rn(silu(v * 0.25f));
    }
    if (Inject && owner) {
        for (int i = static_cast<int>(threadIdx.x); i < 4 * tokens; i += kFuseThreads) {
            const int q = i / tokens, t = i % tokens;
            float v = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) v += s_cta[c][kFuseRows + q][t];
            inj[static_cast<std::int64_t>(t) * kStreams + q] = 2.f * sigmoid(v * 0.25f);
        }
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0; }

} // namespace

void gr_norm(const float* X, std::int32_t tokens, const float* hc_norm,
             const __nv_bfloat16* inject, __nv_bfloat16* Xn, float* inj, cudaStream_t stream) {
    if (tokens <= 0) throw std::invalid_argument("gr_norm: tokens must be positive");
    if (X == nullptr || hc_norm == nullptr || Xn == nullptr || (inject != nullptr && inj == nullptr)) {
        throw std::invalid_argument("gr_norm: null pointer");
    }
    if (!aligned16(X) || !aligned16(hc_norm) || !aligned16(Xn) ||
        (inject != nullptr && (!aligned16(inject) || !aligned16(inj)))) {
        throw std::invalid_argument("gr_norm: pointers must be 16-byte aligned");
    }
    if (inject != nullptr) {
        gr_norm_kernel<true><<<tokens, kThreads, 0, stream>>>(X, hc_norm, inject, Xn, inj);
    } else {
        gr_norm_kernel<false><<<tokens, kThreads, 0, stream>>>(X, hc_norm, nullptr, Xn, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {
template <bool Inject, int Chunk>
void launch_norm_down(const float* X, std::int32_t tokens, const float* hc_norm,
                      const __nv_bfloat16* inject, const std::uint8_t* codes,
                      const __nv_bfloat16* scales, __nv_bfloat16* Xn, float* inj, __nv_bfloat16* r,
                      cudaStream_t stream) {
    CUDA_CHECK(pdl_launch(gr_norm_down_kernel<Inject, Chunk>, dim3(kFuseClusters * 4),
                          dim3(kFuseThreads), 0, stream, X, int(tokens), hc_norm, inject, codes,
                          scales, Xn, inj, r));
}

template <bool Inject>
void dispatch_norm_down(const float* X, std::int32_t tokens, const float* hc_norm,
                        const __nv_bfloat16* inject, const std::uint8_t* codes,
                        const __nv_bfloat16* scales, __nv_bfloat16* Xn, float* inj,
                        __nv_bfloat16* r, cudaStream_t stream) {
    if (tokens == 1) {
        launch_norm_down<Inject, 1>(X, tokens, hc_norm, inject, codes, scales, Xn, inj, r, stream);
    } else if (tokens == 2) {
        launch_norm_down<Inject, 2>(X, tokens, hc_norm, inject, codes, scales, Xn, inj, r, stream);
    } else {
        launch_norm_down<Inject, kFuseChunk>(X, tokens, hc_norm, inject, codes, scales, Xn, inj, r,
                                             stream);
    }
}
} // namespace

void gr_norm_down(const float* X, std::int32_t tokens, const float* hc_norm,
                  const __nv_bfloat16* inject, const std::uint8_t* down_codes,
                  const __nv_bfloat16* down_scales, __nv_bfloat16* Xn, float* inj,
                  __nv_bfloat16* r, cudaStream_t stream) {
    if (tokens <= 0 || tokens > kGrNormDownMaxTokens)
        throw std::invalid_argument("gr_norm_down: tokens must be in [1, 16]");
    if (X == nullptr || hc_norm == nullptr || Xn == nullptr || down_codes == nullptr ||
        down_scales == nullptr || r == nullptr || (inject != nullptr && inj == nullptr)) {
        throw std::invalid_argument("gr_norm_down: null pointer");
    }
    if (!aligned16(X) || !aligned16(hc_norm) || !aligned16(Xn) || !aligned16(down_codes) ||
        (inject != nullptr && (!aligned16(inject) || !aligned16(inj)))) {
        throw std::invalid_argument("gr_norm_down: pointers must be 16-byte aligned");
    }
    if (inject != nullptr) {
        dispatch_norm_down<true>(X, tokens, hc_norm, inject, down_codes, down_scales, Xn, inj, r,
                                 stream);
    } else {
        dispatch_norm_down<false>(X, tokens, hc_norm, nullptr, down_codes, down_scales, Xn,
                                  nullptr, r, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
