// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-common.h @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (kvalues_iq4nl, block_iq4_nl layout). See NOTICE.
// PLE kernels: IQ4_NL n-gram row dequantization, the gated value / dilated conv mixer, and the
// conv-state fold used by speculative rollback.
//
// IQ4_NL semantics follow llama.cpp/ggml (MIT): block_iq4_nl {fp16 d; uint8 qs[16]}, value =
// d * kvalues_iq4nl[nibble] (see third_party/ggml-iq).

#include "kernels/ple/ple.h"

#include "core/device.h"
#include "core/exchange.h"
#include "core/model_dims.h"
#include "kernels/common/math.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

using dims::kHidden;
using dims::kPleConvState;
using dims::kStreamHidden;

inline constexpr int kRowsPerToken   = dims::kPleHeads;                    // 16
inline constexpr int kTokenRawBytes  = kRowsPerToken * dims::kPleRowBytes; // 1440
inline constexpr int kTokenRawChunks = kTokenRawBytes / 16;                // 90
inline constexpr int kIq4nlBlockBytes = 18;
inline constexpr int kDequantThreads  = kHidden / 8; // 320: one 8-value output chunk each
static_assert(kTokenRawBytes % 16 == 0);
static_assert(dims::kPleHeadDim * kRowsPerToken == kHidden);

__constant__ float kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                       1,    13,   25,  38,  53,  69,  89,  113};

__device__ __forceinline__ uint4 load_relaxed_sys_u4(const void* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__launch_bounds__(kDequantThreads) __global__
    void ple_dequant_kernel(const std::uint8_t* rows, __nv_bfloat16* E, PleRowsWait wait) {
    __shared__ __align__(16) std::uint8_t raw[kTokenRawBytes];
    __shared__ int proceed;
    const int t   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    if (wait.ready != nullptr) {
        if (tid == 0) {
            const std::uint32_t expected = exchange_load_acquire_sys(wait.expected);
            const bool ok = exchange_wait_sys(wait.ready, expected, kExchangeTimeoutCycles);
            if (!ok) { exchange_store_release_sys(wait.error, kExchangeErrorPleTimeout); }
            proceed = ok ? 1 : 0;
        }
        __syncthreads();
        if (!proceed) { return; }
    }

    // Rows may live in host-mapped memory published by the host: system-scope loads.
    const std::uint8_t* token_rows = rows + static_cast<std::int64_t>(t) * kTokenRawBytes;
    if (tid < kTokenRawChunks) {
        reinterpret_cast<uint4*>(raw)[tid] = load_relaxed_sys_u4(token_rows + tid * 16);
    }
    __syncthreads();

    // Chunk tid: head h, block b, quarter q (0,1: low nibbles of qs[0..8), qs[8..16);
    // 2,3: high nibbles of the same bytes) -> elements h*160 + b*32 + q*8 .. +8.
    const int h                = tid / 20;
    const int r                = tid - h * 20;
    const int b                = r >> 2;
    const int q                = r & 3;
    const std::uint8_t* block  = raw + h * dims::kPleRowBytes + b * kIq4nlBlockBytes;
    const float d              = __half2float(__ushort_as_half(
        static_cast<unsigned short>(block[0] | (static_cast<unsigned>(block[1]) << 8))));
    const std::uint8_t* qs     = block + 2 + (q & 1) * 8;
    const int shift            = q >= 2 ? 4 : 0;
    __align__(16) __nv_bfloat16 out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        out[e] = __float2bfloat16_rn(d * kIq4nlValues[(qs[e] >> shift) & 0xF]);
    }
    *reinterpret_cast<uint4*>(E + static_cast<std::int64_t>(t) * kHidden + tid * 8) =
        *reinterpret_cast<const uint4*>(out);
}

// ---------------------------------------------------------------- ple_mix

inline constexpr int kMixThreads   = 256;
inline constexpr int kMixPerThread = kHidden / kMixThreads; // 10
static_assert(kMixPerThread * kMixThreads == kHidden);

// Deterministic block sum (fixed shuffle tree, warps summed in order); result in every thread.
__device__ __forceinline__ float ple_block_sum(float v, float* red) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, offset); }
    __syncthreads(); // previous readers of `red` are done
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = v; }
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kMixThreads / 32; ++w) { s += red[w]; }
    return s;
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Phase 1, one CTA per (stream s, token t): Kn, Qn, the gate, GV and c of one 2560 group.
__launch_bounds__(kMixThreads) __global__
    void ple_gate_kernel(const float* X, const __nv_bfloat16* K, const __nv_bfloat16* V,
                         PleMixWeights w, __nv_bfloat16* c_record, float* gate_sig) {
    __shared__ float red[kMixThreads / 32];
    const int s   = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    // K, X and c share the [T][10240] = [T][4][2560] indexing.
    const std::int64_t group = static_cast<std::int64_t>(t) * kStreamHidden + s * kHidden;
    const float* nk          = w.norm_key + s * kHidden;
    const float* nq          = w.norm_query + s * kHidden;
    const float* nc          = w.norm_conv + s * kHidden;

    float kv[kMixPerThread], xv[kMixPerThread];
    float sk = 0.0f, sx = 0.0f;
#pragma unroll
    for (int i = 0; i < kMixPerThread; ++i) {
        const int d = tid + i * kMixThreads;
        kv[i]       = __bfloat162float(K[group + d]);
        xv[i]       = X[group + d];
        sk += kv[i] * kv[i];
        sx += xv[i] * xv[i];
    }
    sk             = ple_block_sum(sk, red);
    sx             = ple_block_sum(sx, red);
    const float rk = 1.0f / sqrtf(sk / static_cast<float>(kHidden) + dims::kRmsEps);
    const float rx = 1.0f / sqrtf(sx / static_cast<float>(kHidden) + dims::kRmsEps);

    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kMixPerThread; ++i) {
        const int d    = tid + i * kMixThreads;
        const float kn = round_bf16(kv[i] * rk * nk[d]);
        const float qn = round_bf16(xv[i] * rx * nq[d]);
        dot += kn * qn;
    }
    dot = ple_block_sum(dot, red);

    float g = dot / 50.596442562694072f; // sqrt(2560)
    g       = g == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g);
    const float sig = 1.0f / (1.0f + expf(-g));

    const __nv_bfloat16* v = V + static_cast<std::int64_t>(t) * kHidden;
    float gv[kMixPerThread];
    float sg = 0.0f;
#pragma unroll
    for (int i = 0; i < kMixPerThread; ++i) {
        const int d = tid + i * kMixThreads;
        gv[i]       = round_bf16(sig * __bfloat162float(v[d]));
        sg += gv[i] * gv[i];
    }
    sg             = ple_block_sum(sg, red);
    const float rc = 1.0f / sqrtf(sg / static_cast<float>(kHidden) + dims::kRmsEps);
#pragma unroll
    for (int i = 0; i < kMixPerThread; ++i) {
        const int d         = tid + i * kMixThreads;
        c_record[group + d] = __float2bfloat16_rn(gv[i] * rc * nc[d]);
    }
    if (tid == 0) { gate_sig[t * dims::kStreams + s] = sig; }
}

inline constexpr int kConvThreads     = 256;
inline constexpr int kChunksPerToken  = kStreamHidden / 8; // 1280

__device__ __forceinline__ void bf16x8_to_float(const uint4& packed, float (&out)[8]) {
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

// Phase 2, one thread per (token, 8 channels): GV + silu(dilated conv over c), added to X.
__launch_bounds__(kConvThreads) __global__
    void ple_conv_kernel(float* X, const __nv_bfloat16* V, const float* conv_weight,
                         const __nv_bfloat16* conv_state, const __nv_bfloat16* c_record,
                         const float* gate_sig, std::int32_t T) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::int64_t>(T) * kChunksPerToken) { return; }
    const int t     = static_cast<int>(idx / kChunksPerToken);
    const int chunk = static_cast<int>(idx - static_cast<std::int64_t>(t) * kChunksPerToken);
    const int ch0   = chunk * 8;
    const int s     = ch0 / kHidden;
    const int d0    = ch0 - s * kHidden;

    const float sig = gate_sig[t * dims::kStreams + s];
    float v[8];
    bf16x8_to_float(*reinterpret_cast<const uint4*>(V + static_cast<std::int64_t>(t) * kHidden + d0),
                    v);

    float taps[dims::kPleConvTaps][8];
#pragma unroll
    for (int j = 0; j < dims::kPleConvTaps; ++j) {
        const int p                = t - kPleConvState + j * dims::kPleConvDilation;
        const __nv_bfloat16* src   = p >= 0
                                         ? c_record + static_cast<std::int64_t>(p) * kStreamHidden
                                         : conv_state + (kPleConvState + p) * kStreamHidden;
        bf16x8_to_float(*reinterpret_cast<const uint4*>(src + ch0), taps[j]);
    }

    float* x = X + static_cast<std::int64_t>(t) * kStreamHidden + ch0;
    const float4* wv = reinterpret_cast<const float4*>(conv_weight) + ch0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float4 wt = wv[e];
        float acc       = 0.0f;
        acc             = fmaf(wt.x, taps[0][e], acc);
        acc             = fmaf(wt.y, taps[1][e], acc);
        acc             = fmaf(wt.z, taps[2][e], acc);
        acc             = fmaf(wt.w, taps[3][e], acc);
        const float gv  = round_bf16(sig * v[e]);
        const float act = acc / (1.0f + expf(-acc));
        x[e] += gv + act;
    }
}

// state[i] = (state || c)[n + i], i < 9; per-thread increasing i reads only indices > i, so the
// update is safe in place.
template <bool kDeviceCount>
__global__ void ple_fold_kernel(__nv_bfloat16* conv_state, const __nv_bfloat16* c_record,
                                const std::int32_t* n_commit, std::int32_t n_scalar,
                                std::int32_t max_commit) {
    const int chunk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (chunk >= kChunksPerToken) { return; }
    int n = kDeviceCount ? *n_commit : n_scalar;
    n     = n < 0 ? 0 : (n > max_commit ? max_commit : n);
    if (n == 0) { return; }
    const int ch0 = chunk * 8;
#pragma unroll
    for (int i = 0; i < kPleConvState; ++i) {
        const int j      = n + i;
        const uint4 val  = j < kPleConvState
                               ? *reinterpret_cast<const uint4*>(conv_state + j * kStreamHidden + ch0)
                               : *reinterpret_cast<const uint4*>(
                                    c_record + static_cast<std::int64_t>(j - kPleConvState) *
                                                   kStreamHidden +
                                    ch0);
        *reinterpret_cast<uint4*>(conv_state + i * kStreamHidden + ch0) = val;
    }
}

inline constexpr int kFoldThreads = 128;

} // namespace

void ple_dequant(const std::uint8_t* rows, std::int32_t T, __nv_bfloat16* E,
                 const PleRowsWait& wait, cudaStream_t stream) {
    if (T == 0) { return; }
    if (reinterpret_cast<std::uintptr_t>(rows) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(E) % 16 != 0) {
        throw std::invalid_argument("ple_dequant: rows and E must be 16-byte aligned");
    }
    if (wait.ready != nullptr && (wait.expected == nullptr || wait.error == nullptr)) {
        throw std::invalid_argument("ple_dequant: a wait needs expected and error cells");
    }
    ple_dequant_kernel<<<static_cast<unsigned int>(T), kDequantThreads, 0, stream>>>(rows, E,
                                                                                    wait);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t ple_mix_workspace_bytes(std::int32_t T) {
    return align_up<256>(static_cast<std::size_t>(T) * dims::kStreams * sizeof(float));
}

void ple_mix(float* X, const __nv_bfloat16* K, const __nv_bfloat16* V, const PleMixWeights& w,
             __nv_bfloat16* conv_state, __nv_bfloat16* c_record, std::int32_t T,
             bool commit_state, void* workspace, std::size_t workspace_bytes,
             cudaStream_t stream) {
    if (T == 0) { return; }
    if (T < 0 || T > 65535) { throw std::invalid_argument("ple_mix: T must be in [0, 65535]"); }
    if (!commit_state && T > dims::kMaxStepTokens) {
        throw std::invalid_argument("ple_mix: record-only (decode/verify) mode needs T <= 16");
    }
    if (workspace_bytes < ple_mix_workspace_bytes(T)) {
        throw std::invalid_argument("ple_mix: workspace too small");
    }
    auto* gate_sig = static_cast<float*>(workspace);
    ple_gate_kernel<<<dim3(dims::kStreams, static_cast<unsigned int>(T)), kMixThreads, 0,
                      stream>>>(X, K, V, w, c_record, gate_sig);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t items = static_cast<std::int64_t>(T) * kChunksPerToken;
    ple_conv_kernel<<<static_cast<unsigned int>(div_up<std::int64_t>(items, kConvThreads)),
                      kConvThreads, 0, stream>>>(X, V, w.conv_weight, conv_state, c_record,
                                                 gate_sig, T);
    CUDA_CHECK(cudaGetLastError());
    if (commit_state) {
        ple_fold_kernel<false><<<div_up(kChunksPerToken, kFoldThreads), kFoldThreads, 0, stream>>>(
            conv_state, c_record, nullptr, T, T);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ple_fold(__nv_bfloat16* conv_state, const __nv_bfloat16* c_record,
              const std::int32_t* n_commit, std::int32_t max_commit, cudaStream_t stream) {
    ple_fold_kernel<true><<<div_up(kChunksPerToken, kFoldThreads), kFoldThreads, 0, stream>>>(
        conv_state, c_record, n_commit, 0, max_commit);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
