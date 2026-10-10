// Modified for fngine from cinference:src/ops/kv_cache/append/k8v4_kernel.cuh @ 383e5db. See NOTICE.
// fngine: single sequence (identity page table), per-token positions, two KV heads, V read in
// place from the fused qkv projection, plus a BF16 cache with a plain copy append.
#include "kernels/qsa/qsa.h"

#include "core/device.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/qsa/fp8_e4m3_row_codec.cuh"
#include "kernels/qsa/hadamard_d256.cuh"
#include "kernels/qsa/nvfp4_group16_codec.cuh"
#include "kernels/qsa/paged_kv_address.cuh"
#include "kernels/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

using Geometry = qsa::KvGeometry;
constexpr int kWarps = 8;

// Row-256 E4M3 codec with an FP16 per-(token, head) scale, after the fixed FP32 normalized
// Hadamard rotation: K of K8V4 and K8V8, and V of K8V8.
__device__ __forceinline__ void append_fp8_row(const __nv_bfloat16* __restrict__ row,
                                               std::uint8_t* __restrict__ codes,
                                               __half* __restrict__ scales, int kv_head,
                                               int physical_page, int page_offset, int lane) {
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = __bfloat162float(row[lane + 32 * r]);
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
    const auto quant = kv_cache_fp8_quant_params(warp_max(local_absmax));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        codes[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_offset)] =
            kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
    }
    if (lane == 0) {
        scales[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_offset)] =
            quant.scale;
    }
}

// Asymmetric K8V4 row: K uses the row-256 E4M3 codec, V the group-16 packed E2M1 codec; both
// operands receive the fixed FP32 normalized Hadamard rotation first.
__device__ __forceinline__ void append_k8v4_row(const __nv_bfloat16* __restrict__ k_row,
                                                const __nv_bfloat16* __restrict__ v_row,
                                                std::uint8_t* __restrict__ cache_k,
                                                std::uint8_t* __restrict__ cache_v,
                                                __half* __restrict__ scale_k,
                                                std::uint8_t* __restrict__ scale_v, int kv_head,
                                                int physical_page, int page_offset, int lane,
                                                float* scratch) {
    append_fp8_row(k_row, cache_k, scale_k, kv_head, physical_page, page_offset, lane);
    float values[8];

#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = __bfloat162float(v_row[lane + 32 * r]);
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) scratch[lane + 32 * r] = values[r];
    __syncwarp();
    if (lane < kKVCacheNvfp4Groups) {
        const auto quantized = kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
        const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
            physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
        store_vec(cache_v + code_offset, make_uint2(quantized.codes_lo, quantized.codes_hi));
        scale_v[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane, page_offset)] =
            quantized.scale;
    }
}

// Programmatic dependents (pdl.cuh): K and V are read after pdl_wait() through coherent loads.
__launch_bounds__(kWarps * 32) __global__
    void qsa_kv_append_k8v4_kernel(const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v, std::int64_t v_ld,
                                   const std::int32_t* __restrict__ positions,
                                   std::uint8_t* __restrict__ cache_k,
                                   std::uint8_t* __restrict__ cache_v,
                                   __half* __restrict__ scale_k,
                                   std::uint8_t* __restrict__ scale_v, std::int32_t tokens) {
    __shared__ float scratch[kWarps][qsa::kHeadDim];
    pdl_trigger();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (unit >= tokens * qsa::kKvHeads) return;
    const int kv_head  = unit % qsa::kKvHeads;
    const int token    = unit / qsa::kKvHeads;
    const int position = positions[token];
    pdl_wait();
    append_k8v4_row(pdl_coherent(k) + (static_cast<std::int64_t>(token) * qsa::kKvHeads + kv_head) *
                                          qsa::kHeadDim,
                    pdl_coherent(v) + static_cast<std::int64_t>(token) * v_ld + kv_head * qsa::kHeadDim,
                    cache_k, cache_v, scale_k, scale_v, kv_head, paged_kv_physical_page(position),
                    position & kPagedKVPageMask, lane, scratch[warp]);
}

// K8V8: K and V both use the row-256 E4M3 codec (one warp per (token, KV head)).
__launch_bounds__(kWarps * 32) __global__
    void qsa_kv_append_k8v8_kernel(const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v, std::int64_t v_ld,
                                   const std::int32_t* __restrict__ positions,
                                   std::uint8_t* __restrict__ cache_k,
                                   std::uint8_t* __restrict__ cache_v,
                                   __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                   std::int32_t tokens) {
    pdl_trigger();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (unit >= tokens * qsa::kKvHeads) return;
    const int kv_head  = unit % qsa::kKvHeads;
    const int token    = unit / qsa::kKvHeads;
    const int position = positions[token];
    const int page     = paged_kv_physical_page(position);
    const int offset   = position & kPagedKVPageMask;
    pdl_wait();
    append_fp8_row(pdl_coherent(k) + (static_cast<std::int64_t>(token) * qsa::kKvHeads + kv_head) *
                                         qsa::kHeadDim,
                   cache_k, scale_k, kv_head, page, offset, lane);
    append_fp8_row(pdl_coherent(v) + static_cast<std::int64_t>(token) * v_ld + kv_head * qsa::kHeadDim,
                   cache_v, scale_v, kv_head, page, offset, lane);
}

// One warp copies one (token, KV head) row of K and of V: 512 bytes each, 16 bytes per lane.
__launch_bounds__(kWarps * 32) __global__
    void qsa_kv_append_bf16_kernel(const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v, std::int64_t v_ld,
                                   const std::int32_t* __restrict__ positions,
                                   __nv_bfloat16* __restrict__ cache_k,
                                   __nv_bfloat16* __restrict__ cache_v, std::int32_t tokens) {
    pdl_trigger();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (unit >= tokens * qsa::kKvHeads) return;
    const int kv_head  = unit % qsa::kKvHeads;
    const int token    = unit / qsa::kKvHeads;
    const int position = positions[token];
    const std::int64_t dst =
        paged_kv_position_offset<qsa::kHeadDim, qsa::kKvHeads>(kv_head, position, lane * 8);
    pdl_wait();
    const __nv_bfloat16* k_row =
        pdl_coherent(k) + (static_cast<std::int64_t>(token) * qsa::kKvHeads + kv_head) * qsa::kHeadDim;
    const __nv_bfloat16* v_row =
        pdl_coherent(v) + static_cast<std::int64_t>(token) * v_ld + kv_head * qsa::kHeadDim;
    store_vec(cache_k + dst, load_vec<int4>(k_row + lane * 8));
    store_vec(cache_v + dst, load_vec<int4>(v_row + lane * 8));
}

std::size_t pages_for(std::int32_t capacity_tokens) {
    if (capacity_tokens < 0) throw std::invalid_argument("qsa kv: negative capacity");
    return static_cast<std::size_t>((capacity_tokens + kQsaKvPageTokens - 1) / kQsaKvPageTokens);
}

}  // namespace

size_t qsa_kv_bytes_per_token(QsaKvFormat format) {
    switch (format) {
    case QsaKvFormat::K8V4: return 2u * 402u;  // per head: K 256 + 2 B scale, V 128 + 16 scales
    case QsaKvFormat::K8V8: return 2u * 516u;  // per head: K 256 + 2 B scale, V 256 + 2 B scale
    case QsaKvFormat::BF16: return 2u * 2u * 256u * 2u;
    }
    throw std::invalid_argument("qsa kv: unknown format");
}

size_t qsa_kv_layer_bytes(QsaKvFormat format, int32_t capacity_tokens) {
    // Every plane is a whole number of 256-byte units per page, so plane offsets stay aligned.
    return pages_for(capacity_tokens) * kQsaKvPageTokens * qsa_kv_bytes_per_token(format);
}

std::array<QsaKvPlane, 4> qsa_kv_planes(QsaKvFormat format, int32_t capacity_tokens) {
    const std::size_t heads_tokens = pages_for(capacity_tokens) * kQsaKvPageTokens * qsa::kKvHeads;
    // Bytes per (token, head) of k, k_scale, v, v_scale.
    std::array<std::size_t, 4> unit{};
    if (format == QsaKvFormat::K8V4) unit = {256, 2, 128, 16};
    else if (format == QsaKvFormat::K8V8) unit = {256, 2, 256, 2};
    else unit = {qsa::kHeadDim * 2, qsa::kHeadDim * 2, 0, 0};
    std::array<QsaKvPlane, 4> planes{};
    std::size_t offset = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        planes[i] = {offset, heads_tokens * unit[i]};
        offset += planes[i].bytes;
    }
    return planes;
}

QsaKvCache qsa_kv_cache_view(QsaKvFormat format, void* base, int32_t capacity_tokens) {
    const auto planes = qsa_kv_planes(format, capacity_tokens);
    auto* bytes       = static_cast<std::uint8_t*>(base);
    QsaKvCache cache;
    cache.format          = format;
    cache.capacity_tokens = static_cast<int32_t>(pages_for(capacity_tokens) * kQsaKvPageTokens);
    cache.k               = bytes + planes[0].offset;
    if (format == QsaKvFormat::BF16) {
        cache.v       = bytes + planes[1].offset;
        cache.k_scale = nullptr;
        cache.v_scale = nullptr;
    } else {
        cache.k_scale = bytes + planes[1].offset;
        cache.v       = bytes + planes[2].offset;
        cache.v_scale = bytes + planes[3].offset;
    }
    return cache;
}

void qsa_kv_append(const QsaKvCache& cache, const __nv_bfloat16* k, const __nv_bfloat16* v,
                   int64_t v_ld, const int32_t* positions, int32_t tokens, cudaStream_t stream) {
    if (tokens <= 0) return;
    const int units = tokens * qsa::kKvHeads;
    const int grid  = (units + kWarps - 1) / kWarps;
    if (cache.format == QsaKvFormat::K8V4) {
        CUDA_CHECK(pdl_launch(qsa_kv_append_k8v4_kernel, dim3(grid), dim3(kWarps * 32), 0, stream, k, v,
                              v_ld, positions, static_cast<std::uint8_t*>(cache.k),
                              static_cast<std::uint8_t*>(cache.v), static_cast<__half*>(cache.k_scale),
                              static_cast<std::uint8_t*>(cache.v_scale), tokens));
    } else if (cache.format == QsaKvFormat::K8V8) {
        CUDA_CHECK(pdl_launch(qsa_kv_append_k8v8_kernel, dim3(grid), dim3(kWarps * 32), 0, stream, k, v,
                              v_ld, positions, static_cast<std::uint8_t*>(cache.k),
                              static_cast<std::uint8_t*>(cache.v), static_cast<__half*>(cache.k_scale),
                              static_cast<__half*>(cache.v_scale), tokens));
    } else {
        CUDA_CHECK(pdl_launch(qsa_kv_append_bf16_kernel, dim3(grid), dim3(kWarps * 32), 0, stream, k, v,
                              v_ld, positions, static_cast<__nv_bfloat16*>(cache.k),
                              static_cast<__nv_bfloat16*>(cache.v), tokens));
    }
}

}  // namespace fngine::kernels
