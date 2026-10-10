// QSA sparse attention over the paged KV cache: selected 4-token blocks plus the tail (or the
// whole causal prefix when the indexer is dense), FP32 online softmax, gated BF16 output.
#include "kernels/qsa/qsa.h"

#include "core/device.h"
#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/qsa/fp8_e4m3_row_codec.cuh"
#include "kernels/qsa/hadamard_d256.cuh"
#include "kernels/qsa/nvfp4_group16_codec.cuh"
#include "kernels/qsa/paged_kv_address.cuh"
#include "kernels/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace fngine::kernels {
namespace {

using Geometry = qsa::KvGeometry;

constexpr int kWarps        = 4;
constexpr int kThreads      = kWarps * 32;
constexpr int kRows         = 16;  // MMA rows: 12 query heads of one KV head, padded
// 32-key chunks keep the CTA at ~46 KB of shared memory: two CTAs per SM within sm_120's
// 100 KB per SM (99 KB per CTA).
constexpr int kChunkKeys    = 32;
constexpr int kD            = qsa::kHeadDim;
constexpr int kStride       = kD + 8;           // halves per smem row (528 B: conflict-free ldmatrix)
constexpr int kSStride      = kChunkKeys + 4;   // floats per score row
constexpr int kPStride      = kChunkKeys + 8;   // halves per P row
constexpr int kOStride      = kD + 4;           // floats per output row (finalization)
constexpr int kSplitTokenLimit = 16;  // decode/verify widths use split-K + merge
// At most 33 splits: every split of a full 2051-key list gets >= 2 chunks, which measured as fast
// as 1-chunk splits while halving the merge traffic.
constexpr int kMaxSplits    = 33;
constexpr float kScaleLog2  = 0.0625f * 1.4426950408889634f;  // 1/16 * log2(e)

struct SharedLayout {
    static constexpr int q     = 0;
    static constexpr int k     = q + kRows * kStride * 2;
    static constexpr int v     = k + kChunkKeys * kStride * 2;
    static constexpr int s     = v + kChunkKeys * kStride * 2;
    static constexpr int p     = s + kRows * kSStride * 4;
    static constexpr int p_lo  = p + kRows * kPStride * 2;
    static constexpr int pos   = p_lo + kRows * kPStride * 2;
    static constexpr int alpha = pos + kChunkKeys * 4;
    static constexpr int l      = alpha + kRows * 4;
    static constexpr int kscale = l + kRows * 4;
    static constexpr int vscale = kscale + kChunkKeys * 4;
    static constexpr int bytes  = vscale + kChunkKeys * 4;
};
static_assert(kRows * kOStride * 4 <= kChunkKeys * kStride * 2, "output staging reuses sK");

// K8V4 and K8V8 store Hadamard-rotated K/V and multiply in FP16; BF16 stores K/V as is.
template <QsaKvFormat Format>
inline constexpr bool kRotated = Format != QsaKvFormat::BF16;

template <QsaKvFormat Format>
struct Elem;
template <>
struct Elem<QsaKvFormat::BF16> {
    using type = __nv_bfloat16;
    __device__ static unsigned pack(float lo, float hi) { return pack_bf16x2(lo, hi); }
    __device__ static type from_float(float x) { return __float2bfloat16_rn(x); }
    __device__ static float to_float(type x) { return __bfloat162float(x); }
    __device__ static float2 unpack(unsigned bits) { return bf16x2_bits_to_float2(bits); }
    __device__ static void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
        mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1);
    }
};
template <>
struct Elem<QsaKvFormat::K8V4> {
    using type = __half;
    __device__ static unsigned pack(float lo, float hi) { return pack_f16x2(lo, hi); }
    __device__ static type from_float(float x) { return __float2half_rn(x); }
    __device__ static float to_float(type x) { return __half2float(x); }
    __device__ static float2 unpack(unsigned bits) { return __half22float2(half2_from_bits(bits)); }
    __device__ static void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
        mma_f16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1);
    }
};
template <>
struct Elem<QsaKvFormat::K8V8> : Elem<QsaKvFormat::K8V4> {};

struct KvPlanes {
    const void* k;
    const void* v;
    const void* k_scale;
    const void* v_scale;
};

// Attended key list of one query: dense (all of 0..p) or the selected blocks then the tail.
struct KeyList {
    int keys;
    int selected_keys;  // 4 * selected blocks (0 when dense)
    int tail_start;
    bool dense;
    __device__ int position(const std::int32_t* __restrict__ sel, int i) const {
        if (dense) return i;
        if (i < selected_keys) return sel[i >> 2] * qsa::kBlock + (i & 3);
        return tail_start + (i - selected_keys);
    }
};

__device__ __forceinline__ KeyList make_key_list(std::int32_t p, std::int32_t count) {
    const int nb = (p + 1) / qsa::kBlock;
    KeyList list;
    list.dense = nb <= qsa::kTopBlocks;
    if (list.dense) {
        list.keys = p + 1;
        list.selected_keys = 0;
        list.tail_start = 0;
    } else {
        list.selected_keys = count * qsa::kBlock;
        list.tail_start    = nb * qsa::kBlock;
        list.keys          = list.selected_keys + (p + 1 - list.tail_start);
    }
    return list;
}

// Gathers one chunk of K and V (kChunkKeys keys, positions in pos[], -1 = padding) into shared
// memory. BF16: copied. K8V4/K8V8: K as its E4M3 codes in FP16 (exact) with the per-key FP16 row
// scale in kscale[] (applied to the FP32 scores). K8V4 V: E2M1 code * E4M3 group scale in FP16
// (exact). K8V8 V: E4M3 codes in FP16 (exact) with the per-key scale in vscale[] (folded into
// the probabilities). Ends with the data visible to the CTA.
template <QsaKvFormat Format>
__device__ __forceinline__ void load_chunk(const KvPlanes& cache, int kv_head, const int* pos,
                                           typename Elem<Format>::type* sk,
                                           typename Elem<Format>::type* sv, float* kscale,
                                           float* vscale, int thread) {
    if constexpr (Format == QsaKvFormat::BF16) {
        const auto* k = static_cast<const __nv_bfloat16*>(cache.k);
        const auto* v = static_cast<const __nv_bfloat16*>(cache.v);
        constexpr int kParts = kD / 8;  // 16-byte parts per row
#pragma unroll
        for (int j = 0; j < kChunkKeys * kParts / kThreads; ++j) {
            const int idx  = thread + j * kThreads;
            const int key  = idx / kParts;
            const int part = idx % kParts;
            const int p    = pos[key];
            const std::int64_t off =
                p >= 0 ? paged_kv_position_offset<kD, qsa::kKvHeads>(kv_head, p, part * 8) : 0;
            cp_async_zfill<16, Cache::cg>(sk + key * kStride + part * 8, k + off, p >= 0 ? 16 : 0);
            cp_async_zfill<16, Cache::cg>(sv + key * kStride + part * 8, v + off, p >= 0 ? 16 : 0);
        }
        cp_commit();
        cp_wait<0>();
    } else if constexpr (Format == QsaKvFormat::K8V8) {
        constexpr int kParts = kD / 16;                         // 16 codes per part
        constexpr int kLoads = kChunkKeys * kParts / kThreads;  // 4 per operand
        const std::uint8_t* planes_codes[2] = {static_cast<const std::uint8_t*>(cache.k),
                                        static_cast<const std::uint8_t*>(cache.v)};
        const __half* planes_scales[2] = {static_cast<const __half*>(cache.k_scale),
                                        static_cast<const __half*>(cache.v_scale)};
        int4 codes[2][kLoads];
        __half scales[2][kLoads];
#pragma unroll
        for (int operand = 0; operand < 2; ++operand) {
#pragma unroll
            for (int j = 0; j < kLoads; ++j) {
                const int idx  = thread + j * kThreads;
                const int key  = idx / kParts;
                const int part = idx % kParts;
                const int p    = pos[key];
                if (p >= 0) {
                    const int page = paged_kv_physical_page(p);
                    const int off  = p & kPagedKVPageMask;
                    codes[operand][j]  = load_vec<int4>(planes_codes[operand] +
                                                        kv_cache_fp8_code_index<Geometry>(page, kv_head, part * 16, off));
                    scales[operand][j] = planes_scales[operand][kv_cache_fp8_scale_index<Geometry>(page, kv_head, off)];
                } else {
                    codes[operand][j]  = make_int4(0, 0, 0, 0);
                    scales[operand][j] = __float2half_rn(0.0f);
                }
            }
        }
#pragma unroll
        for (int operand = 0; operand < 2; ++operand) {
#pragma unroll
            for (int j = 0; j < kLoads; ++j) {
                const int idx  = thread + j * kThreads;
                const int key  = idx / kParts;
                const int part = idx % kParts;
                const auto* c2 = reinterpret_cast<const std::uint16_t*>(&codes[operand][j]);
                unsigned h[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const __half2 value = kv_cache_fp8_code2_to_half2(c2[i]);
                    h[i]                = load_vec<unsigned>(&value);
                }
                if (part == 0) (operand == 0 ? kscale : vscale)[key] = __half2float(scales[operand][j]);
                __half* dst = (operand == 0 ? sk : sv) + key * kStride + part * 16;
                store_vec(dst, make_uint4(h[0], h[1], h[2], h[3]));
                store_vec(dst + 8, make_uint4(h[4], h[5], h[6], h[7]));
            }
        }
    } else {
        const auto* k_codes  = static_cast<const std::uint8_t*>(cache.k);
        const auto* k_scales = static_cast<const __half*>(cache.k_scale);
        const auto* v_codes  = static_cast<const std::uint8_t*>(cache.v);
        const auto* v_scales = static_cast<const std::uint8_t*>(cache.v_scale);
        constexpr int kKParts = kD / 16;  // 16 codes per part
        constexpr int kVParts = kD / 32;  // 32 codes (two groups) per part
        constexpr int kKLoads = kChunkKeys * kKParts / kThreads;  // 8
        constexpr int kVLoads = kChunkKeys * kVParts / kThreads;  // 4
        int4 kc[kKLoads];
        __half ks[kKLoads];
        int4 vc[kVLoads];
        std::uint16_t vs[kVLoads];
#pragma unroll
        for (int j = 0; j < kKLoads; ++j) {
            const int idx  = thread + j * kThreads;
            const int key  = idx / kKParts;
            const int part = idx % kKParts;
            const int p    = pos[key];
            if (p >= 0) {
                const int page = paged_kv_physical_page(p);
                const int off  = p & kPagedKVPageMask;
                kc[j] = load_vec<int4>(k_codes + kv_cache_fp8_code_index<Geometry>(page, kv_head,
                                                                                   part * 16, off));
                ks[j] = k_scales[kv_cache_fp8_scale_index<Geometry>(page, kv_head, off)];
            } else {
                kc[j] = make_int4(0, 0, 0, 0);
                ks[j] = __float2half_rn(0.0f);
            }
        }
#pragma unroll
        for (int j = 0; j < kVLoads; ++j) {
            const int idx  = thread + j * kThreads;
            const int key  = idx / kVParts;
            const int part = idx % kVParts;
            const int p    = pos[key];
            if (p >= 0) {
                const int page = paged_kv_physical_page(p);
                const int off  = p & kPagedKVPageMask;
                vc[j] = load_vec<int4>(v_codes + kv_cache_nvfp4_code_index<Geometry>(
                                                     page, kv_head, part * 32, off));
                vs[j] = load_vec<std::uint16_t>(v_scales + kv_cache_nvfp4_scale_index<Geometry>(
                                                               page, kv_head, part * 2, off));
            } else {
                vc[j] = make_int4(0, 0, 0, 0);
                vs[j] = 0;
            }
        }
#pragma unroll
        for (int j = 0; j < kKLoads; ++j) {
            const int idx  = thread + j * kThreads;
            const int key  = idx / kKParts;
            const int part = idx % kKParts;
            const auto* codes = reinterpret_cast<const std::uint16_t*>(&kc[j]);
            unsigned h[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __half2 value = kv_cache_fp8_code2_to_half2(codes[i]);
                h[i]                = load_vec<unsigned>(&value);
            }
            if (part == 0) kscale[key] = __half2float(ks[j]);
            __half* dst = sk + key * kStride + part * 16;
            store_vec(dst, make_uint4(h[0], h[1], h[2], h[3]));
            store_vec(dst + 8, make_uint4(h[4], h[5], h[6], h[7]));
        }
#pragma unroll
        for (int j = 0; j < kVLoads; ++j) {
            const int idx  = thread + j * kThreads;
            const int key  = idx / kVParts;
            const int part = idx % kVParts;
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&vc[j]);
            const auto lo = kv_cache_nvfp4_dequant_f16x16(bytes, static_cast<std::uint8_t>(vs[j] & 0xff));
            const auto hi = kv_cache_nvfp4_dequant_f16x16(bytes + 8, static_cast<std::uint8_t>(vs[j] >> 8));
            __half* dst = sv + key * kStride + part * 32;
            store_vec(dst, lo.lo);
            store_vec(dst + 8, lo.hi);
            store_vec(dst + 16, hi.lo);
            store_vec(dst + 24, hi.hi);
        }
    }
    __syncthreads();
}

// Final output of one query-head row held as values[r] = dim lane + 32 r (already normalized,
// in the cache basis): inverse Hadamard for K8V4/K8V8, gate, BF16 store.
template <QsaKvFormat Format>
__device__ __forceinline__ void store_output_row(float (&values)[8], int token, int head,
                                                 const __nv_bfloat16* __restrict__ qkv,
                                                 __nv_bfloat16* __restrict__ out, int lane) {
    if constexpr (kRotated<Format>) normalized_hadamard_d256_inplace(values, lane);
    const __nv_bfloat16* gate =
        qkv + static_cast<std::int64_t>(token) * qsa::kQkvLd + head * 2 * kD + kD;
    __nv_bfloat16* dst = out + static_cast<std::int64_t>(token) * qsa::kOutLd + head * kD;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        dst[d]      = __float2bfloat16_rn(values[r] * sigmoid(__bfloat162float(gate[d])));
    }
}

template <QsaKvFormat Format, bool Split>
__global__ __launch_bounds__(kThreads, 2) void qsa_attention_kernel(
    KvPlanes cache, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ qkv,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ selected,
    const std::int32_t* __restrict__ counts, std::int32_t splits, float* __restrict__ part_o,
    float* __restrict__ part_ml, __nv_bfloat16* __restrict__ out) {
    using E = Elem<Format>;
    using T = typename E::type;
    extern __shared__ __align__(16) std::uint8_t smem[];
    T* sq        = reinterpret_cast<T*>(smem + SharedLayout::q);
    T* sk        = reinterpret_cast<T*>(smem + SharedLayout::k);
    T* sv        = reinterpret_cast<T*>(smem + SharedLayout::v);
    float* ss    = reinterpret_cast<float*>(smem + SharedLayout::s);
    T* sp        = reinterpret_cast<T*>(smem + SharedLayout::p);
    T* sp_lo     = reinterpret_cast<T*>(smem + SharedLayout::p_lo);
    int* spos    = reinterpret_cast<int*>(smem + SharedLayout::pos);
    float* salpha = reinterpret_cast<float*>(smem + SharedLayout::alpha);
    float* sl    = reinterpret_cast<float*>(smem + SharedLayout::l);
    float* skscale = reinterpret_cast<float*>(smem + SharedLayout::kscale);
    float* svscale = reinterpret_cast<float*>(smem + SharedLayout::vscale);

    // Programmatic dependent (pdl.cuh): everything it reads comes from earlier kernels.
    pdl_trigger();
    pdl_wait();
    q        = pdl_coherent(q);
    qkv      = pdl_coherent(qkv);
    selected = pdl_coherent(selected);
    counts   = pdl_coherent(counts);
    const int thread  = static_cast<int>(threadIdx.x);
    const int warp    = thread >> 5;
    const int lane    = thread & 31;
    const int split   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int token   = static_cast<int>(blockIdx.z);
    const int g       = lane >> 2;
    const int c       = lane & 3;

    const KeyList list = make_key_list(positions[token], counts[token]);
    const std::int32_t* sel = selected + static_cast<std::int64_t>(token) * qsa::kTopBlocks;
    const int chunks        = (list.keys + kChunkKeys - 1) / kChunkKeys;
    const int per_split     = (chunks + splits - 1) / splits;
    const int chunk_begin   = split * per_split;
    const int chunk_end     = min(chunks, chunk_begin + per_split);
    const std::int64_t unit = (static_cast<std::int64_t>(token) * qsa::kKvHeads + kv_head) * splits + split;

    if constexpr (Split) {
        if (chunk_begin >= chunk_end) {  // no keys: weight 0 and a zero partial for the merge
            if (thread < qsa::kGroup) {
                part_ml[(unit * qsa::kGroup + thread) * 2]     = -INFINITY;
                part_ml[(unit * qsa::kGroup + thread) * 2 + 1] = 0.0f;
            }
            float4* dst = reinterpret_cast<float4*>(part_o + unit * qsa::kGroup * kD);
            for (int i = thread; i < qsa::kGroup * kD / 4; i += kThreads) dst[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            return;
        }
    }

    // Query tile: rows 0..11 = query heads 12*kv_head + row, 12..15 = 0. K8V4: the query is
    // Hadamard-rotated and kept as FP16 hi + lo (staged in sK) so the scores lose nothing to the
    // FP16 rounding of the rotated query.
    T* sq_lo = sk;
    for (int row = warp; row < kRows; row += kWarps) {
        float values[8];
        if (row < qsa::kGroup) {
            const __nv_bfloat16* src =
                q + (static_cast<std::int64_t>(token) * qsa::kHeads + kv_head * qsa::kGroup + row) * kD;
#pragma unroll
            for (int r = 0; r < 8; ++r) values[r] = __bfloat162float(src[lane + 32 * r]);
            if constexpr (kRotated<Format>) normalized_hadamard_d256_inplace(values, lane);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) values[r] = 0.0f;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const T hi = E::from_float(values[r]);
            sq[row * kStride + lane + 32 * r] = hi;
            if constexpr (kRotated<Format>) {
                sq_lo[row * kStride + lane + 32 * r] = E::from_float(values[r] - E::to_float(hi));
            }
        }
    }
    __syncthreads();
    unsigned qa[kD / 16][4];
    unsigned ql[kRotated<Format> ? kD / 16 : 1][4];
#pragma unroll
    for (int ks = 0; ks < kD / 16; ++ks) {
        const int offset = (lane & 15) * kStride + ks * 16 + (lane >> 4) * 8;
        ldmatrix_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], smem_addr(sq + offset));
        if constexpr (kRotated<Format>) {
            ldmatrix_x4(ql[ks][0], ql[ks][1], ql[ks][2], ql[ks][3], smem_addr(sq_lo + offset));
        }
    }

    float o[8][4] = {};
    // Softmax row state, owned by the 8 threads of row thread >> 3.
    const int srow = thread >> 3;
    const int ssub = thread & 7;
    float m_run    = -INFINITY;
    float l_run    = 0.0f;
    // K8V8: O is accumulated in units of u_run (V = code * per-key scale; P carries scale / u).
    float u_run    = 1.0f;

    for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
        const int key_base = chunk * kChunkKeys;
        const int valid    = min(kChunkKeys, list.keys - key_base);
        __syncthreads();  // previous chunk's PV reads of sv/sp are done
        if (thread < kChunkKeys) spos[thread] = thread < valid ? list.position(sel, key_base + thread) : -1;
        __syncthreads();
        load_chunk<Format>(cache, kv_head, spos, sk, sv, skscale, svscale, thread);

        // S = Q K^T for this warp's 8 keys (one n-tile; one x4 ldmatrix covers two k-steps).
        {
            float acc[4] = {};
#pragma unroll
            for (int ks = 0; ks < kD / 16; ks += 2) {
                const int key = warp * 8 + (lane & 7);
                const int col = ks * 16 + (lane >> 3) * 8;
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_addr(sk + key * kStride + col));
                E::mma(acc, qa[ks], b0, b1);
                E::mma(acc, qa[ks + 1], b2, b3);
                if constexpr (kRotated<Format>) {
                    E::mma(acc, ql[ks], b0, b1);
                    E::mma(acc, ql[ks + 1], b2, b3);
                }
            }
            const int key = warp * 8 + 2 * c;
            float scale0 = kScaleLog2, scale1 = kScaleLog2;
            if constexpr (kRotated<Format>) {
                scale0 *= skscale[key];
                scale1 *= skscale[key + 1];
            }
            const float s0 = key < valid ? acc[0] * scale0 : -INFINITY;
            const float s1 = key + 1 < valid ? acc[1] * scale1 : -INFINITY;
            const float s2 = key < valid ? acc[2] * scale0 : -INFINITY;
            const float s3 = key + 1 < valid ? acc[3] * scale1 : -INFINITY;
            store_vec(ss + g * kSStride + key, make_float2(s0, s1));
            store_vec(ss + (g + 8) * kSStride + key, make_float2(s2, s3));
        }
        __syncthreads();

        // Online softmax over the chunk: 8 threads per row, 4 keys each.
        {
            const float4 a = load_vec<float4>(ss + srow * kSStride + ssub * 4);
            const float s4[4] = {a.x, a.y, a.z, a.w};
            float m_local = fmaxf(fmaxf(s4[0], s4[1]), fmaxf(s4[2], s4[3]));
            m_local = fmaxf(m_local, __shfl_xor_sync(kFullWarpMask, m_local, 1));
            m_local = fmaxf(m_local, __shfl_xor_sync(kFullWarpMask, m_local, 2));
            m_local = fmaxf(m_local, __shfl_xor_sync(kFullWarpMask, m_local, 4));
            const float m_new = fmaxf(m_run, m_local);  // finite: every chunk has a valid key
            const float alpha = exp2f(m_run - m_new);
            float p[4];
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = exp2f(s4[i] - m_new);
                sum += p[i];
            }
            sum += __shfl_xor_sync(kFullWarpMask, sum, 1);
            sum += __shfl_xor_sync(kFullWarpMask, sum, 2);
            sum += __shfl_xor_sync(kFullWarpMask, sum, 4);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            // P = hi + lo carries the probabilities to ~2^-17 (bf16) / ~2^-22 (fp16) through the MMA.
            float alpha_out = alpha;
            if constexpr (Format == QsaKvFormat::K8V8) {
                // Fold each key's V scale into its probability, relative to the chunk's largest
                // scale u, so the FP16 halves stay in [0, 1]; rescale O from the old unit to u.
                float u = 0.0f;
#pragma unroll 8
                for (int i = 0; i < kChunkKeys; ++i) u = fmaxf(u, svscale[i]);
                if (u == 0.0f) u = u_run;  // all-zero V rows: every scaled probability is 0
#pragma unroll
                for (int i = 0; i < 4; ++i) p[i] *= svscale[ssub * 4 + i] / u;
                alpha_out = alpha * (u_run / u);
                u_run     = u;
            }
            const unsigned hi[2] = {E::pack(p[0], p[1]), E::pack(p[2], p[3])};
            const float2 h0      = E::unpack(hi[0]);
            const float2 h1      = E::unpack(hi[1]);
            store_vec(sp + srow * kPStride + ssub * 4, make_uint2(hi[0], hi[1]));
            store_vec(sp_lo + srow * kPStride + ssub * 4,
                      make_uint2(E::pack(p[0] - h0.x, p[1] - h0.y), E::pack(p[2] - h1.x, p[3] - h1.y)));
            if (ssub == 0) salpha[srow] = alpha_out;
        }
        __syncthreads();

        // O = alpha * O + P V for this warp's 64 dims.
        {
            const float alpha0 = salpha[g];
            const float alpha1 = salpha[g + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                o[nt][0] *= alpha0;
                o[nt][1] *= alpha0;
                o[nt][2] *= alpha1;
                o[nt][3] *= alpha1;
            }
#pragma unroll
            for (int kk = 0; kk < kChunkKeys / 16; ++kk) {
                const int offset = (lane & 15) * kPStride + kk * 16 + (lane >> 4) * 8;
                unsigned pa[4], pl[4];
                ldmatrix_x4(pa[0], pa[1], pa[2], pa[3], smem_addr(sp + offset));
                ldmatrix_x4(pl[0], pl[1], pl[2], pl[3], smem_addr(sp_lo + offset));
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                    const int dim = warp * 64 + np * 16 + ((lane >> 4) << 3);
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4_t(b0, b1, b2, b3, smem_addr(sv + key * kStride + dim));
                    E::mma(o[2 * np], pa, b0, b1);
                    E::mma(o[2 * np + 1], pa, b2, b3);
                    E::mma(o[2 * np], pl, b0, b1);
                    E::mma(o[2 * np + 1], pl, b2, b3);
                }
            }
        }
    }

    if constexpr (Split) {
        if (ssub == 0 && srow < qsa::kGroup) {
            part_ml[(unit * qsa::kGroup + srow) * 2]     = m_run;
            part_ml[(unit * qsa::kGroup + srow) * 2 + 1] = l_run;
        }
        float* dst = part_o + unit * qsa::kGroup * kD;
        if constexpr (Format == QsaKvFormat::K8V8) {
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) o[nt][i] *= u_run;
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dim = warp * 64 + nt * 8 + 2 * c;
            if (g < qsa::kGroup) store_vec(dst + g * kD + dim, make_float2(o[nt][0], o[nt][1]));
            if (g + 8 < qsa::kGroup) store_vec(dst + (g + 8) * kD + dim, make_float2(o[nt][2], o[nt][3]));
        }
    } else {
        if (ssub == 0) sl[srow] = l_run;
        __syncthreads();  // all PV reads of sk/sv done; sl visible
        float* so = reinterpret_cast<float*>(sk);
        float inv0 = 1.0f / sl[g];
        float inv1 = 1.0f / sl[g + 8];
        if constexpr (Format == QsaKvFormat::K8V8) {
            inv0 = u_run / sl[g];
            inv1 = u_run / sl[g + 8];
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dim = warp * 64 + nt * 8 + 2 * c;
            store_vec(so + g * kOStride + dim, make_float2(o[nt][0] * inv0, o[nt][1] * inv0));
            store_vec(so + (g + 8) * kOStride + dim, make_float2(o[nt][2] * inv1, o[nt][3] * inv1));
        }
        __syncthreads();
        for (int row = warp; row < qsa::kGroup; row += kWarps) {
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) values[r] = so[row * kOStride + lane + 32 * r];
            store_output_row<Format>(values, token, kv_head * qsa::kGroup + row, qkv, out, lane);
        }
    }
}

// Split-K merge: one CTA per (token, query head); thread d owns output dim d. Every split's
// partial (empty splits wrote zeros) is loaded up front, independent of the split weights that
// warp 0 forms meanwhile, so the kernel pays about two global-load latencies.
constexpr int kMergeThreads = kD;
template <QsaKvFormat Format>
__global__ __launch_bounds__(kMergeThreads) void qsa_attention_merge_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_ml, std::int32_t splits,
    const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out) {
    static_assert(kMaxSplits <= 64, "warp 0 reads two split states per lane");
    __shared__ float weight[kMaxSplits];
    __shared__ float row_values[kD];
    __shared__ float inv_total;
    pdl_trigger();
    pdl_wait();
    part_o  = pdl_coherent(part_o);
    part_ml = pdl_coherent(part_ml);
    qkv     = pdl_coherent(qkv);
    const int thread  = static_cast<int>(threadIdx.x);
    const int lane    = thread & 31;
    const int token   = static_cast<int>(blockIdx.x) / qsa::kHeads;
    const int head    = static_cast<int>(blockIdx.x) % qsa::kHeads;
    const int kv_head = head / qsa::kGroup;
    const int row     = head % qsa::kGroup;
    const std::int64_t base = (static_cast<std::int64_t>(token) * qsa::kKvHeads + kv_head) * splits;
    const float* src = part_o + (base * qsa::kGroup + row) * kD + thread;
    float partial[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
        partial[s] = s < splits ? src[static_cast<std::int64_t>(s) * qsa::kGroup * kD] : 0.0f;
    }
    if (thread < 32) {
        float2 ml[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int s = lane + 32 * k;
            ml[k] = s < splits ? load_vec<float2>(part_ml + ((base + s) * qsa::kGroup + row) * 2)
                               : make_float2(-INFINITY, 0.0f);
        }
        float m_max = -INFINITY;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (ml[k].y > 0.0f) m_max = fmaxf(m_max, ml[k].x);
        }
        m_max       = warp_max(m_max);
        float total = 0.0f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int s   = lane + 32 * k;
            const float w = ml[k].y > 0.0f ? exp2f(ml[k].x - m_max) : 0.0f;
            if (s < kMaxSplits) weight[s] = w;
            total += w * ml[k].y;
        }
        total = warp_sum(total);
        if (lane == 0) inv_total = 1.0f / total;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
        if (s < splits) acc += weight[s] * partial[s];
    }
    row_values[thread] = acc * inv_total;
    __syncthreads();
    if (thread < 32) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) values[r] = row_values[lane + 32 * r];
        store_output_row<Format>(values, token, head, qkv, out, lane);
    }
}

std::once_flag g_attention_attributes_once;

template <QsaKvFormat Format>
void launch(const QsaKvCache& cache, const __nv_bfloat16* q, const __nv_bfloat16* qkv,
            const int32_t* positions, const int32_t* selected, const int32_t* counts,
            int32_t tokens, int32_t splits, float* part_o, float* part_ml, __nv_bfloat16* out,
            cudaStream_t stream) {
    const KvPlanes planes{cache.k, cache.v, cache.k_scale, cache.v_scale};
    const dim3 grid(static_cast<unsigned>(splits), qsa::kKvHeads, static_cast<unsigned>(tokens));
    if (splits > 1) {
        CUDA_CHECK(pdl_launch(qsa_attention_kernel<Format, true>, grid, dim3(kThreads),
                              SharedLayout::bytes, stream, planes, q, qkv, positions, selected, counts,
                              splits, part_o, part_ml, out));
        CUDA_CHECK(pdl_launch(qsa_attention_merge_kernel<Format>, dim3(tokens * qsa::kHeads),
                              dim3(kMergeThreads), 0, stream, static_cast<const float*>(part_o),
                              static_cast<const float*>(part_ml), splits, qkv, out));
    } else {
        CUDA_CHECK(pdl_launch(qsa_attention_kernel<Format, false>, grid, dim3(kThreads),
                              SharedLayout::bytes, stream, planes, q, qkv, positions, selected, counts,
                              1, static_cast<float*>(nullptr), static_cast<float*>(nullptr), out));
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int32_t qsa_attention_splits(int32_t tokens) {
    if (tokens > kSplitTokenLimit) return 1;
    // ~2 CTAs per SM on 170 SMs for the two KV heads: tokens * splits ~ 170 (capped at 33).
    const int32_t want = (170 + tokens - 1) / tokens;
    return std::clamp(want, 1, kMaxSplits);
}

static std::size_t split_workspace_bytes(int32_t tokens, int32_t splits) {
    if (splits <= 1) return 0;
    const std::size_t units = static_cast<std::size_t>(tokens) * qsa::kKvHeads * splits * qsa::kGroup;
    return units * (kD + 2) * sizeof(float);
}

size_t qsa_attention_workspace_bytes(int32_t max_tokens) {
    // Enough for every call with 1..max_tokens tokens (splits shrink as T grows, so the
    // requirement is not monotonic in T).
    std::size_t bytes = 0;
    for (int32_t t = 1; t <= std::min(max_tokens, kSplitTokenLimit); ++t) {
        bytes = std::max(bytes, split_workspace_bytes(t, qsa_attention_splits(t)));
    }
    return bytes;
}

void qsa_attention(const QsaKvCache& cache, const __nv_bfloat16* q, const __nv_bfloat16* qkv,
                   const int32_t* positions, const int32_t* selected, const int32_t* counts,
                   int32_t tokens, void* workspace, size_t workspace_bytes, __nv_bfloat16* out,
                   cudaStream_t stream) {
    if (tokens <= 0) return;
    std::call_once(g_attention_attributes_once, [] {
        const void* kernels[] = {
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::K8V4, true>),
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::K8V4, false>),
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::K8V8, true>),
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::K8V8, false>),
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::BF16, true>),
            reinterpret_cast<const void*>(qsa_attention_kernel<QsaKvFormat::BF16, false>),
        };
        for (const void* kernel : kernels) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            SharedLayout::bytes));
        }
    });
    const int32_t splits = qsa_attention_splits(tokens);
    float* part_o  = nullptr;
    float* part_ml = nullptr;
    if (splits > 1) {
        if (workspace_bytes < split_workspace_bytes(tokens, splits)) {
            throw std::invalid_argument("qsa_attention: workspace too small");
        }
        const std::size_t units = static_cast<std::size_t>(tokens) * qsa::kKvHeads * splits * qsa::kGroup;
        part_o  = static_cast<float*>(workspace);
        part_ml = part_o + units * kD;
    }
    if (cache.format == QsaKvFormat::K8V4) {
        launch<QsaKvFormat::K8V4>(cache, q, qkv, positions, selected, counts, tokens, splits,
                                  part_o, part_ml, out, stream);
    } else if (cache.format == QsaKvFormat::K8V8) {
        launch<QsaKvFormat::K8V8>(cache, q, qkv, positions, selected, counts, tokens, splits,
                                  part_o, part_ml, out, stream);
    } else {
        launch<QsaKvFormat::BF16>(cache, q, qkv, positions, selected, counts, tokens, splits,
                                  part_o, part_ml, out, stream);
    }
}

}  // namespace fngine::kernels
