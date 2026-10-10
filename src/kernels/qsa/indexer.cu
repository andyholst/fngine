// QSA indexer: query/block-key preparation with a raw-key ring, block scoring (bf16 MMA) and
// exact top-512 block selection (radix select, ties -> lower block id).
#include "kernels/qsa/qsa.h"

#include "core/device.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/common/score_id_order.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/qsa/qsa_common.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

// ---------------------------------------------------------------------------------------------
// Prepare
// ---------------------------------------------------------------------------------------------
constexpr int kPrepareWarps        = 8;
constexpr int kSingleCtaMaxTokens  = 16;
constexpr int kIdxRotaryPairs      = 16;                // lanes 0..15: dims 0..31; 16..31: 32..63

// RMSNorm (eps 1e-6, FP32 multiplier) of one 128-wide row held as pairs (lane, lane + 32), BF16
// rounding, then NeoX RoPE of dims [0, 64) (the step-0 pairs) at `position`, BF16 rounding.
__device__ __forceinline__ void idx_norm_rope_row(float2 (&x)[2], const float2* weight,
                                                  std::int32_t position, int lane,
                                                  __nv_bfloat162 (&out)[2]) {
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < 2; ++s) sum = __fadd_rn(sum, __fmaf_rn(x[s].x, x[s].x, __fmul_rn(x[s].y, x[s].y)));
    sum             = warp_sum(sum);
    const float inv = rsqrtf(sum / static_cast<float>(qsa::kIdxDim) + qsa::kEps);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const float2 w = weight[lane + 32 * s];
        out[s]         = __floats2bfloat162_rn(x[s].x * inv * w.x, x[s].y * inv * w.y);
    }
    const int frequency = 2 * (lane & (kIdxRotaryPairs - 1));
    float s0, c0, s1, c1;
    qsa_rope_sincos(position, frequency, &s0, &c0);
    qsa_rope_sincos(position, frequency + 1, &s1, &c1);
    const float2 own     = __bfloat1622float2(out[0]);
    const float2 partner = make_float2(__shfl_xor_sync(kFullWarpMask, own.x, kIdxRotaryPairs),
                                       __shfl_xor_sync(kFullWarpMask, own.y, kIdxRotaryPairs));
    const bool first_half = lane < kIdxRotaryPairs;
    const float2 rx =
        rope_rotate_pair(first_half ? own.x : partner.x, first_half ? partner.x : own.x, c0, s0);
    const float2 ry =
        rope_rotate_pair(first_half ? own.y : partner.y, first_half ? partner.y : own.y, c1, s1);
    out[0] = first_half ? __floats2bfloat162_rn(rx.x, ry.x) : __floats2bfloat162_rn(rx.y, ry.y);
}

// Row `row` of the prepare work list: rows [0, 4T) are query rows (token-major, head-minor), rows
// [4T, 4T + blocks) are the batch's completed blocks starting at block p0 / 4.
__device__ __forceinline__ void prepare_row(int row, const __nv_bfloat16* __restrict__ idx,
                                            std::int32_t p0, std::int32_t tokens,
                                            const float2* __restrict__ q_norm,
                                            const float2* __restrict__ k_norm,
                                            const __nv_bfloat16* __restrict__ ring,
                                            __nv_bfloat16* __restrict__ block_keys,
                                            __nv_bfloat16* __restrict__ q_idx, int lane) {
    const int query_rows = tokens * qsa::kIdxHeads;
    if (row < query_rows) {
        const int token = row / qsa::kIdxHeads;
        const int head  = row % qsa::kIdxHeads;
        const auto* src = reinterpret_cast<const __nv_bfloat162*>(
            idx + static_cast<std::int64_t>(token) * qsa::kIdxLd + head * qsa::kIdxDim);
        float2 x[2] = {__bfloat1622float2(src[lane]), __bfloat1622float2(src[lane + 32])};
        __nv_bfloat162 out[2];
        idx_norm_rope_row(x, q_norm, p0 + token, lane, out);
        auto* dst = reinterpret_cast<__nv_bfloat162*>(q_idx + static_cast<std::int64_t>(row) *
                                                                  qsa::kIdxDim);
        dst[lane]      = out[0];
        dst[lane + 32] = out[1];
        return;
    }
    const int block       = p0 / qsa::kBlock + (row - query_rows);
    const int block_end   = (p0 + tokens) / qsa::kBlock;
    if (block >= block_end) return;
    float2 sum[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
#pragma unroll
    for (int j = 0; j < qsa::kBlock; ++j) {
        const int position = block * qsa::kBlock + j;
        const __nv_bfloat16* raw =
            position >= p0 ? idx + static_cast<std::int64_t>(position - p0) * qsa::kIdxLd +
                                 qsa::kIdxHeads * qsa::kIdxDim
                           : ring + (position % qsa::kRing) * qsa::kIdxDim;
        const auto* raw2 = reinterpret_cast<const __nv_bfloat162*>(raw);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const float2 v = __bfloat1622float2(raw2[lane + 32 * s]);
            sum[s].x       = __fadd_rn(sum[s].x, v.x);
            sum[s].y       = __fadd_rn(sum[s].y, v.y);
        }
    }
    // mean in FP32, rounded to BF16 (HF: key_groups.float().mean(1).to(bf16)).
    float2 x[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        x[s] = __bfloat1622float2(__floats2bfloat162_rn(sum[s].x * 0.25f, sum[s].y * 0.25f));
    }
    __nv_bfloat162 out[2];
    idx_norm_rope_row(x, k_norm, block * qsa::kBlock, lane, out);
    auto* dst = reinterpret_cast<__nv_bfloat162*>(block_keys +
                                                  static_cast<std::int64_t>(block) * qsa::kIdxDim);
    dst[lane]      = out[0];
    dst[lane + 32] = out[1];
}

// Ring update: slot p % 16 <- raw key of p for the batch's last min(T, 16) positions.
// 16 threads copy one 256-byte row.
__device__ __forceinline__ void ring_update(const __nv_bfloat16* __restrict__ idx, std::int32_t p0,
                                            std::int32_t tokens, __nv_bfloat16* __restrict__ ring,
                                            int thread, int threads) {
    const int first = max(0, tokens - qsa::kRing);
    const int units = (tokens - first) * 16;
    for (int u = thread; u < units; u += threads) {
        const int token = first + u / 16;
        const int chunk = u % 16;
        const int4 value = load_vec<int4>(idx + static_cast<std::int64_t>(token) * qsa::kIdxLd +
                                          qsa::kIdxHeads * qsa::kIdxDim + chunk * 8);
        store_vec(ring + ((p0 + token) % qsa::kRing) * qsa::kIdxDim + chunk * 8, value);
    }
}

// SingleCta: one CTA processes every row, then (after a barrier that orders the ring reads before
// the writes) updates the ring. Otherwise every CTA processes kPrepareWarps rows and the ring is
// updated by indexer_ring_update_kernel afterwards. Programmatic dependents (pdl.cuh): they wait
// before touching anything, and read idx through coherent loads.
template <bool SingleCta>
__global__ __launch_bounds__(kPrepareWarps * 32) void indexer_prepare_kernel(
    const __nv_bfloat16* __restrict__ idx, const std::int32_t* __restrict__ positions,
    std::int32_t tokens, const float2* __restrict__ q_norm, const float2* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ ring, __nv_bfloat16* __restrict__ block_keys,
    __nv_bfloat16* __restrict__ q_idx) {
    pdl_trigger();
    pdl_wait();
    idx                 = pdl_coherent(idx);
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t p0 = positions[0];
    const int rows      = tokens * qsa::kIdxHeads + (p0 + tokens) / qsa::kBlock - p0 / qsa::kBlock;
    if constexpr (SingleCta) {
        for (int row = warp; row < rows; row += kPrepareWarps) {
            prepare_row(row, idx, p0, tokens, q_norm, k_norm, ring, block_keys, q_idx, lane);
        }
        __syncthreads();
        ring_update(idx, p0, tokens, ring, static_cast<int>(threadIdx.x), kPrepareWarps * 32);
    } else {
        const int row = static_cast<int>(blockIdx.x) * kPrepareWarps + warp;
        if (row < rows) {
            prepare_row(row, idx, p0, tokens, q_norm, k_norm, ring, block_keys, q_idx, lane);
        }
    }
}

__global__ __launch_bounds__(256) void indexer_ring_update_kernel(
    const __nv_bfloat16* __restrict__ idx, const std::int32_t* __restrict__ positions,
    std::int32_t tokens, __nv_bfloat16* __restrict__ ring) {
    pdl_trigger();
    pdl_wait();
    ring_update(pdl_coherent(idx), positions[0], tokens, ring, static_cast<int>(threadIdx.x), 256);
}

// ---------------------------------------------------------------------------------------------
// Score: scores[t][b] = sum_h relu(q[t][h] . k[b]) / sqrt(128), FP32 from bf16 m16n8k16 MMA.
// One CTA = 4 warps = 16 query tokens (warp w: tokens 4w..4w+3, its 16 MMA rows are the 16
// contiguous q_idx rows token*4 + head) x `tiles_per_cta` tiles of 64 blocks, double buffered.
// ---------------------------------------------------------------------------------------------
constexpr int kScoreWarps      = 4;
constexpr int kScoreTokens     = 16;
constexpr int kScoreTileBlocks = 64;
constexpr int kScoreStride     = qsa::kIdxDim + 8;  // halves; 272-byte rows avoid ldmatrix conflicts
constexpr int kQueryChunk      = 256;
constexpr float kScoreDivisor  = 11.313708498984761f;  // float(sqrt(128))

// Scores workspace row stride: padded to 4 floats so the select kernel can load float4.
constexpr int score_ld(int max_blocks) { return (max_blocks + 3) & ~3; }

__device__ __forceinline__ void score_load_tile(__nv_bfloat16* tile,
                                                const __nv_bfloat16* __restrict__ block_keys,
                                                int tile_begin, int valid_blocks, int thread) {
    constexpr int kChunks = kScoreTileBlocks * qsa::kIdxDim / 8;  // 16-byte chunks
#pragma unroll
    for (int i = 0; i < kChunks / (kScoreWarps * 32); ++i) {
        const int chunk = thread + i * kScoreWarps * 32;
        const int row   = chunk / (qsa::kIdxDim / 8);
        const int col   = (chunk % (qsa::kIdxDim / 8)) * 8;
        const int block = tile_begin + row;
        const bool ok   = block < valid_blocks;
        const __nv_bfloat16* src =
            block_keys + static_cast<std::int64_t>(ok ? block : 0) * qsa::kIdxDim + col;
        cp_async_zfill<16, Cache::cg>(tile + row * kScoreStride + col, src, ok ? 16 : 0);
    }
}

__global__ __launch_bounds__(kScoreWarps * 32) void indexer_score_kernel(
    const __nv_bfloat16* __restrict__ q_idx, const __nv_bfloat16* __restrict__ block_keys,
    const std::int32_t* __restrict__ positions, std::int32_t tokens, std::int32_t max_blocks,
    std::int32_t tiles_per_cta, float* __restrict__ scores, std::int32_t ld) {
    __shared__ alignas(16) __nv_bfloat16 tiles[2][kScoreTileBlocks * kScoreStride];
    pdl_trigger();
    pdl_wait();
    q_idx                = pdl_coherent(q_idx);
    block_keys           = pdl_coherent(block_keys);
    const int thread     = static_cast<int>(threadIdx.x);
    const int warp       = thread >> 5;
    const int lane       = thread & 31;
    const int token_base = static_cast<int>(blockIdx.y) * kScoreTokens;

    // Highest block count among this CTA's queries; all-dense CTAs have nothing to score.
    const int last_token = min(tokens, token_base + kScoreTokens) - 1;
    const int nb_max     = min(max_blocks, (positions[last_token] + 1) / qsa::kBlock);
    if (nb_max <= qsa::kTopBlocks) return;
    const int tile_first = static_cast<int>(blockIdx.x) * tiles_per_cta;
    const int tile_last  = min(tile_first + tiles_per_cta, (nb_max + kScoreTileBlocks - 1) /
                                                                kScoreTileBlocks);
    if (tile_first >= tile_last) return;

    // A fragments (16 rows x 128) straight from global: rows token*4 + head are contiguous.
    const int warp_token = token_base + warp * 4;
    const bool warp_active = warp_token < tokens;
    const int g = lane >> 2;
    const int c = lane & 3;
    unsigned a[8][4];
    {
        const int row0   = warp_token * qsa::kIdxHeads + g;
        const int row1   = row0 + 8;
        const bool ok0   = row0 < tokens * qsa::kIdxHeads;
        const bool ok1   = row1 < tokens * qsa::kIdxHeads;
        const auto* q0   = reinterpret_cast<const unsigned*>(q_idx + static_cast<std::int64_t>(row0) * qsa::kIdxDim);
        const auto* q1   = reinterpret_cast<const unsigned*>(q_idx + static_cast<std::int64_t>(row1) * qsa::kIdxDim);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int col = ks * 8 + c;  // in bf16 pairs
            a[ks][0] = ok0 ? q0[col] : 0u;
            a[ks][1] = ok1 ? q1[col] : 0u;
            a[ks][2] = ok0 ? q0[col + 4] : 0u;
            a[ks][3] = ok1 ? q1[col + 4] : 0u;
        }
    }

    score_load_tile(tiles[0], block_keys, tile_first * kScoreTileBlocks, nb_max, thread);
    cp_commit();
    for (int tile = tile_first; tile < tile_last; ++tile) {
        const int stage = (tile - tile_first) & 1;
        if (tile + 1 < tile_last) {
            score_load_tile(tiles[stage ^ 1], block_keys, (tile + 1) * kScoreTileBlocks, nb_max,
                            thread);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (warp_active) {
            const __nv_bfloat16* t = tiles[stage];
#pragma unroll
            for (int np = 0; np < kScoreTileBlocks / 16; ++np) {  // pairs of n-tiles
                float acc[2][4] = {};
#pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int col = ks * 16 + ((lane >> 3) & 1) * 8;
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3, smem_addr(t + key * kScoreStride + col));
                    mma_bf16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a[ks][0], a[ks][1],
                             a[ks][2], a[ks][3], b0, b1);
                    mma_bf16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a[ks][0], a[ks][1],
                             a[ks][2], a[ks][3], b2, b3);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        v[i] = fmaxf(acc[h][i], 0.0f);
                        v[i] += __shfl_xor_sync(kFullWarpMask, v[i], 4);
                        v[i] += __shfl_xor_sync(kFullWarpMask, v[i], 8);
                    }
                    if ((lane & 12) == 0) {
                        const int block = tile * kScoreTileBlocks + np * 16 + h * 8 + 2 * c;
                        const int t0    = warp_token + (lane >> 4);
                        const int t1    = t0 + 2;
                        if (block < nb_max) {  // nb_max is even-safe: guard each column
                            const bool two = block + 1 < nb_max;
                            if (t0 < tokens) {
                                float* dst = scores + static_cast<std::int64_t>(t0) * ld + block;
                                dst[0] = __fdiv_rn(v[0], kScoreDivisor);
                                if (two) dst[1] = __fdiv_rn(v[1], kScoreDivisor);
                            }
                            if (t1 < tokens) {
                                float* dst = scores + static_cast<std::int64_t>(t1) * ld + block;
                                dst[0] = __fdiv_rn(v[2], kScoreDivisor);
                                if (two) dst[1] = __fdiv_rn(v[3], kScoreDivisor);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Select: one CTA per query; exact top-512 of the (score, -block id) keys, emitted ascending.
// A single CTA walking ~25K scores several times is instruction bound, so the common path touches
// every score once: 512 evenly spaced samples give a conservative threshold t (the r-th largest
// sample, r sized so ~2x512 scores are expected at or above it), one pass compacts the scores >= t
// into shared memory, and the exact radix select, tie-break and ordering run on those candidates.
// Whenever at least 512 scores are >= t the top 512 are all candidates, so the result is exact;
// if fewer than 512 land there, a warp's 256-slot segment overflows, or nb > 65536 (contexts above
// 256K), the CTA falls back to a radix select over all scores with an order-preserving emission.
// ---------------------------------------------------------------------------------------------
constexpr int kSelectThreads = 512;
constexpr int kSelectWarps   = kSelectThreads / 32;
constexpr int kCandidates    = 4096;
constexpr int kSelectBatch   = 8;  // independent loads in flight per thread

constexpr int kWarpCandidates = kCandidates / kSelectWarps;  // per-warp compaction segment
constexpr int kBitmapWords    = 2048;  // ascending emission via a block bitmap: nb <= 65536

struct SelectShared {
    int histogram[256];
    int digit[3];
    std::uint32_t sample[kSelectThreads];
    std::uint32_t cand_key[kCandidates];  // warp w fills [w*256, w*256 + n_w); the rest key 0
    int cand_id[kCandidates];
    unsigned bitmap[kBitmapWords];
    int warp_count[kSelectWarps + 1];
    int warp_gt[kSelectWarps];
    int warp_eq[kSelectWarps];
};

// Result of an MSB-first radix select for the `want`-th largest of n keys: keys whose masked value
// exceeds `prefix` are above it; `want` of the `bucket` keys whose masked value equals `prefix`
// complete the top set (want == bucket when the bucket is taken whole and the select stopped early).
struct RadixResult {
    std::uint32_t prefix;
    std::uint32_t mask;
    int want;
    int bucket;
};

// `visit(fn)` must call fn(key, valid) for every key assigned to the calling thread, the same
// number of times on every lane of a warp (the histogram update uses warp collectives).
template <class Visit>
__device__ RadixResult radix_select(int want, Visit visit, SelectShared& sh) {
    const int thread = static_cast<int>(threadIdx.x);
    const int lane   = thread & 31;
    RadixResult r{0u, 0u, want, 0};
    for (int i = thread; i < 256; i += kSelectThreads) sh.histogram[i] = 0;
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        visit([&](std::uint32_t key, bool valid) {
            const bool match      = valid && (key & r.mask) == r.prefix;
            const unsigned active = __ballot_sync(kFullWarpMask, match);
            if (match) {
                const int digit      = static_cast<int>((key >> shift) & 255u);
                const unsigned peers = __match_any_sync(active, digit);
                if ((__ffs(peers) - 1) == lane) atomicAdd(&sh.histogram[digit], __popc(peers));
            }
        });
        __syncthreads();
        // Warp 0 finds the digit holding the want-th largest key (lane l owns digits 255-8l ..
        // 248-8l, so its exclusive prefix counts the keys in larger digits) and clears the
        // histogram for the next pass.
        if (thread < 32) {
            int h[8];
            int local = 0;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                h[b] = sh.histogram[255 - (lane * 8 + b)];
                local += h[b];
            }
            int inclusive = local;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int v = __shfl_up_sync(kFullWarpMask, inclusive, offset);
                if (lane >= offset) inclusive += v;
            }
            int greater = inclusive - local;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (greater < r.want && greater + h[b] >= r.want) {
                    sh.digit[0] = 255 - (lane * 8 + b);
                    sh.digit[1] = greater;
                    sh.digit[2] = h[b];
                }
                greater += h[b];
            }
#pragma unroll
            for (int b = 0; b < 8; ++b) sh.histogram[255 - (lane * 8 + b)] = 0;
        }
        __syncthreads();
        const int digit = sh.digit[0];
        r.want -= sh.digit[1];
        r.bucket = sh.digit[2];
        r.prefix |= static_cast<std::uint32_t>(digit) << shift;
        r.mask |= 255u << shift;
        if (r.bucket == r.want) break;  // the whole bucket is taken
    }
    return r;
}

__global__ __launch_bounds__(kSelectThreads) void indexer_select_kernel(
    const float* __restrict__ scores, std::int32_t max_blocks, std::int32_t ld,
    const std::int32_t* __restrict__ positions, std::int32_t* __restrict__ selected,
    std::int32_t* __restrict__ counts) {
    static_assert(kSelectThreads * 4 == kBitmapWords, "four bitmap words per thread");
    __shared__ SelectShared sh;
    pdl_trigger();
    pdl_wait();
    scores           = pdl_coherent(scores);
    const int thread = static_cast<int>(threadIdx.x);
    const int lane   = thread & 31;
    const int warp   = thread >> 5;
    const int token  = static_cast<int>(blockIdx.x);
    const int nb     = min(max_blocks, (positions[token] + 1) / qsa::kBlock);
    std::int32_t* out = selected + static_cast<std::int64_t>(token) * qsa::kTopBlocks;
    if (nb <= qsa::kTopBlocks) {
        for (int i = thread; i < nb; i += kSelectThreads) out[i] = i;
        if (thread == 0) counts[token] = nb;
        return;
    }
    if (thread == 0) counts[token] = qsa::kTopBlocks;
    const float* row  = scores + static_cast<std::int64_t>(token) * ld;  // 16-byte aligned rows
    auto global_key   = [&](int i) -> std::uint32_t { return ordered_score_bits(row[i]); };
    const unsigned lower_lanes = (1u << lane) - 1u;

    // ---- Threshold from samples (every score is a candidate when they all fit).
    std::uint32_t threshold = 0;
    if (nb > kCandidates) {
        sh.sample[thread] = global_key(static_cast<int>((static_cast<std::int64_t>(thread) * nb) / kSelectThreads));
        __syncthreads();
        const int rank = min(kSelectThreads, static_cast<int>((2LL * qsa::kTopBlocks * kSelectThreads + nb - 1) / nb) + 8);
        threshold = radix_select(rank, [&](auto&& fn) { fn(sh.sample[thread], true); }, sh).prefix;
    }

    // ---- Compact the candidates (key >= threshold) into per-warp segments: no shared atomics.
    // Unused slots keep key 0, below every score key (scores are >= +0, key >= 0x80000000), so
    // with >= 512 real candidates they can never reach the selection.
    for (int i = thread; i < kCandidates; i += kSelectThreads) sh.cand_key[i] = 0u;
    __syncthreads();
    // Rows are padded to a multiple of 4 floats: lane loads 4 consecutive scores per access.
    int warp_found = 0;
    const int vectors = (nb + 3) / 4;
    for (int base = warp * 32; base < vectors; base += kSelectThreads * kSelectBatch) {
        float4 v[kSelectBatch];
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int vi = base + j * kSelectThreads + lane;
            v[j]         = vi < vectors ? load_vec<float4>(row + 4 * vi) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int vi       = base + j * kSelectThreads + lane;
            const float e[4]   = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i            = 4 * vi + k;
                const std::uint32_t kv = ordered_score_bits(e[k]);
                const bool take        = i < nb && kv >= threshold;
                const unsigned bits    = __ballot_sync(kFullWarpMask, take);
                const int slot         = warp_found + __popc(bits & lower_lanes);
                if (take && slot < kWarpCandidates) {
                    sh.cand_key[warp * kWarpCandidates + slot] = kv;
                    sh.cand_id[warp * kWarpCandidates + slot]  = i;
                }
                warp_found += __popc(bits);
            }
        }
    }
    if (lane == 0) sh.warp_count[warp] = warp_found;
    __syncthreads();
    int candidates = 0;
    bool overflow  = false;
    for (int w = 0; w < kSelectWarps; ++w) {
        candidates += sh.warp_count[w];
        overflow |= sh.warp_count[w] > kWarpCandidates;
    }

    if (candidates >= qsa::kTopBlocks && !overflow && nb <= kBitmapWords * 32) {
        // Each warp histograms its own segment's valid candidates.
        const int own = sh.warp_count[warp];
        const RadixResult r = radix_select(
            qsa::kTopBlocks,
            [&](auto&& fn) {
                for (int base = 0; base < own; base += 32) {
                    const int i = base + lane;
                    fn(i < own ? sh.cand_key[warp * kWarpCandidates + i] : 0u, i < own);
                }
            },
            sh);
        for (int i = thread; i < kBitmapWords; i += kSelectThreads) sh.bitmap[i] = 0u;
        __syncthreads();
        for (int j = lane; j < own; j += 32) {
            const int c             = warp * kWarpCandidates + j;
            const std::uint32_t key = sh.cand_key[c] & r.mask;
            bool take               = key > r.prefix;
            if (key == r.prefix) {
                if (r.want == r.bucket) {
                    take = true;
                } else {  // exact score ties: the lowest block ids win
                    const int id = sh.cand_id[c];
                    int rank     = 0;
                    for (int o = 0; o < kCandidates; ++o) {
                        rank += (sh.cand_key[o] & r.mask) == r.prefix && sh.cand_id[o] < id;
                    }
                    take = rank < r.want;
                }
            }
            if (take) atomicOr(&sh.bitmap[sh.cand_id[c] >> 5], 1u << (sh.cand_id[c] & 31));
        }
        __syncthreads();
        // Ascending emission: thread t owns bitmap words [4t, 4t + 4); an exclusive scan of the
        // per-thread bit counts gives each thread its first output slot.
        constexpr int kWordsPerThread = kBitmapWords / kSelectThreads;
        unsigned words[kWordsPerThread];
        int bits = 0;
#pragma unroll
        for (int k = 0; k < kWordsPerThread; ++k) {
            words[k] = sh.bitmap[thread * kWordsPerThread + k];
            bits += __popc(words[k]);
        }
        int inclusive = bits;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int v = __shfl_up_sync(kFullWarpMask, inclusive, offset);
            if (lane >= offset) inclusive += v;
        }
        if (lane == 31) sh.warp_count[warp] = inclusive;
        __syncthreads();
        int slot = inclusive - bits;
        for (int w = 0; w < warp; ++w) slot += sh.warp_count[w];
#pragma unroll
        for (int k = 0; k < kWordsPerThread; ++k) {
            unsigned word = words[k];
            while (word != 0u) {
                const int b = __ffs(word) - 1;
                out[slot++] = (thread * kWordsPerThread + k) * 32 + b;
                word &= word - 1u;
            }
        }
        return;
    }

    // ---- Fallback: radix select over every score, then an order-preserving emission. Warp w owns
    // the contiguous range [w*segment, (w+1)*segment); per-warp totals give each warp its output
    // offset and equal-rank base, then warps emit with ballots only.
    const RadixResult r = radix_select(
        qsa::kTopBlocks,
        [&](auto&& fn) {
            for (int base = 0; base < nb; base += kSelectThreads * kSelectBatch) {
                std::uint32_t keys[kSelectBatch];
#pragma unroll
                for (int j = 0; j < kSelectBatch; ++j) {
                    const int i = base + j * kSelectThreads + thread;
                    keys[j]     = i < nb ? global_key(i) : 0u;
                }
#pragma unroll
                for (int j = 0; j < kSelectBatch; ++j) fn(keys[j], base + j * kSelectThreads + thread < nb);
            }
        },
        sh);
    const int segment = (((nb + kSelectWarps - 1) / kSelectWarps) + 31) & ~31;
    const int begin   = warp * segment;
    const int end     = min(nb, begin + segment);
    int gt = 0, eq = 0;
    for (int base = begin; base < end; base += 32 * kSelectBatch) {
        std::uint32_t keys[kSelectBatch];
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int i = base + 32 * j + lane;
            keys[j]     = i < end ? (global_key(i) & r.mask) : 0u;
        }
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int i = base + 32 * j + lane;
            gt += __popc(__ballot_sync(kFullWarpMask, i < end && keys[j] > r.prefix));
            eq += __popc(__ballot_sync(kFullWarpMask, i < end && keys[j] == r.prefix));
        }
    }
    if (lane == 0) {
        sh.warp_gt[warp] = gt;
        sh.warp_eq[warp] = eq;
    }
    __syncthreads();
    int eq_rank = 0, slot = 0;
    for (int w = 0; w < warp; ++w) {
        slot += sh.warp_gt[w] + min(sh.warp_eq[w], max(0, r.want - eq_rank));
        eq_rank += sh.warp_eq[w];
    }
    for (int base = begin; base < end; base += 32 * kSelectBatch) {
        std::uint32_t keys[kSelectBatch];
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int i = base + 32 * j + lane;
            keys[j]     = i < end ? (global_key(i) & r.mask) : 0u;
        }
#pragma unroll
        for (int j = 0; j < kSelectBatch; ++j) {
            const int i              = base + 32 * j + lane;
            const bool is_eq         = i < end && keys[j] == r.prefix;
            const unsigned eq_bits   = __ballot_sync(kFullWarpMask, is_eq);
            const bool take          = i < end && (keys[j] > r.prefix ||
                                          (is_eq && eq_rank + __popc(eq_bits & lower_lanes) < r.want));
            const unsigned take_bits = __ballot_sync(kFullWarpMask, take);
            if (take) out[slot + __popc(take_bits & lower_lanes)] = i;
            eq_rank += __popc(eq_bits);
            slot += __popc(take_bits);
        }
    }
}

}  // namespace

size_t indexer_state_bytes(int32_t capacity_tokens) {
    const std::size_t blocks = static_cast<std::size_t>(capacity_tokens) / qsa::kBlock;
    return (qsa::kRing + blocks) * qsa::kIdxDim * sizeof(__nv_bfloat16);
}

void indexer_prepare(const __nv_bfloat16* idx, const int32_t* positions, int32_t tokens,
                     const float* q_norm, const float* k_norm, __nv_bfloat16* ring,
                     __nv_bfloat16* block_keys, __nv_bfloat16* q_idx, cudaStream_t stream) {
    if (tokens <= 0) return;
    const auto* qn = reinterpret_cast<const float2*>(q_norm);
    const auto* kn = reinterpret_cast<const float2*>(k_norm);
    if (tokens <= kSingleCtaMaxTokens) {
        CUDA_CHECK(pdl_launch(indexer_prepare_kernel<true>, dim3(1), dim3(kPrepareWarps * 32), 0, stream,
                              idx, positions, tokens, qn, kn, ring, block_keys, q_idx));
    } else {
        const int max_rows = tokens * qsa::kIdxHeads + tokens / qsa::kBlock + 1;
        const int grid     = (max_rows + kPrepareWarps - 1) / kPrepareWarps;
        CUDA_CHECK(pdl_launch(indexer_prepare_kernel<false>, dim3(grid), dim3(kPrepareWarps * 32), 0,
                              stream, idx, positions, tokens, qn, kn, ring, block_keys, q_idx));
        CUDA_CHECK(pdl_launch(indexer_ring_update_kernel, dim3(1), dim3(256), 0, stream, idx,
                              positions, tokens, ring));
    }
    CUDA_CHECK(cudaGetLastError());
}

size_t indexer_score_workspace_bytes(int32_t tokens, int32_t max_blocks) {
    const std::size_t rows = static_cast<std::size_t>(std::min(tokens, kQueryChunk));
    return rows * static_cast<std::size_t>(score_ld(std::max(max_blocks, 0))) * sizeof(float);
}

void indexer_score_topk(const __nv_bfloat16* q_idx, const __nv_bfloat16* block_keys,
                        const int32_t* positions, int32_t tokens, int32_t max_blocks,
                        void* workspace, size_t workspace_bytes, int32_t* selected,
                        int32_t* counts, cudaStream_t stream) {
    if (tokens <= 0) return;
    if (max_blocks < 0) throw std::invalid_argument("indexer_score_topk: negative max_blocks");
    const bool need_scores = max_blocks > qsa::kTopBlocks;
    if (need_scores && workspace_bytes < indexer_score_workspace_bytes(tokens, max_blocks)) {
        throw std::invalid_argument("indexer_score_topk: workspace too small");
    }
    auto* scores          = static_cast<float*>(workspace);
    const int ld          = score_ld(max_blocks);
    const int total_tiles = (max_blocks + kScoreTileBlocks - 1) / kScoreTileBlocks;
    for (int begin = 0; begin < tokens; begin += kQueryChunk) {
        const int chunk = std::min(kQueryChunk, tokens - begin);
        if (need_scores) {
            const int tiles_per_cta = chunk <= kScoreTokens ? 2 : 8;
            const dim3 grid(static_cast<unsigned>((total_tiles + tiles_per_cta - 1) / tiles_per_cta),
                            static_cast<unsigned>((chunk + kScoreTokens - 1) / kScoreTokens));
            CUDA_CHECK(pdl_launch(indexer_score_kernel, grid, dim3(kScoreWarps * 32), 0, stream,
                                  static_cast<const __nv_bfloat16*>(
                                      q_idx + static_cast<std::int64_t>(begin) * qsa::kIdxHeads * qsa::kIdxDim),
                                  static_cast<const __nv_bfloat16*>(block_keys), positions + begin, chunk,
                                  max_blocks, tiles_per_cta, scores, ld));
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(pdl_launch(indexer_select_kernel, dim3(chunk), dim3(kSelectThreads), 0, stream,
                              static_cast<const float*>(scores), max_blocks, ld, positions + begin,
                              selected + static_cast<std::int64_t>(begin) * qsa::kTopBlocks, counts + begin));
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace fngine::kernels
