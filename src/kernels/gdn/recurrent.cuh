// Modified for fngine from cinference:src/ops/linear_attention/gated_delta_net/recurrent.cuh @ 383e5db. See NOTICE.
//
// fngine changes: single sequence, fixed 16/48 head geometry, strided q/k/v views into the conv
// output, device-resident slot/commit control, and one staged kernel shared by the T=1 decode,
// the verify record and the fold so that their arithmetic is identical by construction. Every
// multiply-add in the recurrence is an explicit fmaf, so no path depends on compiler contraction.

#pragma once

#include "kernels/common/pdl.cuh"
#include "kernels/gdn/common.cuh"
#include "kernels/gdn/gdn.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels::gdn {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;
inline constexpr int kStateTiles = kStateDim / kBlockDv;
inline constexpr int kGroup      = kValueHeadCount / kQkHeadCount;
inline constexpr int kMaxWidth   = gdn_dims::kMaxRecordWidth;

inline constexpr float kQkL2NormEps = 1.0e-6f;
inline constexpr float kOutputScale = 0.08838834764831845f; // 1/sqrt(128)

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateTiles == 8);
static_assert(kGroup == 3);

using gdn_dims::kConvChannels;
using gdn_dims::kLayers;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

// Lane partial of a 128-wide head's sum of squares.
__device__ __forceinline__ float lane_sumsq(const float (&value)[kQkPerLane]) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kQkPerLane; ++i) { sum = fmaf(value[i], value[i], sum); }
    return sum;
}

// Normalization scale of one head, formed on lane 0 and broadcast (all paths use this form).
__device__ __forceinline__ float head_l2_scale(float lane_partial, int lane) {
    const float sum = warp_reduce_sum(lane_partial);
    float inv       = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
    return __shfl_sync(kFullWarpMask, inv, 0);
}

__device__ __forceinline__ void row_partials(const float (&state)[kDvPerWarp][kQkPerLane],
                                             const float (&vec)[kQkPerLane],
                                             float (&partial)[kDvPerWarp]) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        partial[r] = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial[r] = fmaf(state[r][c], vec[c], partial[r]); }
    }
}

// Sums the lane partials of a warp's four state rows with the pairwise tree of four warp_sum
// butterflies (offsets 16, 8, 4, 2, 1), exchanging only the rows a lane keeps: two rows at offset
// 16, one at offset 8, then one row to the end. Floating-point addition is commutative, so each
// row sum equals its warp_sum result bit for bit, with six shuffles instead of twenty. Returns the
// sum of row (lane >> 3) & 3, which lanes 8r..8r+7 hold for row r.
__device__ __forceinline__ float gdn_row_sums(const float (&partial)[kDvPerWarp], int lane) {
    static_assert(kDvPerWarp == 4 && kWarpSize == 32);
    const bool upper16 = (lane & 16) != 0;
    const float pair0  = (upper16 ? partial[2] : partial[0]) +
                         __shfl_xor_sync(kFullWarpMask, upper16 ? partial[0] : partial[2], 16);
    const float pair1  = (upper16 ? partial[3] : partial[1]) +
                         __shfl_xor_sync(kFullWarpMask, upper16 ? partial[1] : partial[3], 16);
    const bool upper8  = (lane & 8) != 0;
    float sum =
        (upper8 ? pair1 : pair0) + __shfl_xor_sync(kFullWarpMask, upper8 ? pair0 : pair1, 8);
    sum += __shfl_xor_sync(kFullWarpMask, sum, 4);
    sum += __shfl_xor_sync(kFullWarpMask, sum, 2);
    sum += __shfl_xor_sync(kFullWarpMask, sum, 1);
    return sum;
}

__device__ __forceinline__ float gdn_delta(float beta, float value, float alpha, float projection) {
    return beta * fmaf(-alpha, projection, value);
}

__device__ __forceinline__ float gdn_update(float alpha, float state, float delta, float key) {
    return fmaf(delta, key, alpha * state);
}

__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const float* head_base, std::uint32_t dv_base,
                                                std::uint32_t dqk_base) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_vec(state[r], load_vec<float4>(head_base +
                                             static_cast<std::int64_t>(dv_base + r) * kStateDim +
                                             dqk_base));
    }
}

__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 float* head_base, std::uint32_t dv_base,
                                                 std::uint32_t dqk_base) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_vec(head_base + static_cast<std::int64_t>(dv_base + r) * kStateDim + dqk_base,
                  load_vec<float4>(state[r]));
    }
}

__device__ __forceinline__ std::int64_t state_layer_offset(std::int32_t slot, std::int32_t layer) {
    return (static_cast<std::int64_t>(slot) * kLayers + layer) *
           static_cast<std::int64_t>(gdn_dims::kRecurrentLayerElems);
}

__device__ __forceinline__ std::int64_t conv_layer_offset(std::int32_t slot, std::int32_t layer) {
    return (static_cast<std::int64_t>(slot) * kLayers + layer) *
           static_cast<std::int64_t>(gdn_dims::kConvLayerElems);
}

// Record column of (layer, token) in every record plane.
__device__ __forceinline__ std::int64_t record_column(std::int32_t layer, std::int32_t token) {
    return static_cast<std::int64_t>(layer) * kMaxWidth + token;
}

// ---------------------------------------------------------------------------------------------
// Staged form (decode T=1, verify record T<=16, fold). One CTA owns one value head and a 16-row
// state tile (4 warps x 4 rows, each lane 4 key columns). The CTA copies its whole raw block into
// shared memory with independent asynchronous copies, forms each Q/K column's normalization scale
// once, and runs the per-token transition from shared memory.

enum class StagedMode { Decode, Record, Fold };

struct StagedParams {
    // Decode/Record sources: q/k/v views with a shared row stride, g/beta [T][48].
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    std::int64_t qkv_ld;
    const float* g;
    const float* beta;
    // Records (written by Record, read by Fold).
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    const __nv_bfloat16* conv_record;
    // State pool.
    float* recurrent;
    __nv_bfloat16* conv_state;
    const GdnStepControl* control;
    __nv_bfloat16* out;
    std::int32_t layer; // Decode/Record
    std::int32_t width;
};

// Fold: conv history of the committed prefix, tail3(source history ‖ record[0:commit]).
__device__ __forceinline__ void fold_conv_history(const StagedParams& p, std::int32_t layer,
                                                  std::int32_t tile_block, int tid,
                                                  const GdnStepControl& control,
                                                  std::int32_t commit) {
    if (tile_block >= kConvChannels / 128) { return; }
    const std::int32_t channel = tile_block * 128 + tid;
    const __nv_bfloat16* source =
        p.conv_state + conv_layer_offset(control.source_slot, layer) + channel;
    __nv_bfloat16* destination =
        p.conv_state + conv_layer_offset(control.destination_slot, layer) + channel;
    const __nv_bfloat16* record = p.conv_record + record_column(layer, 0) * kConvChannels + channel;
    const auto tap              = [&](std::int32_t token) {
        return record[static_cast<std::int64_t>(token) * kConvChannels];
    };
    __nv_bfloat16 h0;
    __nv_bfloat16 h1;
    __nv_bfloat16 h2;
    if (commit == 0) {
        h0 = source[0];
        h1 = source[kConvChannels];
        h2 = source[2 * kConvChannels];
    } else if (commit == 1) {
        h0 = source[kConvChannels];
        h1 = source[2 * kConvChannels];
        h2 = tap(0);
    } else if (commit == 2) {
        h0 = source[2 * kConvChannels];
        h1 = tap(0);
        h2 = tap(1);
    } else {
        h0 = tap(commit - 3);
        h1 = tap(commit - 2);
        h2 = tap(commit - 1);
    }
    destination[0]                 = h0;
    destination[kConvChannels]     = h1;
    destination[2 * kConvChannels] = h2;
}

// One tile (value head, z) of gdn_staged_kernel, z = state tile (layer * kStateTiles + state tile
// in a fold). `valid` columns of the step are applied.
template <StagedMode Mode>
__device__ __forceinline__ void staged_tile(const StagedParams& p, const GdnStepControl& control,
                                            std::int32_t valid, int tile) {
    constexpr bool kFold          = Mode == StagedMode::Fold;
    constexpr int kThreads        = kWarpSize * kNumWarps;
    constexpr int kPacksPerColumn = kStateDim / 8;

    const int lane                = static_cast<int>(threadIdx.x);
    const int warp                = static_cast<int>(threadIdx.y);
    const int thread              = warp * kWarpSize + lane;
    const std::uint32_t value_head = static_cast<std::uint32_t>(tile % kValueHeadCount);
    const std::int32_t z           = tile / kValueHeadCount;
    const std::uint32_t qk_head    = value_head / kGroup;
    const std::int32_t layer       = kFold ? z / kStateTiles : p.layer;
    const std::int32_t state_tile  = kFold ? z % kStateTiles : z;
    const std::uint32_t tile_dv  = static_cast<std::uint32_t>(state_tile * kBlockDv);
    const std::uint32_t dv_base  = tile_dv + static_cast<std::uint32_t>(warp * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);

    const std::int64_t head_offset =
        static_cast<std::int64_t>(value_head) * kStateDim * kStateDim;
    const float* state_read =
        p.recurrent + state_layer_offset(control.source_slot, layer) + head_offset;

    // Column sources. Fold reads the layer's records; Decode/Record read the strided views.
    const __nv_bfloat16* key_base;
    std::int64_t key_ld;
    const __nv_bfloat16* value_base;
    std::int64_t value_ld;
    if constexpr (kFold) {
        key_base   = p.key_record + record_column(layer, 0) * gdn_dims::kKeyDim +
                     static_cast<std::int64_t>(qk_head) * kStateDim;
        key_ld     = gdn_dims::kKeyDim;
        value_base = p.value_record + record_column(layer, 0) * gdn_dims::kValueDim +
                     static_cast<std::int64_t>(value_head) * kStateDim;
        value_ld   = gdn_dims::kValueDim;
    } else {
        key_base   = p.k + static_cast<std::int64_t>(qk_head) * kStateDim;
        key_ld     = p.qkv_ld;
        value_base = p.v + static_cast<std::int64_t>(value_head) * kStateDim;
        value_ld   = p.qkv_ld;
    }

    __shared__ __align__(16) __nv_bfloat16 key_raw[kMaxWidth][kStateDim];
    __shared__ __align__(16) __nv_bfloat16 query_raw[kFold ? 1 : kMaxWidth][kStateDim];
    __shared__ __align__(16) __nv_bfloat16 value_raw[kMaxWidth][kBlockDv];
    __shared__ float key_scale[kMaxWidth];
    __shared__ float query_scale[kMaxWidth];
    __shared__ float decay[kMaxWidth];
    __shared__ __align__(8) uint2 gate_raw[kMaxWidth];

    // The state tile is independent of the staged block, so its DRAM latency overlaps staging.
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, state_read, dv_base, dqk_base);

    for (int task = thread; task < valid * kPacksPerColumn; task += kThreads) {
        const int token = task / kPacksPerColumn;
        const int pack  = task - token * kPacksPerColumn;
        cp_async<16>(&key_raw[token][pack * 8], key_base + token * key_ld + pack * 8);
        if constexpr (!kFold) {
            cp_async<16>(&query_raw[token][pack * 8],
                         p.q + token * p.qkv_ld + static_cast<std::int64_t>(qk_head) * kStateDim +
                             pack * 8);
        }
    }
    for (int task = thread; task < valid * (kBlockDv / 8); task += kThreads) {
        const int token = task / (kBlockDv / 8);
        const int pack  = task - token * (kBlockDv / 8);
        cp_async<16>(&value_raw[token][pack * 8], value_base + token * value_ld + tile_dv + pack * 8);
    }
    for (int token = thread; token < valid; token += kThreads) {
        if constexpr (kFold) {
            cp_async<8>(&gate_raw[token],
                        p.gate_record + record_column(layer, token) * kValueHeadCount + value_head);
        } else {
            const std::int64_t offset = static_cast<std::int64_t>(token) * kValueHeadCount + value_head;
            cp_async<4>(&gate_raw[token].x, p.g + offset);
            cp_async<4>(&gate_raw[token].y, p.beta + offset);
        }
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // Each warp forms whole columns' scales.
    for (int token = warp; token < valid; token += kNumWarps) {
        const RawQkLane key = load_raw_qk_lane(key_raw[token], dqk_base);
        const float ks      = head_l2_scale(lane_sumsq(key.value), lane);
        float qs            = 0.0f;
        if constexpr (!kFold) {
            const RawQkLane query = load_raw_qk_lane(query_raw[token], dqk_base);
            qs                    = head_l2_scale(lane_sumsq(query.value), lane);
        }
        if (lane == 0) {
            key_scale[token]   = ks;
            query_scale[token] = qs;
            decay[token]       = expf(__uint_as_float(gate_raw[token].x));
        }
    }

    if constexpr (Mode == StagedMode::Record) {
        if (state_tile == 0 && value_head % kGroup == 0) {
            for (int task = thread; task < valid * kPacksPerColumn; task += kThreads) {
                const int token = task / kPacksPerColumn;
                const int pack  = task - token * kPacksPerColumn;
                __nv_bfloat16* destination =
                    p.key_record + record_column(layer, token) * gdn_dims::kKeyDim +
                    static_cast<std::int64_t>(qk_head) * kStateDim;
                store_vec(destination + pack * 8, load_vec<uint4>(&key_raw[token][pack * 8]));
            }
        }
        for (int task = thread; task < valid * (kBlockDv / 8); task += kThreads) {
            const int token = task / (kBlockDv / 8);
            const int pack  = task - token * (kBlockDv / 8);
            __nv_bfloat16* destination =
                p.value_record + record_column(layer, token) * gdn_dims::kValueDim +
                static_cast<std::int64_t>(value_head) * kStateDim;
            store_vec(destination + tile_dv + pack * 8, load_vec<uint4>(&value_raw[token][pack * 8]));
        }
        if (state_tile == 0) {
            for (int token = thread; token < valid; token += kThreads) {
                p.gate_record[record_column(layer, token) * kValueHeadCount + value_head] =
                    gate_raw[token];
            }
        }
    }
    __syncthreads();

    const int local_dv  = warp * kDvPerWarp;
    const int owned_row = (lane >> 3) & (kDvPerWarp - 1);
#pragma unroll 4
    for (std::int32_t token = 0; token < valid; ++token) {
        const RawQkLane key_lane = load_raw_qk_lane(key_raw[token], dqk_base);
        const float key_factor   = key_scale[token];
        float key[kQkPerLane];
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { key[c] = key_lane.value[c] * key_factor; }
        const float alpha = decay[token];
        const float beta  = __uint_as_float(gate_raw[token].y);

        float partial[kDvPerWarp];
        row_partials(state, key, partial);
        const float projection  = gdn_row_sums(partial, lane);
        const float v_r         = __bfloat162float(value_raw[token][local_dv + owned_row]);
        const float owned_delta = gdn_delta(beta, v_r, alpha, projection);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            const float delta = __shfl_sync(kFullWarpMask, owned_delta, r * 8);
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                state[r][c] = gdn_update(alpha, state[r][c], delta, key[c]);
            }
        }

        if constexpr (!kFold) {
            const RawQkLane query_lane = load_raw_qk_lane(query_raw[token], dqk_base);
            const float query_factor   = query_scale[token];
            float query[kQkPerLane];
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { query[c] = query_lane.value[c] * query_factor; }
            row_partials(state, query, partial);
            const float attn_val = gdn_row_sums(partial, lane);
            if ((lane & 7) == 0) {
                p.out[static_cast<std::int64_t>(token) * gdn_dims::kValueDim +
                      static_cast<std::int64_t>(value_head) * kStateDim + dv_base + owned_row] =
                    __float2bfloat16(attn_val * kOutputScale);
            }
        }
    }

    if constexpr (Mode == StagedMode::Decode) {
        store_state_tile(state, p.recurrent + state_layer_offset(control.source_slot, layer) +
                                    head_offset,
                         dv_base, dqk_base);
    } else if constexpr (kFold) {
        if (valid > 0 || control.source_slot != control.destination_slot) {
            store_state_tile(state,
                             p.recurrent + state_layer_offset(control.destination_slot, layer) +
                                 head_offset,
                             dv_base, dqk_base);
            fold_conv_history(p, layer, static_cast<std::int32_t>(value_head) * kStateTiles + state_tile,
                              thread, control, valid);
        }
    }
}

// Decode and record launch one CTA per tile (grid kValueHeadCount x 1 x kStateTiles); a fold grid
// (kValueHeadCount x 1 x kLayers * kStateTiles, or any smaller one, e.g. for a background launch
// that leaves SMs to concurrent work) sweeps the tiles in turn. Every tile's arithmetic is
// independent of the grid.
template <StagedMode Mode>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    gdn_staged_kernel(const __grid_constant__ StagedParams p) {
    constexpr bool kFold = Mode == StagedMode::Fold;
    constexpr int kTiles = kValueHeadCount * (kFold ? kLayers : 1) * kStateTiles;
    // Programmatic dependent (pdl.cuh): the step control, state and columns all come from earlier
    // kernels (back-to-back launches update the same state tile), so it waits first.
    pdl_trigger();
    pdl_wait();
    const GdnStepControl control = *p.control;
    std::int32_t valid           = p.width;
    if constexpr (kFold) { valid = min(max(control.commit, 0), p.width); }
    const int ctas = static_cast<int>(gridDim.x * gridDim.z);
    for (int tile = static_cast<int>(blockIdx.z * gridDim.x + blockIdx.x); tile < kTiles; tile += ctas) {
        staged_tile<Mode>(p, control, valid, tile);
        if (tile + ctas < kTiles) __syncthreads();  // the next tile restages the shared columns
    }
}

} // namespace fngine::kernels::gdn
