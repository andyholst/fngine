// Modified for fngine from cinference:src/ops/linear/fp8/fp8_a8_mma.cuh @ 383e5db. See NOTICE.
// fngine: the epilogue stages the finished FP32 tile (no BF16 rounding before the output policy)
// and emits it through fngine output policies (pairs, 4-row groups, FP32 outputs); the
// separate Epilogue functor and the row-range policy are dropped.
// Modified by satellitedown for Cinference: evict-first weight fills; tile-finishing outputs;
// row-range launches.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

// Row-scaled E4M3 weight x materialized row-scaled E4M3 activation Tensor Core GEMM.
//
// The MMA tile is oriented as [token,K] x [K,output-row]. This makes the accumulator's
// contiguous axis the public output-row axis. The epilogue stages the finished FP32 tile and
// emits 8-row vectors through the output policy, which remains replaceable by a fused consumer.

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/linear/linear_output.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace fngine::kernels::detail {

enum class Fp8MmaFragmentPipeline : std::uint8_t {
    Serial,
    PingPong,
};

enum class Fp8MmaRaster : std::uint8_t {
    RowFast,
    TokenFast,
    Grouped,
};

struct Fp8MmaIdentityRows {
    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

template <int BlockTokens, int BlockRows, int BlockK, int WarpsTokens, int WarpsRows, int Stages,
          int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache,
          Fp8MmaFragmentPipeline FragmentPipeline, Fp8MmaRaster Raster, int RasterGroupRows = 1>
struct Fp8MmaSchedule {
    static constexpr int kBlockTokens                         = BlockTokens;
    static constexpr int kBlockRows                           = BlockRows;
    static constexpr int kBlockK                              = BlockK;
    static constexpr int kWarpsTokens                         = WarpsTokens;
    static constexpr int kWarpsRows                           = WarpsRows;
    static constexpr int kStages                              = Stages;
    static constexpr int kMinBlocksPerSm                      = MinBlocksPerSm;
    static constexpr Cache kWeightCache                       = WeightCache;
    static constexpr Cache kActivationCache                   = ActivationCache;
    static constexpr Fp8MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Fp8MmaRaster kRaster                     = Raster;
    static constexpr int kRasterGroupRows                     = RasterGroupRows;

    static constexpr int kWarps          = kWarpsTokens * kWarpsRows;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpTokens     = kBlockTokens / kWarpsTokens;
    static constexpr int kWarpRows       = kBlockRows / kWarpsRows;
    static constexpr int kMmaTokens      = kWarpTokens / 16;
    static constexpr int kMmaRows        = kWarpRows / 8;
    static constexpr int kMmaK           = kBlockK / 32;
    static constexpr int kSegmentsPerRow = kBlockK / 16;
    static constexpr int kSharedBytes    = kStages * (kBlockTokens + kBlockRows) * kBlockK;

    static_assert(kBlockTokens > 0 && kBlockRows > 0 && kBlockK > 0);
    static_assert((kBlockTokens % kWarpsTokens) == 0 && (kBlockRows % kWarpsRows) == 0);
    static_assert((kWarpTokens % 16) == 0 && (kWarpRows % 8) == 0);
    static_assert((kBlockK % 32) == 0);
    static_assert((kSegmentsPerRow & (kSegmentsPerRow - 1)) == 0);
    static_assert(kStages >= 2 && kStages <= 8);
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
    static_assert(kRaster != Fp8MmaRaster::Grouped || kRasterGroupRows > 0);
};

template <class Schedule>
__device__ __forceinline__ int fp8_mma_shared_byte(int row, int logical_byte) {
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Schedule>
__device__ __forceinline__ void
fp8_mma_tile_coordinates(std::int32_t linear, std::int32_t row_tiles, std::int32_t token_tiles,
                         std::int32_t& row_tile, std::int32_t& token_tile) {
    if constexpr (Schedule::kRaster == Fp8MmaRaster::RowFast) {
        token_tile = linear / row_tiles;
        row_tile   = linear - token_tile * row_tiles;
    } else if constexpr (Schedule::kRaster == Fp8MmaRaster::TokenFast) {
        row_tile   = linear / token_tiles;
        token_tile = linear - row_tile * token_tiles;
    } else {
        constexpr int group_rows       = Schedule::kRasterGroupRows;
        const std::int32_t group_span  = group_rows * token_tiles;
        const std::int32_t group       = linear / group_span;
        const std::int32_t first_row   = group * group_rows;
        const std::int32_t active_rows = min(group_rows, row_tiles - first_row);
        const std::int32_t within      = linear - group * group_span;
        row_tile                       = first_row + within % active_rows;
        token_tile                     = within / active_rows;
    }
}

template <class Geometry, class Schedule, bool FullTokens, class Output,
          class RowPolicy = Fp8MmaIdentityRows, bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_mma_kernel(
    const std::uint8_t* __restrict__ activation_codes, const float* __restrict__ activation_scales,
    const std::uint8_t* __restrict__ weight_codes, const __nv_bfloat16* __restrict__ weight_scales,
    std::int32_t tokens, Output output, RowPolicy row_policy = {}) {
    constexpr int TILES_K = Geometry::kInputRows / Schedule::kBlockK;
    constexpr int BM      = Schedule::kBlockTokens;
    constexpr int BN      = Schedule::kBlockRows;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int S       = Schedule::kStages;
    constexpr int THREADS = Schedule::kThreads;
    static_assert((Geometry::kInputRows % BK) == 0);
    static_assert((Geometry::kOutputRows % BN) == 0);
    static_assert(!PairRows || (BN % 2) == 0);
    static_assert(!PairRows || ((Geometry::kOutputRows / 2) % (BN / 2)) == 0);
    static_assert(TILES_K >= S);
    constexpr int output_stride = BN + 4;  // FP32 staging row stride (keeps float4 alignment)
    static_assert(Schedule::kSharedBytes >= BM * output_stride * sizeof(float));

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* activation_shared = reinterpret_cast<std::uint8_t*>(shared_raw);
    auto* weight_shared     = activation_shared + S * BM * BK;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int warp_token = warp / Schedule::kWarpsRows;
    const int warp_row   = warp - warp_token * Schedule::kWarpsRows;

    constexpr int row_tiles = Geometry::kOutputRows / BN;
    const int token_tiles   = tokens / BM + static_cast<int>(tokens % BM != 0);
    int row_tile            = 0;
    int token_tile          = 0;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), row_tiles, token_tiles,
                                       row_tile, token_tile);
    constexpr int rows_per_block = PairRows ? BN / 2 : BN;
    const int row_begin          = row_tile * rows_per_block;
    const int token_begin        = token_tile * BM;

    const unsigned long long weight_policy = l2_weight_policy(token_tiles == 1);
    auto stage_inputs = [&](int stage, int k_tile) {
        const int k_begin      = k_tile * BK;
        auto* activation_stage = activation_shared + stage * BM * BK;
        auto* weight_stage     = weight_shared + stage * BN * BK;

#pragma unroll 1
        for (int task = tid; task < BM * Schedule::kSegmentsPerRow; task += THREADS) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int logical_byte    = logical_segment * 16;
            const int physical_byte   = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            auto* destination         = activation_stage + row * BK + physical_byte;
            const int token           = token_begin + row;
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, activation_codes +
                                     static_cast<std::int64_t>(token) * Geometry::kInputRows +
                                     k_begin + logical_byte);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    activation_codes +
                        static_cast<std::int64_t>(valid ? token : 0) * Geometry::kInputRows +
                        k_begin + logical_byte,
                    valid ? 16 : 0);
            }
        }

#pragma unroll 1
        for (int task = tid; task < BN * Schedule::kSegmentsPerRow; task += THREADS) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int logical_byte    = logical_segment * 16;
            const int physical_byte   = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            const int weight_row      = row_policy.weight_row(row_begin, row);
            const auto* weight_source =
                weight_codes + static_cast<std::int64_t>(weight_row) * Geometry::kInputRows +
                k_begin + logical_byte;
            if constexpr (Schedule::kWeightCache == Cache::cg) {
                cp_async_cg_policy(weight_stage + row * BK + physical_byte, weight_source,
                                   weight_policy);
            } else {
                cp_async<16, Schedule::kWeightCache>(weight_stage + row * BK + physical_byte,
                                                     weight_source);
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    const int a_matrix                                              = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;

#pragma unroll 1
    for (int k_tile = 0; k_tile < TILES_K; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= TILES_K) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_fragments)[Schedule::kMmaTokens][4],
                                  unsigned(&b_fragments)[Schedule::kMmaRows][2]) {
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
                const int logical_byte  = k_step * 32 + a_column_byte;
                const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
                ldmatrix_x4(
                    a_fragments[mma_token][0], a_fragments[mma_token][1], a_fragments[mma_token][2],
                    a_fragments[mma_token][3],
                    smem_addr(activation_shared + stage * BM * BK + row * BK + physical_byte));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                const int row = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
                const int logical_byte  = k_step * 32 + b_column_byte;
                const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
                ldmatrix_x2(b_fragments[mma_row][0], b_fragments[mma_row][1],
                            smem_addr(weight_shared + stage * BN * BK + row * BK + physical_byte));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Fp8MmaFragmentPipeline::PingPong) {
            unsigned a_fragments[2][Schedule::kMmaTokens][4];
            unsigned b_fragments[2][Schedule::kMmaRows][2];
            load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
            for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
                const int slot = k_step & 1;
                if (k_step + 1 < Schedule::kMmaK) {
                    load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
                }
#pragma unroll
                for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                        mma_fp8_e4m3(
                            accumulators[mma_token][mma_row][0],
                            accumulators[mma_token][mma_row][1],
                            accumulators[mma_token][mma_row][2],
                            accumulators[mma_token][mma_row][3], a_fragments[slot][mma_token][0],
                            a_fragments[slot][mma_token][1], a_fragments[slot][mma_token][2],
                            a_fragments[slot][mma_token][3], b_fragments[slot][mma_row][0],
                            b_fragments[slot][mma_row][1]);
                    }
                }
            }
        } else {
            unsigned a_fragments[Schedule::kMmaTokens][4];
            unsigned b_fragments[Schedule::kMmaRows][2];
#pragma unroll
            for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
                load_fragments(k_step, a_fragments, b_fragments);
#pragma unroll
                for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                        mma_fp8_e4m3(accumulators[mma_token][mma_row][0],
                                     accumulators[mma_token][mma_row][1],
                                     accumulators[mma_token][mma_row][2],
                                     accumulators[mma_token][mma_row][3], a_fragments[mma_token][0],
                                     a_fragments[mma_token][1], a_fragments[mma_token][2],
                                     a_fragments[mma_token][3], b_fragments[mma_row][0],
                                     b_fragments[mma_row][1]);
                    }
                }
            }
        }

        __syncthreads();
        const int next_k_tile = k_tile + S;
        if (next_k_tile < TILES_K) {
            stage_inputs(stage, next_k_tile);
            cp_commit();
        }
    }

    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
    auto* shared_output         = reinterpret_cast<float*>(shared_raw);
#pragma unroll
    for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
        const int token0 =
            token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
        const int token1 = token0 + 8;
        const float activation_scale0 =
            (FullTokens || token0 < tokens) ? activation_scales[token0] : 0.0F;
        const float activation_scale1 =
            (FullTokens || token1 < tokens) ? activation_scales[token1] : 0.0F;
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            const std::uint32_t scale_bits = load_vec<std::uint32_t>(weight_scales + parent_row0);
            const float2 weight_scale      = bf16x2_bits_to_float2(scale_bits);
            const float value00 =
                accumulators[mma_token][mma_row][0] * activation_scale0 * weight_scale.x;
            const float value01 =
                accumulators[mma_token][mma_row][1] * activation_scale0 * weight_scale.y;
            const float value10 =
                accumulators[mma_token][mma_row][2] * activation_scale1 * weight_scale.x;
            const float value11 =
                accumulators[mma_token][mma_row][3] * activation_scale1 * weight_scale.y;
            store_vec(shared_output + (token0 - token_begin) * output_stride + local_row0,
                      make_float2(value00, value01));
            store_vec(shared_output + (token1 - token_begin) * output_stride + local_row0,
                      make_float2(value10, value11));
        }
    }
    __syncthreads();

    constexpr int stored_rows       = PairRows ? BN / 2 : BN;
    constexpr int vectors_per_token = stored_rows / 8;
    constexpr int output_vectors    = BM * vectors_per_token;
    for (int task = tid; task < output_vectors; task += THREADS) {
        const int token_local = task / vectors_per_token;
        const int row_vector  = task - token_local * vectors_per_token;
        const int token       = token_begin + token_local;
        if (!FullTokens && token >= tokens) continue;
        const float* source = shared_output + token_local * output_stride + row_vector * 8;
        float values[8];
        float paired[8];
        const float4 v0 = load_vec<float4>(source);
        const float4 v1 = load_vec<float4>(source + 4);
        values[0] = v0.x, values[1] = v0.y, values[2] = v0.z, values[3] = v0.w;
        values[4] = v1.x, values[5] = v1.y, values[6] = v1.z, values[7] = v1.w;
        if constexpr (PairRows) {
            const float4 p0 = load_vec<float4>(source + stored_rows);
            const float4 p1 = load_vec<float4>(source + stored_rows + 4);
            paired[0] = p0.x, paired[1] = p0.y, paired[2] = p0.z, paired[3] = p0.w;
            paired[4] = p1.x, paired[5] = p1.y, paired[6] = p1.z, paired[7] = p1.w;
        }
        emit_rows8<PairRows>(output, row_begin + row_vector * 8, token, values, paired);
    }
}

} // namespace fngine::kernels::detail
