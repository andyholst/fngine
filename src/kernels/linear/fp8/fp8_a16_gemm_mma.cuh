// Modified for fngine from cinference:src/ops/linear/fp8/fp8_a16_gemm_mma.cuh @ 383e5db. See NOTICE.
// fngine: token tiles on grid.x (weights fetched once per call); row policies (SwiGLU pair rows);
// the finished FP32 tile is staged in shared memory and emitted through any fngine output policy
// (elementwise, pairs, 4-row groups) in 8-row vectors.
#pragma once

// Large-T row-scaled E4M3 weight x BF16 activation Tensor Core GEMM.
//
//   out[T,M] = policy((scale[M] * e4m3_codes[M,K]) . x[T,K])
//
// Persistent E4M3 codes are staged and widened exactly to swizzled BF16 MMA operands. Activations
// remain BF16. Tensor Cores accumulate the complete K reduction in FP32, then the represented BF16
// row scale is applied once. Every output element is finished by exactly one thread (no atomics).

#include "kernels/common/memory.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/linear/fp8/fp8_a16_codec.cuh"
#include "kernels/linear/fp8/fp8_gemv.cuh"
#include "kernels/linear/linear_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels::detail {

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm, Cache WeightCache = Cache::cg,
          Cache ActivationCache = Cache::cg, bool RowFast = false>
struct Fp8A16GemmSchedule {
    // RowFast: row tiles on grid.x (CTAs resident together share one activation tile; for
    // narrow-N, long-K shapes whose activations, not weights, dominate the traffic).
    static constexpr bool kRowFast          = RowFast;
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;

    static constexpr int kWarpsRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps       = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads     = kWarps * 32;
    static constexpr int kMmaRows     = kWarpRows / 16;
    static constexpr int kMmaTokens   = kWarpTokens / 8;
    static constexpr int kMmaK        = kBlockK / 16;
    static constexpr int kOperandBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kBlockRows * kBlockK;
    // FP32 epilogue staging [token][row], row stride kBlockRows + 4 (float4-aligned rows).
    static constexpr int kStageStride = kBlockRows + 4;
    static constexpr int kStageBytes  = kBlockTokens * kStageStride * static_cast<int>(sizeof(float));
    static constexpr int kSharedBytes = kOperandBytes > kStageBytes ? kOperandBytes : kStageBytes;

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert((kBlockRows % kWarpRows) == 0 && (kBlockTokens % kWarpTokens) == 0);
    static_assert((kWarpRows % 16) == 0 && (kWarpTokens % 8) == 0);
    static_assert(kBlockK == 64 || kBlockK == 128);
    static_assert(kActivationStages == 1 || kActivationStages == 2);
    static_assert(kMinBlocksPerSm > 0);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Geometry, class Schedule, bool FullTokens, class Output,
          class RowPolicy = Fp8GemvIdentityRows, bool PairRows = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, std::int32_t tokens,
    RowPolicy row_policy = {}) {
    constexpr int K       = Geometry::kInputRows;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpTokens;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int KSUB    = Schedule::kMmaK;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int kStoredRows = PairRows ? BM / 2 : BM;
    static_assert((Geometry::kOutputRows % BM) == 0);
    static_assert((K % BK) == 0);
    static_assert((kStoredRows % 8) == 0);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* weight_shared     = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared = weight_shared + BM * BK;
    auto* code_shared =
        reinterpret_cast<std::uint8_t*>(activation_shared + Schedule::kActivationStages * BN * BK);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // By default token tiles run fastest (grid.x) so the CTAs resident together share one weight
    // row tile and every weight byte is fetched from DRAM once; the activations stay L2-resident.
    const int row_tile    = static_cast<int>(Schedule::kRowFast ? blockIdx.x : blockIdx.y);
    const int token_tile  = static_cast<int>(Schedule::kRowFast ? blockIdx.y : blockIdx.x);
    const int row_begin   = row_tile * kStoredRows;
    const int token_begin = token_tile * BN;

    float accumulators[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_col_offset = ((lane >> 3) & 1) << 3;

    const auto stage_activation = [&](int stage, int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int local_token = item / (BK / 8);
            const int k8          = item - local_token * (BK / 8);
            const int token       = token_begin + local_token;
            const int kk          = k8 * 8;
            auto* destination     = &activation_shared[stage * BN * BK + local_token * BK +
                                                   fp8_a16_shared_col_64(local_token, kk)];
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += THREADS) {
            const int local_row = item / (BK / 16);
            const int chunk     = item - local_row * (BK / 16);
            const int row       = row_policy.weight_row(row_begin, local_row);
            cp_async<16, Schedule::kWeightCache>(
                &code_shared[local_row * BK + chunk * 16],
                weight_codes + static_cast<std::int64_t>(row) * K + k_begin + chunk * 16);
        }
    };

    const auto widen_codes = [&]() {
        const int half      = lane >> 4;
        const int half_lane = lane & 15;
        for (int row_pair = warp * 2; row_pair < BM; row_pair += Schedule::kWarps * 2) {
            const int row = row_pair + half;
#pragma unroll
            for (int col32 = 0; col32 < BK; col32 += 32) {
                const int col         = col32 + half_lane * 2;
                const unsigned packed = static_cast<unsigned>(
                    *reinterpret_cast<const std::uint16_t*>(&code_shared[row * BK + col]));
                const unsigned widened = fp8_e4m3x2_to_bf16x2_bits(packed);
                store_vec(&weight_shared[row * BK + fp8_a16_shared_col_64(row, col)], widened);
            }
        }
    };

    stage_activation(0, 0);
    stage_codes(0);
    cp_commit();

    constexpr int kTiles = K / BK;
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kActivationStages;
        cp_wait<0>();
        __syncthreads();

        widen_codes();
        __syncthreads();

        const int next = k_tile + 1;
        if (next < kTiles) {
            if constexpr (Schedule::kActivationStages == 2) { stage_activation(next & 1, next); }
            stage_codes(next);
            cp_commit();
        }

        unsigned a_fragments[2][MT][4];
        unsigned b_fragments[2][NT][2];
        const auto load_fragments = [&](int slot, int k_step) {
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                const int row = wm * WM + mma_row * 16 + a_row_offset;
                const int col = k_step * 16 + a_col_offset;
                ldmatrix_x4(a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                            a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                            smem_addr(&weight_shared[row * BK + fp8_a16_shared_col_64(row, col)]));
            }
#pragma unroll
            for (int mma_token = 0; mma_token < NT; ++mma_token) {
                const int row = wn * WN + mma_token * 8 + b_inner_row;
                const int col = k_step * 16 + b_col_offset;
                ldmatrix_x2(b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1],
                            smem_addr(&activation_shared[stage * BN * BK + row * BK +
                                                         fp8_a16_shared_col_64(row, col)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) { load_fragments(slot ^ 1, k_step + 1); }
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
#pragma unroll
                for (int mma_token = 0; mma_token < NT; ++mma_token) {
                    mma_bf16(
                        accumulators[mma_row][mma_token][0], accumulators[mma_row][mma_token][1],
                        accumulators[mma_row][mma_token][2], accumulators[mma_row][mma_token][3],
                        a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                        a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                        b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1]);
                }
            }
        }

        if constexpr (Schedule::kActivationStages == 1) {
            if (next < kTiles) {
                __syncthreads();
                stage_activation(0, next);
                cp_commit();
            }
        }
    }

    // Stage the scaled FP32 tile ([token][local row]) over the operand buffers.
    __syncthreads();
    constexpr int kStride = Schedule::kStageStride;
    auto* staged          = reinterpret_cast<float*>(shared_raw);
#pragma unroll
    for (int mma_row = 0; mma_row < MT; ++mma_row) {
        const int local_row0 = wm * WM + mma_row * 16 + gid;
        const int local_row1 = local_row0 + 8;
        const float scale0 =
            __bfloat162float(row_scales[row_policy.weight_row(row_begin, local_row0)]);
        const float scale1 =
            __bfloat162float(row_scales[row_policy.weight_row(row_begin, local_row1)]);
#pragma unroll
        for (int mma_token = 0; mma_token < NT; ++mma_token) {
            const int local_token0 = wn * WN + mma_token * 8 + 2 * lid;
            const float* values    = accumulators[mma_row][mma_token];
            staged[local_token0 * kStride + local_row0]       = values[0] * scale0;
            staged[(local_token0 + 1) * kStride + local_row0] = values[1] * scale0;
            staged[local_token0 * kStride + local_row1]       = values[2] * scale1;
            staged[(local_token0 + 1) * kStride + local_row1] = values[3] * scale1;
        }
    }
    __syncthreads();

    constexpr int vectors_per_token = kStoredRows / 8;
#pragma unroll 1
    for (int task = tid; task < BN * vectors_per_token; task += THREADS) {
        const int token_local = task / vectors_per_token;
        const int row_vector  = task - token_local * vectors_per_token;
        const int token       = token_begin + token_local;
        if (!FullTokens && token >= tokens) continue;
        const float* source = staged + token_local * kStride + row_vector * 8;
        float values[8];
        float paired[8];
        const float4 v0 = load_vec<float4>(source);
        const float4 v1 = load_vec<float4>(source + 4);
        values[0] = v0.x, values[1] = v0.y, values[2] = v0.z, values[3] = v0.w;
        values[4] = v1.x, values[5] = v1.y, values[6] = v1.z, values[7] = v1.w;
        if constexpr (PairRows) {
            const float4 p0 = load_vec<float4>(source + kStoredRows);
            const float4 p1 = load_vec<float4>(source + kStoredRows + 4);
            paired[0] = p0.x, paired[1] = p0.y, paired[2] = p0.z, paired[3] = p0.w;
            paired[4] = p1.x, paired[5] = p1.y, paired[6] = p1.z, paired[7] = p1.w;
        }
        emit_rows8<PairRows>(output, row_begin + row_vector * 8, token, values, paired);
    }
}

} // namespace fngine::kernels::detail
