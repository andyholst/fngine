// Modified for fngine from cinference:src/ops/linear/fp8/fp8_simt.cuh @ 383e5db. See NOTICE.
// fngine: fngine output policies (row groups, pairs, elementwise) through one finishing path;
// 4-value lanes; intra-CTA warp K-split; the row-vector finalization is dropped.

#pragma once

// Reusable row-scaled FP8 CUDA-core mainloop for a compile-time number of BF16 activation
// columns. A CTA owns one row tile and one compile-time token tile, reusing each decoded weight
// pair across that token tile before advancing K. Output and row policies let fused consumers
// retain their observable semantics without duplicating the contraction.

#include "kernels/linear/fp8/fp8_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels::detail {

template <int Values>
struct Fp8ActivationPack {
    static_assert(Values == 4 || Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 2];
};

template <int Values>
__device__ __forceinline__ Fp8ActivationPack<Values>
load_fp8_activation_pack(const __nv_bfloat16* pointer) {
    Fp8ActivationPack<Values> result;
    if constexpr (Values == 4) {
        const uint2 packed = load_vec<uint2>(pointer);
        result.words[0]    = packed.x;
        result.words[1]    = packed.y;
    } else {
#pragma unroll
        for (int chunk = 0; chunk < Values / 8; ++chunk) {
            const uint4 packed          = load_vec<uint4>(pointer + chunk * 8);
            result.words[chunk * 4]     = packed.x;
            result.words[chunk * 4 + 1] = packed.y;
            result.words[chunk * 4 + 2] = packed.z;
            result.words[chunk * 4 + 3] = packed.w;
        }
    }
    return result;
}

template <class Schedule>
struct Fp8SimtSharedStorage {
    static constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase
            ? Schedule::kTokenTile * kValuesPerPhase
            : 8;
    alignas(16) __nv_bfloat16 activation[kActivationElements];
};

// ActiveTokens is the compile-time column capacity; with RuntimeColumns the live column count is
// `columns` (<= ActiveTokens) and padded columns re-read the last live column without storing.
template <class Geometry, int ActiveTokens, class Schedule, class Output,
          class RowPolicy = Fp8GemvIdentityRows, bool PairRows = false,
          bool RuntimeColumns = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, RowPolicy row_policy = {},
    int columns = ActiveTokens) {
    const int live_tokens = RuntimeColumns ? columns : ActiveTokens;
    static_assert(ActiveTokens >= 2);
    static_assert(Schedule::kTokenTile <= ActiveTokens);
    static_assert(!PairRows || (Schedule::kRowsPerWarp % 2) == 0);
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    static_assert((kPhases % Schedule::kKSplit) == 0);
    constexpr int kSlicePhases = kPhases / Schedule::kKSplit;
    constexpr int kStoredRowsPerWarp =
        PairRows ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    constexpr int kStoredRowsPerCta = Schedule::kRowWarps * kStoredRowsPerWarp;
    constexpr int kRowBlocks        = Geometry::kOutputRows / Schedule::kRowsPerCta;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kRows       = Schedule::kRowsPerWarp;
    constexpr int kTile       = Schedule::kTokenTile;

    const int linear_block = static_cast<int>(blockIdx.x);
    int row_block;
    int token_tile;
    if constexpr (Schedule::kBlockOrder == Fp8SimtBlockOrder::RowsContiguous) {
        token_tile = linear_block / kRowBlocks;
        row_block  = linear_block - token_tile * kRowBlocks;
    } else {
        row_block  = linear_block / kTokenTiles;
        token_tile = linear_block - row_block * kTokenTiles;
    }
    const int token0 = kTokenTiles == 1 ? 0 : token_tile * kTile;

    using Reduce = WarpKSplitReduce<Schedule::kKSplit, Schedule::kRowWarps, kRows * kTile>;
    __shared__ Fp8SimtSharedStorage<Schedule> shared;
    __shared__ typename Reduce::Storage reduce_storage;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int row_warp  = warp % Schedule::kRowWarps;
    const int k_slice   = warp / Schedule::kRowWarps;
    const int row_begin = row_block * kStoredRowsPerCta + row_warp * kStoredRowsPerWarp;
    float accumulators[kRows][kTile][Schedule::kAccumulatorChains] = {};
    using Pack = Fp8CodePack<Schedule::kValuesPerLane>;
    // `pinned` (pre-wait) loads use ld.global.cg, which ptxas keeps above the wait (pdl.cuh).
    const auto load_phase = [&](int slice_phase, Pack (&codes)[kRows], bool pinned) {
        const int value_begin =
            (k_slice * kSlicePhases + slice_phase) * kValuesPerPhase + lane * Schedule::kValuesPerLane;
#pragma unroll
        for (int local_row = 0; local_row < kRows; ++local_row) {
            const int weight_row     = row_policy.weight_row(row_begin, local_row);
            const std::uint8_t* row = weight_codes +
                                      static_cast<std::int64_t>(weight_row) * Geometry::kInputRows +
                                      value_begin;
            codes[local_row] =
                pinned ? load_fp8_codes<Fp8CodeCache::Streaming, Schedule::kValuesPerLane>(row)
                       : load_fp8_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(row);
        }
    };
    const __nv_bfloat16* const activations = pdl_coherent(x);
    const auto run_phase = [&](int slice_phase, const Pack (&row_codes)[kRows]) {
        const int phase = k_slice * kSlicePhases + slice_phase;
        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            constexpr int kPacksPerToken = kValuesPerPhase / 8;
            constexpr int kPacks         = kTile * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(shared.activation);
            for (int task = static_cast<int>(threadIdx.x); task < kPacks;
                 task += Schedule::kThreads) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                const int token       = token0 + local_token;
                if (token < ActiveTokens) {
                    destination[task] =
                        load_vec<uint4>(activations +
                                        static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                                            Geometry::kInputRows +
                                        phase * kValuesPerPhase + local_pack * 8);
                }
            }
        }

        const int value_begin = phase * kValuesPerPhase + lane * Schedule::kValuesPerLane;
        Fp8ActivationPack<Schedule::kValuesPerLane> activation[kTile];
        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            __syncthreads();
#pragma unroll
            for (int local_token = 0; local_token < kTile; ++local_token) {
                if (token0 + local_token < ActiveTokens) {
                    activation[local_token] = load_fp8_activation_pack<Schedule::kValuesPerLane>(
                        shared.activation + local_token * kValuesPerPhase +
                        lane * Schedule::kValuesPerLane);
                }
            }
        } else {
#pragma unroll
            for (int local_token = 0; local_token < kTile; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    activation[local_token] = load_fp8_activation_pack<Schedule::kValuesPerLane>(
                        activations +
                        static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                            Geometry::kInputRows +
                        value_begin);
                }
            }
        }

        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
        for (int pair = 0; pair < Schedule::kValuesPerLane / 2; ++pair) {
            float2 weights[kRows];
#pragma unroll
            for (int local_row = 0; local_row < kRows; ++local_row) {
                const std::uint32_t word   = row_codes[local_row].words[pair >> 1];
                const std::uint16_t packed = static_cast<std::uint16_t>(word >> ((pair & 1) * 16));
                weights[local_row]         = decode_fp8_e4m3x2(packed);
            }
#pragma unroll
            for (int local_token = 0; local_token < kTile; ++local_token) {
                if (token0 + local_token >= ActiveTokens) { continue; }
                const float2 value = bf16x2_bits_to_float2(activation[local_token].words[pair]);
#pragma unroll
                for (int local_row = 0; local_row < kRows; ++local_row) {
                    accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                        fmaf(weights[local_row].x, value.x,
                             accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                    accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                        fmaf(weights[local_row].y, value.y,
                             accumulators[local_row][local_token][(2 * pair + 1) & kChainMask]);
                }
            }
        }

        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            __syncthreads();
        }
    };

    // PDL (pdl.cuh): the codes of the first phases (up to 16 words per lane) are requested before
    // pdl_wait(), the activations after it. The FP32 order is unchanged.
    constexpr int kWordsPerPhase = kRows * Schedule::kValuesPerLane / 4;
    constexpr int kPreload       = std::min(kSlicePhases, std::max(1, 16 / kWordsPerPhase));
    pdl_trigger();
    Pack early[kPreload][kRows];
#pragma unroll
    for (int slice_phase = 0; slice_phase < kPreload; ++slice_phase) {
        load_phase(slice_phase, early[slice_phase], true);
    }
    pdl_wait();
#pragma unroll
    for (int slice_phase = 0; slice_phase < kPreload; ++slice_phase) {
        run_phase(slice_phase, early[slice_phase]);
    }
#pragma unroll Schedule::kPhaseUnroll
    for (int slice_phase = kPreload; slice_phase < kSlicePhases; ++slice_phase) {
        Pack row_codes[kRows];
        load_phase(slice_phase, row_codes, false);
        run_phase(slice_phase, row_codes);
    }

    // totals[local_token * kRows + local_row]. Butterfly reduction: every lane holds every total,
    // so lane l finishes token l of the tile and the epilogue's loads / stores (GrMix reads Xn,
    // GrWrite updates X) run lane-parallel.
    static_assert(kTile <= 32);
    float totals[kRows * kTile];
#pragma unroll
    for (int local_token = 0; local_token < kTile; ++local_token) {
#pragma unroll
        for (int local_row = 0; local_row < kRows; ++local_row) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[local_row][local_token][chain];
            }
            totals[local_token * kRows + local_row] = warp_sum(total);
        }
    }
    if (!Reduce::template run<true>(reduce_storage, row_warp, k_slice, lane, totals)) return;

    float row_totals[kRows];
#pragma unroll
    for (int local_token = 0; local_token < kTile; ++local_token) {
        if (lane == local_token) {
#pragma unroll
            for (int local_row = 0; local_row < kRows; ++local_row) {
                row_totals[local_row] = totals[local_token * kRows + local_row];
            }
        }
    }
    const int token = token0 + lane;
    if (lane < kTile && token < live_tokens) {
        finish_token<PairRows>(output, row_policy, Fp8RowScale{row_scales}, row_begin, token,
                               row_totals);
    }
}

} // namespace fngine::kernels::detail
