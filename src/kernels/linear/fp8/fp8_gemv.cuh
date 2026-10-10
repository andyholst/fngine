// Modified for fngine from cinference:src/ops/linear/fp8/fp8_gemv.cuh @ 383e5db. See NOTICE.
// fngine: outputs are fngine output policies (linear_output.cuh) fed the finished FP32 value,
// including 4-row groups (GrMix); 4-value lanes; intra-CTA warp K-split with an ordered
// shared-memory reduction.

#pragma once

// Reusable row-scaled FP8 T=1 CUDA-core mainloop. Each warp owns contiguous output rows and reads
// naturally aligned E4M3 code packs. The persistent codes are decoded exactly, multiplied by the
// represented BF16 activation with FP32 FMA, reduced within the warp, and scaled once by the
// owning BF16 row multiplier. Output owns the physical row mapping so fused projections reuse the
// same weight/decode mainloop without materializing a packed parent output.

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/linear/fp8/fp8_config.h"
#include "kernels/linear/linear_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

namespace fngine::kernels::detail {

struct Fp8GemvIdentityRows {
    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

// SwiGLU pair rows: local rows below RowsPerBranch read gate rows, the rest the matching up rows.
template <int RowsPerBranch, int IntermediateRows>
struct Fp8SwiGluRows {
    static_assert(RowsPerBranch > 0 && (RowsPerBranch & (RowsPerBranch - 1)) == 0);

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (RowsPerBranch - 1)) +
               (local_row >= RowsPerBranch ? IntermediateRows : 0);
    }
};

template <int Values>
struct alignas(Values) Fp8CodePack {
    static_assert(Values == 4 || Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 4];
};

static_assert(sizeof(Fp8CodePack<4>) == 4);
static_assert(sizeof(Fp8CodePack<8>) == 8);
static_assert(sizeof(Fp8CodePack<16>) == 16);
static_assert(sizeof(Fp8CodePack<32>) == 32);

template <Fp8CodeCache Cache, int Values>
__device__ __forceinline__ Fp8CodePack<Values> load_fp8_codes(const std::uint8_t* pointer) {
    if constexpr (Values == 32) {
        Fp8CodePack<Values> result;
        const Fp8CodePack<16> low  = load_fp8_codes<Cache, 16>(pointer);
        const Fp8CodePack<16> high = load_fp8_codes<Cache, 16>(pointer + 16);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            result.words[word]     = low.words[word];
            result.words[word + 4] = high.words[word];
        }
        return result;
    } else if constexpr (Cache == Fp8CodeCache::Default) {
        return load_vec<Fp8CodePack<Values>>(pointer);
    } else if constexpr (Values == 4) {
        Fp8CodePack<Values> result;
        asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(result.words[0]) : "l"(pointer));
        return result;
    } else if constexpr (Values == 8) {
        Fp8CodePack<Values> result;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1])
                     : "l"(pointer));
        return result;
    } else {
        Fp8CodePack<Values> result;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1]), "=r"(result.words[2]),
                       "=r"(result.words[3])
                     : "l"(pointer));
        return result;
    }
}

__device__ __forceinline__ float2 decode_fp8_e4m3x2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

template <int Values, int Rows, int AccumulatorChains>
__device__ __forceinline__ void accumulate_rows(const Fp8CodePack<Values> (&codes)[Rows],
                                                const std::uint32_t* activation_pairs,
                                                float (&accumulators)[Rows][AccumulatorChains]) {
    constexpr int kChainMask = AccumulatorChains - 1;
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 activation = bf16x2_bits_to_float2(activation_pairs[pair]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const std::uint32_t word   = codes[row].words[pair >> 1];
            const std::uint16_t packed = static_cast<std::uint16_t>(word >> ((pair & 1) * 16));
            const float2 weight        = decode_fp8_e4m3x2(packed);
            accumulators[row][(2 * pair) & kChainMask] =
                fmaf(weight.x, activation.x, accumulators[row][(2 * pair) & kChainMask]);
            accumulators[row][(2 * pair + 1) & kChainMask] =
                fmaf(weight.y, activation.y, accumulators[row][(2 * pair + 1) & kChainMask]);
        }
    }
}

// BF16 row multiplier of the finished FP32 dot products.
struct Fp8RowScale {
    const __nv_bfloat16* __restrict__ scales;
    __device__ __forceinline__ float operator()(int row) const {
        return __bfloat162float(scales[row]);
    }
};

template <class Geometry, class Schedule, class Output, class RowPolicy = Fp8GemvIdentityRows,
          bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, RowPolicy row_policy = {}) {
    constexpr int kValuesPerPhase = kWarpSize * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert(!PairRows || (Schedule::kRowsPerWarp % 2) == 0);
    static_assert(!PairRows || ((Geometry::kOutputRows / 2) % (Schedule::kRowsPerCta / 2)) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    static_assert((kPhases % Schedule::kKSplit) == 0);
    constexpr int kSlicePhases = kPhases / Schedule::kKSplit;
    constexpr int kStoredRowsPerWarp =
        PairRows ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    constexpr int kStoredRowsPerCta = Schedule::kRowWarps * kStoredRowsPerWarp;
    using Reduce = WarpKSplitReduce<Schedule::kKSplit, Schedule::kRowWarps, Schedule::kRowsPerWarp>;
    __shared__ typename Reduce::Storage reduce_storage;

    const int lane     = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp     = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_warp = warp % Schedule::kRowWarps;
    const int k_slice  = warp / Schedule::kRowWarps;
    const int row_begin =
        static_cast<int>(blockIdx.x) * kStoredRowsPerCta + row_warp * kStoredRowsPerWarp;
    const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(pdl_coherent(x));
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    using Pack = Fp8CodePack<Schedule::kValuesPerLane>;
    // `pinned` (pre-wait) loads use ld.global.cg, which ptxas keeps above the wait (pdl.cuh).
    const auto load_phase = [&](int slice_phase, Pack (&codes)[Schedule::kRowsPerWarp],
                                bool pinned) {
        const int value_begin =
            (k_slice * kSlicePhases + slice_phase) * kValuesPerPhase + lane * Schedule::kValuesPerLane;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int weight_row     = row_policy.weight_row(row_begin, local_row);
            const std::uint8_t* row = weight_codes +
                                      static_cast<std::int64_t>(weight_row) * Geometry::kInputRows +
                                      value_begin;
            codes[local_row] =
                pinned ? load_fp8_codes<Fp8CodeCache::Streaming, Schedule::kValuesPerLane>(row)
                       : load_fp8_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(row);
        }
    };
    // PDL (pdl.cuh): the codes of the first phases (up to 16 words per lane) are requested before
    // pdl_wait(), the activations and epilogue inputs after it. The FP32 order is unchanged.
    constexpr int kWordsPerPhase = Schedule::kRowsPerWarp * Schedule::kValuesPerLane / 4;
    constexpr int kPreload       = std::min(kSlicePhases, std::max(1, 16 / kWordsPerPhase));
    pdl_trigger();
    Pack early[kPreload][Schedule::kRowsPerWarp];
#pragma unroll
    for (int slice_phase = 0; slice_phase < kPreload; ++slice_phase) {
        load_phase(slice_phase, early[slice_phase], true);
    }
    pdl_wait();
    // Epilogue inputs (GrMix Xn) are fetched now so their latency hides behind the weight stream.
    const auto preloaded =
        lane_preload<PairRows, Schedule::kRowsPerWarp>(output, row_policy, row_begin, 0, lane);

#pragma unroll
    for (int slice_phase = 0; slice_phase < kPreload; ++slice_phase) {
        const int value_begin =
            (k_slice * kSlicePhases + slice_phase) * kValuesPerPhase + lane * Schedule::kValuesPerLane;
        accumulate_rows(early[slice_phase], activation_pairs + value_begin / 2, accumulators);
    }
#pragma unroll Schedule::kPhaseUnroll
    for (int slice_phase = kPreload; slice_phase < kSlicePhases; ++slice_phase) {
        const int value_begin =
            (k_slice * kSlicePhases + slice_phase) * kValuesPerPhase + lane * Schedule::kValuesPerLane;
        Pack row_codes[Schedule::kRowsPerWarp];
        load_phase(slice_phase, row_codes, false);
        accumulate_rows(row_codes, activation_pairs + value_begin / 2, accumulators);
    }

    // Butterfly reduction: every lane holds every row total; lane u finishes stored unit u.
    float totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        totals[local_row] = warp_sum(total);
    }
    if (Reduce::template run<true>(reduce_storage, row_warp, k_slice, lane, totals)) {
        finish_token_lanes<PairRows>(output, row_policy, Fp8RowScale{row_scales}, row_begin, 0,
                                     totals, lane, preloaded);
    }
}

} // namespace fngine::kernels::detail
