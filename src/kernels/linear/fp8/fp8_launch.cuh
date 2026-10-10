// Modified for fngine from cinference:src/ops/linear/fp8/fp8_launch.cuh @ 383e5db. See NOTICE.
// fngine: launches are parameterized by the epilogue kind (output policy + SwiGLU row policy) and
// take raw pointers; adds the K-split MMA and tiled A16 MMA launches used by lm_head.

#pragma once

#include "core/device.h"
#include "kernels/common/math.h"
#include "kernels/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "kernels/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "kernels/linear/fp8/fp8_a8.h"
#include "kernels/linear/fp8/fp8_a8_mma.cuh"
#include "kernels/linear/fp8/fp8_gemv.cuh"
#include "kernels/linear/fp8/fp8_shapes.h"
#include "kernels/linear/fp8/fp8_simt.cuh"
#include "kernels/linear/linear_output.cuh"

#include <algorithm>
#include <stdexcept>

namespace fngine::kernels::detail {

// Decode-width launches (GEMV, SIMT, K-split) are programmatic dependents of the preceding kernel
// (kernels/common/pdl.cuh): their first weight codes stream in while that kernel finishes.
template <class Geometry, LinearEpilogue Kind, class Schedule>
void launch_fp8_gemv(const Fp8Call& call) {
    using Traits                 = EpilogueTraits<Kind>;
    constexpr bool kPair         = Traits::kPair;
    constexpr int kStoredPerWarp = kPair ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    using Rows = std::conditional_t<kPair, Fp8SwiGluRows<kStoredPerWarp, Geometry::kOutputRows / 2>,
                                    Fp8GemvIdentityRows>;
    const auto output = Traits::make(*call.epilogue, Geometry::kOutputRows);
    CUDA_CHECK(pdl_launch(fp8_gemv_kernel<Geometry, Schedule, typename Traits::Output, Rows, kPair>,
                          dim3(Geometry::kOutputRows / Schedule::kRowsPerCta), dim3(Schedule::kThreads),
                          0, call.stream, call.x, call.codes, call.scales, output, Rows{}));
}

template <class Geometry, LinearEpilogue Kind, int Capacity, class Schedule,
          bool FullColumns = false>
void launch_fp8_simt(const Fp8Call& call) {
    using Traits                 = EpilogueTraits<Kind>;
    constexpr bool kPair         = Traits::kPair;
    constexpr int kStoredPerWarp = kPair ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    using Rows = std::conditional_t<kPair, Fp8SwiGluRows<kStoredPerWarp, Geometry::kOutputRows / 2>,
                                    Fp8GemvIdentityRows>;
    constexpr int blocks = Geometry::kOutputRows / Schedule::kRowsPerCta *
                           ((Capacity + Schedule::kTokenTile - 1) / Schedule::kTokenTile);
    const auto output = Traits::make(*call.epilogue, Geometry::kOutputRows);
    CUDA_CHECK(pdl_launch(fp8_simt_kernel<Geometry, Capacity, Schedule, typename Traits::Output, Rows,
                                          kPair, !FullColumns>,
                          dim3(blocks), dim3(Schedule::kThreads), 0, call.stream, call.x, call.codes,
                          call.scales, output, Rows{}, int(call.tokens)));
}

// Splits T into launches of at most Chunk tokens; Select(count) picks each launch.
template <int Chunk, Fp8LaunchFn (*Select)(std::int32_t), class Geometry>
void launch_fp8_a16_chunks(const Fp8Call& call) {
    for (std::int32_t offset = 0; offset < call.tokens; offset += Chunk) {
        const std::int32_t count = std::min<std::int32_t>(Chunk, call.tokens - offset);
        const LinearEpilogueParams epilogue =
            offset_epilogue_tokens(*call.epilogue, offset, Geometry::kOutputRows);
        Fp8Call slice  = call;
        slice.x        = call.x + static_cast<std::int64_t>(offset) * Geometry::kInputRows;
        slice.tokens   = count;
        slice.epilogue = &epilogue;
        Select(count)(slice);
    }
}

// K-split BF16 MMA over sixteen-row CTAs (elementwise outputs only); T <= TileTokens.
template <class Geometry, LinearEpilogue Kind, int TileTokens, class Schedule>
void launch_fp8_ksplit(const Fp8Call& call) {
    using Traits = EpilogueTraits<Kind>;
    static_assert(!Traits::kPair && output_row_group<typename Traits::Output>() == 1);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const auto output     = Traits::make(*call.epilogue, Geometry::kOutputRows);
    CUDA_CHECK(pdl_launch(fp8_a16_ksplit_mma_kernel<Geometry, TileTokens, Schedule,
                                                    typename Traits::Output, true>,
                          dim3(kBlocks), dim3(Schedule::kThreads), 0, call.stream, call.x, call.codes,
                          call.scales, output, int(call.tokens)));
}

// Tiled A16 BF16 MMA for any epilogue (pairs, 4-row groups, elementwise).
template <class Geometry, LinearEpilogue Kind, class Schedule, bool FullTokens>
void launch_fp8_a16_gemm_tiles(const Fp8Call& call) {
    using Traits         = EpilogueTraits<Kind>;
    constexpr bool kPair = Traits::kPair;
    using Rows = std::conditional_t<kPair,
                                    Fp8SwiGluRows<Schedule::kBlockRows / 2, Geometry::kOutputRows / 2>,
                                    Fp8GemvIdentityRows>;
    using Output          = typename Traits::Output;
    constexpr auto kernel = fp8_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Output, Rows, kPair>;
    constexpr int row_tiles = Geometry::kOutputRows / Schedule::kBlockRows;
    static_assert(Schedule::kRowFast || row_tiles <= 65535);
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    // The fast tile index is grid.x (no practical limit).
    const unsigned token_tiles = static_cast<unsigned>(div_up(call.tokens, Schedule::kBlockTokens));
    const dim3 grid = Schedule::kRowFast ? dim3(row_tiles, token_tiles, 1U)
                                         : dim3(token_tiles, row_tiles, 1U);
    kernel<<<grid, Schedule::kThreads, Schedule::kSharedBytes, call.stream>>>(
        call.x, call.codes, call.scales, Traits::make(*call.epilogue, Geometry::kOutputRows),
        call.tokens, Rows{});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, LinearEpilogue Kind, class Schedule>
void launch_fp8_a16_gemm(const Fp8Call& call) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    if ((call.tokens % Schedule::kBlockTokens) == 0) {
        launch_fp8_a16_gemm_tiles<Geometry, Kind, Schedule, true>(call);
    } else {
        launch_fp8_a16_gemm_tiles<Geometry, Kind, Schedule, false>(call);
    }
}

// Prefill A16 GEMM schedules (64-row tiles divide every registered N, incl. 320 and the SwiGLU
// half 640): 64-token tiles up to T = 64, 128-token tiles above.
using Fp8A16GemmTile64  = Fp8A16GemmSchedule<64, 64, 64, 64, 16, 2, 2>;
using Fp8A16GemmTile128 = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
// Narrow-N shapes (N <= 2560) at medium T: 32 x 32 tiles multiply the CTA count (a 64 x 64 grid
// leaves most SMs idle below T = 256); the 128-wide K step halves the per-tile latency chain when
// the grid is smallest (T <= 64).
using Fp8A16GemmTile32     = Fp8A16GemmSchedule<32, 32, 64, 32, 8, 2, 4>;
using Fp8A16GemmTile32K128 = Fp8A16GemmSchedule<32, 32, 128, 32, 8, 2, 4>;

// A16 for any T: Small (GEMV / SIMT / K-split, decode-verify widths) for T <= 16, the tiled A16
// GEMM above.
template <class Geometry, LinearEpilogue Kind, Fp8LaunchFn Small,
          class Medium = Fp8A16GemmTile64, class Large = Fp8A16GemmTile128,
          int MediumMaxTokens = Medium::kBlockTokens>
void launch_fp8_a16_any(const Fp8Call& call) {
    if (call.tokens <= kFp8A16MaxVerifyTokens) return Small(call);
    if (call.tokens <= MediumMaxTokens) return launch_fp8_a16_gemm<Geometry, Kind, Medium>(call);
    launch_fp8_a16_gemm<Geometry, Kind, Large>(call);
}

// launch_fp8_a16_any for narrow-N shapes: 32 x 32 tiles up to T = 256, 64 x 128 above.
template <class Geometry, LinearEpilogue Kind, Fp8LaunchFn Small>
void launch_fp8_a16_narrow(const Fp8Call& call) {
    if (call.tokens > kFp8A16MaxVerifyTokens && call.tokens <= 64) {
        return launch_fp8_a16_gemm<Geometry, Kind, Fp8A16GemmTile32K128>(call);
    }
    launch_fp8_a16_any<Geometry, Kind, Small, Fp8A16GemmTile32, Fp8A16GemmTile128, 256>(call);
}

template <class Geometry, LinearEpilogue Kind, class Schedule, bool FullTokens>
void launch_fp8_a8_mma(const Fp8Call& call) {
    using Traits         = EpilogueTraits<Kind>;
    constexpr bool kPair = Traits::kPair;
    using Rows = std::conditional_t<kPair,
                                    Fp8SwiGluRows<Schedule::kBlockRows / 2, Geometry::kOutputRows / 2>,
                                    Fp8MmaIdentityRows>;
    using Output = typename Traits::Output;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    constexpr auto kernel = fp8_mma_kernel<Geometry, Schedule, FullTokens, Output, Rows, kPair>;
    const int row_tiles   = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles = div_up(call.tokens, Schedule::kBlockTokens);
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    kernel<<<row_tiles * token_tiles, Schedule::kThreads, Schedule::kSharedBytes, call.stream>>>(
        call.a8.codes, call.a8.scales, call.codes, call.scales, call.tokens,
        Traits::make(*call.epilogue, Geometry::kOutputRows), Rows{});
    CUDA_CHECK(cudaGetLastError());
}

// Quantizes the activations, then runs Small for T <= SmallLimit and Large otherwise.
template <class Geometry, LinearEpilogue Kind, class Small, int SmallLimit, class Large>
void launch_fp8_a8(const Fp8Call& call) {
    launch_fp8_a8_quantize(call.x, call.tokens, Geometry::kInputRows, call.a8, call.stream);
    const auto run = [&]<class Schedule>() {
        if ((call.tokens % Schedule::kBlockTokens) == 0) {
            launch_fp8_a8_mma<Geometry, Kind, Schedule, true>(call);
        } else {
            launch_fp8_a8_mma<Geometry, Kind, Schedule, false>(call);
        }
    };
    if (call.tokens <= SmallLimit) {
        run.template operator()<Small>();
    } else {
        run.template operator()<Large>();
    }
}

} // namespace fngine::kernels::detail
