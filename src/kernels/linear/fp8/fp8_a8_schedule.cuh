// Modified for fngine from cinference:src/ops/linear/fp8/fp8_a8_schedule.cuh @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: add and retune the small-token A8 MMA schedule.
// See NOTICE and upstream-provenance.json for upstream attribution.
// fngine: K-128 small-token and 64-row large schedules for the narrow fngine shapes.

#pragma once
#include "kernels/linear/fp8/fp8_a8_mma.cuh"

namespace fngine::kernels::detail {
using Fp8A8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// Small token counts are weight-bandwidth bound. A 16-token tile removes the idle token half of
// the default tile. 32-row tiles multiply the CTA count, so four resident CTAs per SM each keep a
// 256-wide K tile in flight behind the one they multiply; narrow-N fngine shapes keep this tile
// up to a measured per-shape token limit. Every output keeps the same k32 MMA sequence under any
// row or K tiling.
using Fp8A8SmallTokenSchedule =
    Fp8MmaSchedule<16, 32, 256, 1, 4, 2, 4, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// K = 384 / 640 are not multiples of 256.
using Fp8A8SmallTokenK128Schedule =
    Fp8MmaSchedule<16, 32, 128, 1, 4, 2, 4, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// N = 320 is not a multiple of 128 rows.
using Fp8A8Rows64Schedule =
    Fp8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;
} // namespace fngine::kernels::detail
