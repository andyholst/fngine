// Modified for fngine from cinference:src/ops/linear/fp8/shapes/n16384_k5120.cu @ 383e5db. See NOTICE.
// fngine: one schedule set for the [N,2560] BF16-store projections (gdn.qkvz, qsa.qkv, ple.key,
// ple.value / mtp.fc_*): T <= 16 K-split MMA with identical per-token arithmetic at every width,
// tiled A16 GEMM above; A8 small-token / default MMA when forced.

#pragma once

#include "kernels/linear/fp8/fp8_a8_schedule.cuh"
#include "kernels/linear/fp8/fp8_launch.cuh"

namespace fngine::kernels::detail::k2560_store {

constexpr auto kKind = LinearEpilogue::StoreBf16;
using Tile8          = Fp8A16KSplitSchedule<8, 8, 2, Fp8A16KSplitCache::Default,
                                            Fp8A16KSplitCache::Streaming,
                                            Fp8A16KSplitActivationStage::ActiveOnly, 2>;
using Tile16         = Fp8A16KSplitSchedule<8, 16, 2>;

template <int N>
void launch_a16_tile(const Fp8Call& call) {
    using Geometry = Fp8Geometry<N, 2560>;
    if (call.tokens <= 8) return launch_fp8_ksplit<Geometry, kKind, 8, Tile8>(call);
    return launch_fp8_ksplit<Geometry, kKind, 16, Tile16>(call);
}

// A16 for any T: K-split for T <= 16, the tiled A16 GEMM above.
template <int N>
void launch_a16(const Fp8Call& call) {
    if constexpr (N <= 2560) {
        launch_fp8_a16_narrow<Fp8Geometry<N, 2560>, kKind, launch_a16_tile<N>>(call);
    } else {
        launch_fp8_a16_any<Fp8Geometry<N, 2560>, kKind, launch_a16_tile<N>>(call);
    }
}

template <int N, int SmallLimit>
void launch_a8(const Fp8Call& call) {
    launch_fp8_a8<Fp8Geometry<N, 2560>, kKind, Fp8A8SmallTokenSchedule, SmallLimit,
                  Fp8A8DefaultSchedule>(call);
}

} // namespace fngine::kernels::detail::k2560_store
