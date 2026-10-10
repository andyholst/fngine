// Modified for fngine from cinference:src/ops/linear/fp8/shapes/n16384_k5120.cu @ 383e5db. See NOTICE.
// fngine: GR down [320,10240] -> r = bf16(silu(acc / 4)), ld 384. Narrow N: warps split K.

#include "kernels/linear/fp8/fp8_a8_schedule.cuh"
#include "kernels/linear/fp8/fp8_launch.cuh"

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<320, 10240>;
constexpr auto kKind = LinearEpilogue::SiluQuarter;
// 20 phases of 512; ten K-slice warps per two-row CTA -> 160 CTAs.
using Gemv = Fp8GemvSchedule<1, 2, 16, 1, Fp8CodeCache::Default, 2, 4, 10>;
using C2   = Fp8SimtSchedule<1, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2, 10>;
using C4   = Fp8SimtSchedule<1, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2, 10>;
using C8   = Fp8SimtSchedule<1, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2, 10>;
using C16  = Fp8SimtSchedule<1, 2, 16, 16, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1, 10>;

Fp8LaunchFn select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, kKind, Gemv>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, kKind, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, kKind, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, kKind, 8, C8>;
    return launch_fp8_simt<Geometry, kKind, 16, C16>;
}
// T > 16: 32-row x 64-token tiles, rows fastest (10 row tiles share each activation tile in L2;
// token-fast order re-streamed the [T][10240] activations once per row tile).
using Gemm      = Fp8A16GemmSchedule<32, 64, 64, 32, 16, 2, 4, Cache::cg, Cache::cg, true>;
using GemmSmall = Fp8A16GemmSchedule<32, 32, 128, 32, 8, 2, 4, Cache::cg, Cache::cg, true>;
} // namespace

const Fp8ShapeEntry kFp8N320K10240{
    320, 10240, kKind, launch_fp8_a16_any<Geometry, kKind, launch_fp8_a16_chunks<16, select_a16, Geometry>, GemmSmall, Gemm, 256>,
    launch_fp8_a8<Geometry, kKind, Fp8A8SmallTokenSchedule, 1024, Fp8A8Rows64Schedule>,
    kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
