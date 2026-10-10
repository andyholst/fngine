// Modified for fngine from cinference:src/ops/linear/fp8/shapes/n16384_k5120.cu @ 383e5db. See NOTICE.
// fngine: permuted GR up [10240,384] with the 4-row-group GR mix epilogue -> mix bf16 [T][2560].

#include "kernels/linear/fp8/fp8_a8_schedule.cuh"
#include "kernels/linear/fp8/fp8_launch.cuh"

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<10240, 384>;
constexpr auto kKind = LinearEpilogue::GrMix;
// K = 384 = 3 phases of 128; each warp owns whole 4-row groups.
using Gemv = Fp8GemvSchedule<4, 4, 4, 1, Fp8CodeCache::Default, 4, 2>;
using C2   = Fp8SimtSchedule<4, 4, 4, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4   = Fp8SimtSchedule<4, 4, 4, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
// T 5..16: eight 8-row warps share each staged activation phase (activation re-reads, not the
// 3.9 MB weight stream, bound the token-packed form here).
using C8   = Fp8SimtSchedule<8, 8, 4, 8, 1, Fp8SimtActivationAccess::SharedPhase,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C16  = Fp8SimtSchedule<8, 8, 4, 16, 1, Fp8SimtActivationAccess::SharedPhase,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8LaunchFn select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, kKind, Gemv>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, kKind, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, kKind, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, kKind, 8, C8>;
    return launch_fp8_simt<Geometry, kKind, 16, C16>;
}
} // namespace

const Fp8ShapeEntry kFp8N10240K384{
    10240, 384, kKind, launch_fp8_a16_any<Geometry, kKind, launch_fp8_a16_chunks<16, select_a16, Geometry>>,
    launch_fp8_a8<Geometry, kKind, Fp8A8SmallTokenK128Schedule, 64, Fp8A8DefaultSchedule>,
    kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
