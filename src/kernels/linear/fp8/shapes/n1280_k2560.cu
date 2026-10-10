// Modified for fngine from cinference:src/ops/linear_swiglu/fp8/fp8_linear_swiglu_{decode,small_t,a8}.cu @ 383e5db. See NOTICE.
// fngine: shared-expert gate|up [1280,2560] with the SwiGLU pair epilogue -> h bf16 [T][640].

#include "kernels/linear/fp8/fp8_a8_schedule.cuh"
#include "kernels/linear/fp8/fp8_launch.cuh"

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<1280, 2560>;
constexpr auto kKind = LinearEpilogue::SwiGlu;
// Each warp owns one gate row and its up row (RowsPerWarp 2 = one pair).
using Gemv = Fp8GemvSchedule<4, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using C2   = Fp8SimtSchedule<4, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4   = Fp8SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8   = Fp8SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C16  = Fp8SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                             Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;

Fp8LaunchFn select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, kKind, Gemv>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, kKind, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, kKind, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, kKind, 8, C8>;
    return launch_fp8_simt<Geometry, kKind, 16, C16>;
}
} // namespace

const Fp8ShapeEntry kFp8N1280K2560{
    1280, 2560, kKind, launch_fp8_a16_narrow<Geometry, kKind, launch_fp8_a16_chunks<16, select_a16, Geometry>>,
    launch_fp8_a8<Geometry, kKind, Fp8A8SmallTokenSchedule, 256, Fp8A8DefaultSchedule>,
    kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
