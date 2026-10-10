// Modified for fngine from cinference:src/ops/linear/fp8/shapes/n5120_k6144.cu @ 383e5db. See NOTICE.
// fngine: [2560,6144] with the GR residual-stream write (gdn.out / qsa.o).

#include "kernels/linear/fp8/fp8_a8_schedule.cuh"
#include "kernels/linear/fp8/fp8_launch.cuh"

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<2560, 6144>;
constexpr auto kKind = LinearEpilogue::GrWrite;
// Decode and verify share the same K partition and reduction order.
using Tile8  = Fp8A16KSplitSchedule<8, 8, 1, Fp8A16KSplitCache::Default,
                                    Fp8A16KSplitCache::Streaming,
                                    Fp8A16KSplitActivationStage::ActiveOnly, 2>;
using Tile16 = Fp8A16KSplitSchedule<8, 16, 1, Fp8A16KSplitCache::Default,
                                    Fp8A16KSplitCache::Streaming,
                                    Fp8A16KSplitActivationStage::ActiveOnly, 2>;

Fp8LaunchFn select_a16(std::int32_t tokens) {
    if (tokens <= 8) return launch_fp8_ksplit<Geometry, kKind, 8, Tile8>;
    return launch_fp8_ksplit<Geometry, kKind, 16, Tile16>;
}
} // namespace

const Fp8ShapeEntry kFp8N2560K6144{
    2560, 6144, kKind, launch_fp8_a16_narrow<Geometry, kKind, launch_fp8_a16_chunks<16, select_a16, Geometry>>,
    launch_fp8_a8<Geometry, kKind, Fp8A8SmallTokenSchedule, 128, Fp8A8DefaultSchedule>,
    kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
