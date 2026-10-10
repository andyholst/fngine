// fngine: MTP draft head [32768,2560] -> FP32 logits (the draft-vocabulary rows of lm_head).
// Decode/verify widths only (the MTP bridge runs at most 16 columns); same small-T schedules as
// lm_head.

#include "kernels/linear/fp8/fp8_launch.cuh"

#include <stdexcept>

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<32768, 2560>;
constexpr auto kKind = LinearEpilogue::StoreF32;
using Gemv = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using KSplit8  = Fp8A16KSplitSchedule<8, 8, 1>;
using KSplit16 = Fp8A16KSplitSchedule<8, 16, 1>;

void launch_a16(const Fp8Call& call) {
    if (call.tokens == 1) return launch_fp8_gemv<Geometry, kKind, Gemv>(call);
    if (call.tokens <= 8) return launch_fp8_ksplit<Geometry, kKind, 8, KSplit8>(call);
    if (call.tokens <= 16) return launch_fp8_ksplit<Geometry, kKind, 16, KSplit16>(call);
    throw std::invalid_argument("draft head: at most 16 columns");
}
} // namespace

const Fp8ShapeEntry kFp8N32768K2560{32768, 2560, kKind, launch_a16, nullptr, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
