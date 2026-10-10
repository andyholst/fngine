// Modified for fngine from cinference:src/ops/linear/fp8/shapes/n248320_k5120.cu @ 383e5db. See NOTICE.
// fngine: lm_head [248320,2560] -> FP32 logits; A16 only (BF16 activations keep logit precision).

#include "kernels/linear/fp8/fp8_launch.cuh"

#include <limits>

namespace fngine::kernels::detail {
namespace {
using Geometry       = Fp8Geometry<248320, 2560>;
constexpr auto kKind = LinearEpilogue::StoreF32;
using Gemv = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;

template <int Tile>
using KSplit = Fp8A16KSplitSchedule<(Tile <= 8 ? 8 : (Tile <= 24 ? 8 : 4)), Tile, Tile <= 8 ? 1 : 2>;

// Measured cinference schedules for [248320,K]: the 128-token schedule is the large-T core; the
// 64- and 96-token schedules avoid a mostly empty final token tile.
using Main128 = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
using Tail64  = Fp8A16GemmSchedule<128, 64, 64, 64, 16, 2, 2>;
using Tail96  = Fp8A16GemmSchedule<64, 96, 64, 64, 16, 2, 2>;

void launch_small(const Fp8Call& call) {
    if (call.tokens == 1) return launch_fp8_gemv<Geometry, kKind, Gemv>(call);
    if (call.tokens <= 8) return launch_fp8_ksplit<Geometry, kKind, 8, KSplit<8>>(call);
    if (call.tokens <= 16) return launch_fp8_ksplit<Geometry, kKind, 16, KSplit<16>>(call);
    if (call.tokens <= 24) return launch_fp8_ksplit<Geometry, kKind, 24, KSplit<24>>(call);
    if (call.tokens <= 32) return launch_fp8_ksplit<Geometry, kKind, 32, KSplit<32>>(call);
    if (call.tokens <= 40) return launch_fp8_ksplit<Geometry, kKind, 40, KSplit<40>>(call);
    return launch_fp8_ksplit<Geometry, kKind, 48, KSplit<48>>(call);
}

constexpr std::int32_t kGemmMinTokens = 42;

void launch_tail(const Fp8Call& call) {
    if (call.tokens < kGemmMinTokens) return launch_small(call);
    if (call.tokens <= Tail64::kBlockTokens) return launch_fp8_a16_gemm<Geometry, kKind, Tail64>(call);
    launch_fp8_a16_gemm<Geometry, kKind, Tail96>(call);
}

void launch_a16(const Fp8Call& call) {
    const std::int32_t tokens = call.tokens;
    if (tokens < kGemmMinTokens) return launch_small(call);
    if (tokens <= Tail64::kBlockTokens) return launch_fp8_a16_gemm<Geometry, kKind, Tail64>(call);
    if (tokens <= Tail96::kBlockTokens) return launch_fp8_a16_gemm<Geometry, kKind, Tail96>(call);
    if (tokens <= Main128::kBlockTokens) return launch_fp8_a16_gemm<Geometry, kKind, Main128>(call);
    if ((tokens >= 161 && tokens <= 192) || (tokens >= 257 && tokens <= 288)) {
        return launch_fp8_a16_gemm<Geometry, kKind, Tail96>(call);
    }
    const std::int32_t tail = tokens % Main128::kBlockTokens;
    if (tail == 0 || tail > Tail96::kBlockTokens) {
        return launch_fp8_a16_gemm<Geometry, kKind, Main128>(call);
    }
    Fp8Call prefix = call;
    prefix.tokens  = tokens - tail;
    launch_fp8_a16_gemm<Geometry, kKind, Main128>(prefix);

    const LinearEpilogueParams epilogue =
        offset_epilogue_tokens(*call.epilogue, prefix.tokens, Geometry::kOutputRows);
    Fp8Call rest  = call;
    rest.x        = call.x + static_cast<std::int64_t>(prefix.tokens) * Geometry::kInputRows;
    rest.tokens   = tail;
    rest.epilogue = &epilogue;
    launch_tail(rest);
}
} // namespace

const Fp8ShapeEntry kFp8N248320K2560{248320, 2560, kKind, launch_a16, nullptr, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
