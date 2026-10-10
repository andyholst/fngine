// Modified for fngine from cinference:src/ops/linear/fp8/fp8_shapes.h @ 383e5db. See NOTICE.
// fngine: entries carry the fused epilogue and the measured A8 threshold.

#pragma once

#include "kernels/linear/fp8/fp8_a8.h"
#include "kernels/linear/linear.h"
#include "kernels/linear/linear_params.h"

#include <cstdint>
#include <limits>

namespace fngine::kernels::detail {

struct Fp8Call {
    const __nv_bfloat16* x;
    std::int32_t tokens;
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
    const LinearEpilogueParams* epilogue;
    Fp8A8Workspace a8;
    cudaStream_t stream;
};

using Fp8LaunchFn = void (*)(const Fp8Call&);

struct Fp8ShapeEntry {
    std::int32_t n, k;
    LinearEpilogue epilogue;
    Fp8LaunchFn a16;               // any T >= 1
    Fp8LaunchFn a8;                // any T >= 1 (expects quantization workspace); null if none
    std::int32_t a8_min_tokens;    // Auto selects A8 for T >= this
};

// A16 launches use the decode/verify kernels (GEMV, SIMT, K-split MMA) up to this width and the
// tiled A16 GEMM above it.
inline constexpr std::int32_t kFp8A16MaxVerifyTokens = 16;

// Auto keeps BF16 activations (A16) at every T on every shape: A8's per-token E4M3 activations
// add 1-4% rms per projection (measured on the 8192-token golden: mean KLD 0.049 with A8 prefill
// vs 0.034 with A16), while the A16 GEMM stays within 1.3-2.0x of A8 at T = 4096 (rule: keep A16
// unless > 2.5x slower). Fp8LinearPath::A8 remains available explicitly.
inline constexpr std::int32_t kFp8AutoNeverA8 = std::numeric_limits<std::int32_t>::max();

extern const Fp8ShapeEntry kFp8N16384K2560;
extern const Fp8ShapeEntry kFp8N13312K2560;
extern const Fp8ShapeEntry kFp8N2560K6144;
extern const Fp8ShapeEntry kFp8N1280K2560;
extern const Fp8ShapeEntry kFp8N2560K640;
extern const Fp8ShapeEntry kFp8N320K10240;
extern const Fp8ShapeEntry kFp8N10240K384;
extern const Fp8ShapeEntry kFp8N10240K2560;
extern const Fp8ShapeEntry kFp8N2560K2560;
extern const Fp8ShapeEntry kFp8N248320K2560;
extern const Fp8ShapeEntry kFp8N32768K2560;  // MTP draft head (draft-vocabulary lm_head rows)

} // namespace fngine::kernels::detail
