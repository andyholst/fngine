// Gating launcher (modified for fngine from cinference:src/ops/launcher/gdn_gating.cu @ 383e5db,
// see NOTICE) and the new gated output norm.

#include "kernels/gdn/gating.cuh"
#include "kernels/gdn/gdn.h"

#include "core/device.h"
#include "kernels/common/bf16_vector.cuh"
#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace {

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

constexpr int kNormWarps = 4;

// One warp per (token, value head); each lane owns 4 consecutive dims. Programmatic dependent
// (pdl.cuh): the weight loads before pdl_wait(), core and z after it.
__global__ void __launch_bounds__(kNormWarps* kWarpSize)
    gdn_out_norm_kernel(const __nv_bfloat16* __restrict__ core, const __nv_bfloat16* __restrict__ z,
                        std::int64_t z_ld, const float* __restrict__ w,
                        __nv_bfloat16* __restrict__ out, std::int64_t rows) {
    pdl_trigger();
    const int lane         = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kNormWarps + threadIdx.x / kWarpSize;
    if (row >= rows) { return; }
    const std::int64_t t = row / gdn_dims::kValueHeads;
    const std::int64_t h = row - t * gdn_dims::kValueHeads;
    const int d          = lane * 4;

    const std::int64_t core_offset = t * gdn_dims::kValueDim + h * gdn_dims::kHeadDim + d;
    const float4 weight            = load_vec<float4>(w + d);
    pdl_wait();
    const Bf16x4Pack c_pack = load_vec<Bf16x4Pack>(pdl_coherent(core) + core_offset);
    const Bf16x4Pack z_pack =
        load_vec<Bf16x4Pack>(pdl_coherent(z) + t * z_ld + h * gdn_dims::kHeadDim + d);
    const float2 c01 = __bfloat1622float2(c_pack.pair[0]);
    const float2 c23 = __bfloat1622float2(c_pack.pair[1]);
    const float2 z01 = __bfloat1622float2(z_pack.pair[0]);
    const float2 z23 = __bfloat1622float2(z_pack.pair[1]);

    float sum = c01.x * c01.x;
    sum       = fmaf(c01.y, c01.y, sum);
    sum       = fmaf(c23.x, c23.x, sum);
    sum       = fmaf(c23.y, c23.y, sum);
    sum       = warp_sum(sum);
    const float inv = rsqrtf(sum * (1.0f / gdn_dims::kHeadDim) + 1.0e-6f);

    const auto gated = [&](float c, float wd, float zv) {
        const float n = __bfloat162float(__float2bfloat16_rn(c * inv));
        return n * wd * sigmoid(zv);
    };
    Bf16x4Pack result;
    result.pair[0] = __floats2bfloat162_rn(gated(c01.x, weight.x, z01.x), gated(c01.y, weight.y, z01.y));
    result.pair[1] = __floats2bfloat162_rn(gated(c23.x, weight.z, z23.x), gated(c23.y, weight.w, z23.y));
    store_vec(out + core_offset, result);
}

} // namespace

void gdn_gating(const float* ab, const float* A_log, const float* dt_bias, float* g, float* beta,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_gating";
    require(aligned(ab, 4) && aligned(A_log, 4) && aligned(dt_bias, 4) && aligned(g, 4) &&
                aligned(beta, 4),
            kOp, "invalid tensor pointer");
    require(tokens >= 1, kOp, "tokens must be positive");
    constexpr int kThreads = 256;
    const std::int64_t n   = static_cast<std::int64_t>(tokens) * gdn_dims::kValueHeads;
    CUDA_CHECK(pdl_launch(gdn::gating_kernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads)),
                          dim3(kThreads), 0, stream, ab, A_log, dt_bias, g, beta, n));
}

void gdn_out_norm(const __nv_bfloat16* core, const __nv_bfloat16* z, std::int32_t z_ld,
                  const float* w, __nv_bfloat16* out, std::int32_t tokens, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_out_norm";
    require(aligned(core, 8) && aligned(z, 8) && aligned(w, 16) && aligned(out, 8), kOp,
            "invalid tensor pointer");
    require(z_ld >= gdn_dims::kValueDim && z_ld % 4 == 0, kOp, "invalid z row stride");
    require(tokens >= 1, kOp, "tokens must be positive");
    const std::int64_t rows = static_cast<std::int64_t>(tokens) * gdn_dims::kValueHeads;
    CUDA_CHECK(pdl_launch(gdn_out_norm_kernel, dim3(static_cast<unsigned>((rows + kNormWarps - 1) / kNormWarps)),
                          dim3(kNormWarps * kWarpSize), 0, stream, core, z, std::int64_t(z_ld), w, out, rows));
}

} // namespace fngine::kernels
