// Modified for fngine from cinference:src/ops/launcher/causal_conv1d.cu @ 383e5db. See NOTICE.

#include "kernels/gdn/causal_conv1d.cuh"
#include "kernels/gdn/gdn.h"

#include "core/device.h"

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

void require_common(const __nv_bfloat16* qkvz, const float* conv_w, const __nv_bfloat16* conv_out,
                    const char* op) {
    require(aligned(qkvz, 4) && aligned(conv_w, 16) && aligned(conv_out, 4), op,
            "invalid tensor pointer");
}

void require_pool(const GdnStatePool& pool, std::int32_t layer, const GdnStepControl* control,
                  const char* op) {
    require(aligned(pool.conv, 4) && pool.slots > 0, op, "invalid state pool");
    require(layer >= 0 && layer < gdn_dims::kLayers, op, "layer out of range");
    require(aligned(control, 16), op, "invalid control pointer");
}

constexpr int kThreads = 256;

} // namespace

void gdn_conv_decode(const __nv_bfloat16* qkvz, const float* conv_w, const GdnStatePool& pool,
                     std::int32_t layer, const GdnStepControl* control, __nv_bfloat16* conv_out,
                     cudaStream_t stream) {
    constexpr const char* kOp = "gdn_conv_decode";
    require_common(qkvz, conv_w, conv_out, kOp);
    require_pool(pool, layer, control, kOp);
    CUDA_CHECK(pdl_launch(gdn::conv_decode_kernel, dim3(gdn_dims::kConvChannels / kThreads),
                          dim3(kThreads), 0, stream, qkvz, conv_w, pool.conv, layer, control, conv_out));
}

void gdn_conv_record(const __nv_bfloat16* qkvz, const float* conv_w, const GdnStatePool& pool,
                     std::int32_t layer, const GdnStepControl* control, const GdnRecords& records,
                     std::int32_t tokens, __nv_bfloat16* conv_out, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_conv_record";
    require_common(qkvz, conv_w, conv_out, kOp);
    require_pool(pool, layer, control, kOp);
    require(aligned(records.conv, 4), kOp, "invalid conv record");
    require(tokens >= 1 && tokens <= gdn_dims::kMaxRecordWidth, kOp, "tokens must be 1..16");
    const dim3 grid(gdn_dims::kConvChannels / gdn::kConvChannelTile);
    const dim3 block(gdn::kConvChannelTile, static_cast<unsigned>(tokens));
    CUDA_CHECK(pdl_launch(gdn::conv_record_kernel, grid, block, 0, stream, qkvz, conv_w, pool.conv,
                          layer, control, records.conv, conv_out));
}

void gdn_conv_prefill(const __nv_bfloat16* qkvz, const float* conv_w,
                      const __nv_bfloat16* state_in, __nv_bfloat16* state_out,
                      __nv_bfloat16* conv_out, std::int32_t tokens, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_conv_prefill";
    require_common(qkvz, conv_w, conv_out, kOp);
    require(aligned(state_in, 4) && aligned(state_out, 4), kOp, "invalid state pointer");
    require(tokens >= 1, kOp, "tokens must be positive");
    constexpr std::int64_t pair_blocks = (gdn_dims::kConvChannels / 2 + kThreads - 1) / kThreads;
    const std::int64_t blocks          = pair_blocks * tokens;
    require(blocks <= 0x7fffffff, kOp, "grid too large");
    gdn::conv_prefill_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        qkvz, conv_w, state_in, conv_out, tokens);
    CUDA_CHECK(cudaGetLastError());
    gdn::conv_prefill_state_kernel<<<gdn_dims::kConvChannels / kThreads, kThreads, 0, stream>>>(
        qkvz, state_in, state_out, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
