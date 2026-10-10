// Modified for fngine from cinference:src/ops/linear_attention/gated_delta_net/recurrent.cu and
// cinference:src/ops/linear_attention/gated_delta_net/replay.cpp @ 383e5db. See NOTICE.

#include "kernels/gdn/gdn.h"
#include "kernels/gdn/recurrent.cuh"

#include "core/device.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace {

using gdn::StagedMode;
using gdn::StagedParams;

constexpr dim3 kBlock(kWarpSize, gdn::kNumWarps, 1);

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_pool(const GdnStatePool& pool, std::int32_t layer, const GdnStepControl* control,
                  const char* op) {
    require(aligned(pool.recurrent, 16) && pool.slots > 0, op, "invalid state pool");
    require(layer >= 0 && layer < gdn_dims::kLayers, op, "layer out of range");
    require(aligned(control, 16), op, "control must be a 16-byte aligned device pointer");
}

void require_records(const GdnRecords& records, const char* op) {
    require(aligned(records.conv, 16) && aligned(records.key, 16) && aligned(records.value, 16) &&
                aligned(records.gate, 16),
            op, "invalid record storage");
}

StagedParams source_params(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                           const GdnStatePool& pool, std::int32_t layer,
                           const GdnStepControl* control, __nv_bfloat16* core,
                           std::int32_t width, const char* op) {
    require_pool(pool, layer, control, op);
    require(aligned(conv_out, 16) && aligned(g, 4) && aligned(beta, 4) && aligned(core, 2), op,
            "invalid tensor pointer");
    StagedParams p{};
    p.q         = conv_out;
    p.k         = conv_out + gdn_dims::kKeyDim;
    p.v         = conv_out + 2 * gdn_dims::kKeyDim;
    p.qkv_ld    = gdn_dims::kConvChannels;
    p.g         = g;
    p.beta      = beta;
    p.recurrent = pool.recurrent;
    p.control   = control;
    p.out       = core;
    p.layer     = layer;
    p.width     = width;
    return p;
}

__global__ void gdn_swap_slots_kernel(GdnStepControl* control) {
    const std::int32_t source  = control->source_slot;
    control->source_slot      = control->destination_slot;
    control->destination_slot = source;
}

} // namespace

void gdn_recurrent_decode(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                          const GdnStatePool& pool, std::int32_t layer,
                          const GdnStepControl* control, __nv_bfloat16* core,
                          cudaStream_t stream) {
    const StagedParams p =
        source_params(conv_out, g, beta, pool, layer, control, core, 1, "gdn_recurrent_decode");
    const dim3 grid(gdn::kValueHeadCount, 1, gdn::kStateTiles);
    CUDA_CHECK(pdl_launch(gdn::gdn_staged_kernel<StagedMode::Decode>, grid, kBlock, 0, stream, p));
}

void gdn_recurrent_record(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                          const GdnStatePool& pool, std::int32_t layer,
                          const GdnStepControl* control, const GdnRecords& records,
                          std::int32_t tokens, __nv_bfloat16* core, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_recurrent_record";
    require(tokens >= 1 && tokens <= gdn_dims::kMaxRecordWidth, kOp, "tokens must be 1..16");
    require_records(records, kOp);
    StagedParams p = source_params(conv_out, g, beta, pool, layer, control, core, tokens, kOp);
    p.key_record   = records.key;
    p.value_record = records.value;
    p.gate_record  = reinterpret_cast<uint2*>(records.gate);
    const dim3 grid(gdn::kValueHeadCount, 1, gdn::kStateTiles);
    CUDA_CHECK(pdl_launch(gdn::gdn_staged_kernel<StagedMode::Record>, grid, kBlock, 0, stream, p));
}

void gdn_fold(const GdnStatePool& pool, const GdnRecords& records, const GdnStepControl* control,
              std::int32_t width, cudaStream_t stream) {
    constexpr const char* kOp = "gdn_fold";
    require_pool(pool, 0, control, kOp);
    require(aligned(pool.conv, 16), kOp, "invalid conv state pool");
    require_records(records, kOp);
    require(width >= 1 && width <= gdn_dims::kMaxRecordWidth, kOp, "width must be 1..16");
    StagedParams p{};
    p.key_record   = records.key;
    p.value_record = records.value;
    p.gate_record  = reinterpret_cast<uint2*>(records.gate);
    p.conv_record  = records.conv;
    p.recurrent    = pool.recurrent;
    p.conv_state   = pool.conv;
    p.control      = control;
    p.width        = width;
    const dim3 grid(gdn::kValueHeadCount, 1, gdn_dims::kLayers * gdn::kStateTiles);
    CUDA_CHECK(pdl_launch(gdn::gdn_staged_kernel<StagedMode::Fold>, grid, kBlock, 0, stream, p));
}

void gdn_swap_slots(GdnStepControl* control, cudaStream_t stream) {
    require(aligned(control, 16), "gdn_swap_slots", "invalid control pointer");
    gdn_swap_slots_kernel<<<1, 1, 0, stream>>>(control);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace fngine::kernels
