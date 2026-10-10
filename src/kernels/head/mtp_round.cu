// Modified for fngine from cinference:src/ops/kernel/mtp_round.cuh @ 383e5db. See NOTICE.
// (Also merges cinference:src/ops/launcher/mtp_round.cu.) Single request; no mRoPE deltas.

#include "core/device.h"
#include "kernels/head/head.h"

#include <stdexcept>

namespace fngine::kernels {
namespace {

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchor, const std::int32_t* accepted,
    const std::int32_t* updated_frontier, const std::int32_t* remaining_budget,
    const std::int32_t* licensed_count, std::int32_t* alignment_ids, std::int32_t* next_extent,
    std::int32_t* ar_positions, std::int32_t* ar_valid, std::int32_t k,
    std::int32_t max_context) {
    const int T = k + 1;
    int a       = *accepted;
    a           = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(threadIdx.x); j < T; j += blockDim.x) {
        alignment_ids[j] = j < a ? verify_ids[j + 1] : *next_anchor;
    }
    if (threadIdx.x == 0) {
        const int remaining      = *remaining_budget - *licensed_count;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - *updated_frontier - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        *next_extent             = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            ar_positions[s] = *updated_frontier + s;
            ar_valid[s]     = s + 1 < next ? 1 : 0;
        }
    }
}

} // namespace

void mtp_prepare_next_round(const std::int32_t* verify_ids, std::int32_t k,
                            const std::int32_t* next_anchor, const std::int32_t* accepted,
                            const std::int32_t* updated_frontier,
                            const std::int32_t* remaining_budget,
                            const std::int32_t* licensed_count, std::int32_t max_context,
                            std::int32_t* alignment_ids, std::int32_t* next_extent,
                            std::int32_t* ar_positions, std::int32_t* ar_valid,
                            cudaStream_t stream) {
    if (k < 1 || k + 1 > 32) {
        throw std::invalid_argument("mtp_prepare_next_round: k must be in [1, 31]");
    }
    mtp_prepare_next_round_kernel<<<1, 32, 0, stream>>>(
        verify_ids, next_anchor, accepted, updated_frontier, remaining_budget, licensed_count,
        alignment_ids, next_extent, ar_positions, ar_valid, k, max_context);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace fngine::kernels
