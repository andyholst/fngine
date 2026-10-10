// Modified for fngine from cinference:src/ops/common/sampling_workspace.h @ 383e5db. See NOTICE.
#pragma once

// Private workspace layout shared by token sampling and stochastic speculative acceptance.
// The same builder supplies the public sizing query and launcher-side binding.
//
// fngine runs one request whose logits are FP32 [T][248320]; every route is the two-launch
// partial/group pipeline (token domains in (256, 248320], at most 16 columns).

#include "kernels/common/math.h"

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

inline constexpr int kSamplerBlock               = 256;
inline constexpr int kSamplerTileItems           = 256;
inline constexpr int kSamplerItemsPerThread      = 2;
inline constexpr int kSamplerPartialTileItems    = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerGroupBlock          = 256;
inline constexpr int kSamplerGroupItemsPerThread = 2;
inline constexpr int kSamplerGroupTileItems      = kSamplerGroupBlock * kSamplerGroupItemsPerThread;
inline constexpr int kSamplerPartialsPerGroup    = 25;
inline constexpr int kSamplerCandidateCap        = 20;
inline constexpr int kSamplerMaxColumns          = 16;

static_assert(kSamplerPartialsPerGroup * kSamplerCandidateCap <= kSamplerGroupTileItems,
              "group merge tile must hold one group's candidates");

#if defined(__CUDACC__)
#    define FNGINE_SAMPLER_HD __host__ __device__
#else
#    define FNGINE_SAMPLER_HD
#endif

FNGINE_SAMPLER_HD inline int sampler_partial_blocks(int token_domain) {
    return div_up(token_domain, kSamplerPartialTileItems);
}

FNGINE_SAMPLER_HD inline int sampler_group_count(int partial_blocks) {
    return div_up(partial_blocks, kSamplerPartialsPerGroup);
}

#undef FNGINE_SAMPLER_HD

// The multi-block route is deliberately finite: a single final merge tile must hold every group
// candidate, and at most sixteen columns share one launch.
inline bool sampler_route_ok(int token_domain, int cols) {
    if (token_domain <= kSamplerTileItems || cols <= 0 || cols > kSamplerMaxColumns) {
        return false;
    }
    const int groups = sampler_group_count(sampler_partial_blocks(token_domain));
    return groups * kSamplerCandidateCap <= kSamplerGroupTileItems;
}

struct SamplingWorkspace {
    unsigned long long* partial_keys         = nullptr; // [cols][partial_stride][20]
    std::int32_t* dist_idx                   = nullptr; // [cols][20]
    float* dist_prob                         = nullptr; // [cols][20]
    std::int32_t* dist_support               = nullptr; // [cols]
    std::int32_t* group_done                 = nullptr; // [cols]
    std::int32_t* speculative_finalize_count = nullptr; // [1]
    std::int32_t partial_stride              = 0;
};

struct SamplingWorkspaceLayout {
    std::size_t partial_keys               = 0;
    std::size_t dist_idx                   = 0;
    std::size_t dist_prob                  = 0;
    std::size_t dist_support               = 0;
    std::size_t group_done                 = 0;
    std::size_t speculative_finalize_count = 0;
    std::size_t bytes                      = 0;
    std::int32_t partial_stride            = 0;

    [[nodiscard]] SamplingWorkspace bind(void* backing) const {
        auto* base = static_cast<unsigned char*>(backing);
        return {
            reinterpret_cast<unsigned long long*>(base + partial_keys),
            reinterpret_cast<std::int32_t*>(base + dist_idx),
            reinterpret_cast<float*>(base + dist_prob),
            reinterpret_cast<std::int32_t*>(base + dist_support),
            reinterpret_cast<std::int32_t*>(base + group_done),
            reinterpret_cast<std::int32_t*>(base + speculative_finalize_count),
            partial_stride,
        };
    }
};

// Requires sampler_route_ok(token_domain, columns).
inline SamplingWorkspaceLayout make_sampling_workspace_layout(std::int32_t token_domain,
                                                              std::int32_t columns) {
    SamplingWorkspaceLayout out;
    const std::int32_t partial_blocks = sampler_partial_blocks(token_domain);
    out.partial_stride = partial_blocks + sampler_group_count(partial_blocks);

    std::size_t offset = 0;
    const auto add     = [&](std::size_t bytes) {
        const std::size_t at = offset;
        offset               = align_up<256>(offset + bytes);
        return at;
    };
    out.partial_keys = add(sizeof(unsigned long long) * kSamplerCandidateCap *
                           static_cast<std::size_t>(out.partial_stride) * columns);
    out.dist_idx     = add(sizeof(std::int32_t) * kSamplerCandidateCap * columns);
    out.dist_prob    = add(sizeof(float) * kSamplerCandidateCap * columns);
    out.dist_support = add(sizeof(std::int32_t) * columns);
    out.group_done   = add(sizeof(std::int32_t) * columns);
    out.speculative_finalize_count = add(sizeof(std::int32_t));
    out.bytes                      = offset;
    return out;
}

} // namespace fngine::kernels
