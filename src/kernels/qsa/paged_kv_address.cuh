// Modified for fngine from cinference:src/ops/kernel/paged_kv_address.cuh @ 383e5db. See NOTICE.
#pragma once

// fngine serves one sequence, so the page table is the identity: position p lives in physical
// page p >> 6 at page offset p & 63. The PageMajor plane layout [page][head][64][Leading] is kept
// so a real block table could be reintroduced by changing paged_kv_physical_page only.

#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels {

inline constexpr int kPagedKVPageSize  = 64;
inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

__device__ __forceinline__ std::int32_t paged_kv_physical_page(std::int32_t position) {
    return position >> kPagedKVPageShift;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

// Offset of element `leading` of `head` at absolute `position` (identity page table).
template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_position_offset(std::int32_t head,
                                                                 std::int32_t position,
                                                                 std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(position), head, position & kPagedKVPageMask, leading);
}

} // namespace fngine::kernels
