#include "kernels/gdn/gdn_test_common.h"

namespace fngine::test::gdn {
namespace {

__global__ void count_mismatch_kernel(const std::uint32_t* a, const std::uint32_t* b, size_t n,
                                      unsigned long long* count) {
    unsigned long long local = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        local += a[i] != b[i];
    if (local) atomicAdd(count, local);
}

}  // namespace

unsigned long long device_mismatches(const void* a, const void* b, size_t bytes) {
    if (bytes % 4 != 0) throw std::invalid_argument("device_mismatches: bytes % 4 != 0");
    unsigned long long* d = nullptr;
    FNGINE_TEST_CUDA(cudaMalloc(&d, sizeof(*d)));
    FNGINE_TEST_CUDA(cudaMemset(d, 0, sizeof(*d)));
    count_mismatch_kernel<<<512, 256>>>(static_cast<const std::uint32_t*>(a),
                                        static_cast<const std::uint32_t*>(b), bytes / 4, d);
    FNGINE_TEST_CUDA(cudaGetLastError());
    unsigned long long h = 0;
    FNGINE_TEST_CUDA(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
    FNGINE_TEST_CUDA(cudaFree(d));
    return h;
}

}  // namespace fngine::test::gdn
