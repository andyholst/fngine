// Modified for fngine from cinference:src/ops/linear/fp8/fp8_a8.cu @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: share the token quantization with fused producers.
// See NOTICE and upstream-provenance.json for upstream attribution.
// fngine: raw pointers; the fngine activation widths; thread count chosen per width.

#include "kernels/linear/fp8/fp8_a8.h"

#include "core/device.h"
#include "kernels/common/math.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/linear/fp8/fp8_a8_quantize.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels::detail {
namespace {

template <int InputRows, int Threads>
__global__ __launch_bounds__(Threads) void fp8_a8_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales) {
    static_assert((InputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_token  = InputRows / 2;
    constexpr int pairs_per_thread = pairs_per_token / Threads;
    constexpr int warps            = Threads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float token_scale;

    const int token         = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    const int lane          = tid & 31;
    const int warp          = tid >> 5;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(
        input + static_cast<std::int64_t>(token) * InputRows);
    auto* output_pairs =
        reinterpret_cast<std::uint16_t*>(codes + static_cast<std::int64_t>(token) * InputRows);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        values[item]   = bf16x2_bits_to_float2(input_pairs[pair]);
        maximum        = fmaxf(maximum, fabsf(values[item].x));
        maximum        = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = warp_max(maximum);
        if (lane == 0) { token_scale = fp8_a8_token_scale(maximum); }
    }
    __syncthreads();

    const float scale   = token_scale;
    const float inverse = fp8_a8_inverse_scale(scale);
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair     = tid + item * Threads;
        output_pairs[pair] = fp8_a8_encode_pair(values[item], inverse);
    }
    if (tid == 0) { scales[token] = scale; }
}

template <int InputRows, int Threads>
void launch_exact(const __nv_bfloat16* x, std::int32_t tokens, Fp8A8Workspace workspace,
                  cudaStream_t stream) {
    fp8_a8_quantize_kernel<InputRows, Threads>
        <<<tokens, Threads, 0, stream>>>(x, workspace.codes, workspace.scales);
    CUDA_CHECK(cudaGetLastError());
}

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

} // namespace

std::size_t fp8_a8_workspace_bytes(std::int32_t tokens, std::int32_t input_rows) {
    if (tokens <= 0 || input_rows <= 0) {
        throw std::invalid_argument("fp8 A8 workspace: T and K must be positive");
    }
    return align256(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(input_rows)) +
           align256(static_cast<std::size_t>(tokens) * sizeof(float));
}

Fp8A8Workspace fp8_a8_workspace(void* base, std::size_t bytes, std::int32_t tokens,
                                std::int32_t input_rows) {
    const std::size_t need = fp8_a8_workspace_bytes(tokens, input_rows);
    if (base == nullptr || bytes < need) {
        throw std::invalid_argument("fp8 A8 linear: workspace is missing or too small");
    }
    if ((reinterpret_cast<std::uintptr_t>(base) & 255) != 0) {
        throw std::invalid_argument("fp8 A8 linear: workspace must be 256-byte aligned");
    }
    auto* bytes_base = static_cast<std::uint8_t*>(base);
    return {bytes_base,
            reinterpret_cast<float*>(bytes_base + align256(static_cast<std::size_t>(tokens) *
                                                           static_cast<std::size_t>(input_rows)))};
}

void launch_fp8_a8_quantize(const __nv_bfloat16* x, std::int32_t tokens, std::int32_t input_rows,
                            Fp8A8Workspace workspace, cudaStream_t stream) {
    switch (input_rows) {
    case 384: return launch_exact<384, 64>(x, tokens, workspace, stream);
    case 640: return launch_exact<640, 64>(x, tokens, workspace, stream);
    case 2560: return launch_exact<2560, 256>(x, tokens, workspace, stream);
    case 6144: return launch_exact<6144, 256>(x, tokens, workspace, stream);
    case 10240: return launch_exact<10240, 256>(x, tokens, workspace, stream);
    default: throw std::invalid_argument("fp8 A8 quantize: unsupported K");
    }
}

} // namespace fngine::kernels::detail
