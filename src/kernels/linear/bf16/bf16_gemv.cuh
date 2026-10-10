#pragma once

// BF16 weight x BF16 activation CUDA-core GEMV for small T (<= 16), FP32 accumulation. Each warp
// owns RowsPerWarp rows over one K slice; lanes walk K in 8-value (16-byte) packs, so any K that is
// a multiple of 8 (per slice) is accepted, as is any N (rows >= n are not loaded or stored for
// elementwise outputs). K-split warps are reduced in a fixed order (deterministic).

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/linear/linear_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels::detail {

struct Bf16IdentityRows {
    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

// Pair rows: local rows below RowsPerBranch are gate rows, the rest the matching up rows.
template <int RowsPerBranch>
struct Bf16PairRows {
    int intermediate;
    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row % RowsPerBranch) +
               (local_row >= RowsPerBranch ? intermediate : 0);
    }
};

struct Bf16UnitScale {
    __device__ __forceinline__ float operator()(int) const { return 1.0F; }
};

// Programmatic dependent (pdl.cuh): each lane's first weight step loads before pdl_wait(), the
// activations after it; the accumulation order does not depend on it.
template <int Tokens, int RowsPerWarp, int RowWarps, int KSplit, class Output, class RowPolicy,
          bool PairRows>
__global__ __launch_bounds__(RowWarps* KSplit * 32) void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    std::int32_t n, std::int32_t k, std::int32_t live_tokens, Output output,
    RowPolicy row_policy) {
    constexpr int kStoredPerWarp = PairRows ? RowsPerWarp / 2 : RowsPerWarp;
    using Reduce                 = WarpKSplitReduce<KSplit, RowWarps, RowsPerWarp * Tokens>;
    __shared__ typename Reduce::Storage reduce_storage;

    pdl_trigger();
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int row_warp  = warp % RowWarps;
    const int k_slice   = warp / RowWarps;
    const int row_begin = (static_cast<int>(blockIdx.x) * RowWarps + row_warp) * kStoredPerWarp;
    // The pair/group policies are only registered for N that the grid tiles exactly.
    const int row_limit = PairRows ? n / 2 : n;
    const int slice_k   = k / KSplit;
    const int k_begin   = k_slice * slice_k;
    const int k_end     = k_begin + slice_k;

    const auto load_weights = [&](int kk, float (&w)[RowsPerWarp][8]) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            const int row = row_policy.weight_row(row_begin, r);
            uint4 packed  = make_uint4(0, 0, 0, 0);
            if (PairRows || row < n) {
                packed = load_vec<uint4>(weight + static_cast<std::int64_t>(row) * k + kk);
            }
            const float2 a = bf16x2_bits_to_float2(packed.x);
            const float2 b = bf16x2_bits_to_float2(packed.y);
            const float2 c = bf16x2_bits_to_float2(packed.z);
            const float2 d = bf16x2_bits_to_float2(packed.w);
            w[r][0] = a.x, w[r][1] = a.y, w[r][2] = b.x, w[r][3] = b.y;
            w[r][4] = c.x, w[r][5] = c.y, w[r][6] = d.x, w[r][7] = d.y;
        }
    };

    float acc[RowsPerWarp][Tokens] = {};
    float w[RowsPerWarp][8];
    int kk = k_begin + lane * 8;
    if (kk < k_end) load_weights(kk, w);
    pdl_wait();
    const __nv_bfloat16* const xs = pdl_coherent(x);
    while (kk < k_end) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const int source = t < live_tokens ? t : live_tokens - 1;
            const uint4 packed =
                load_vec<uint4>(xs + static_cast<std::int64_t>(source) * k + kk);
            const float2 a = bf16x2_bits_to_float2(packed.x);
            const float2 b = bf16x2_bits_to_float2(packed.y);
            const float2 c = bf16x2_bits_to_float2(packed.z);
            const float2 d = bf16x2_bits_to_float2(packed.w);
            const float v[8] = {a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][t] = fmaf(w[r][i], v[i], acc[r][t]);
            }
        }
        kk += 256;
        if (kk < k_end) load_weights(kk, w);
    }

    // Butterfly reduction: every lane holds every total; lane t finishes token t.
    static_assert(Tokens <= 32);
    float totals[RowsPerWarp * Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) totals[t * RowsPerWarp + r] = warp_sum(acc[r][t]);
    }
    if (!Reduce::template run<true>(reduce_storage, row_warp, k_slice, lane, totals)) return;
    if (row_begin >= row_limit) return;
    float row_totals[RowsPerWarp];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        if (lane == t) {
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) row_totals[r] = totals[t * RowsPerWarp + r];
        }
    }
    if (lane < Tokens && lane < live_tokens) {
        finish_token<PairRows>(output, row_policy, Bf16UnitScale{}, row_begin, lane, row_totals, n);
    }
}

// Applies an epilogue policy to an FP32 GEMM result acc [T][n] (one thread per stored unit:
// row, pair, or 4-row group).
template <class Output, bool PairRows>
__global__ __launch_bounds__(256) void bf16_epilogue_kernel(const float* __restrict__ acc,
                                                            std::int32_t n, std::int32_t units,
                                                            Output output) {
    const int unit  = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (unit >= units) return;
    const float* row = acc + static_cast<std::int64_t>(token) * n;
    if constexpr (PairRows) {
        output.store_pair(unit, token, row[unit], row[unit + n / 2]);
    } else if constexpr (output_row_group<Output>() == 4) {
        output.store_group4(unit * 4, token, load_vec<float4>(row + unit * 4));
    } else {
        output.store(unit, token, row[unit]);
    }
}

} // namespace fngine::kernels::detail
