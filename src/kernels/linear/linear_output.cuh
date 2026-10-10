// Modified for fngine from cinference:src/ops/linear/fp8/fp8_output.cuh,
// src/ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh and
// src/ops/linear_add/fp8/fp8_linear_add_epilogue.cuh @ 383e5db. See NOTICE.
#pragma once

// Output policies shared by every FP8 and BF16 projection kernel. A kernel finishes the complete K
// reduction of an output element (row scale applied) in FP32 and hands it to the policy:
//   store(row, token, v)                 elementwise policies
//   store8(row, token, v[8])             optional vector form of 8 consecutive rows
//   store_pair(row, token, gate, up)     PairRows policies (SwiGLU); `row` is the stored row
//   store_group4(row, token, float4)     kRowGroup == 4 policies (GrMix); `row` % 4 == 0
//   Preload preload(row, token)          optional: epilogue inputs of a unit (GrWrite X / inj,
//                                        GrMix Xn) that a kernel may fetch before its main loop
//                                        and pass back as the last argument of store / store_group4
// Each output element is produced by exactly one call, so read-modify-write policies (GrWrite)
// are race-free and deterministic.

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/linear/linear.h"

#include <cuda_bf16.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace fngine::kernels::detail {

template <class Output>
constexpr int output_row_group() {
    if constexpr (requires { Output::kRowGroup; }) {
        return Output::kRowGroup;
    } else {
        return 1;
    }
}

template <class Output>
constexpr bool output_has_store8() {
    return requires(const Output& o, const float (&v)[8]) { o.store8(0, 0, v); };
}

struct OutBf16 {
    __nv_bfloat16* data;
    std::int32_t ld;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v) const {
        data[static_cast<std::int64_t>(token) * ld + row] = __float2bfloat16_rn(v);
    }
    __device__ __forceinline__ void store8(std::int32_t row, std::int32_t token,
                                           const float (&v)[8]) const {
        const uint4 packed = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                                        pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
        store_vec(data + static_cast<std::int64_t>(token) * ld + row, packed);
    }
};

struct OutF32 {
    float* data;
    std::int32_t ld;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v) const {
        data[static_cast<std::int64_t>(token) * ld + row] = v;
    }
    __device__ __forceinline__ void store8(std::int32_t row, std::int32_t token,
                                           const float (&v)[8]) const {
        float* d = data + static_cast<std::int64_t>(token) * ld + row;
        store_vec(d, make_float4(v[0], v[1], v[2], v[3]));
        store_vec(d + 4, make_float4(v[4], v[5], v[6], v[7]));
    }
};

// X[t][s][row] += inj[t][s] * v for the four residual streams (X FP32 [T][4][rows]).
// Preload: the row's four X values and the token's inj, fetched before a kernel's main loop.
struct OutGrWrite {
    struct Preload {
        float4 inj;
        float4 x;
    };
    float* x;
    const float* inj;
    std::int32_t rows;

    __device__ __forceinline__ Preload preload(std::int32_t row, std::int32_t token) const {
        const float* base = x + static_cast<std::int64_t>(token) * 4 * rows + row;
        return {load_vec<float4>(inj + static_cast<std::int64_t>(token) * 4),
                make_float4(base[0], base[rows], base[2 * rows], base[3 * rows])};
    }
    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v,
                                          const Preload& p) const {
        float* base    = x + static_cast<std::int64_t>(token) * 4 * rows + row;
        base[0]        = fmaf(p.inj.x, v, p.x.x);
        base[rows]     = fmaf(p.inj.y, v, p.x.y);
        base[2 * rows] = fmaf(p.inj.z, v, p.x.z);
        base[3 * rows] = fmaf(p.inj.w, v, p.x.w);
    }
    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v) const {
        store(row, token, v, preload(row, token));
    }
    __device__ __forceinline__ void store8(std::int32_t row, std::int32_t token,
                                           const float (&v)[8]) const {
        const float4 g = load_vec<float4>(inj + static_cast<std::int64_t>(token) * 4);
        const float gs[4] = {g.x, g.y, g.z, g.w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            float* d = x + (static_cast<std::int64_t>(token) * 4 + s) * rows + row;
            float4 a = load_vec<float4>(d);
            float4 b = load_vec<float4>(d + 4);
            a.x = fmaf(gs[s], v[0], a.x);
            a.y = fmaf(gs[s], v[1], a.y);
            a.z = fmaf(gs[s], v[2], a.z);
            a.w = fmaf(gs[s], v[3], a.w);
            b.x = fmaf(gs[s], v[4], b.x);
            b.y = fmaf(gs[s], v[5], b.y);
            b.z = fmaf(gs[s], v[6], b.z);
            b.w = fmaf(gs[s], v[7], b.w);
            store_vec(d, a);
            store_vec(d + 4, b);
        }
    }
};

// h[t][row] = bf16(silu(gate) * up); gate is weight row `row`, up is weight row rows + `row`.
struct OutSwiGlu {
    __nv_bfloat16* data;
    std::int32_t ld;

    __device__ __forceinline__ void store_pair(std::int32_t row, std::int32_t token, float gate,
                                               float up) const {
        data[static_cast<std::int64_t>(token) * ld + row] = __float2bfloat16_rn(silu(gate) * up);
    }
};

// y[t][row] = sigmoid(gate_logit[t * stride]) * v (FP32).
struct OutSharedScale {
    float* data;
    const float* gate_logit;
    std::int32_t gate_stride;
    std::int32_t ld;

    __device__ __forceinline__ float gate(std::int32_t token) const {
        return sigmoid(gate_logit[static_cast<std::int64_t>(token) * gate_stride]);
    }
    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v) const {
        data[static_cast<std::int64_t>(token) * ld + row] = gate(token) * v;
    }
    __device__ __forceinline__ void store8(std::int32_t row, std::int32_t token,
                                           const float (&v)[8]) const {
        const float g = gate(token);
        float* d      = data + static_cast<std::int64_t>(token) * ld + row;
        store_vec(d, make_float4(g * v[0], g * v[1], g * v[2], g * v[3]));
        store_vec(d + 4, make_float4(g * v[4], g * v[5], g * v[6], g * v[7]));
    }
};

// r[t][row] = bf16(silu(v / 4)) with leading dimension `ld` (384 for the GR down projection).
struct OutSiluQuarter {
    __nv_bfloat16* data;
    std::int32_t ld;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float v) const {
        data[static_cast<std::int64_t>(token) * ld + row] = __float2bfloat16_rn(silu(v * 0.25f));
    }
    __device__ __forceinline__ void store8(std::int32_t row, std::int32_t token,
                                           const float (&v)[8]) const {
        float s[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = silu(v[i] * 0.25f);
        const uint4 packed = make_uint4(pack_bf16x2(s[0], s[1]), pack_bf16x2(s[2], s[3]),
                                        pack_bf16x2(s[4], s[5]), pack_bf16x2(s[6], s[7]));
        store_vec(data + static_cast<std::int64_t>(token) * ld + row, packed);
    }
};

// mix[t][d] = bf16(0.25 * sum_s sigmoid(v[4d + s]) * Xn[t][s * dims + d]), dims = rows / 4.
// Preload: a kernel may fetch a group's four Xn values before its main loop (preload) and hand
// them back to store_group4, hiding their latency behind the weight stream.
struct OutGrMix {
    static constexpr int kRowGroup = 4;
    using Preload                  = float4;
    __nv_bfloat16* data;       // [T][dims]
    const __nv_bfloat16* xn;   // [T][4 * dims]
    std::int32_t dims;

    __device__ __forceinline__ float4 preload(std::int32_t row, std::int32_t token) const {
        const __nv_bfloat16* xt = xn + static_cast<std::int64_t>(token) * 4 * dims + (row >> 2);
        return make_float4(__bfloat162float(xt[0]), __bfloat162float(xt[dims]),
                           __bfloat162float(xt[2 * dims]), __bfloat162float(xt[3 * dims]));
    }
    __device__ __forceinline__ void store_group4(std::int32_t row, std::int32_t token, float4 v,
                                                 float4 x) const {
        float sum = sigmoid(v.x) * x.x;
        sum       = fmaf(sigmoid(v.y), x.y, sum);
        sum       = fmaf(sigmoid(v.z), x.z, sum);
        sum       = fmaf(sigmoid(v.w), x.w, sum);
        data[static_cast<std::int64_t>(token) * dims + (row >> 2)] = __float2bfloat16_rn(0.25f * sum);
    }
    __device__ __forceinline__ void store_group4(std::int32_t row, std::int32_t token,
                                                 float4 v) const {
        store_group4(row, token, v, preload(row, token));
    }
};

// Emits 8 consecutive finished rows [row, row + 8) of one token through any policy
// (`paired` holds the up rows for PairRows policies).
template <bool PairRows, class Output>
__device__ __forceinline__ void emit_rows8(const Output& output, std::int32_t row,
                                           std::int32_t token, const float (&v)[8],
                                           const float (&paired)[8]) {
    if constexpr (PairRows) {
#pragma unroll
        for (int i = 0; i < 8; ++i) output.store_pair(row + i, token, v[i], paired[i]);
    } else if constexpr (output_row_group<Output>() == 4) {
        output.store_group4(row, token, make_float4(v[0], v[1], v[2], v[3]));
        output.store_group4(row + 4, token, make_float4(v[4], v[5], v[6], v[7]));
    } else if constexpr (output_has_store8<Output>()) {
        output.store8(row, token, v);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) output.store(row + i, token, v[i]);
    }
}

// Hands one token's finished warp rows (FP32 dot products) to the policy after multiplying by
// scale(weight_row). Rows are the warp's local rows (weight rows through `row_policy`); PairRows
// stores Rows / 2 pairs. Elementwise policies skip rows >= row_limit.
template <bool PairRows, int Rows, class Output, class RowPolicy, class Scale>
__device__ __forceinline__ void finish_token(const Output& output, const RowPolicy& row_policy,
                                             const Scale& scale, int row_begin, int token,
                                             const float (&totals)[Rows],
                                             int row_limit = INT_MAX) {
    if constexpr (PairRows) {
        static_assert((Rows % 2) == 0);
#pragma unroll
        for (int local = 0; local < Rows / 2; ++local) {
            const int gate_row = row_policy.weight_row(row_begin, local);
            const int up_row   = row_policy.weight_row(row_begin, Rows / 2 + local);
            output.store_pair(row_begin + local, token, totals[local] * scale(gate_row),
                              totals[Rows / 2 + local] * scale(up_row));
        }
    } else if constexpr (output_row_group<Output>() == 4) {
        static_assert((Rows % 4) == 0, "4-row group outputs need whole groups per warp");
#pragma unroll
        for (int group = 0; group < Rows / 4; ++group) {
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = row_policy.weight_row(row_begin, group * 4 + i);
                v[i]          = totals[group * 4 + i] * scale(row);
            }
            output.store_group4(row_policy.weight_row(row_begin, group * 4), token,
                                make_float4(v[0], v[1], v[2], v[3]));
        }
    } else {
#pragma unroll
        for (int local = 0; local < Rows; ++local) {
            const int row = row_policy.weight_row(row_begin, local);
            if (row < row_limit) output.store(row, token, totals[local] * scale(row));
        }
    }
}

// Lane-parallel form of finish_token for one token when every lane holds all totals: lane u
// finishes stored unit u (row, pair or 4-row group), so the policy's global loads / stores of the
// units overlap instead of running back to back in one lane. Selection uses predicated register
// moves (no divergent per-unit code). `preloaded` (Output::Preload of the lane's unit, see
// lane_preload) is passed to 4-row group policies that support it.
template <bool PairRows, int Rows, class Output, class RowPolicy, class Scale,
          class Preloaded = std::nullptr_t>
__device__ __forceinline__ void finish_token_lanes(const Output& output,
                                                   const RowPolicy& row_policy,
                                                   const Scale& scale, int row_begin, int token,
                                                   const float (&totals)[Rows], int lane,
                                                   const Preloaded& preloaded = nullptr) {
    if constexpr (PairRows) {
        static_assert((Rows % 2) == 0);
        constexpr int kUnits = Rows / 2;
        float gate = 0.0F, up = 0.0F;
#pragma unroll
        for (int unit = 0; unit < kUnits; ++unit) {
            if (lane == unit) {
                gate = totals[unit];
                up   = totals[kUnits + unit];
            }
        }
        if (lane < kUnits) {
            const int gate_row = row_policy.weight_row(row_begin, lane);
            const int up_row   = row_policy.weight_row(row_begin, kUnits + lane);
            output.store_pair(row_begin + lane, token, gate * scale(gate_row), up * scale(up_row));
        }
    } else if constexpr (output_row_group<Output>() == 4) {
        static_assert((Rows % 4) == 0, "4-row group outputs need whole groups per warp");
        constexpr int kUnits = Rows / 4;
        float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
        for (int unit = 0; unit < kUnits; ++unit) {
            if (lane == unit) {
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = totals[unit * 4 + i];
            }
        }
        if (lane < kUnits) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= scale(row_policy.weight_row(row_begin, lane * 4 + i));
            const int group_row = row_policy.weight_row(row_begin, lane * 4);
            const float4 values = make_float4(v[0], v[1], v[2], v[3]);
            if constexpr (std::is_same_v<Preloaded, std::nullptr_t>) {
                output.store_group4(group_row, token, values);
            } else {
                output.store_group4(group_row, token, values, preloaded);
            }
        }
    } else {
        float v = 0.0F;
#pragma unroll
        for (int unit = 0; unit < Rows; ++unit) {
            if (lane == unit) v = totals[unit];
        }
        if (lane < Rows) {
            const int row = row_policy.weight_row(row_begin, lane);
            if constexpr (std::is_same_v<Preloaded, std::nullptr_t>) {
                output.store(row, token, v * scale(row));
            } else {
                output.store(row, token, v * scale(row), preloaded);
            }
        }
    }
}

// The Output::Preload of the unit (row, or 4-row group) a lane finishes in finish_token_lanes;
// nullptr when the policy has none (pair policies never preload).
template <bool PairRows, int Rows, class Output, class RowPolicy>
__device__ __forceinline__ auto lane_preload(const Output& output, const RowPolicy& row_policy,
                                             int row_begin, int token, int lane) {
    if constexpr (!PairRows && requires { typename Output::Preload; }) {
        constexpr int kGroup = output_row_group<Output>();
        typename Output::Preload value{};
        if (lane < Rows / kGroup) {
            value = output.preload(row_policy.weight_row(row_begin, lane * kGroup), token);
        }
        return value;
    } else {
        return nullptr;
    }
}

// Ordered cross-warp K-split reduction of warp totals (slice 0 adds slices 1.. in order, so
// results are deterministic). Totals must be complete in lane 0 (AllLanes = false) or in every
// lane (AllLanes = true). Every thread of the CTA must call run(); it returns true in the lanes of
// the slice-0 warps that own finished totals (lane 0, or every lane with AllLanes).
template <int KSplit, int RowWarps, int Count>
struct WarpKSplitReduce {
    struct Storage {
        float partial[KSplit > 1 ? KSplit - 1 : 1][KSplit > 1 ? RowWarps : 1]
                     [KSplit > 1 ? Count : 1];
    };

    template <bool AllLanes = false>
    __device__ __forceinline__ static bool run(Storage& storage, int row_warp, int k_slice,
                                               int lane, float (&totals)[Count]) {
        if constexpr (KSplit == 1) {
            return AllLanes || lane == 0;
        } else {
            if (lane == 0 && k_slice > 0) {
#pragma unroll
                for (int i = 0; i < Count; ++i) storage.partial[k_slice - 1][row_warp][i] = totals[i];
            }
            __syncthreads();
            if (k_slice != 0 || (!AllLanes && lane != 0)) return false;
#pragma unroll
            for (int slice = 0; slice < KSplit - 1; ++slice) {
#pragma unroll
                for (int i = 0; i < Count; ++i) totals[i] += storage.partial[slice][row_warp][i];
            }
            return true;
        }
    }
};

template <LinearEpilogue Kind>
struct EpilogueTraits;

template <>
struct EpilogueTraits<LinearEpilogue::StoreBf16> {
    using Output                  = OutBf16;
    static constexpr bool kPair   = false;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<__nv_bfloat16*>(p.out), rows};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::StoreF32> {
    using Output                  = OutF32;
    static constexpr bool kPair   = false;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<float*>(p.out), rows};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::GrWrite> {
    using Output                  = OutGrWrite;
    static constexpr bool kPair   = false;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<float*>(p.out), p.inj, rows};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::SwiGlu> {
    using Output                  = OutSwiGlu;
    static constexpr bool kPair   = true;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<__nv_bfloat16*>(p.out), rows / 2};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::SharedScale> {
    using Output                  = OutSharedScale;
    static constexpr bool kPair   = false;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<float*>(p.out), p.gate_logit, p.gate_stride, rows};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::SiluQuarter> {
    using Output                  = OutSiluQuarter;
    static constexpr bool kPair   = false;
    static constexpr std::int32_t kLd = 384;
    static Output make(const LinearEpilogueParams& p, std::int32_t) {
        return {static_cast<__nv_bfloat16*>(p.out), kLd};
    }
};
template <>
struct EpilogueTraits<LinearEpilogue::GrMix> {
    using Output                  = OutGrMix;
    static constexpr bool kPair   = false;
    static Output make(const LinearEpilogueParams& p, std::int32_t rows) {
        return {static_cast<__nv_bfloat16*>(p.out), p.xn, rows / 4};
    }
};

} // namespace fngine::kernels::detail
