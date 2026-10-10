#pragma once

// Gated-residual (hyper-connection) read side: the grouped RMSNorm of the four residual streams
// and the fused 4-row inject GEMV. The down / up projections and the residual write are FP8
// linears with the SiluQuarter / GrMix / GrWrite epilogues (kernels/linear/linear.h).
//
//   Xn[t][s*2560 + d] = bf16(X[t][s][d] * rsqrt(mean_d X[t][s][d]^2 + 1e-6) * hc_norm[s*2560 + d])
//   inj[t][s]         = 2 * sigmoid((sum_j inject[s][j] * Xn[t][j]) / 4)    (BF16-rounded Xn)
//
// Graphs: `tokens` is a host scalar baked into the launch (one CUDA graph per step width, prefill
// is eager), the same convention as the linear kernels.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels {

// X FP32 [T][4][2560]; hc_norm FP32 [10240] (effective 1 + w multipliers); inject BF16 [4][10240]
// or null (final mixer: inj is then not written and may be null); Xn BF16 [T][10240];
// inj FP32 [T][4]. All pointers 16-byte aligned.
void gr_norm(const float* X, std::int32_t tokens, const float* hc_norm,
             const __nv_bfloat16* inject, __nv_bfloat16* Xn, float* inj, cudaStream_t stream);

// Decode/verify GR read front half (1 <= tokens <= kGrNormDownMaxTokens) in one launch: gr_norm
// (Xn, inj exactly as above) fused with the GR down projection
//   r[t][j] = bf16(silu(scale[j] * sum_k down[j][k] * Xn[t][k] / 4)),  j < 320,
// down = FP8 E4M3 codes [320][10240] plus BF16 row scales; r has leading dimension 384 and its
// columns 320..383 are not written. 40 clusters of four CTAs: CTA rank s normalizes stream s and
// reduces the stream-s slice of eight down rows; the four stream partials of each row meet in
// rank 0 over distributed shared memory. Deterministic, and each token's results are independent
// of `tokens`. Versus gr_norm + the SiluQuarter linear, only FP32 summation orders differ.
inline constexpr std::int32_t kGrNormDownMaxTokens = 16;
void gr_norm_down(const float* X, std::int32_t tokens, const float* hc_norm,
                  const __nv_bfloat16* inject, const std::uint8_t* down_codes,
                  const __nv_bfloat16* down_scales, __nv_bfloat16* Xn, float* inj,
                  __nv_bfloat16* r, cudaStream_t stream);

}  // namespace fngine::kernels
