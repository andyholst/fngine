#pragma once

// MTP draft-layer glue (vLLM qwen4_exp nvidia/mtp.py):
//   e = fc_embedding(rmsnorm_2560(embed(tok_{t+1}); pre_fc_norm_embedding))
//   h = fc_hidden(rmsnorm_10240(X_pre_mixer; pre_fc_norm_hidden)) applied per stream
//   X_mtp[t][s] = h[t][s] + e[t]
// The fc projections are linear-area kernels; this area owns the norms and the merge.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels {

// out[t][i] = bf16(float(x[t][i]) * rsqrt(mean_i x[t][i]^2 + 1e-6) * mult[i]) for one norm over
// all n_cols values of each row; mult is the FP32 effective multiplier (1+w).
// BF16 input: the 2560-wide embedding norm. FP32 input: the 10240-wide norm over the
// pre-mixer streams X [T][4][2560]. n_cols must be a positive multiple of 8; rows are
// contiguous; x and out must be 16-byte aligned.
void rmsnorm_bf16(const __nv_bfloat16* x, std::int32_t T, std::int32_t n_cols, const float* mult,
                  __nv_bfloat16* out, cudaStream_t stream);
void rmsnorm_bf16(const float* x, std::int32_t T, std::int32_t n_cols, const float* mult,
                  __nv_bfloat16* out, cudaStream_t stream);

// X_mtp[t][s][d] = float(bf16(float(h[t][s][d]) + float(e[t][d]))): h bf16 [T][4][2560]
// (fc_hidden output over T*4 rows), e bf16 [T][2560], X_mtp FP32 [T][4][2560].
void mtp_merge_streams(const __nv_bfloat16* h, const __nv_bfloat16* e, std::int32_t T,
                       float* X_mtp, cudaStream_t stream);

} // namespace fngine::kernels
