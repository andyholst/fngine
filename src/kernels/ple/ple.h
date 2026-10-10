#pragma once

// Per-layer n-gram embedding (PLE), applied before layer 1's attention GR.
//
//   E   = 16 IQ4_NL rows x 160 (ple_dequant)
//   K   = key(E) bf16 [T][10240], V = value(E) bf16 [T][2560]          (linear area)
//   Kn  = bf16(rmsnorm_2560-groups(K) * norm_key)
//   Qn  = bf16(rmsnorm_2560-groups(X) * norm_query)
//   g_s = sum_d Kn_s * Qn_s / sqrt(2560);  g_s = sign(g_s) * sqrt(max(|g_s|, 1e-6))
//   GV_s = bf16(sigmoid(g_s) * V)
//   c   = bf16(rmsnorm_2560-groups(GV) * norm_conv)
//   out = GV + silu(w0*c[t-9] + w1*c[t-6] + w2*c[t-3] + w3*c[t])   (depthwise, 10240 channels)
//   X  += out
// All norms use eps 1e-6 with FP32 effective multipliers (1+w) from the pack. The conv state is
// the last 9 c vectors, BF16 [9][10240] oldest first (zero at sequence start).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

// Optional device-side wait for rows staged by the host (exchange protocol, core/exchange.h):
// every CTA acquire-spins until *ready == *expected; after ~10 s it release-stores
// kExchangeErrorPleTimeout into *error and returns without writing E.
struct PleRowsWait {
    const std::uint32_t* ready    = nullptr; // &ExchangeControl::ple_ready.value; null = no wait
    const std::uint32_t* expected = nullptr; // u32 holding this step's sequence number
    std::uint32_t* error          = nullptr; // &ExchangeControl::error.value
};

// E[t][h*160 + i] = bf16(d * kvalues_iq4nl[nibble]) from raw IQ4_NL rows [T][16][90] (5 ggml
// block_iq4_nl per row; low nibbles are elements 0..15, high nibbles 16..31 of each block).
// `rows` may be host-mapped and must be 16-byte aligned. E is BF16 [T][2560].
void ple_dequant(const std::uint8_t* rows, std::int32_t T, __nv_bfloat16* E,
                 const PleRowsWait& wait, cudaStream_t stream);

struct PleMixWeights {
    const float* norm_key    = nullptr; // [10240] (1+w)
    const float* norm_query  = nullptr; // [10240] (1+w)
    const float* norm_conv   = nullptr; // [10240] (1+w)
    const float* conv_weight = nullptr; // [10240][4], tap 3 multiplies c[t]
};

// Workspace for ple_mix over T tokens (per-(token, stream) gate values).
[[nodiscard]] std::size_t ple_mix_workspace_bytes(std::int32_t T);

/**
 * X FP32 [T][4][2560] += out (see above). K bf16 [T][10240], V bf16 [T][2560]. c_record
 * receives c, BF16 [T][10240]; column t's conv reads c_record for t-9+3j >= 0 and
 * conv_state[9 + t-9+3j] otherwise.
 *   commit_state == false (decode/verify, T <= 16): conv_state is read only; ple_fold later
 *     commits the accepted prefix of c_record.
 *   commit_state == true (prefill chunk, any T): afterwards conv_state = last 9 of
 *     (conv_state || c[0:T]).
 */
void ple_mix(float* X, const __nv_bfloat16* K, const __nv_bfloat16* V, const PleMixWeights& w,
             __nv_bfloat16* conv_state, __nv_bfloat16* c_record, std::int32_t T,
             bool commit_state, void* workspace, std::size_t workspace_bytes,
             cudaStream_t stream);

// conv_state = last 9 of (conv_state || c_record[0:n]) with n = clamp(*n_commit, 0, max_commit)
// read from device memory (accepted drafts + 1). c_record is BF16 [>= max_commit][10240].
void ple_fold(__nv_bfloat16* conv_state, const __nv_bfloat16* c_record,
              const std::int32_t* n_commit, std::int32_t max_commit, cudaStream_t stream);

} // namespace fngine::kernels
