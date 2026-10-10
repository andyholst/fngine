#pragma once

// RoPE inverse frequencies for the QSA heads and the indexer (rotary dim 64, theta 1e7), as the
// exact float32 bit patterns HF computes: 1.0f / (1e7f ** (float(2i) / 64)) in float32
// (transformers 5.17 Qwen4ExpTextRotaryEmbedding, CPU). Shared by the kernels and test oracles.
#include <cstdint>

namespace fngine::kernels {

inline constexpr int kQsaRopePairs = 32;  // rotary dim 64 -> 32 frequencies, pairs (i, i + 32)

inline constexpr uint32_t kQsaRopeInvFreqBits[kQsaRopePairs] = {
    0x3f800000u, 0x3f1ab32bu, 0x3ebaf81bu, 0x3e61f835u, 0x3e088d77u, 0x3da50956u, 0x3d47763fu,
    0x3cf11177u, 0x3c91ad39u, 0x3c301052u, 0x3bd4ca15u, 0x3b80967du, 0x3b1b690du, 0x3abbd3edu,
    0x3a6301e2u, 0x3a092e02u, 0x39a5cb60u, 0x394860c1u, 0x38f22ce2u, 0x3892587eu, 0x3830df52u,
    0x37d5c441u, 0x37812dabu, 0x371c1fc4u, 0x36bcb0c1u, 0x36640cc6u, 0x3609cf4au, 0x35a68e4cu,
    0x35494c57u, 0x34f3499du, 0x3493048eu, 0x3431af44u,
};

}  // namespace fngine::kernels
