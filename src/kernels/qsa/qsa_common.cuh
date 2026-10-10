// Modified for fngine from cinference:src/ops/kernel/rope.cuh @ 383e5db. See NOTICE.
// Only the text 1-D rotation is kept; the frequency table is HF's float32 table (rope_freq.h).
#pragma once

#include "kernels/qsa/rope_freq.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels {

namespace qsa {

inline constexpr int kHeads       = 24;
inline constexpr int kKvHeads     = 2;
inline constexpr int kGroup       = kHeads / kKvHeads;  // 12 query heads per KV head
inline constexpr int kHeadDim     = 256;
inline constexpr int kQkvLd       = 13312;
inline constexpr int kKeyCol      = 12288;
inline constexpr int kValueCol    = 12800;
inline constexpr int kOutLd       = kHeads * kHeadDim;  // 6144
inline constexpr int kIdxHeads    = 4;
inline constexpr int kIdxDim      = 128;
inline constexpr int kIdxLd       = (kIdxHeads + 1) * kIdxDim;  // 640
inline constexpr int kBlock       = 4;
inline constexpr int kTopBlocks   = 512;
inline constexpr int kRing        = 16;
inline constexpr int kMaxKeys     = kTopBlocks * kBlock + kBlock - 1;  // 2051
inline constexpr float kEps       = 1e-6f;

// Geometry tag for the ported cinference KV codecs (Geometry::KVHeads).
struct KvGeometry {
    static constexpr int KVHeads = kKvHeads;
};

}  // namespace qsa

static __device__ __constant__ std::uint32_t kQsaRopeInvFreqDevice[kQsaRopePairs] = {
    kQsaRopeInvFreqBits[0],  kQsaRopeInvFreqBits[1],  kQsaRopeInvFreqBits[2],
    kQsaRopeInvFreqBits[3],  kQsaRopeInvFreqBits[4],  kQsaRopeInvFreqBits[5],
    kQsaRopeInvFreqBits[6],  kQsaRopeInvFreqBits[7],  kQsaRopeInvFreqBits[8],
    kQsaRopeInvFreqBits[9],  kQsaRopeInvFreqBits[10], kQsaRopeInvFreqBits[11],
    kQsaRopeInvFreqBits[12], kQsaRopeInvFreqBits[13], kQsaRopeInvFreqBits[14],
    kQsaRopeInvFreqBits[15], kQsaRopeInvFreqBits[16], kQsaRopeInvFreqBits[17],
    kQsaRopeInvFreqBits[18], kQsaRopeInvFreqBits[19], kQsaRopeInvFreqBits[20],
    kQsaRopeInvFreqBits[21], kQsaRopeInvFreqBits[22], kQsaRopeInvFreqBits[23],
    kQsaRopeInvFreqBits[24], kQsaRopeInvFreqBits[25], kQsaRopeInvFreqBits[26],
    kQsaRopeInvFreqBits[27], kQsaRopeInvFreqBits[28], kQsaRopeInvFreqBits[29],
    kQsaRopeInvFreqBits[30], kQsaRopeInvFreqBits[31],
};

// HF angle: float(position) * inv_freq rounded once to float, then exact-rounded sin/cos.
__device__ __forceinline__ void qsa_rope_sincos(std::int32_t position, int frequency, float* sine,
                                                float* cosine) {
    const float angle =
        __fmul_rn(static_cast<float>(position), __uint_as_float(kQsaRopeInvFreqDevice[frequency]));
    sincosf(angle, sine, cosine);
}

// The rotation is pinned to one contraction so every kernel that rotates a pair agrees bit for bit:
// products with the first-half value are fused, products with the second-half value rounded.
__device__ __forceinline__ float2 rope_rotate_pair(float first, float second, float cosine,
                                                   float sine) {
    return make_float2(__fmaf_rn(first, cosine, -__fmul_rn(second, sine)),
                       __fmaf_rn(first, sine, __fmul_rn(second, cosine)));
}

}  // namespace fngine::kernels
