#pragma once

// Fixed dimensions of the Qwen3.8-Flash-Next (qwen4_exp) text model. fngine runs
// only this model; the pack loader validates manifest.json against these values.
#include <cstdint>

namespace fngine::dims {

inline constexpr int32_t kHidden = 2560;
inline constexpr int32_t kStreams = 4;
inline constexpr int32_t kStreamHidden = kStreams * kHidden;  // 10240
inline constexpr int32_t kLayers = 48;
inline constexpr int32_t kVocab = 248320;
inline constexpr float kRmsEps = 1e-6f;

// Layer i is sparse attention (QSA) when (i + 1) % 4 == 0; the other 36 are GDN.
inline constexpr int32_t kAttentionInterval = 4;
inline constexpr int32_t kQsaLayers = 12;
inline constexpr int32_t kGdnLayers = 36;
constexpr bool is_qsa_layer(int32_t layer) { return (layer + 1) % kAttentionInterval == 0; }
// Dense index of a layer among its own kind (GDN 0..35, QSA 0..11).
constexpr int32_t qsa_index(int32_t layer) { return layer / kAttentionInterval; }
constexpr int32_t gdn_index(int32_t layer) { return layer - layer / kAttentionInterval; }

// Gated residual (hyper-connection) mixing.
inline constexpr int32_t kGrLowRank = 320;
inline constexpr int32_t kGrLowRankPadded = 384;

// Gated DeltaNet.
inline constexpr int32_t kGdnQkHeads = 16;
inline constexpr int32_t kGdnValueHeads = 48;
inline constexpr int32_t kGdnHeadDim = 128;
inline constexpr int32_t kGdnKeyDim = kGdnQkHeads * kGdnHeadDim;      // 2048
inline constexpr int32_t kGdnValueDim = kGdnValueHeads * kGdnHeadDim;  // 6144
inline constexpr int32_t kGdnConvChannels = 2 * kGdnKeyDim + kGdnValueDim;  // 10240
inline constexpr int32_t kGdnConvWidth = 4;
inline constexpr int32_t kGdnQkvzRows = kGdnConvChannels + kGdnValueDim;  // 16384

// Sparse attention (QSA).
inline constexpr int32_t kQsaHeads = 24;
inline constexpr int32_t kQsaKvHeads = 2;
inline constexpr int32_t kQsaHeadDim = 256;
inline constexpr int32_t kQsaRotaryDim = 64;
inline constexpr float kRopeTheta = 1e7f;
inline constexpr int32_t kQsaQProjRows = kQsaHeads * kQsaHeadDim * 2;  // 12288 (query|gate per head)
inline constexpr int32_t kQsaQkvRows = kQsaQProjRows + 2 * kQsaKvHeads * kQsaHeadDim;  // 13312
inline constexpr int32_t kQsaAttnOut = kQsaHeads * kQsaHeadDim;  // 6144
inline constexpr int32_t kIndexerHeads = 4;
inline constexpr int32_t kIndexerHeadDim = 128;
inline constexpr int32_t kIndexerRows = (kIndexerHeads + 1) * kIndexerHeadDim;  // 640
inline constexpr int32_t kIndexerBlock = 4;         // compress ratio
inline constexpr int32_t kIndexerTopBlocks = 512;   // budget 2048 / 4
inline constexpr int32_t kIndexerRawRing = 16;
// A query attends at most 512 selected blocks plus a tail of up to 3 tokens.
inline constexpr int32_t kQsaMaxKeys = kIndexerTopBlocks * kIndexerBlock + kIndexerBlock - 1;  // 2051

// Mixture of experts.
inline constexpr int32_t kExperts = 512;
inline constexpr int32_t kTopK = 10;
inline constexpr int32_t kExpertInter = 640;
inline constexpr int32_t kSharedInter = 640;
inline constexpr int32_t kRouterRows = kExperts + 1;  // router | shared-expert gate

// Per-layer n-gram embedding (PLE), applied before layer 1's attention GR.
inline constexpr int32_t kPleLayer = 1;
inline constexpr int32_t kPleHeads = 16;
inline constexpr int32_t kPleHeadDim = 160;
inline constexpr int32_t kPleRowBytes = 90;  // 5 IQ4_NL blocks
inline constexpr int32_t kPleConvTaps = 4;
inline constexpr int32_t kPleConvDilation = 3;
inline constexpr int32_t kPleConvState = (kPleConvTaps - 1) * kPleConvDilation;  // 9
inline constexpr int32_t kEosToken = 248044;

// Decode/verify step widths captured as CUDA graphs.
inline constexpr int32_t kMaxStepTokens = 16;
inline constexpr int32_t kMaxGraphWidth = 9;

}  // namespace fngine::dims
