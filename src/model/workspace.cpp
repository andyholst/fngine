#include "model/workspace.h"

#include "core/device.h"
#include "core/model_dims.h"
#include "kernels/gdn/gdn.h"
#include "kernels/head/head.h"
#include "kernels/ple/ple.h"
#include "kernels/qsa/qsa.h"

#include <algorithm>
#include <stdexcept>

namespace fngine::model {
namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Bump carver: with base == nullptr it only measures.
struct Carver {
  unsigned char* base = nullptr;
  std::size_t off = 0;
  template <class T>
  T* take(std::size_t count) {
    const std::size_t at = off;
    off = align_up(off + count * sizeof(T));
    return base ? reinterpret_cast<T*>(base + at) : nullptr;
  }
  void* bytes(std::size_t n) { return take<unsigned char>(n); }
};

using kernels::LinearEpilogue;

struct LinearShape {
  int32_t n, k;
  LinearEpilogue ep;
  int32_t rows;  // tokens the projection runs over
};

std::size_t linear_ws_bytes(const WorkspaceOptions& o) {
  const int32_t T = o.max_tokens;
  const LinearShape fp8[] = {
      {16384, 2560, LinearEpilogue::StoreBf16, T},  {13312, 2560, LinearEpilogue::StoreBf16, T},
      {2560, 6144, LinearEpilogue::GrWrite, T},      {1280, 2560, LinearEpilogue::SwiGlu, T},
      {2560, 640, LinearEpilogue::SharedScale, T},   {320, 10240, LinearEpilogue::SiluQuarter, T},
      {10240, 384, LinearEpilogue::GrMix, T},        {10240, 2560, LinearEpilogue::StoreBf16, T},
      {2560, 2560, LinearEpilogue::StoreBf16, 4 * T}, {248320, 2560, LinearEpilogue::StoreF32, o.logit_rows},
  };
  const LinearShape bf16_always[] = {
      {513, 2560, LinearEpilogue::StoreF32, T},
      {96, 2560, LinearEpilogue::StoreF32, T},
      {640, 2560, LinearEpilogue::StoreBf16, T},
  };
  std::size_t best = 0;
  for (const auto& s : fp8) {
    best = std::max(best, kernels::linear_fp8_workspace_bytes(s.n, s.k, s.rows));
    if (o.dense_bf16) best = std::max(best, kernels::linear_bf16_workspace_bytes(s.n, s.k, s.ep, s.rows));
  }
  for (const auto& s : bf16_always)
    best = std::max(best, kernels::linear_bf16_workspace_bytes(s.n, s.k, s.ep, s.rows));
  return best;
}

Workspace carve(const WorkspaceOptions& o, unsigned char* base, std::size_t* total) {
  if (o.max_tokens < dims::kMaxStepTokens) throw std::invalid_argument("workspace max_tokens < 16");
  if (o.logit_rows < dims::kMaxStepTokens) throw std::invalid_argument("workspace logit_rows < 16");
  const std::size_t T = std::size_t(o.max_tokens);
  Carver c{base};
  Workspace w;
  w.options = o;
  w.tokens = c.take<int32_t>(T + 1);
  w.positions = c.take<int32_t>(T);
  w.seq = c.take<uint32_t>(4);

  w.X = c.take<float>(T * dims::kStreamHidden);
  w.Xn = c.take<__nv_bfloat16>(T * dims::kStreamHidden);
  w.inj = c.take<float>(T * dims::kStreams);
  w.r = c.take<__nv_bfloat16>(T * dims::kGrLowRankPadded);
  w.mix = c.take<__nv_bfloat16>(T * dims::kHidden);
  w.emb = c.take<__nv_bfloat16>(T * dims::kHidden);

  // GDN and QSA scratch share one region.
  const std::size_t mixer_base = c.off;
  w.qkvz = c.take<__nv_bfloat16>(T * dims::kGdnQkvzRows);
  w.ab = c.take<float>(T * 2 * dims::kGdnValueHeads);
  w.conv_out = c.take<__nv_bfloat16>(T * dims::kGdnConvChannels);
  w.g = c.take<float>(T * dims::kGdnValueHeads);
  w.beta = c.take<float>(T * dims::kGdnValueHeads);
  w.core = c.take<__nv_bfloat16>(T * dims::kGdnValueDim);
  w.normed = c.take<__nv_bfloat16>(T * dims::kGdnValueDim);
  const std::size_t gdn_end = c.off;
  c.off = mixer_base;
  w.qkv = c.take<__nv_bfloat16>(T * dims::kQsaQkvRows);
  w.q = c.take<__nv_bfloat16>(T * dims::kQsaAttnOut);
  w.k = c.take<__nv_bfloat16>(T * dims::kQsaKvHeads * dims::kQsaHeadDim);
  w.idx = c.take<__nv_bfloat16>(T * dims::kIndexerRows);
  w.q_idx = c.take<__nv_bfloat16>(T * dims::kIndexerHeads * dims::kIndexerHeadDim);
  w.selected = c.take<int32_t>(T * dims::kIndexerTopBlocks);
  w.counts = c.take<int32_t>(T);
  w.attn = c.take<__nv_bfloat16>(T * dims::kQsaAttnOut);
  c.off = std::max(c.off, gdn_end);

  w.router = c.take<float>(T * dims::kRouterRows);
  w.shared_h = c.take<__nv_bfloat16>(T * dims::kSharedInter);
  w.y_shared = c.take<float>(T * dims::kHidden);
  {
    void* p = c.bytes(kernels::moe_decode_workspace_bytes());
    if (base) w.moe_decode = kernels::make_moe_decode_workspace(p);
  }
  {
    void* p = c.bytes(kernels::moe_prefill_workspace_bytes(o.max_tokens));
    if (base) w.moe_prefill = kernels::make_moe_prefill_workspace(p, o.max_tokens);
  }

  w.E = c.take<__nv_bfloat16>(T * dims::kHidden);
  w.K = c.take<__nv_bfloat16>(T * dims::kStreamHidden);
  w.V = c.take<__nv_bfloat16>(T * dims::kHidden);
  w.c_record = c.take<__nv_bfloat16>(T * dims::kStreamHidden);

  w.X_mtp = c.take<float>(T * dims::kStreamHidden);
  w.hid_n = c.take<__nv_bfloat16>(T * dims::kStreamHidden);
  w.h = c.take<__nv_bfloat16>(T * dims::kStreamHidden);
  w.e = c.take<__nv_bfloat16>(T * dims::kHidden);
  w.emb_n = c.take<__nv_bfloat16>(T * dims::kHidden);

  w.logits = c.take<float>(std::size_t(o.logit_rows) * dims::kVocab);

  w.linear_ws_bytes = linear_ws_bytes(o);
  w.linear_ws = w.linear_ws_bytes ? c.bytes(w.linear_ws_bytes) : nullptr;
  w.gdn_ws_bytes = kernels::gdn_prefill_workspace_bytes(o.max_tokens);
  w.gdn_ws = w.gdn_ws_bytes ? c.bytes(w.gdn_ws_bytes) : nullptr;
  w.ple_ws_bytes = kernels::ple_mix_workspace_bytes(o.max_tokens);
  w.ple_ws = w.ple_ws_bytes ? c.bytes(w.ple_ws_bytes) : nullptr;
  w.score_ws_bytes = kernels::indexer_score_workspace_bytes(o.max_tokens, o.max_blocks);
  w.score_ws = w.score_ws_bytes ? c.bytes(w.score_ws_bytes) : nullptr;
  w.attn_ws_bytes = kernels::qsa_attention_workspace_bytes(o.max_tokens);
  w.attn_ws = w.attn_ws_bytes ? c.bytes(w.attn_ws_bytes) : nullptr;
  w.sample_ws_bytes = kernels::sampling_workspace_bytes(dims::kVocab, dims::kMaxStepTokens);
  w.sample_ws = w.sample_ws_bytes ? c.bytes(w.sample_ws_bytes) : nullptr;
  w.accept_ws_bytes = kernels::speculative_accept_workspace_bytes(dims::kVocab, dims::kMaxStepTokens - 1);
  w.accept_ws = w.accept_ws_bytes ? c.bytes(w.accept_ws_bytes) : nullptr;

  *total = c.off;
  return w;
}

}  // namespace

std::size_t WorkspaceBuffer::device_bytes(const WorkspaceOptions& options) {
  std::size_t total = 0;
  carve(options, nullptr, &total);
  return total;
}

Workspace WorkspaceBuffer::view(const WorkspaceOptions& options, void* memory, std::size_t bytes) {
  if (!memory || bytes < device_bytes(options)) throw std::invalid_argument("workspace loan too small");
  std::size_t total = 0;
  return carve(options, static_cast<unsigned char*>(memory), &total);
}

void WorkspaceBuffer::initialize(const Workspace& w, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(w.r, 0, std::size_t(w.options.max_tokens) *
                            dims::kGrLowRankPadded * sizeof(__nv_bfloat16), stream));
}

WorkspaceBuffer::WorkspaceBuffer(const WorkspaceOptions& options) {
  std::size_t total = 0;
  carve(options, nullptr, &total);
  buffer_ = DeviceBuffer(total);
  // Zero once: the padded GR low-rank columns (320..383) must stay zero forever.
  buffer_.fill(0);
  ws_ = carve(options, static_cast<unsigned char*>(buffer_.p), &total);
}

}  // namespace fngine::model
