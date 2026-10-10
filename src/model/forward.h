#pragma once

// Composition of the model forward from the kernel areas: embedding, the 48 decoder layers
// (PLE at layer 1, attention GR, GDN or QSA mixer, MLP GR, MoE), the final mixer + lm_head, and
// the MTP draft layer. Every launch goes to the caller's stream; nothing here allocates or
// synchronizes, so decode/verify bodies can be captured into CUDA graphs.
#include "core/exchange.h"
#include "kernels/linear/linear.h"
#include "core/arena.h"
#include "kernels/moe/moe.h"
#include "model/placement.h"
#include "model/l2_prefetch.h"
#include "model/prefill_stream.h"
#include "model/state.h"
#include "model/weights.h"
#include "model/workspace.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace fngine::model {

// Prefill: eager chunk, states committed in place, the CPU tier through the prefill exchange.
// Decode: one token, states updated in place (graph).
// Verify: T > 1 tokens, GDN/PLE produce records and leave state untouched (graph; the fold
//         commits the accepted prefix afterwards).
enum class Pass : uint8_t { Prefill, Decode, Verify };

struct StepIo {
  int32_t T = 0;
  const int32_t* tokens = nullptr;     // device [T]
  const int32_t* positions = nullptr;  // device [T], consecutive
  const uint32_t* seq = nullptr;       // device, exchange sequence number of this invocation
  Pass pass = Pass::Prefill;
  // Prefill only: host copy of positions[T-1] (sizes the indexer score grid).
  int32_t last_position = 0;
};

struct ForwardContext {
  const DenseWeights* weights = nullptr;
  const ExpertPlacement* placement = nullptr;  // null: every routed expert on the GPU is required
  ModelState* state = nullptr;
  Workspace* ws = nullptr;
  // Device-visible pointers of the host-mapped exchange (null: no CPU tier).
  DecodeExchange* exchange = nullptr;
  PrefillExchangeView prefill{};
  const uint8_t* ple_rows = nullptr;  // device-visible pointer of the staged raw n-gram rows
  bool dense_bf16 = false;
  // FP8 activation path (Auto = per-shape thresholds; A16 keeps BF16 activations everywhere).
  kernels::Fp8LinearPath fp8_path = kernels::Fp8LinearPath::Auto;
  kernels::CublasLtContext* cublas = nullptr;
  long long moe_timeout_cycles = kernels::kMoeExchangeTimeoutCycles;
  // Device u32 [48][512] routing counters (null: no telemetry); main layers add 1 per (token, expert).
  uint32_t* routing_counts = nullptr;
  // Prefill expert blob tables and streaming ring (required for Pass::Prefill).
  PrefillStream* prefill_stream = nullptr;
  // Per-layer expert formats (from the pack).
  std::array<kernels::MoeFormat, 48> gate_up_format{};
  std::array<kernels::MoeFormat, 48> down_format{};
  // Abliteration (pack ablation section): device per-layer unit refusal directions, FP32
  // [48][2560] (null when the pack has none), and the per-layer scales (host values: they are
  // kernel arguments).
  const float* ablation_directions = nullptr;
  std::array<float, 48> ablation_scales{};
  // Decode/verify: after each main layer's routing, a parallel branch on this stream prefetches up
  // to `prefetch_budget` bytes of the next layer's dense weights into L2 while the layer waits for
  // the CPU expert tier (null or 0: no prefetch). The branch joins in join_prefetch().
  cudaStream_t prefetch_stream = nullptr;
  uint64_t prefetch_budget = 0;
  // Decode/verify: independent short chains run as a parallel graph branch on this stream (GDN
  // ab + gating, QSA indexer projection + prepare, MoE shared expert), joined inside the layer
  // (null: serial). Not used with forced A8 linears, which share the linear workspace.
  cudaStream_t aux_stream = nullptr;
};

class Forward {
 public:
  explicit Forward(const ForwardContext& ctx);
  ~Forward();
  Forward(const Forward&) = delete;
  Forward& operator=(const Forward&) = delete;
  // Eager prefill may use borrowed scratch. Captured decode graphs always use the permanent view.
  void set_workspace(Workspace& workspace) noexcept { ctx_.ws = &workspace; }

  // X (all 4 streams) and emb from the token embedding of io.tokens.
  void embed(const StepIo& io, cudaStream_t s);
  // Main decoder layer l (0..47) on ws.X.
  void layer(int32_t l, const StepIo& io, cudaStream_t s);
  // mix_out[0..rows) = final hyper-connection mixer of X rows (main or MTP mixer weights).
  void final_mixer(const MixerWeights& w, const float* X, int32_t rows, __nv_bfloat16* mix_out,
                   cudaStream_t s);
  // logits[0..rows) = lm_head(mix rows); rows <= ws.logit_rows.
  void lm_head(const __nv_bfloat16* mix, int32_t rows, float* logits, cudaStream_t s);
  // logits[0..rows) = draft-vocabulary rows of lm_head (requires a built draft head).
  void draft_head(const __nv_bfloat16* mix, int32_t rows, float* logits, cudaStream_t s);
  // MTP draft layer over io.T columns: hidden x_src [T][4][2560] (pre-mixer streams of the
  // positions in io.positions), io.tokens = the tokens that follow those positions. Leaves the
  // MTP's own pre-mixer streams in ws.X_mtp. x_src may not alias ws.X_mtp.
  void mtp_layer(const float* x_src, const StepIo& io, cudaStream_t s);
  // Joins the L2 prefetch branch forked by the layers enqueued since the last join into `s`
  // (required before the end of a graph capture; no-op when nothing was forked).
  void join_prefetch(cudaStream_t s);

  [[nodiscard]] const ForwardContext& context() const noexcept { return ctx_; }

 private:
  void dense(const __nv_bfloat16* x, int32_t T, const Fp8Weight& w,
             const kernels::LinearEpilogueParams& ep, cudaStream_t s);
  void dense(const __nv_bfloat16* x, int32_t T, const Bf16Weight& w,
             const kernels::LinearEpilogueParams& ep, cudaStream_t s);
  // GR read front half into ws.Xn / ws.r (+ ws.inj when `inject` is set): the fused decode kernel
  // for T <= 16 on FP8 A16 weights, gr_norm plus the SiluQuarter linear otherwise.
  void gr_front(const float* X, int32_t T, const float* hc_norm, const __nv_bfloat16* inject,
                const Fp8Weight& down, cudaStream_t s);
  void gr_read(const GrWeights& w, const float* X, int32_t T, cudaStream_t s);
  void ple(const StepIo& io, cudaStream_t s);
  void gdn(int32_t l, const StepIo& io, cudaStream_t s);
  void qsa(const QsaWeights& w, int32_t cache, float* X, const StepIo& io, cudaStream_t s);
  void moe(const MoeWeights& w, int32_t l, float* X, const StepIo& io, cudaStream_t s);
  void build_prefetch_lists();
  void prefetch_next(int32_t l, cudaStream_t s);
  // Parallel branch for decode/verify: returns the aux stream after ordering it behind `s` (or
  // null for serial execution); join_aux orders `s` behind the branch.
  cudaStream_t fork_aux(const StepIo& io, cudaStream_t s);
  void join_aux(cudaStream_t s);

  ForwardContext ctx_;
  std::array<kernels::MoeExpertLayout, 48> layouts_{};
  kernels::MoeExpertLayout mtp_layout_{};
  DeviceBuffer prefetch_lists_;  // L2PrefetchList [48]: weights consumed by layer l + 1
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t join_ = nullptr;
  bool pending_join_ = false;
  cudaEvent_t aux_fork_ = nullptr;
  cudaEvent_t aux_join_ = nullptr;
};

}  // namespace fngine::model
