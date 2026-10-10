#pragma once

// Decode/verify steps. One CUDA graph per width T (1..9) runs: input refresh, embedding, the 48
// layers (decode or verify pass), final mixer + lm_head, sampling or speculative acceptance, the
// GDN/PLE folds, and (when MTP is enabled) the MTP bridge plus two autoregressive draft steps,
// then stores the step outputs in device-mapped host memory. CPU experts and the n-gram rows join
// the graph through the host-mapped exchange (no host launches inside a step).
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/host_memory.h"
#include "kernels/head/head.h"
#include "model/runtime.h"
#include "model/step_kernels.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace fngine::model {

struct StepResult {
  int32_t licensed[kStepMaxTokens] = {};
  int32_t count = 0;     // licensed tokens (accepted + 1)
  int32_t accepted = 0;  // accepted drafts
  int32_t mtp_drafts[kMtpDrafts] = {};
  int32_t mtp_count = 0;  // valid drafts; zero when drafting would exceed the context
  float target_gap[kStepMaxTokens] = {};  // populated only with logit diagnostics enabled
  double seconds = 0;    // launch -> outputs available
  [[nodiscard]] std::span<const int32_t> tokens() const { return {licensed, std::size_t(count)}; }
};

class StepRunner {
 public:
  StepRunner(ModelRuntime& rt, bool use_graphs, bool mtp, bool logit_diagnostics = false);
  StepRunner(const StepRunner&) = delete;
  StepRunner& operator=(const StepRunner&) = delete;

  [[nodiscard]] bool mtp() const noexcept { return mtp_; }

  // Sampling parameters for the request; resets the generated-token counts (penalties).
  void set_sampling(const kernels::SamplingConfig& config);

  // After a prefill: logits of the last prefilled row, sample the first generated token, and
  // (MTP) bridge + draft. `length` = committed tokens (= prompt length).
  StepResult first_token(int32_t length);

  // One decode (k = 0) or verify step: tokens [anchor, drafts[0..k)] at positions
  // rt.position() .. +k. Commits the licensed tokens into the runtime (position/history).
  StepResult step(int32_t anchor, std::span<const int32_t> drafts);

  // Drops captured graphs (e.g. after the KV/state buffers were reallocated).
  void reset_graphs();

 private:
  void enqueue_step(int32_t T, cudaStream_t s, bool draft);
  void enqueue_mtp(int32_t T, const float* x_src, cudaStream_t s);
  void draft_logits_argmax(int32_t rows, cudaStream_t s);
  StepResult finish(double seconds, bool draft) const;

  ModelRuntime& rt_;
  bool use_graphs_;
  bool mtp_;
  bool logit_diagnostics_;
  DeviceBuffer dev_;
  MappedHostBuffer host_;  // StepInDev, then StepOutDev
  StepInDev* in_dev_ = nullptr;
  StepOutDev* out_dev_ = nullptr;
  StepScratch* scratch_ = nullptr;
  kernels::SamplingConfig* config_dev_ = nullptr;
  int32_t* token_counts_ = nullptr;
  float* x_ar_ = nullptr;
  StepInDev* in_host_ = nullptr;
  StepOutDev* out_host_ = nullptr;
  const StepInDev* in_mapped_ = nullptr;  // device aliases of in_host_ / out_host_
  StepOutDev* out_mapped_ = nullptr;
  std::array<std::unique_ptr<DecodeGraphExecutable>, kStepMaxTokens + 1> graphs_{};
};

}  // namespace fngine::model
