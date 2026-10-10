#pragma once

// Activation buffers and kernel scratch for one forward invocation of up to `max_tokens` tokens
// (the prefill chunk; decode/verify graphs use the leading rows of the same buffers). GDN and QSA
// scratch share one region because a layer is one or the other; the MTP layer reuses the main
// scratch after the main layers finished.
#include "core/arena.h"
#include "kernels/linear/linear.h"
#include "kernels/moe/moe.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace fngine::model {

struct WorkspaceOptions {
  int32_t max_tokens = 4096;   // prefill chunk (>= 16)
  int32_t logit_rows = 16;     // rows of the FP32 logits buffer (>= 16)
  int32_t max_blocks = 32768;  // indexer blocks (KV capacity / 4)
  bool dense_bf16 = false;     // size the BF16 linear workspaces too
};

struct Workspace {
  WorkspaceOptions options;

  // Inputs (device).
  int32_t* tokens = nullptr;     // [max_tokens + 1], next prompt token for MTP
  int32_t* positions = nullptr;  // [max_tokens]
  uint32_t* seq = nullptr;       // [1]

  // Residual streams and GR.
  float* X = nullptr;            // [T][4][2560]
  __nv_bfloat16* Xn = nullptr;   // [T][10240]
  float* inj = nullptr;          // [T][4]
  __nv_bfloat16* r = nullptr;    // [T][384], cols 320.. zero
  __nv_bfloat16* mix = nullptr;  // [T][2560]
  __nv_bfloat16* emb = nullptr;  // [T][2560]

  // GDN scratch (aliases the QSA scratch).
  __nv_bfloat16* qkvz = nullptr;      // [T][16384]
  float* ab = nullptr;                // [T][96]
  __nv_bfloat16* conv_out = nullptr;  // [T][10240]
  float* g = nullptr;                 // [T][48]
  float* beta = nullptr;              // [T][48]
  __nv_bfloat16* core = nullptr;      // [T][6144]
  __nv_bfloat16* normed = nullptr;    // [T][6144]

  // QSA scratch.
  __nv_bfloat16* qkv = nullptr;    // [T][13312]
  __nv_bfloat16* q = nullptr;      // [T][24][256]
  __nv_bfloat16* k = nullptr;      // [T][2][256]
  __nv_bfloat16* idx = nullptr;    // [T][640]
  __nv_bfloat16* q_idx = nullptr;  // [T][4][128]
  int32_t* selected = nullptr;     // [T][512]
  int32_t* counts = nullptr;       // [T]
  __nv_bfloat16* attn = nullptr;   // [T][6144]

  // MoE.
  float* router = nullptr;            // [T][513]
  __nv_bfloat16* shared_h = nullptr;  // [T][640]
  float* y_shared = nullptr;          // [T][2560]
  kernels::MoeDecodeWorkspace moe_decode{};
  kernels::MoePrefillWorkspace moe_prefill{};

  // PLE.
  __nv_bfloat16* E = nullptr;         // [T][2560]
  __nv_bfloat16* K = nullptr;         // [T][10240]
  __nv_bfloat16* V = nullptr;         // [T][2560]
  __nv_bfloat16* c_record = nullptr;  // [T][10240]

  // MTP.
  float* X_mtp = nullptr;           // [T][4][2560]
  __nv_bfloat16* hid_n = nullptr;   // [T][10240]
  __nv_bfloat16* h = nullptr;       // [T*4][2560]
  __nv_bfloat16* e = nullptr;       // [T][2560]
  __nv_bfloat16* emb_n = nullptr;   // [T][2560]

  // Head.
  float* logits = nullptr;  // [logit_rows][248320]

  // Kernel scratch.
  void* linear_ws = nullptr;   std::size_t linear_ws_bytes = 0;
  void* gdn_ws = nullptr;      std::size_t gdn_ws_bytes = 0;
  void* ple_ws = nullptr;      std::size_t ple_ws_bytes = 0;
  void* score_ws = nullptr;    std::size_t score_ws_bytes = 0;
  void* attn_ws = nullptr;     std::size_t attn_ws_bytes = 0;
  void* sample_ws = nullptr;   std::size_t sample_ws_bytes = 0;
  void* accept_ws = nullptr;   std::size_t accept_ws_bytes = 0;
};

class WorkspaceBuffer {
 public:
  static std::size_t device_bytes(const WorkspaceOptions& options);
  // Non-owning view into a placement loan; does not initialize/overwrite resident experts.
  static Workspace view(const WorkspaceOptions& options, void* memory, std::size_t bytes);
  // Initialize GR padding each time the loan changes from expert storage to scratch.
  static void initialize(const Workspace& workspace, cudaStream_t stream);
  explicit WorkspaceBuffer(const WorkspaceOptions& options);
  WorkspaceBuffer(const WorkspaceBuffer&) = delete;
  WorkspaceBuffer& operator=(const WorkspaceBuffer&) = delete;

  [[nodiscard]] Workspace& get() noexcept { return ws_; }
  [[nodiscard]] const Workspace& get() const noexcept { return ws_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return buffer_.bytes; }

 private:
  DeviceBuffer buffer_;
  Workspace ws_;
};

}  // namespace fngine::model
