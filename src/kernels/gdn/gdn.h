#pragma once

// Gated DeltaNet (GDN) kernels for the 36 linear-attention layers of Qwen3.8-Flash-Next.
//
// Per-layer data flow (T = tokens in the step/chunk, all buffers token-major):
//   qkvz  BF16 [T][16384] = q(2048) | k(2048) | v(6144) | z(6144)        (Linear agent output)
//   ab    FP32 [T][96]    = a(48) | b(48)                                (BF16 GEMV output)
//   conv:     conv_out BF16 [T][10240] = silu(causal depthwise conv4(qkvz[:, 0:10240]))
//             weight F32 [10240][4] channel-major, tap 3 multiplies x[t]; history BF16 [3][10240]
//             oldest first.
//   gating:   g[t][h] = -exp(A_log[h]) * softplus(a[t][h] + dt_bias[h]), beta[t][h] = sigmoid(b)
//             (FP32 [T][48] each).
//   core:     q = conv_out[:, 0:2048], k = conv_out[:, 2048:4096], v = conv_out[:, 4096:10240]
//             are read in place (row stride 10240, no copies); q/k are L2-normalized
//             (x * rsqrt(sum x^2 + 1e-6)) per 128-wide head, value head h uses q/k head h/3,
//             S = S*exp(g); delta = beta*(v - S k); S += delta k^T; o = (S q) / sqrt(128).
//             Output core BF16 [T][6144]; state FP32 [48][128 v][128 k] per layer.
//   out_norm: out[t][h*128+d] = bf16(bf16(core*rsqrt(mean_h(core^2)+1e-6)) * w[d] * sigmoid(z)).
//
// Decode paths (CUDA-graph capturable; kernels never allocate or synchronize):
//   * plain T=1 step:   gdn_conv_decode + gdn_recurrent_decode update the state of
//                       control->source_slot in place.
//   * verify step T<=16: gdn_conv_record + gdn_recurrent_record leave every state untouched and
//                       write per-column records; after acceptance, gdn_fold (one launch, all 36
//                       layers) replays control->commit records from control->source_slot into
//                       control->destination_slot. Records + fold are bit-identical to `commit`
//                       sequential plain T=1 steps (state, conv history and outputs).
//   * gdn_swap_slots flips source/destination for ping-pong state slots (optional; the fold is
//     also correct with source == destination).
// Capture-time constants: tensor pointers, `layer`, `tokens` (the graph width) and the pool
// geometry. Dynamic per-replay values are read from the device-resident GdnStepControl
// (source/destination slot, commit count).
//
// Prefill paths (eager; host scalars are read at launch): gdn_conv_prefill and gdn_prefill
// (chunked for the leading T - T%64 tokens, chunk 64, then a recurrent tail; any T >= 1).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

namespace gdn_dims {
inline constexpr std::int32_t kLayers         = 36;
inline constexpr std::int32_t kQkHeads        = 16;
inline constexpr std::int32_t kValueHeads     = 48;
inline constexpr std::int32_t kHeadDim        = 128;
inline constexpr std::int32_t kKeyDim         = kQkHeads * kHeadDim;               // 2048
inline constexpr std::int32_t kValueDim       = kValueHeads * kHeadDim;            // 6144
inline constexpr std::int32_t kConvChannels   = 2 * kKeyDim + kValueDim;           // 10240
inline constexpr std::int32_t kConvHistory    = 3;                                 // width 4 - 1
inline constexpr std::int32_t kQkvzLd         = kConvChannels + kValueDim;         // 16384
inline constexpr std::int32_t kZOffset        = kConvChannels;                     // z columns
inline constexpr std::int32_t kAbLd           = 2 * kValueHeads;                   // 96
inline constexpr std::int32_t kMaxRecordWidth = 16;                                // verify width
inline constexpr std::int32_t kChunk          = 64;
inline constexpr std::size_t kRecurrentLayerElems =
    static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim;                   // 786432
inline constexpr std::size_t kConvLayerElems =
    static_cast<std::size_t>(kConvHistory) * kConvChannels;                        // 30720
} // namespace gdn_dims

// ---------------------------------------------------------------------------------------------
// State pool: `slots` complete GDN state images (all 36 layers). Slot-major so one slot is two
// contiguous ranges (session snapshots copy a slot with two memcpys):
//   recurrent FP32 [slots][36][48][128 v][128 k]   (3 MiB per layer, 108 MiB per slot)
//   conv      BF16 [slots][36][3][10240]           (60 KiB per layer)
struct GdnStatePool {
    float* recurrent         = nullptr;
    __nv_bfloat16* conv      = nullptr;
    std::int32_t slots       = 0;
};

// Total backing bytes for a pool of `slots` (>= 1) slots; regions are 256-byte aligned.
std::size_t gdn_state_pool_bytes(std::int32_t slots);
// Bytes of one slot's recurrent / conv image (all 36 layers).
std::size_t gdn_state_slot_recurrent_bytes();
std::size_t gdn_state_slot_conv_bytes();
// Binds caller-owned device memory (>= gdn_state_pool_bytes(slots), 256-byte aligned).
GdnStatePool gdn_state_pool_bind(void* backing, std::size_t backing_bytes, std::int32_t slots);
// Per-(slot, layer) pointers; layer is the dense GDN index 0..35.
float* gdn_recurrent_state(const GdnStatePool& pool, std::int32_t slot, std::int32_t layer);
__nv_bfloat16* gdn_conv_state(const GdnStatePool& pool, std::int32_t slot, std::int32_t layer);

// Device-resident per-step control, read by decode kernels at graph replay.
struct alignas(16) GdnStepControl {
    std::int32_t source_slot;       // committed state at the start of the step
    std::int32_t destination_slot;  // fold output (may equal source_slot)
    std::int32_t commit;            // columns the fold commits (clamped to [0, record width])
    std::int32_t reserved;
};

// ---------------------------------------------------------------------------------------------
// Verify records for all 36 layers, width capacity 16 (column index = layer * 16 + t):
//   conv  BF16 [36][16][10240]   raw conv inputs x[t] (qkvz cols 0..10239)
//   key   BF16 [36][16][16][128] raw (unnormalized) keys
//   value BF16 [36][16][48][128]
//   gate  FP32 [36][16][48][2]   {g, beta}
struct GdnRecords {
    __nv_bfloat16* conv  = nullptr;
    __nv_bfloat16* key   = nullptr;
    __nv_bfloat16* value = nullptr;
    float* gate          = nullptr;
};

std::size_t gdn_records_bytes();
GdnRecords gdn_records_bind(void* backing, std::size_t backing_bytes);

// ---------------------------------------------------------------------------------------------
// Convolution. `qkvz` points at row 0 of the projection output (row stride 16384); only columns
// 0..10239 are read. conv_out is BF16 [T][10240]. conv_w is F32 [10240][4].

// T = 1, in place on the conv history of (control->source_slot, layer).
void gdn_conv_decode(const __nv_bfloat16* qkvz, const float* conv_w, const GdnStatePool& pool,
                     std::int32_t layer, const GdnStepControl* control, __nv_bfloat16* conv_out,
                     cudaStream_t stream);

// T = 1..16 verify columns starting from (control->source_slot, layer); no state write. Writes
// conv_out and the layer's conv record columns 0..T-1.
void gdn_conv_record(const __nv_bfloat16* qkvz, const float* conv_w, const GdnStatePool& pool,
                     std::int32_t layer, const GdnStepControl* control, const GdnRecords& records,
                     std::int32_t tokens, __nv_bfloat16* conv_out, cudaStream_t stream);

// Prefill (eager), any T >= 1: reads history `state_in`, writes conv_out and the trailing
// history to `state_out` (may alias state_in).
void gdn_conv_prefill(const __nv_bfloat16* qkvz, const float* conv_w,
                      const __nv_bfloat16* state_in, __nv_bfloat16* state_out,
                      __nv_bfloat16* conv_out, std::int32_t tokens, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Gating: ab FP32 [T][96] -> g, beta FP32 [T][48]. A_log, dt_bias F32 [48].
void gdn_gating(const float* ab, const float* A_log, const float* dt_bias, float* g, float* beta,
                std::int32_t tokens, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Recurrence. conv_out BF16 [T][10240] supplies q/k/v; g/beta FP32 [T][48]; core BF16 [T][6144].

// T = 1, in place on (control->source_slot, layer).
void gdn_recurrent_decode(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                          const GdnStatePool& pool, std::int32_t layer,
                          const GdnStepControl* control, __nv_bfloat16* core,
                          cudaStream_t stream);

// T = 1..16 verify columns from (control->source_slot, layer); no state write. Column t's output
// is bit-identical to t+1 sequential gdn_recurrent_decode steps. Writes the layer's key/value/
// gate records.
void gdn_recurrent_record(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                          const GdnStatePool& pool, std::int32_t layer,
                          const GdnStepControl* control, const GdnRecords& records,
                          std::int32_t tokens, __nv_bfloat16* core, cudaStream_t stream);

// One launch for all 36 layers: replays record columns [0, commit) from the source slot into the
// destination slot (recurrent state and conv history = tail3(old history ‖ conv record[0:commit])).
// commit = 0 copies the source image. `width` is the record width of the step (1..16).
void gdn_fold(const GdnStatePool& pool, const GdnRecords& records, const GdnStepControl* control,
              std::int32_t width, cudaStream_t stream);

// Swaps control->source_slot and control->destination_slot (single-thread kernel).
void gdn_swap_slots(GdnStepControl* control, cudaStream_t stream);

// Prefill recurrence (eager), any T >= 1: chunked WY form (chunk 64) over the leading T - T%64
// tokens with FP32 intermediates and FP32 FMA throughout, recurrent kernel for the tail.
// state_in/state_out are one layer's FP32 [48][128][128] (may alias). Workspace (FP32
// normalized q/k, U, W, intra-chunk Q.K, decays) is ~74.5 KiB per token, 0 for T < 64.
std::size_t gdn_prefill_workspace_bytes(std::int32_t tokens);
void gdn_prefill(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                 const float* state_in, float* state_out, __nv_bfloat16* core,
                 std::int32_t tokens, void* workspace, std::size_t workspace_bytes,
                 cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// Gated output norm. core BF16 [T][6144]; z = qkvz + 10240 with row stride z_ld (16384);
// w F32 [128] (plain weight); out BF16 [T][6144] (may not alias core).
void gdn_out_norm(const __nv_bfloat16* core, const __nv_bfloat16* z, std::int32_t z_ld,
                  const float* w, __nv_bfloat16* out, std::int32_t tokens, cudaStream_t stream);

} // namespace fngine::kernels
