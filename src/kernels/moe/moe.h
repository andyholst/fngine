#pragma once

// Routed-expert MoE kernels: router top-k, GPU<->CPU exchange publish/combine, hot-expert decode
// GEMVs in the native GGUF formats, and the prefill grouped GEMM.
//
// Model semantics (per token t, MoE input x = mix bf16 [T][2560]):
//   top-10 over router_logits[t][0..511] (FP32; ties -> lower expert id),
//   w_k = softmax-renormalized over the 10 (norm_topk_prob), kept in FP32,
//   y[t] = sum_k w_k * down_e(silu(gate_e . x) * (up_e . x)) + y_shared[t] + y_cpu[t],
//   X[t][s] += inj[t][s] * y[t]   (GR write).
//
// Expert storage: a GPU-resident expert lives in a per-layer slot array; expert e of layer l is in
// slot slot_of[e] (int32, -1 = computed on the CPU) at `slots + slot * blob_stride`. A blob is the
// verbatim GGUF bytes gate[640 rows of 2560] | up[640 rows of 2560] | down[2560 rows of 640].
// Prefill addresses blobs through a per-layer device table expert_blob[512] instead (nullptr =
// computed on the CPU), so blobs may live anywhere in device memory (slot arrays, the prefill
// streaming ring).
//
// Decode (T <= 16, CUDA-graph capturable; T is a capture-time constant, `seq` is read from device
// memory at replay):
//   moe_select_publish -> moe_hot_gate_up -> moe_hot_down -> (shared expert, other kernels)
//   -> moe_combine.
// Prefill (eager; T and tile counts may be large, scalars are host values):
//   moe_route -> moe_grouped_gate_up -> moe_grouped_down -> moe_prefill_combine.
//
// Numerics:
//   decode: activations x and h are quantized per 32 values to int8 with an FP32 scale
//           (ggml quantize_q8_1 semantics: d = amax/127, q = roundf(v/d); the scale is kept FP32
//           instead of FP16); weights are consumed as exact integers via dp4a (ggml vec_dot_*_q8_1
//           with the IQ2_S/IQ3_XXS half-scale terms computed exactly instead of truncated);
//           FP32 accumulation; h is FP32. Sums over experts use a fixed slot order (deterministic).
//   prefill: weights dequantized exactly (ggml reference values) then rounded to BF16, BF16
//           activations, mma.m16n8k16 with FP32 accumulation; h stored BF16; the down GEMM
//           atomically adds fixed-point w*out into int64 y (exact, order-independent: prefill is
//           bit-reproducible).
#include "core/exchange.h"
#include "core/model_dims.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

// GGML type ids of the expert formats present in the pack.
enum class MoeFormat : int32_t {
  IQ3_XXS = 18,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  Q2_0 = 42,
};

// Bytes of one row of `k` elements in format `f` (k must be a multiple of the block size).
int64_t moe_row_bytes(MoeFormat f, int32_t k);
bool moe_gate_up_format_supported(MoeFormat f);  // IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL
bool moe_down_format_supported(MoeFormat f);     // IQ4_NL, Q2_0

// Per-layer expert storage descriptor (chosen once per layer at graph build). `slots` and
// `blob_stride` are used by decode only; the blob offsets and formats by both paths.
struct MoeExpertLayout {
  const uint8_t* slots = nullptr;  // device base of the layer's slot array (decode)
  int64_t blob_stride = 0;         // bytes between consecutive slots
  int64_t gate_offset = 0;         // byte offsets of gate / up / down inside a blob
  int64_t up_offset = 0;
  int64_t down_offset = 0;
  MoeFormat gate_up_format = MoeFormat::IQ3_XXS;
  MoeFormat down_format = MoeFormat::IQ4_NL;
};

// Standard layout for a pack blob (gate | up | down, packed back to back).
MoeExpertLayout make_moe_expert_layout(const uint8_t* slots, int64_t blob_stride,
                                       MoeFormat gate_up_format, MoeFormat down_format);

// Timeout of the GPU wait on the CPU expert pool (~10 s at the RTX 5090 boost clock).
inline constexpr long long kMoeExchangeTimeoutCycles = 10ll * 2'500'000'000ll;

// ------------------------------------------------------------------------------------- decode
inline constexpr int32_t kMoeDecodeMaxTokens = dims::kMaxStepTokens;                 // 16
inline constexpr int32_t kMoeDecodeMaxPairs = kMoeDecodeMaxTokens * dims::kTopK;     // 160
inline constexpr int32_t kMoeQ8Block = 32;
inline constexpr int32_t kMoeHiddenBlocks = dims::kHidden / kMoeQ8Block;       // 80
inline constexpr int32_t kMoeInterBlocks = dims::kExpertInter / kMoeQ8Block;   // 20

// Device-resident decode work list written by moe_select_publish. Hot work items are the distinct
// GPU slots of the step, sorted ascending; pairs are (token, expert) assignments ordered by item and
// then by token. token_pairs[t] lists token t's hot pairs in slot order (the combine order).
struct MoeDecodeWork {
  int32_t n_items;
  int32_t n_pairs;
  int32_t n_cold;
  int32_t tokens;
  int32_t item_slot[kMoeDecodeMaxPairs];
  int32_t item_pair_begin[kMoeDecodeMaxPairs + 1];
  int32_t pair_token[kMoeDecodeMaxPairs];
  float pair_weight[kMoeDecodeMaxPairs];
  int32_t token_pair_count[kMoeDecodeMaxTokens];
  int32_t token_pairs[kMoeDecodeMaxTokens][dims::kTopK];
  int32_t token_cold_count[kMoeDecodeMaxTokens];
  int32_t topk_expert[kMoeDecodeMaxTokens][dims::kTopK];  // router result, rank order
  float topk_weight[kMoeDecodeMaxTokens][dims::kTopK];
};

// Decode workspace (device memory, fixed size for T <= 16).
struct MoeDecodeWorkspace {
  MoeDecodeWork* work = nullptr;
  int8_t* xq = nullptr;     // [16][80][32] int8 activations
  float* xd = nullptr;      // [16][80] FP32 block scales
  float* h = nullptr;       // [160 pairs][640] FP32 silu(g)*u
  float* partial = nullptr; // [160 pairs][2560] FP32, routing weight applied
  uint32_t* relay = nullptr; // [48 layers][2] combine's CPU-tier wait relay {seq, ok}; zero-filled once
};
std::size_t moe_decode_workspace_bytes();
MoeDecodeWorkspace make_moe_decode_workspace(void* base);  // base: 256-byte aligned device memory

// Router top-k, hot/cold classification and exchange publish (single routing CTA plus T CTAs that
// quantize x). `slot_of` may be null (expert e is in slot e: MTP layer). `exchange` may be null (no
// CPU tier: cold pairs are not allowed). When `exchange` is set, the kernel writes
// publish[layer] = {T, n_cold, per-token cold lists in rank order, x rows of tokens with >= 1 cold
// pair}, executes __threadfence_system() and release-stores control.ready[layer] = *seq (always,
// also when n_cold == 0). `tokens` (1..16) is a capture-time constant. `routing_counts` (optional,
// device u32 [512], wraps) gets +1 for every routed (token, expert) pair of the step: the decode
// routing telemetry, counted here instead of in a separate launch.
cudaError_t moe_select_publish(const float* router_logits, const __nv_bfloat16* x,
                               const int32_t* slot_of, int32_t layer, const uint32_t* seq,
                               DecodeExchange* exchange, int32_t tokens,
                               const MoeDecodeWorkspace& ws, cudaStream_t stream,
                               uint32_t* routing_counts = nullptr);

// h[pair][j] = silu(gate_j . x_t) * (up_j . x_t) for every hot pair (persistent CTAs over tiles of
// 8 rows x item; each tile's gate/up rows stream into shared memory with cp.async in five K slices
// while the previous tile is decoded).
cudaError_t moe_hot_gate_up(const MoeExpertLayout& layout, int32_t tokens,
                            const MoeDecodeWorkspace& ws, cudaStream_t stream);
// partial[pair][n] = w_pair * (down_n . q8(h[pair])) (persistent CTAs over contiguous ranges of
// 32-row x item tiles, two staged). Partials are stored per hot pair ([160][2560]) rather than
// [items][T][2560].
cudaError_t moe_hot_down(const MoeExpertLayout& layout, int32_t tokens,
                         const MoeDecodeWorkspace& ws, cudaStream_t stream);

// y[t] = sum of token t's hot partials in slot order + y_shared[t]
//        + (cold pairs of t ? exchange.result[layer].y[t] : 0),
// X[t][s][d] += inj[t][s] * y[t][d]. When the step has cold pairs the kernel waits for
// control.done[layer] == *seq; on timeout it sets control.error = kExchangeErrorMoeTimeout |
// (layer + 1) and omits the CPU contribution. `y_out` (optional, FP32 [T][2560]) receives y.
// `ablate_direction` (optional, FP32 [2560], unit): the routed-expert part of y (GPU partials +
// CPU tier, not y_shared) is transformed along the direction before it is written,
// y_routed -= ablate_scale * r * (r . y_routed). This equals editing every routed expert's down
// projection (W -= scale * r r^T W) while the packed expert bytes stay unchanged. y_shared is added
// untransformed: the caller's shared-expert weights already carry the edit.
cudaError_t moe_combine(const MoeDecodeWorkspace& ws, const float* y_shared,
                        DecodeExchange* exchange, int32_t layer, const uint32_t* seq,
                        const float* inj, float* X, float* y_out, int32_t tokens,
                        long long timeout_cycles, const float* ablate_direction,
                        float ablate_scale, cudaStream_t stream);

// ------------------------------------------------------------------------------------ prefill
// Grouped GEMM tiles: 64 tokens x 64 rows (gate/up: 64 intermediate rows = 64 gate + 64 up weight
// rows; down: 64 output rows).
inline constexpr int32_t kMoeTileTokens = 64;
inline constexpr int32_t kMoeTileRows = 64;

// Prefill workspace (device memory) for chunks of up to `capacity` tokens.
struct MoePrefillWorkspace {
  int32_t capacity = 0;
  int32_t max_tiles = 0;            // ceil(capacity*10/64) + 512
  int32_t* meta = nullptr;          // [0]=n_tiles [1]=n_hot_pairs [2]=n_cold [3]=n_rows
  int32_t* expert_count = nullptr;  // [512] hot pairs per expert
  int32_t* expert_offset = nullptr; // [513]
  int32_t* expert_cursor = nullptr; // [512]
  int32_t* topk_expert = nullptr;   // [capacity][10]
  float* topk_weight = nullptr;     // [capacity][10]
  int32_t* token_cold = nullptr;    // [capacity] cold pairs per token
  int32_t* row_of_token = nullptr;  // [capacity] compacted exchange row or -1
  int32_t* pair_token = nullptr;    // [capacity*10] hot pairs grouped by expert
  float* pair_weight = nullptr;     // [capacity*10]
  int4* tiles = nullptr;            // [max_tiles] {pair_begin, pair_end, 0, expert}
  __nv_bfloat16* h = nullptr;       // [capacity*10][640] per hot pair
  long long* y_fixed = nullptr;     // [capacity][2560] hot-expert sum, fixed point (see below)
  uint32_t* relay = nullptr;        // [48 layers][2] combine's CPU-tier wait relay {seq, ok}; zero-filled once
};

// The prefill hot-expert sum is accumulated with 64-bit integer atomics in fixed point
// (value * 2^kMoeYFixedShift, each addend rounded once from FP32), which makes the result exact
// and independent of the summation order (bit-reproducible runs). Requires |y| < 2^31.
inline constexpr int32_t kMoeYFixedShift = 32;
inline float moe_y_fixed_to_float(long long v) {
  return static_cast<float>(v) * (1.0f / 4294967296.0f);  // one rounding; the scale is exact
}
std::size_t moe_prefill_workspace_bytes(int32_t capacity);
MoePrefillWorkspace make_moe_prefill_workspace(void* base, int32_t capacity);

// Top-k, hot/cold split, per-expert token lists and 64-token tiles; zeroes ws.y_fixed; publishes the
// compacted cold rows (row_token, cold lists, x) into `view` (header = {T, n_rows, n_cold}) and
// release-stores control->ready[layer] = *seq (always). `control`/`view.header` may be null when
// no CPU tier exists (then no cold pairs are allowed). `expert_blob`: device [512] blob addresses
// of the layer (nullptr = cold, computed on the CPU). Eager: `tokens` <= ws.capacity.
cudaError_t moe_route(const float* router_logits, const __nv_bfloat16* x,
                      const uint8_t* const* expert_blob, int32_t layer, const uint32_t* seq,
                      ExchangeControl* control, const PrefillExchangeView& view, int32_t tokens,
                      const MoePrefillWorkspace& ws, cudaStream_t stream);

// h[pair] = bf16(silu(gate . x) * (up . x)) over 64-token x 64-row tiles (x: bf16 [T][2560]).
// Tile expert e reads blob expert_blob[e] (the table passed to moe_route; it must not change in
// between).
cudaError_t moe_grouped_gate_up(const MoeExpertLayout& layout, const uint8_t* const* expert_blob,
                                const __nv_bfloat16* x, int32_t tokens,
                                const MoePrefillWorkspace& ws, cudaStream_t stream);
// ws.y_fixed[token] += fixed(w * (down . h[pair])) over 64-token x 64-row tiles (deterministic).
cudaError_t moe_grouped_down(const MoeExpertLayout& layout, const uint8_t* const* expert_blob,
                             int32_t tokens, const MoePrefillWorkspace& ws, cudaStream_t stream);

// y[t] = float(ws.y_fixed[t]) + y_shared[t] + (row_of_token[t] >= 0 ? view.y[row] : 0). When `control` is set
// the kernel always waits for control->done[layer] == *seq (the CPU pool releases done for every
// prefill chunk, also without cold pairs, because one exchange region serves all layers); on
// timeout it sets the error as in moe_combine and omits the CPU rows.
// X[t][s] += inj[t][s] * y[t]. `y_out` optional FP32 [T][2560]. `ablate_direction` as in moe_combine.
cudaError_t moe_prefill_combine(const MoePrefillWorkspace& ws, const float* y_shared,
                                ExchangeControl* control, const PrefillExchangeView& view,
                                int32_t layer, const uint32_t* seq, const float* inj, float* X,
                                float* y_out, int32_t tokens, long long timeout_cycles,
                                const float* ablate_direction, float ablate_scale,
                                cudaStream_t stream);

}  // namespace fngine::kernels
