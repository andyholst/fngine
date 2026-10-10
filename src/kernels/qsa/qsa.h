#pragma once

// QSA (sparse attention) kernels for the 12 QSA layers and the MTP layer of Qwen3.8-Flash-Next.
//
// Per QSA layer and step the model runs, in order:
//   1. qsa_qk_norm_rope      qkv[T][13312] -> q[T][24][256], k[T][2][256]
//   2. qsa_kv_append         k, v (= qkv cols 12800.., ld 13312) -> paged KV cache
//   3. indexer_prepare       idx[T][640] -> q_idx[T][4][128]; block keys + raw-key ring
//   4. indexer_score_topk    q_idx, block keys -> selected[T][512] blocks, counts[T]
//   5. qsa_attention         q, cache, selection, gate (qkv) -> out[T][6144] (o_proj input)
//
// Conventions (all kernels):
//   * Single sequence. `positions[t]` (device int32) is the absolute position of token t;
//     positions inside one call are consecutive (positions[t] = positions[0] + t). They are read
//     on the device, so a CUDA graph captured for a fixed T replays at any position.
//   * `tokens` (T) is a host scalar baked into a captured graph (one graph per width).
//   * Kernels never allocate or synchronize; scratch comes from caller-provided workspace.
//   * RMSNorm multipliers are the pack's FP32 effective multipliers (1 + w already applied).
//   * RoPE: NeoX rotation of dims [0, 64) of each head, pairs (i, i + 32), angle
//     float(float(pos) * inv_freq[i]) with inv_freq the HF float32 table (theta 1e7, dim 64).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

// ---------------------------------------------------------------------------------------------
// 1. Query/key RMSNorm + RoPE.
// ---------------------------------------------------------------------------------------------
// qkv: bf16 [T][13312] (q_proj interleaved per head: query = cols h*512 .. h*512+255,
// gate = h*512+256 .. h*512+511; k = cols 12288..12799 as [2][256]; v = cols 12800..13311).
// q_norm/k_norm: FP32 [256] multipliers. Output q bf16 [T][24][256], k bf16 [T][2][256]:
//   x_n = bf16(x * rsqrt(mean(x^2) + 1e-6) * mult);  out = bf16(rope(x_n, positions[t])).
void qsa_qk_norm_rope(const __nv_bfloat16* qkv, const float* q_norm, const float* k_norm,
                      const int32_t* positions, int32_t tokens, __nv_bfloat16* q_out,
                      __nv_bfloat16* k_out, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// 2. Paged KV cache (one per QSA layer). Single sequence with an identity page table: token
//    position p lives in page p / 64 at page offset p % 64. Each plane is laid out
//    [page][kv_head (2)][64 tokens][leading], cinference's PageMajor layout, so a real block table
//    can be introduced later without changing the codecs.
//      K8V4 (default, 402 B per token per KV head; cinference Fp8KeyNvfp4Value):
//        k      uint8 E4M3 codes [pages][2][64][256]   of H·k (normalized Sylvester H256)
//        k_scale FP16           [pages][2][64]        scale = fp16(absmax / 448)
//        v      uint8 packed E2M1 [pages][2][64][128] of H·v (element 2j low nibble)
//        v_scale uint8 E4M3     [pages][2][64][16]    per 16-element group, scale = absmax / 6
//      K8V8 (516 B per token per KV head): K exactly as K8V4; V like K:
//        v      uint8 E4M3 codes [pages][2][64][256]   of H·v
//        v_scale FP16           [pages][2][64]        scale = fp16(absmax / 448) per (token, head)
//      BF16 (validation option): k, v bf16 [pages][2][64][256]; k_scale = v_scale = nullptr.
// ---------------------------------------------------------------------------------------------
enum class QsaKvFormat : int32_t { K8V4 = 0, BF16 = 1, K8V8 = 2 };

struct QsaKvCache {
    QsaKvFormat format = QsaKvFormat::K8V4;
    int32_t capacity_tokens = 0;  // multiple of 64
    void* k = nullptr;
    void* v = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
};

inline constexpr int32_t kQsaKvPageTokens = 64;

// Bytes per token per layer (both KV heads): K8V4 804, K8V8 1032, BF16 2048.
size_t qsa_kv_bytes_per_token(QsaKvFormat format);
// Bytes of one layer's cache for `capacity_tokens` (rounded up to whole 64-token pages).
size_t qsa_kv_layer_bytes(QsaKvFormat format, int32_t capacity_tokens);
// Carves the planes out of one allocation of qsa_kv_layer_bytes(format, capacity) bytes
// (base 256-byte aligned). capacity_tokens is rounded up to a page multiple.
QsaKvCache qsa_kv_cache_view(QsaKvFormat format, void* base, int32_t capacity_tokens);
// Byte extents of those planes in address order: k, k_scale, v, v_scale (BF16: k, v; then two
// empty planes). Each is position-major: tokens < n use its first pages(n) * 64 * bytes / tokens.
struct QsaKvPlane {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};
std::array<QsaKvPlane, 4> qsa_kv_planes(QsaKvFormat format, int32_t capacity_tokens);

// Writes token t's K/V at positions[t]. k: bf16 [T][2][256] (ld 512); v: bf16 rows with
// v + t * v_ld + head * 256 (v_ld = 13312 to read in place from qkv).
void qsa_kv_append(const QsaKvCache& cache, const __nv_bfloat16* k, const __nv_bfloat16* v,
                   int64_t v_ld, const int32_t* positions, int32_t tokens, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// 3. Indexer prepare.
// ---------------------------------------------------------------------------------------------
// Per-layer indexer state:
//   ring        bf16 [16][128]          raw key of position p at slot p % 16
//   block_keys  bf16 [max_blocks][128]  block b = RoPE(k_norm(bf16(mean_fp32(raw[4b..4b+3]))), 4b)
// Zero-initialize nothing: every slot is written before it is read.
size_t indexer_state_bytes(int32_t capacity_tokens);  // ring + block keys for capacity/4 blocks

// idx: bf16 [T][640] = 4 query heads x 128 | raw key x 128. q_norm/k_norm: FP32 [128].
// Outputs q_idx bf16 [T][4][128] = bf16(rope(bf16(rmsnorm(q_h) * q_norm), positions[t])).
// For every position p in the batch with p % 4 == 3, writes block_keys[p / 4]; raw keys of
// in-batch positions come from `idx`, earlier ones from the ring. Afterwards the ring holds the
// raw keys of the batch's last min(T, 16) positions.
//
// Rollback safety. A verify batch of T tokens at positions p0 .. p0+T-1 may later commit only
// n >= 1 of them; the next batch starts at p0+n and reads the ring only for positions in
// [4*floor((p0+n)/4), p0+n), i.e. at most 3 positions back, all >= p0+n-3 >= p0-2.
//   * Positions >= p0 among them are committed (< p0+n) and were written correctly by this batch.
//   * Positions p0-2 .. p0-1 sit in slots (p0-2) % 16, (p0-1) % 16, which this batch overwrites
//     only if p0+j == p0-k (mod 16) for some j < T, k <= 2, i.e. j >= 14: never for T <= 14.
// Block keys of rejected positions are garbage but are rewritten when the positions are
// re-processed, before any later query can select them (a query at p only scores b < (p+1)/4).
// So verify batches of up to 14 tokens are rollback safe; fngine uses <= 9 (graph widths) and the
// plan's bound of 12 holds. Prefill batches (any T) are always fully committed.
void indexer_prepare(const __nv_bfloat16* idx, const int32_t* positions, int32_t tokens,
                     const float* q_norm, const float* k_norm, __nv_bfloat16* ring,
                     __nv_bfloat16* block_keys, __nv_bfloat16* q_idx, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// 4. Indexer score + top-k.
// ---------------------------------------------------------------------------------------------
// For query t at position p: nb = (p + 1) / 4 complete blocks; score_b = sum_h relu(q_h . k_b)
// / sqrt(128) (FP32, from bf16 inputs); select the top min(512, nb) blocks (exact ties -> lower
// block id), written sorted ascending to selected[t][0 .. counts[t]) (int32 [T][512]).
// When nb <= 512 every block is selected (dense; scores are not computed): selected = 0..nb-1,
// counts = nb. The attention kernel recognizes the dense case from positions alone.
// `max_blocks` is a host bound >= nb of every query in the call (decode graphs: capacity / 4;
// prefill: (last position + 1) / 4); it sizes the grid and the scores workspace row stride.
// Queries are processed 256 at a time through an FP32 scores workspace [min(T,256)][max_blocks
// rounded up to 4]. Selection is exact: a sampled threshold compacts ~1-2K candidates per query,
// with a full radix-select fallback (taken when sampling misjudges the threshold, or always when
// nb > 65536, i.e. contexts above 256K).
size_t indexer_score_workspace_bytes(int32_t tokens, int32_t max_blocks);
void indexer_score_topk(const __nv_bfloat16* q_idx, const __nv_bfloat16* block_keys,
                        const int32_t* positions, int32_t tokens, int32_t max_blocks,
                        void* workspace, size_t workspace_bytes, int32_t* selected,
                        int32_t* counts, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// 5. Sparse attention.
// ---------------------------------------------------------------------------------------------
// Query t at position p attends to the keys of its selected blocks (4 tokens each) plus the tail
// [4*nb, p] (dense: all of 0..p when nb <= 512), at most 2051 keys. One CTA per (token, KV head,
// split) covers that KV head's 12 query heads (padded to 16 MMA rows); keys/values are gathered
// from the paged cache into shared memory, 32 keys per step, with FP32 online softmax
// (scale 1/16); probabilities enter the P.V MMA as hi + lo halves (~2^-17 relative or better).
// BF16 cache: bf16 MMA. K8V4: the query is Hadamard-rotated and held as FP16 hi + lo, K enters
// the f16 MMA as its exact E4M3 codes with the per-key FP16 scale applied to the FP32 score, V as
// exact FP16 (E2M1 code * E4M3 group scale); the output is inverse-rotated. Results match an FP64
// oracle over the stored (dequantized) cache to ~1e-5 of rms beyond the final bf16 rounding.
// Output: out[t][h*256+d] = bf16(attn * sigmoid(gate)), gate = qkv cols h*512+256+d (ld 13312).
// For small T the key range is split across `qsa_attention_splits(T)` CTAs with an FP32 merge.
// qsa_attention_workspace_bytes(max_tokens) covers every call with 1..max_tokens tokens.
int32_t qsa_attention_splits(int32_t tokens);
size_t qsa_attention_workspace_bytes(int32_t max_tokens);
void qsa_attention(const QsaKvCache& cache, const __nv_bfloat16* q, const __nv_bfloat16* qkv,
                   const int32_t* positions, const int32_t* selected, const int32_t* counts,
                   int32_t tokens, void* workspace, size_t workspace_bytes, __nv_bfloat16* out,
                   cudaStream_t stream);

}  // namespace fngine::kernels
