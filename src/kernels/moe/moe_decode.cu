// Decode/verify MoE (T <= 16): router select + exchange publish, hot-expert GEMVs over the native
// GGUF formats (dp4a against q8 activations), deterministic combine with the CPU tier.
#include "kernels/moe/iq_device.cuh"
#include "kernels/moe/moe.h"
#include "kernels/moe/moe_common.cuh"

#include "kernels/common/math.cuh"
#include "kernels/common/pdl.cuh"

#include <cuda_bf16.h>

#include <algorithm>

namespace fngine::kernels {
namespace {

using moe_iq::Sub32;
using moe_iq::decode_sub32;
using moe_iq::kRowBytes;
using moe_iq::quantize_q8_lane;
using moe_iq::sub32_dot;
using moe_detail::warp_topk10;

constexpr int kHidden = dims::kHidden;
constexpr int kInter = dims::kExpertInter;
constexpr int kTopK = dims::kTopK;
constexpr int kExperts = dims::kExperts;
constexpr int kMaxTokens = kMoeDecodeMaxTokens;
constexpr int kMaxPairs = kMoeDecodeMaxPairs;
constexpr int kSelectThreads = 512;
static_assert(kSelectThreads == kExperts, "select kernel maps one thread per expert / slot");

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

struct DecodeLayout {
  std::size_t work, xq, xd, h, partial, relay, total;
};
constexpr DecodeLayout decode_layout() {
  DecodeLayout l{};
  std::size_t off = 0;
  l.work = off;    off = align256(off + sizeof(MoeDecodeWork));
  l.xq = off;      off = align256(off + std::size_t(kMaxTokens) * kHidden);
  l.xd = off;      off = align256(off + std::size_t(kMaxTokens) * kMoeHiddenBlocks * sizeof(float));
  l.h = off;       off = align256(off + std::size_t(kMaxPairs) * kInter * sizeof(float));
  l.partial = off; off = align256(off + std::size_t(kMaxPairs) * kHidden * sizeof(float));
  l.relay = off;   off = align256(off + std::size_t(kExchangeLayers) * 2 * sizeof(uint32_t));
  l.total = off;
  return l;
}

// ------------------------------------------------------------------------- select + publish
// Block 0 routes, counts and publishes; blocks 1..T quantize x[t] to q8 (32-value blocks, FP32
// scale). Programmatic dependent (pdl.cuh): it waits first; the slot map (rewritten between
// steps), router logits and x are read after it.
__global__ void __launch_bounds__(kSelectThreads)
select_publish_kernel(const float* __restrict__ logits, const __nv_bfloat16* __restrict__ x,
                      const int32_t* __restrict__ slot_of, int32_t layer,
                      const uint32_t* __restrict__ seq, DecodeExchange* exchange, int32_t T,
                      MoeDecodeWork* __restrict__ work, int8_t* __restrict__ xq,
                      float* __restrict__ xd, uint32_t* __restrict__ counts) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (blockIdx.x > 0) {
    pdl_wait();
    const __nv_bfloat16* const xs = pdl_coherent(x);
    const int t = blockIdx.x - 1;
    for (int b = warp; b < kMoeHiddenBlocks; b += kSelectThreads / 32) {
      const float v = __bfloat162float(xs[std::size_t(t) * kHidden + b * 32 + lane]);
      float d;
      const int8_t q = quantize_q8_lane(v, &d);
      xq[std::size_t(t) * kHidden + b * 32 + lane] = q;
      if (lane == 0) xd[t * kMoeHiddenBlocks + b] = d;
    }
    return;
  }

  __shared__ int s_id[kMaxTokens][kTopK];
  __shared__ float s_w[kMaxTokens][kTopK];
  __shared__ int s_slot[kMaxTokens][kTopK];
  __shared__ int s_pair[kMaxTokens][kTopK];
  __shared__ int s_cnt[kExperts];
  __shared__ int s_scan[32];
  __shared__ int s_pair_begin[kExperts];
  __shared__ int s_slot_of[kExperts];
  __shared__ int s_hot_slot[kMaxPairs];
  __shared__ int s_hot_count[kMaxPairs];

  pdl_wait();
  s_slot_of[threadIdx.x] = slot_of ? pdl_coherent(slot_of)[threadIdx.x] : int(threadIdx.x);
  s_cnt[threadIdx.x] = 0;
  const __nv_bfloat16* const xs = pdl_coherent(x);
  float lv[moe_detail::kPerLane];
  if (warp < T)
    moe_detail::warp_load_logits(pdl_coherent(logits) + std::size_t(warp) * dims::kRouterRows, lv);
  __syncthreads();
  if (warp < T) {
    int ids[kTopK];
    float w[kTopK];
    moe_detail::warp_topk10(lv, ids, w);
    if (lane < kTopK) {
      int id = 0;
      float wt = 0.f;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        if (k == lane) {
          id = ids[k];
          wt = w[k];
        }
      }
      s_id[warp][lane] = id;
      s_w[warp][lane] = wt;
      s_slot[warp][lane] = s_slot_of[id];
    }
  }
  __syncthreads();

  const int p = threadIdx.x;
  const int pt = p / kTopK, pk = p % kTopK;
  const bool valid = p < T * kTopK;
  const int my_slot = valid ? s_slot[pt][pk] : -1;
  if (my_slot >= 0) atomicAdd(&s_cnt[s_id[pt][pk]], 1);
  if (counts && valid) atomicAdd(&counts[s_id[pt][pk]], 1u);
  __syncthreads();

  // Shared arrays are indexed by expert, never physical slot: reserved loans and the swap
  // spare can have sparse indexes above 511. Compact hot experts, then rank only that compact
  // list by physical slot to preserve deterministic decode reduction order.
  int n_pairs = 0, n_items = 0;
  const int c = s_cnt[threadIdx.x];
  moe_detail::block_exclusive_scan(c, s_scan, &n_pairs);
  const int compact = moe_detail::block_exclusive_scan(c > 0 ? 1 : 0, s_scan, &n_items);
  if (c > 0) {
    s_hot_slot[compact] = s_slot_of[threadIdx.x];
    s_hot_count[compact] = c;
  }
  __syncthreads();
  if (c > 0) {
    const int slot = s_slot_of[threadIdx.x];
    int item = 0, pair_begin = 0;
    for (int i = 0; i < n_items; ++i) {
      if (s_hot_slot[i] < slot) {
        ++item;
        pair_begin += s_hot_count[i];
      }
    }
    s_pair_begin[threadIdx.x] = pair_begin;
    work->item_slot[item] = slot;
    work->item_pair_begin[item] = pair_begin;
  }
  if (threadIdx.x == 0) {
    work->item_pair_begin[n_items] = n_pairs;
    work->n_items = n_items;
    work->n_pairs = n_pairs;
    work->tokens = T;
  }
  __syncthreads();

  // Place hot pairs: within a slot, tokens ascending.
  if (valid) {
    int idx = -1;
    if (my_slot >= 0) {
      int rank = 0;
      for (int t2 = 0; t2 < pt; ++t2)
        for (int k2 = 0; k2 < kTopK; ++k2) rank += s_slot[t2][k2] == my_slot ? 1 : 0;
      idx = s_pair_begin[s_id[pt][pk]] + rank;
      work->pair_token[idx] = pt;
      work->pair_weight[idx] = s_w[pt][pk];
    }
    s_pair[pt][pk] = idx;
    work->topk_expert[pt][pk] = s_id[pt][pk];
    work->topk_weight[pt][pk] = s_w[pt][pk];
  }
  __syncthreads();

  MoeDecodePublish* pub = exchange ? &exchange->publish[layer] : nullptr;
  __shared__ int s_cold[kMaxTokens];
  if (threadIdx.x < T) {
    const int t = threadIdx.x;
    int idx[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) idx[k] = s_pair[t][k];
    int nh = 0, nc = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      if (idx[k] >= 0) {
        // Slot order == pair-index order: rank among this token's hot pairs.
        int rank = 0;
#pragma unroll
        for (int k2 = 0; k2 < kTopK; ++k2) rank += (idx[k2] >= 0 && idx[k2] < idx[k]) ? 1 : 0;
        work->token_pairs[t][rank] = idx[k];
        ++nh;
      } else {
        if (pub) {
          pub->cold_expert[t][nc] = s_id[t][k];
          pub->cold_weight[t][nc] = s_w[t][k];
        }
        ++nc;
      }
    }
    work->token_pair_count[t] = nh;
    work->token_cold_count[t] = nc;
    if (pub) pub->cold_count[t] = nc;
    s_cold[t] = nc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n_cold = 0;
    for (int t = 0; t < T; ++t) n_cold += s_cold[t];
    work->n_cold = n_cold;
    if (pub) {
      pub->tokens = T;
      pub->n_cold = n_cold;
    }
  }
  if (pub) {
    // x rows of tokens with cold pairs, 16-byte chunks.
    constexpr int kChunks = kHidden * 2 / 16;
    bool wrote = threadIdx.x < T || threadIdx.x == 0;  // cold lists / header above
    for (int i = threadIdx.x; i < T * kChunks; i += kSelectThreads) {
      const int t = i / kChunks, ch = i % kChunks;
      if (s_cold[t] == 0) continue;
      const int4 v = reinterpret_cast<const int4*>(xs + std::size_t(t) * kHidden)[ch];
      reinterpret_cast<int4*>(pub->x[t])[ch] = v;
      wrote = true;
    }
    // Every thread that wrote host memory makes its writes visible system-wide before the
    // barrier, so the flag store after it needs no further fence (one PCIe flush, not three).
    if (wrote) __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      exchange_store_relaxed_sys(&exchange->control.ready[layer].value, *seq);
    }
  }
}

// ------------------------------------------------------------------------------- gate/up
constexpr int kGuThreads = 128;                     // 4 warps x 2 rows
constexpr int kGuRowsPerCta = kGuThreads / 32 * 2;  // 8
constexpr int kDownThreads = 256;                   // 8 warps x 8 rows
constexpr int kDownRowsPerCta = kDownThreads / 32 * 8;  // 64

// Shared-memory stride for one staged weight row of `bytes` (copied from the 16-byte aligned
// address at or below its start).
constexpr int staged_stride(int bytes) { return (bytes + 15 + 15) / 16 * 16; }

// Warp-cooperative copy of one weight row into shared memory with 16-byte cp.async (L1 bypass);
// bytes at or past `limit` are zero-filled. Returns the row's offset inside `dst`.
template <int Stride>
__device__ __forceinline__ int warp_stage_row(uint8_t* dst, const uint8_t* row, const uint8_t* limit,
                                              int lane) {
  const uint8_t* base =
      reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(row) & ~uintptr_t{15});
  for (int ch = lane; ch < Stride / 16; ch += 32) {
    const uint8_t* src = base + ch * 16;
    const int64_t left = limit - src;
    cp_async_zfill<16, Cache::cg>(dst + ch * 16, src, left >= 16 ? 16 : (left > 0 ? int(left) : 0));
  }
  return int(row - base);
}

// q8 activation sub-block (32 int8 as 8 words) of token t (written by select_publish, so read
// with coherent loads: see pdl.cuh).
__device__ __forceinline__ void load_q8(const int8_t* xq, int t, int sub, int K, int (&u)[8]) {
  const int4* p = reinterpret_cast<const int4*>(xq + std::size_t(t) * K + sub * 32);
  const int4 u0 = p[0], u1 = p[1];
  u[0] = u0.x; u[1] = u0.y; u[2] = u0.z; u[3] = u0.w;
  u[4] = u1.x; u[5] = u1.y; u[6] = u1.z; u[7] = u1.w;
}

// One CTA = (8 intermediate rows, work item); each warp first stages the gate and up rows of its
// two rows into shared memory (coalesced 16-byte copies) while the CTA stages the format's
// codebook, then half-warps own one row: each of 16 lanes covers 5 of the 80 sub-blocks; weights
// are decoded once and applied to every token.
template <MoeFormat F, int TMAX>
__global__ void __launch_bounds__(kGuThreads)
hot_gate_up_kernel(const uint8_t* __restrict__ slots, int64_t blob_stride, int64_t gate_off,
                   int64_t up_off, int64_t down_off, const MoeDecodeWork* __restrict__ work,
                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                   float* __restrict__ h) {
  // Programmatic dependent of select_publish (pdl.cuh): everything it reads comes from routing, so
  // it waits first, and only then lets hot_down launch, because hot_down reads the work list
  // before its own pdl_wait().
  pdl_wait();
  pdl_trigger();
  const MoeDecodeWork* const wk = pdl_coherent(work);
  // The work-list loads are issued together (all indices are in bounds; entries past n_items are
  // stale but unused) so that only one dependent round trip precedes the weight staging.
  const int item = blockIdx.y;
  const int n_items = wk->n_items;
  const int pb = wk->item_pair_begin[item];
  const int n = wk->item_pair_begin[item + 1] - pb;
  const int slot = wk->item_slot[item];
  if (item >= n_items) return;
  constexpr int kRow = kRowBytes<F, kHidden>;
  constexpr int kStride = staged_stride(kRow);
  constexpr int kGrid = moe_iq::kGridBytes<F>;
  __shared__ __align__(16) uint8_t s_raw[kGuThreads / 32][4][kStride];
  __shared__ __align__(16) uint8_t s_grid[kGrid > 0 ? kGrid : 16];
  __shared__ int s_tok[TMAX];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int half = lane >> 4, l16 = lane & 15;
  const uint8_t* blob = slots + int64_t(slot) * blob_stride;
  const uint8_t* limit = blob + down_off;
  const int r0 = blockIdx.x * kGuRowsPerCta + warp * 2;
  int off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int64_t mat = (s & 1) ? up_off : gate_off;
    off[s] = warp_stage_row<kStride>(s_raw[warp][s], blob + mat + int64_t(r0 + (s >> 1)) * kRow,
                                     limit, lane);
  }
  if constexpr (kGrid > 0) {
    const auto* grid = static_cast<const uint8_t*>(moe_iq::grid_table<F>());
    for (int c = threadIdx.x; c < kGrid / 16; c += kGuThreads) cp_async<16>(s_grid + 16 * c, grid + 16 * c);
  }
  cp_commit();
  if (threadIdx.x < TMAX && threadIdx.x < n) s_tok[threadIdx.x] = wk->pair_token[pb + threadIdx.x];
  cp_wait<0>();
  __syncthreads();

  const int r = r0 + half;
  const uint8_t* grow = s_raw[warp][2 * half] + off[2 * half];
  const uint8_t* urow = s_raw[warp][2 * half + 1] + off[2 * half + 1];
  const int8_t* const xqs = pdl_coherent(xq);
  const float* const xds = pdl_coherent(xd);
  float ag[TMAX], au[TMAX];
#pragma unroll
  for (int j = 0; j < TMAX; ++j) ag[j] = au[j] = 0.f;
#pragma unroll
  for (int i = 0; i < kMoeHiddenBlocks / 16; ++i) {
    const int sub = l16 + 16 * i;
    Sub32 sg, su;
    decode_sub32<F>(grow, sub, sg, s_grid);
    decode_sub32<F>(urow, sub, su, s_grid);
#pragma unroll
    for (int j = 0; j < TMAX; ++j) {
      if (j < n) {
        const int t = s_tok[j];
        int u[8];
        load_q8(xqs, t, sub, kHidden, u);
        const float dx = xds[t * kMoeHiddenBlocks + sub];
        ag[j] += sg.d * dx * float(sub32_dot(sg, u));
        au[j] += su.d * dx * float(sub32_dot(su, u));
      }
    }
  }
#pragma unroll
  for (int j = 0; j < TMAX; ++j) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
      ag[j] += __shfl_xor_sync(0xffffffffu, ag[j], o);
      au[j] += __shfl_xor_sync(0xffffffffu, au[j], o);
    }
  }
  if (l16 == 0) {
#pragma unroll
    for (int j = 0; j < TMAX; ++j)
      if (j < n) h[std::size_t(pb + j) * kInter + r] = silu(ag[j]) * au[j];
  }
}

// --------------------------------------------------------------------------------- down
// One CTA = (64 output rows, work item). Each warp stages its 8 consecutive weight rows (one
// contiguous byte range) into shared memory while h of the item's pairs is quantized to q8;
// quads of lanes own one row (5 of the 20 sub-blocks each). Launched as a programmatic dependent
// of hot_gate_up: the weight staging (independent of h) starts while gate/up finishes, and
// griddepcontrol.wait orders the h reads after it.
template <MoeFormat F, int TMAX>
__global__ void __launch_bounds__(kDownThreads)
hot_down_kernel(const uint8_t* __restrict__ slots, int64_t blob_stride, int64_t down_off,
                const MoeDecodeWork* __restrict__ work, const float* __restrict__ h,
                float* __restrict__ partial) {
  const int item = blockIdx.y;
  // Read before pdl_wait(): hot_gate_up triggers only after select_publish completed (pdl.cuh).
  const int n_items = pdl_load_settled(&work->n_items);
  const int pb = pdl_load_settled(&work->item_pair_begin[item]);
  const int n = pdl_load_settled(&work->item_pair_begin[item + 1]) - pb;
  const int slot = pdl_load_settled(&work->item_slot[item]);
  if (item >= n_items) return;
  constexpr int kRow = kRowBytes<F, kInter>;
  constexpr int kStride = staged_stride(8 * kRow);
  __shared__ __align__(16) uint8_t s_raw[kDownThreads / 32][kStride];
  __shared__ int4 s_q[TMAX][kMoeInterBlocks][2];
  __shared__ float s_d[TMAX][kMoeInterBlocks];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint8_t* mat = slots + int64_t(slot) * blob_stride + down_off;
  const int r0 = blockIdx.x * kDownRowsPerCta + warp * 8;
  const int off = warp_stage_row<kStride>(s_raw[warp], mat + int64_t(r0) * kRow,
                                          mat + int64_t(kHidden) * kRow, lane);
  cp_commit();
  pdl_wait();
  const float* const hs = pdl_coherent(h);  // hot_gate_up's output (pdl.cuh)
  for (int task = warp; task < n * kMoeInterBlocks; task += kDownThreads / 32) {
    const int j = task / kMoeInterBlocks, b = task % kMoeInterBlocks;
    const float v = hs[std::size_t(pb + j) * kInter + b * 32 + lane];
    float d;
    const int8_t q = quantize_q8_lane(v, &d);
    reinterpret_cast<int8_t*>(&s_q[j][b][0])[lane] = q;
    if (lane == 0) s_d[j][b] = d;
  }
  cp_wait<0>();
  __syncthreads();

  const int g4 = lane >> 2, l4 = lane & 3;
  const int r = r0 + g4;
  const uint8_t* row = s_raw[warp] + off + g4 * kRow;
  float acc[TMAX];
#pragma unroll
  for (int j = 0; j < TMAX; ++j) acc[j] = 0.f;
#pragma unroll
  for (int i = 0; i < kMoeInterBlocks / 4; ++i) {
    const int sub = l4 + 4 * i;
    Sub32 s;
    decode_sub32<F>(row, sub, s, moe_iq::grid_table<F>());
#pragma unroll
    for (int j = 0; j < TMAX; ++j) {
      if (j < n) {
        int u[8];
        const int4 u0 = s_q[j][sub][0], u1 = s_q[j][sub][1];
        u[0] = u0.x; u[1] = u0.y; u[2] = u0.z; u[3] = u0.w;
        u[4] = u1.x; u[5] = u1.y; u[6] = u1.z; u[7] = u1.w;
        acc[j] += s.d * s_d[j][sub] * float(sub32_dot(s, u));
      }
    }
  }
#pragma unroll
  for (int j = 0; j < TMAX; ++j) {
    acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
    acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
  }
  if (l4 == 0) {
#pragma unroll
    for (int j = 0; j < TMAX; ++j)
      if (j < n) partial[std::size_t(pb + j) * kHidden + r] = work->pair_weight[pb + j] * acc[j];
  }
}

// ------------------------------------------------------------------------------- combine
constexpr int kCombineThreads = 256;
constexpr int kVec = kHidden / 4;  // 640 float4 per token

// Programmatic dependent of hot_down (and the shared expert's down projection): it waits first,
// since it only reads their outputs. pdl_trigger() once the CPU tier is in: the next layer's GR
// read (a programmatic dependent) streams its weights in while this kernel adds the results.
// Triggering before the CPU wait would let whole chains of programmatic dependents sit resident
// through it and starve the L2 prefetch. Only CTA (0, 0) polls done[layer] in host memory; the
// others wait on its device-memory relay (exchange_wait_sys_relayed).
__global__ void __launch_bounds__(kCombineThreads)
combine_kernel(const MoeDecodeWork* __restrict__ work, const float* __restrict__ partial,
               const float* __restrict__ y_shared, DecodeExchange* exchange, int32_t layer,
               const uint32_t* __restrict__ seq, const float* __restrict__ inj,
               float* __restrict__ X, float* __restrict__ y_out, uint32_t* relay,
               long long timeout_cycles) {
  pdl_wait();
  const MoeDecodeWork* const wk = pdl_coherent(work);
  const int t = blockIdx.y;
  const int i = blockIdx.x * kCombineThreads + threadIdx.x;
  __shared__ int s_cpu;
  if (threadIdx.x == 0) {
    int use_cpu = 0;
    if (exchange && wk->n_cold > 0) {
      if (exchange_wait_sys_relayed(&exchange->control.done[layer].value, *seq,
                                    blockIdx.x == 0 && blockIdx.y == 0, relay + 2 * layer,
                                    timeout_cycles)) {
        use_cpu = 1;
      } else {
        exchange_store_release_sys(&exchange->control.error.value,
                                   kExchangeErrorMoeTimeout | uint32_t(layer + 1));
      }
    }
    s_cpu = use_cpu && wk->token_cold_count[t] > 0;
  }
  __syncthreads();
  pdl_trigger();
  if (i >= kVec) return;

  const float* const ps = pdl_coherent(partial);
  float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
  const int np = wk->token_pair_count[t];
  for (int j = 0; j < np; ++j) {
    const float4 v = reinterpret_cast<const float4*>(ps + std::size_t(wk->token_pairs[t][j]) * kHidden)[i];
    y.x += v.x; y.y += v.y; y.z += v.z; y.w += v.w;
  }
  const float4 s = reinterpret_cast<const float4*>(pdl_coherent(y_shared) + std::size_t(t) * kHidden)[i];
  y.x += s.x; y.y += s.y; y.z += s.z; y.w += s.w;
  if (s_cpu) {
    const float4 c = exchange_load_relaxed_sys(
        reinterpret_cast<const float4*>(exchange->result[layer].y[t]) + i);
    y.x += c.x; y.y += c.y; y.z += c.z; y.w += c.w;
  }
  if (y_out) reinterpret_cast<float4*>(y_out + std::size_t(t) * kHidden)[i] = y;
#pragma unroll
  for (int st = 0; st < dims::kStreams; ++st) {
    const float g = pdl_coherent(inj)[t * dims::kStreams + st];
    float4* xp = reinterpret_cast<float4*>(X + (std::size_t(t) * dims::kStreams + st) * kHidden) + i;
    float4 xv = *xp;
    xv.x += g * y.x; xv.y += g * y.y; xv.z += g * y.z; xv.w += g * y.w;
    *xp = xv;
  }
}

// Abliterated variant: identical, except that one CTA owns the whole token so the routed-expert
// sum can be transformed along the refusal direction before it reaches the residual stream:
// y_routed -= scale * r * (r . y_routed). The routed experts keep their packed bytes; the edit is
// exact on their sum because it is linear in each expert's output. The shared expert is NOT
// transformed here: its weights are already the edited ones, and for scale != 1 the transform is
// not idempotent, so applying it twice would be wrong.
__global__ void __launch_bounds__(kVec)
combine_ablate_kernel(const MoeDecodeWork* __restrict__ work, const float* __restrict__ partial,
                      const float* __restrict__ y_shared, DecodeExchange* exchange, int32_t layer,
                      const uint32_t* __restrict__ seq, const float* __restrict__ inj,
                      float* __restrict__ X, float* __restrict__ y_out, uint32_t* relay,
                      const float* __restrict__ direction, float scale, long long timeout_cycles) {
  pdl_wait();
  const MoeDecodeWork* const wk = pdl_coherent(work);
  const int t = blockIdx.y;
  const int i = threadIdx.x;  // blockDim.x == kVec: one float4 per thread
  __shared__ int s_cpu;
  if (threadIdx.x == 0) {
    int use_cpu = 0;
    if (exchange && wk->n_cold > 0) {
      if (exchange_wait_sys_relayed(&exchange->control.done[layer].value, *seq, blockIdx.y == 0,
                                    relay + 2 * layer, timeout_cycles)) {
        use_cpu = 1;
      } else {
        exchange_store_release_sys(&exchange->control.error.value,
                                   kExchangeErrorMoeTimeout | uint32_t(layer + 1));
      }
    }
    s_cpu = use_cpu && wk->token_cold_count[t] > 0;
  }
  __syncthreads();
  pdl_trigger();

  const float* const ps = pdl_coherent(partial);
  float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
  const int np = wk->token_pair_count[t];
  for (int j = 0; j < np; ++j) {
    const float4 v = reinterpret_cast<const float4*>(ps + std::size_t(wk->token_pairs[t][j]) * kHidden)[i];
    y.x += v.x; y.y += v.y; y.z += v.z; y.w += v.w;
  }
  if (s_cpu) {
    const float4 c = exchange_load_relaxed_sys(
        reinterpret_cast<const float4*>(exchange->result[layer].y[t]) + i);
    y.x += c.x; y.y += c.y; y.z += c.z; y.w += c.w;
  }
  // y holds the routed sum (GPU-resident + CPU-tier experts).
  __shared__ float s_red[32];
  const float4 r = reinterpret_cast<const float4*>(direction)[i];
  const float dot = moe_detail::block_reduce_sum(y.x * r.x + y.y * r.y + y.z * r.z + y.w * r.w, s_red);
  const float k = -scale * dot;
  y.x += k * r.x; y.y += k * r.y; y.z += k * r.z; y.w += k * r.w;
  const float4 s = reinterpret_cast<const float4*>(pdl_coherent(y_shared) + std::size_t(t) * kHidden)[i];
  y.x += s.x; y.y += s.y; y.z += s.z; y.w += s.w;

  if (y_out) reinterpret_cast<float4*>(y_out + std::size_t(t) * kHidden)[i] = y;
#pragma unroll
  for (int st = 0; st < dims::kStreams; ++st) {
    const float g = pdl_coherent(inj)[t * dims::kStreams + st];
    float4* xp = reinterpret_cast<float4*>(X + (std::size_t(t) * dims::kStreams + st) * kHidden) + i;
    float4 xv = *xp;
    xv.x += g * y.x; xv.y += g * y.y; xv.z += g * y.z; xv.w += g * y.w;
    *xp = xv;
  }
}

// ------------------------------------------------------------------------------ dispatch
int tmax_for(int tokens) {
  if (tokens <= 1) return 1;
  if (tokens <= 2) return 2;
  if (tokens <= 4) return 4;
  if (tokens <= 8) return 8;
  return 16;
}

int max_items(int tokens) { return std::min(tokens * kTopK, kExperts); }

template <MoeFormat F, int TMAX>
cudaError_t launch_gate_up(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                           cudaStream_t stream) {
  const dim3 grid(kInter / kGuRowsPerCta, max_items(tokens));
  return pdl_launch(hot_gate_up_kernel<F, TMAX>, grid, dim3(kGuThreads), 0, stream, L.slots,
                    L.blob_stride, L.gate_offset, L.up_offset, L.down_offset,
                    static_cast<const MoeDecodeWork*>(ws.work), static_cast<const int8_t*>(ws.xq),
                    static_cast<const float*>(ws.xd), ws.h);
}

template <MoeFormat F>
cudaError_t gate_up_t(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                      cudaStream_t stream) {
  switch (tmax_for(tokens)) {
    case 1: return launch_gate_up<F, 1>(L, tokens, ws, stream);
    case 2: return launch_gate_up<F, 2>(L, tokens, ws, stream);
    case 4: return launch_gate_up<F, 4>(L, tokens, ws, stream);
    case 8: return launch_gate_up<F, 8>(L, tokens, ws, stream);
    default: return launch_gate_up<F, 16>(L, tokens, ws, stream);
  }
}

template <MoeFormat F, int TMAX>
cudaError_t launch_down(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                        cudaStream_t stream) {
  return pdl_launch(hot_down_kernel<F, TMAX>, dim3(kHidden / kDownRowsPerCta, max_items(tokens)),
                    dim3(kDownThreads), 0, stream, L.slots, L.blob_stride, L.down_offset,
                    static_cast<const MoeDecodeWork*>(ws.work), static_cast<const float*>(ws.h),
                    ws.partial);
}

template <MoeFormat F>
cudaError_t down_t(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                   cudaStream_t stream) {
  switch (tmax_for(tokens)) {
    case 1: return launch_down<F, 1>(L, tokens, ws, stream);
    case 2: return launch_down<F, 2>(L, tokens, ws, stream);
    case 4: return launch_down<F, 4>(L, tokens, ws, stream);
    case 8: return launch_down<F, 8>(L, tokens, ws, stream);
    default: return launch_down<F, 16>(L, tokens, ws, stream);
  }
}

bool valid_tokens(int tokens) { return tokens >= 1 && tokens <= kMaxTokens; }

}  // namespace

std::size_t moe_decode_workspace_bytes() { return decode_layout().total; }

MoeDecodeWorkspace make_moe_decode_workspace(void* base) {
  constexpr DecodeLayout l = decode_layout();
  auto* b = static_cast<uint8_t*>(base);
  MoeDecodeWorkspace ws;
  ws.work = reinterpret_cast<MoeDecodeWork*>(b + l.work);
  ws.xq = reinterpret_cast<int8_t*>(b + l.xq);
  ws.xd = reinterpret_cast<float*>(b + l.xd);
  ws.h = reinterpret_cast<float*>(b + l.h);
  ws.partial = reinterpret_cast<float*>(b + l.partial);
  ws.relay = reinterpret_cast<uint32_t*>(b + l.relay);
  return ws;
}

cudaError_t moe_select_publish(const float* router_logits, const __nv_bfloat16* x,
                               const int32_t* slot_of, int32_t layer, const uint32_t* seq,
                               DecodeExchange* exchange, int32_t tokens,
                               const MoeDecodeWorkspace& ws, cudaStream_t stream,
                               uint32_t* routing_counts) {
  if (!valid_tokens(tokens) || (exchange && (layer < 0 || layer >= kExchangeLayers || !seq)))
    return cudaErrorInvalidValue;
  return pdl_launch(select_publish_kernel, dim3(1 + tokens), dim3(kSelectThreads), 0, stream,
                    router_logits, x, slot_of, layer, seq, exchange, tokens, ws.work, ws.xq, ws.xd,
                    routing_counts);
}

cudaError_t moe_hot_gate_up(const MoeExpertLayout& layout, int32_t tokens,
                            const MoeDecodeWorkspace& ws, cudaStream_t stream) {
  if (!valid_tokens(tokens)) return cudaErrorInvalidValue;
  switch (layout.gate_up_format) {
    case MoeFormat::IQ2_S: return gate_up_t<MoeFormat::IQ2_S>(layout, tokens, ws, stream);
    case MoeFormat::IQ3_XXS: return gate_up_t<MoeFormat::IQ3_XXS>(layout, tokens, ws, stream);
    case MoeFormat::IQ3_S: return gate_up_t<MoeFormat::IQ3_S>(layout, tokens, ws, stream);
    case MoeFormat::IQ4_XS: return gate_up_t<MoeFormat::IQ4_XS>(layout, tokens, ws, stream);
    case MoeFormat::IQ4_NL: return gate_up_t<MoeFormat::IQ4_NL>(layout, tokens, ws, stream);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t moe_hot_down(const MoeExpertLayout& layout, int32_t tokens,
                         const MoeDecodeWorkspace& ws, cudaStream_t stream) {
  if (!valid_tokens(tokens)) return cudaErrorInvalidValue;
  switch (layout.down_format) {
    case MoeFormat::IQ4_NL: return down_t<MoeFormat::IQ4_NL>(layout, tokens, ws, stream);
    case MoeFormat::Q2_0: return down_t<MoeFormat::Q2_0>(layout, tokens, ws, stream);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t moe_combine(const MoeDecodeWorkspace& ws, const float* y_shared,
                        DecodeExchange* exchange, int32_t layer, const uint32_t* seq,
                        const float* inj, float* X, float* y_out, int32_t tokens,
                        long long timeout_cycles, const float* ablate_direction,
                        float ablate_scale, cudaStream_t stream) {
  if (!valid_tokens(tokens) || (exchange && (layer < 0 || layer >= kExchangeLayers || !seq)))
    return cudaErrorInvalidValue;
  if (ablate_direction) {
    const dim3 grid(1, tokens);
    return pdl_launch(combine_ablate_kernel, grid, dim3(kVec), 0, stream,
                      static_cast<const MoeDecodeWork*>(ws.work),
                      static_cast<const float*>(ws.partial), y_shared, exchange, layer, seq, inj, X,
                      y_out, ws.relay, ablate_direction, ablate_scale, timeout_cycles);
  }
  const dim3 grid((kVec + kCombineThreads - 1) / kCombineThreads, tokens);
  return pdl_launch(combine_kernel, grid, dim3(kCombineThreads), 0, stream,
                    static_cast<const MoeDecodeWork*>(ws.work), static_cast<const float*>(ws.partial),
                    y_shared, exchange, layer, seq, inj, X, y_out, ws.relay, timeout_cycles);
}

}  // namespace fngine::kernels
