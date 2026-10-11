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
constexpr int kGuThreads = 128;                           // 4 warps x 2 rows
constexpr int kGuRowsPerTile = kGuThreads / 32 * 2;       // 8
constexpr int kGuTilesPerItem = kInter / kGuRowsPerTile;  // 80
// A gate/up row is staged in kGuSlices K slices: slice i is row bytes [i * Row / 5, (i + 1) *
// Row / 5), whole quantization blocks holding sub-blocks [16 i, 16 i + 16), which is what a lane
// decodes in iteration i of the tile loop.
constexpr int kGuSlices = 5;
static_assert(kMoeHiddenBlocks == 16 * kGuSlices, "one slice per 16 sub-blocks of a hidden row");

constexpr std::size_t align16(std::size_t v) { return (v + 15) & ~std::size_t{15}; }

// Shared-memory stride for one staged weight row of `bytes` (copied from the 16-byte aligned
// address at or below its start).
constexpr int staged_stride(int bytes) { return (bytes + 15 + 15) / 16 * 16; }

// Warp-cooperative copy of one weight row into shared memory with 16-byte cp.async (L1 bypass);
// bytes at or past `limit` are zero-filled. The row starts staged_offset(row) bytes into `dst`.
template <int Stride>
__device__ __forceinline__ void warp_stage_row(uint8_t* dst, const uint8_t* row, const uint8_t* limit,
                                               int lane) {
  const uint8_t* base =
      reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(row) & ~uintptr_t{15});
  for (int ch = lane; ch < Stride / 16; ch += 32) {
    const uint8_t* src = base + ch * 16;
    const int64_t left = limit - src;
    cp_async_zfill<16, Cache::cg>(dst + ch * 16, src, left >= 16 ? 16 : (left > 0 ? int(left) : 0));
  }
}

__device__ __forceinline__ int staged_offset(const uint8_t* row) {
  return int(reinterpret_cast<uintptr_t>(row) & 15);
}

// Warp-cooperative cp.async of slice i of one staged row (the buffer layout of warp_stage_row):
// the 16-byte chunks whose first row byte lies in the slice (chunk 0 always joins slice 0), so
// slices 0..i of a row are complete once their copies are. Whatever the row's offset, slice i's
// chunks lie below staged byte (i + 1) * S + 31 (S = Row / 5) and the chunks holding bytes of slice
// i + 2 start at or above (i + 2) * S - 15: hot_gate_up refills slice i of the next tile as soon as
// slice i + 1 of the current one is decoded.
template <int Row>
__device__ __forceinline__ void warp_stage_slice(uint8_t* dst, const uint8_t* row, const uint8_t* limit,
                                                 int i, int lane) {
  constexpr int kSliceBytes = Row / kGuSlices;
  static_assert(Row % kGuSlices == 0, "slices hold whole quantization blocks");
  static_assert(kSliceBytes >= 46, "a refilled slice i must end before the chunks of slice i + 2");
  const int off = staged_offset(row);
  const uint8_t* base = row - off;
  const int lo = i == 0 ? 0 : (i * kSliceBytes + off + 15) / 16;
  const int hi = ((i + 1) * kSliceBytes + off + 15) / 16;
  for (int ch = lo + lane; ch < hi; ch += 32) {
    const uint8_t* src = base + ch * 16;
    const int64_t left = limit - src;
    cp_async_zfill<16, Cache::cg>(dst + ch * 16, src, left >= 16 ? 16 : (left > 0 ? int(left) : 0));
  }
}

// q8 activation sub-block (32 int8 as 8 words) of token t (written by select_publish, so read
// with coherent loads: see pdl.cuh).
__device__ __forceinline__ void load_q8(const int8_t* xq, int t, int sub, int K, int (&u)[8]) {
  const int4* p = reinterpret_cast<const int4*>(xq + std::size_t(t) * K + sub * 32);
  const int4 u0 = p[0], u1 = p[1];
  u[0] = u0.x; u[1] = u0.y; u[2] = u0.z; u[3] = u0.w;
  u[4] = u1.x; u[5] = u1.y; u[6] = u1.z; u[7] = u1.w;
}

// Persistent grid (as many CTAs as fit on every SM) over tiles (work item, 8 intermediate rows),
// tile index item * 80 + row group; CTA c takes tiles c, c + G, ... Each warp stages the gate and
// up rows of its two rows (coalesced 16-byte copies; the CTA stages the format's codebook once),
// then half-warps own one row: each of 16 lanes covers 5 of the 80 sub-blocks, one per K slice;
// weights are decoded once and applied to every token of the item. A single tile buffer streams:
// once slice i of the current tile is decoded, slice i - 1 of the next tile is requested in its
// place, so DRAM stays busy through the decode (which needs several CTAs per SM, too many for a
// second buffer) and through what used to be the gap between waves. Per lane the FP32 order is
// that of one tile per CTA.
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
  constexpr int kItems = TMAX * kTopK;  // work items and pairs of a step
  constexpr int kRow = kRowBytes<F, kHidden>;
  constexpr int kStride = staged_stride(kRow);
  constexpr int kGrid = moe_iq::kGridBytes<F>;
  __shared__ __align__(16) uint8_t s_raw[kGuThreads / 32][4][kStride];
  __shared__ __align__(16) uint8_t s_grid[kGrid > 0 ? kGrid : 16];
  __shared__ int s_slot[kItems];       // item -> slot
  __shared__ int s_tok[kItems];        // pair -> token
  __shared__ int s_begin[kItems + 1];  // item -> first pair

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int half = lane >> 4, l16 = lane & 15;
  if constexpr (kGrid > 0) {
    const auto* grid = static_cast<const uint8_t*>(moe_iq::grid_table<F>());
    for (int c = threadIdx.x; c < kGrid / 16; c += kGuThreads) cp_async<16>(s_grid + 16 * c, grid + 16 * c);
  }
  // One dependent round trip (the work list; entries past n_items are stale but in bounds)
  // precedes the weight staging.
  const MoeDecodeWork* const wk = pdl_coherent(work);
  const int n_tiles = wk->n_items * kGuTilesPerItem;
  for (int i = threadIdx.x; i <= kItems; i += kGuThreads) {
    if (i < kItems) {
      s_slot[i] = wk->item_slot[i];
      s_tok[i] = wk->pair_token[i];
    }
    s_begin[i] = wk->item_pair_begin[i];
  }
  __syncthreads();

  const int stride = int(gridDim.x);
  auto blob_of = [&](int t) { return slots + int64_t(s_slot[t / kGuTilesPerItem]) * blob_stride; };
  // Staged row s of tile t: gate (s even) or up (s odd) row r0 + s / 2 of the warp.
  auto row_of = [&](int t, int s) {
    const int r = (t % kGuTilesPerItem) * kGuRowsPerTile + warp * 2 + (s >> 1);
    return blob_of(t) + ((s & 1) ? up_off : gate_off) + int64_t(r) * kRow;
  };
  // Requests slice i of the warp's four rows of tile t; always commits one group, so that a tile's
  // slices are groups in order.
  auto stage_slice = [&](int t, int i) {
    if (t < n_tiles) {
#pragma unroll
      for (int s = 0; s < 4; ++s)
        warp_stage_slice<kRow>(s_raw[warp][s], row_of(t, s), blob_of(t) + down_off, i, lane);
    }
    cp_commit();
  };
  int t = blockIdx.x;
#pragma unroll
  for (int i = 0; i < kGuSlices; ++i) stage_slice(t, i);  // group 0 also holds the codebook

  const int8_t* const xqs = pdl_coherent(xq);
  const float* const xds = pdl_coherent(xd);
  for (bool first = true; t < n_tiles; first = false, t += stride) {
    const int item = t / kGuTilesPerItem;
    const int pb = s_begin[item];
    const int n = s_begin[item + 1] - pb;
    const int r = (t % kGuTilesPerItem) * kGuRowsPerTile + warp * 2 + half;
    const uint8_t* grow = s_raw[warp][2 * half] + staged_offset(row_of(t, 2 * half));
    const uint8_t* urow = s_raw[warp][2 * half + 1] + staged_offset(row_of(t, 2 * half + 1));
    float ag[TMAX], au[TMAX];
#pragma unroll
    for (int j = 0; j < TMAX; ++j) ag[j] = au[j] = 0.f;
#pragma unroll
    for (int i = 0; i < kGuSlices; ++i) {
      // Groups committed after slice i of this tile: its later slices, plus slices 0..i - 2 of the
      // next tile requested so far.
      if (i == 0) cp_wait<kGuSlices - 1>();
      else cp_wait<kGuSlices - 2>();
      if (first && i == 0) __syncthreads();  // the codebook was staged by the whole CTA
      else __syncwarp();                     // otherwise a warp reads only the rows it staged
      const int sub = l16 + 16 * i;
      Sub32 sg, su;
      decode_sub32<F>(grow, sub, sg, s_grid);
      decode_sub32<F>(urow, sub, su, s_grid);
#pragma unroll
      for (int j = 0; j < TMAX; ++j) {
        if (j < n) {
          const int tok = s_tok[pb + j];
          int u[8];
          load_q8(xqs, tok, sub, kHidden, u);
          const float dx = xds[tok * kMoeHiddenBlocks + sub];
          ag[j] += sg.d * dx * float(sub32_dot(sg, u));
          au[j] += su.d * dx * float(sub32_dot(su, u));
        }
      }
      if (i > 0) {
        __syncwarp();  // every lane is done with slices i - 1 and i of this tile
        stage_slice(t + stride, i - 1);
      }
    }
    stage_slice(t + stride, kGuSlices - 1);
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
}

// --------------------------------------------------------------------------------- down
// hot_down runs a persistent grid (as many CTAs as fit on every SM) over fixed-size tiles and
// keeps kStages tiles staged per CTA (cp.async double buffering): the next tile's weights stream
// in while the current one is decoded, so DRAM stays busy instead of idling through every wave's
// decode and the next wave's launch. A tile's arithmetic, and therefore every FP32 summation
// order, is the same as with one tile per CTA.
constexpr int kStages = 2;  // 3 and 4 measured no faster
constexpr int kDownThreads = 128;                              // 4 warps x 8 rows
constexpr int kDownRowsPerTile = kDownThreads / 32 * 8;        // 32
constexpr int kDownTilesPerItem = kHidden / kDownRowsPerTile;  // 80

// Dynamic shared memory of hot_down_kernel<F, TMAX>: kStages x [4 warps] staged 8-row ranges, the
// q8 h blocks of one item's pairs, and the work list's item slots and pair ranges.
template <MoeFormat F, int TMAX>
struct DownSmem {
  static constexpr int kItems = TMAX * kTopK;
  static constexpr int kStride = staged_stride(8 * kRowBytes<F, kInter>);
  static constexpr std::size_t kRaw = std::size_t(kStages) * (kDownThreads / 32) * kStride;
  static constexpr std::size_t kQ = std::size_t(TMAX) * kMoeInterBlocks * 32;  // int8 blocks
  static constexpr std::size_t kD = align16(sizeof(float) * TMAX * kMoeInterBlocks);
  static constexpr std::size_t kBytes = kRaw + kQ + kD + align16(sizeof(int) * (2 * kItems + 1));
};

// Tile = (work item, 32 output rows), tile index item * 80 + row group; CTA c takes a contiguous
// tile range, so it meets one or two items and quantizes each one's h to q8 once. Each warp stages
// its 8 consecutive weight rows (one contiguous byte range); quads of lanes own one row (5 of the
// 20 sub-blocks each). Programmatic dependent of hot_gate_up: the first tiles' weight staging
// (independent of h) starts while gate/up finishes, and griddepcontrol.wait orders the h reads
// after it. It triggers first thing: the combine's CPU-tier half runs beside it.
template <MoeFormat F, int TMAX>
__global__ void __launch_bounds__(kDownThreads)
hot_down_kernel(const uint8_t* __restrict__ slots, int64_t blob_stride, int64_t down_off,
                const MoeDecodeWork* __restrict__ work, const float* __restrict__ h,
                float* __restrict__ partial) {
  pdl_trigger();
  using S = DownSmem<F, TMAX>;
  constexpr int kRow = kRowBytes<F, kInter>;
  extern __shared__ __align__(16) uint8_t smem[];
  int8_t* const s_q = reinterpret_cast<int8_t*>(smem + S::kRaw);       // [TMAX][20][32]
  float* const s_d = reinterpret_cast<float*>(smem + S::kRaw + S::kQ);  // [TMAX][20]
  int* const s_slot = reinterpret_cast<int*>(smem + S::kRaw + S::kQ + S::kD);
  int* const s_begin = s_slot + S::kItems;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // Read before pdl_wait(): hot_gate_up triggers only after select_publish completed (pdl.cuh).
  const int n_tiles = pdl_load_settled(&work->n_items) * kDownTilesPerItem;
  for (int i = threadIdx.x; i <= S::kItems; i += kDownThreads) {
    if (i < S::kItems) s_slot[i] = pdl_load_settled(&work->item_slot[i]);
    s_begin[i] = pdl_load_settled(&work->item_pair_begin[i]);
  }
  __syncthreads();
  const int t_begin = int(int64_t(blockIdx.x) * n_tiles / int(gridDim.x));
  const int t_end = int(int64_t(blockIdx.x + 1) * n_tiles / int(gridDim.x));

  auto mat_of = [&](int item) { return slots + int64_t(s_slot[item]) * blob_stride + down_off; };
  auto first_row = [&](int t) { return (t % kDownTilesPerItem) * kDownRowsPerTile + warp * 8; };
  auto stage_tile = [&](int t, int stage) {
    if (t < t_end) {
      const uint8_t* mat = mat_of(t / kDownTilesPerItem);
      warp_stage_row<S::kStride>(smem + (std::size_t(stage) * (kDownThreads / 32) + warp) * S::kStride,
                                 mat + int64_t(first_row(t)) * kRow, mat + int64_t(kHidden) * kRow, lane);
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < kStages; ++i) stage_tile(t_begin + i, i);
  pdl_wait();

  const float* const hs = pdl_coherent(h);  // hot_gate_up's output (pdl.cuh)
  constexpr int kWarps = kDownThreads / 32;
  constexpr int kBatch = std::min(8, (TMAX * kMoeInterBlocks + kWarps - 1) / kWarps);
  const int g4 = lane >> 2, l4 = lane & 3;
  int q_item = -1;  // item whose q8 h blocks are in s_q / s_d
  for (int t = t_begin, k = 0; t < t_end; ++t, ++k) {
    const int stage = k % kStages;  // the loop refills the stage it just consumed
    const int item = t / kDownTilesPerItem;
    const int pb = s_begin[item];
    const int n = s_begin[item + 1] - pb;
    if (item != q_item) {
      // q8 blocks of the item's h rows: block task = j * kMoeInterBlocks + b of pair pb + j. A warp
      // issues the loads of a whole batch before quantizing it.
      if (q_item >= 0) __syncthreads();  // every warp is done with the previous item's blocks
      const float* const hp = hs + std::size_t(pb) * kInter + lane;
      const int n_tasks = n * kMoeInterBlocks;
      for (int t0 = warp; t0 < n_tasks; t0 += kWarps * kBatch) {
        float v[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
          const int task = t0 + b * kWarps;
          v[b] = task < n_tasks ? hp[task * 32] : 0.f;
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
          const int task = t0 + b * kWarps;
          if (task < n_tasks) {  // warp-uniform: quantize_q8_lane reduces over the warp
            float d;
            s_q[task * 32 + lane] = quantize_q8_lane(v[b], &d);
            if (lane == 0) s_d[task] = d;
          }
        }
      }
      q_item = item;
      __syncthreads();
    }
    cp_wait<kStages - 1>();
    __syncwarp();  // a warp reads only the rows it staged

    const int r = first_row(t) + g4;
    const uint8_t* row = smem + (std::size_t(stage) * (kDownThreads / 32) + warp) * S::kStride +
                         staged_offset(mat_of(item) + int64_t(first_row(t)) * kRow) + g4 * kRow;
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
          const int4* q = reinterpret_cast<const int4*>(s_q + (j * kMoeInterBlocks + sub) * 32);
          const int4 u0 = q[0], u1 = q[1];
          const int u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
          acc[j] += s.d * s_d[j * kMoeInterBlocks + sub] * float(sub32_dot(s, u));
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
    __syncwarp();  // every lane is done with this stage before it is refilled
    stage_tile(t + kStages, stage);
  }
}

// ------------------------------------------------------------------------------- combine
constexpr int kCombineThreads = 256;
constexpr int kVec = kHidden / 4;  // 640 float4 per token

// CPU-tier half of a combine, run before pdl_wait(): done[layer] (not stream order) orders the CPU
// result, and the work list and seq settled before hot_down could launch (pdl.cuh), so with
// hot_down triggering first thing the wait and the PCIe read of y overlap the hot experts. Only
// the poller (one thread per grid) polls done[layer] in host memory; the others wait on its
// device-memory relay (exchange_wait_sys_relayed). `use` tells whether token t has CPU-tier pairs;
// `y` is then this thread's float4 of their sum.
struct CpuPart {
  bool use;
  float4 y;
};
__device__ __forceinline__ CpuPart combine_cpu_part(const MoeDecodeWork* work, DecodeExchange* exchange,
                                                    int32_t layer, const uint32_t* seq, uint32_t* relay,
                                                    long long timeout_cycles, int t, int i, bool poller) {
  __shared__ int s_cpu;
  if (threadIdx.x == 0) {
    int use_cpu = 0;
    if (exchange && pdl_load_settled(&work->n_cold) > 0) {
      const uint32_t expected = uint32_t(pdl_load_settled(reinterpret_cast<const int*>(seq)));
      if (exchange_wait_sys_relayed(&exchange->control.done[layer].value, expected, poller,
                                    relay + 2 * layer, timeout_cycles)) {
        use_cpu = 1;
      } else {
        exchange_store_release_sys(&exchange->control.error.value,
                                   kExchangeErrorMoeTimeout | uint32_t(layer + 1));
      }
    }
    s_cpu = use_cpu && pdl_load_settled(&work->token_cold_count[t]) > 0;
  }
  __syncthreads();
  CpuPart c{s_cpu != 0, make_float4(0.f, 0.f, 0.f, 0.f)};
  if (c.use && i < kVec)
    c.y = exchange_load_relaxed_sys(reinterpret_cast<const float4*>(exchange->result[layer].y[t]) + i);
  return c;
}

// Programmatic dependent of hot_down (and, through a full edge, of the shared expert's down
// projection): the CPU-tier half runs first (combine_cpu_part), then it waits for their outputs.
// pdl_trigger() once both are in: the next layer's GR read (a programmatic dependent) streams its
// weights in while this kernel adds the results. Triggering before the CPU wait would let whole
// chains of programmatic dependents sit resident through it and starve the L2 prefetch.
__global__ void __launch_bounds__(kCombineThreads)
combine_kernel(const MoeDecodeWork* __restrict__ work, const float* __restrict__ partial,
               const float* __restrict__ y_shared, DecodeExchange* exchange, int32_t layer,
               const uint32_t* __restrict__ seq, const float* __restrict__ inj,
               float* __restrict__ X, float* __restrict__ y_out, uint32_t* relay,
               long long timeout_cycles) {
  const int t = blockIdx.y;
  const int i = blockIdx.x * kCombineThreads + threadIdx.x;
  const CpuPart cpu = combine_cpu_part(work, exchange, layer, seq, relay, timeout_cycles, t, i,
                                       blockIdx.x == 0 && blockIdx.y == 0);
  pdl_wait();
  pdl_trigger();
  const MoeDecodeWork* const wk = pdl_coherent(work);
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
  if (cpu.use) {
    y.x += cpu.y.x; y.y += cpu.y.y; y.z += cpu.y.z; y.w += cpu.y.w;
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
  const int t = blockIdx.y;
  const int i = threadIdx.x;  // blockDim.x == kVec: one float4 per thread
  const float4 r = reinterpret_cast<const float4*>(direction)[i];  // immutable: loaded first
  const CpuPart cpu =
      combine_cpu_part(work, exchange, layer, seq, relay, timeout_cycles, t, i, blockIdx.y == 0);
  pdl_wait();
  pdl_trigger();
  const MoeDecodeWork* const wk = pdl_coherent(work);

  const float* const ps = pdl_coherent(partial);
  float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
  const int np = wk->token_pair_count[t];
  for (int j = 0; j < np; ++j) {
    const float4 v = reinterpret_cast<const float4*>(ps + std::size_t(wk->token_pairs[t][j]) * kHidden)[i];
    y.x += v.x; y.y += v.y; y.z += v.z; y.w += v.w;
  }
  if (cpu.use) {
    y.x += cpu.y.x; y.y += cpu.y.y; y.z += cpu.y.z; y.w += cpu.y.w;
  }
  // y holds the routed sum (GPU-resident + CPU-tier experts).
  __shared__ float s_red[32];
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

// Persistent grid of a hot-expert kernel: as many CTAs as fit on every SM at once. Computed once
// per instantiation; the dynamic shared-memory limit is raised first (some hot_down layouts exceed
// the 48 KiB default).
struct PersistentGrid {
  int ctas = 0;
  cudaError_t error = cudaSuccess;
};

template <class Kernel>
PersistentGrid persistent_grid(Kernel kernel, int threads, std::size_t smem) {
  PersistentGrid g;
  int device = 0, sms = 0, per_sm = 0;
  if ((g.error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem))) ||
      (g.error = cudaGetDevice(&device)) ||
      (g.error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) ||
      (g.error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem)))
    return g;
  if (per_sm < 1) g.error = cudaErrorInvalidConfiguration;
  g.ctas = sms * per_sm;
  return g;
}

template <MoeFormat F, int TMAX>
cudaError_t launch_gate_up(const MoeExpertLayout& L, const MoeDecodeWorkspace& ws, cudaStream_t stream) {
  static const PersistentGrid grid = persistent_grid(hot_gate_up_kernel<F, TMAX>, kGuThreads, 0);
  if (grid.error != cudaSuccess) return grid.error;
  return pdl_launch(hot_gate_up_kernel<F, TMAX>, dim3(grid.ctas), dim3(kGuThreads), 0, stream, L.slots,
                    L.blob_stride, L.gate_offset, L.up_offset, L.down_offset,
                    static_cast<const MoeDecodeWork*>(ws.work), static_cast<const int8_t*>(ws.xq),
                    static_cast<const float*>(ws.xd), ws.h);
}

template <MoeFormat F>
cudaError_t gate_up_t(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                      cudaStream_t stream) {
  switch (tmax_for(tokens)) {
    case 1: return launch_gate_up<F, 1>(L, ws, stream);
    case 2: return launch_gate_up<F, 2>(L, ws, stream);
    case 4: return launch_gate_up<F, 4>(L, ws, stream);
    case 8: return launch_gate_up<F, 8>(L, ws, stream);
    default: return launch_gate_up<F, 16>(L, ws, stream);
  }
}

template <MoeFormat F, int TMAX>
cudaError_t launch_down(const MoeExpertLayout& L, const MoeDecodeWorkspace& ws, cudaStream_t stream) {
  constexpr std::size_t kSmem = DownSmem<F, TMAX>::kBytes;
  static const PersistentGrid grid = persistent_grid(hot_down_kernel<F, TMAX>, kDownThreads, kSmem);
  if (grid.error != cudaSuccess) return grid.error;
  return pdl_launch(hot_down_kernel<F, TMAX>, dim3(grid.ctas), dim3(kDownThreads), kSmem, stream,
                    L.slots, L.blob_stride, L.down_offset, static_cast<const MoeDecodeWork*>(ws.work),
                    static_cast<const float*>(ws.h), ws.partial);
}

template <MoeFormat F>
cudaError_t down_t(const MoeExpertLayout& L, int tokens, const MoeDecodeWorkspace& ws,
                   cudaStream_t stream) {
  switch (tmax_for(tokens)) {
    case 1: return launch_down<F, 1>(L, ws, stream);
    case 2: return launch_down<F, 2>(L, ws, stream);
    case 4: return launch_down<F, 4>(L, ws, stream);
    case 8: return launch_down<F, 8>(L, ws, stream);
    default: return launch_down<F, 16>(L, ws, stream);
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
