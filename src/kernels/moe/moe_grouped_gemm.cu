// Prefill grouped expert GEMMs: 64-token x 64-row tiles over per-expert token lists. Weights are
// dequantized exactly (ggml reference arithmetic) into BF16 shared memory, activations are gathered
// with cp.async, mma.m16n8k16 BF16 with FP32 accumulation.
#include "kernels/moe/iq_device.cuh"
#include "kernels/moe/moe.h"

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/mma.cuh"

#include <cuda_bf16.h>

namespace fngine::kernels {
namespace {

using moe_iq::dequant8;
using moe_iq::kRowBytes;

constexpr int kHidden = dims::kHidden;
constexpr int kInter = dims::kExpertInter;
constexpr int kThreads = 256;
constexpr int kBK = 64;              // K elements per pipeline step
constexpr int kChunksPerRow = kBK / 8;  // 16-byte chunks per smem row
constexpr int kTileM = kMoeTileTokens;  // tokens
constexpr int kTileN = kMoeTileRows;    // weight rows (per matrix)
constexpr int kGuRowTiles = kInter / kTileN;     // 10
constexpr int kDownRowTiles = kHidden / kTileN;  // 40

// Swizzled 16-byte chunk offset (in bf16 elements) inside a [rows][64] bf16 tile.
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * kBK + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ int4 pack_bf16x8(const float v[8]) {
  int4 r;
  r.x = int(pack_bf16x2(v[0], v[1]));
  r.y = int(pack_bf16x2(v[2], v[3]));
  r.z = int(pack_bf16x2(v[4], v[5]));
  r.w = int(pack_bf16x2(v[6], v[7]));
  return r;
}

// Dequantize `Groups` consecutive 8-element groups of one staged weight row into BF16 registers.
// The values are materialized here (asm barrier), ahead of the volatile MMA sequence that follows
// in the callers: letting the compiler interleave the dequant table lookups with the MMAs stalls
// the tensor pipe (measured ~3.5x slower gate/up GEMM).
template <MoeFormat F, int Groups>
__device__ __forceinline__ void dequant_groups_pinned(const uint8_t* row, int g0, int4 (&v)[Groups]) {
#pragma unroll
  for (int i = 0; i < Groups; ++i) {
    float f[8];
    dequant8<F>(row, g0 + i, f);
    v[i] = pack_bf16x8(f);
    asm volatile("" : "+r"(v[i].x), "+r"(v[i].y), "+r"(v[i].z), "+r"(v[i].w));
  }
}

// Store `Groups` 16-byte chunks (chunk0..) of one row of a swizzled [rows][64] BF16 tile.
template <int Groups>
__device__ __forceinline__ void store_groups(__nv_bfloat16* tile, int row, int chunk0,
                                             const int4 (&v)[Groups]) {
#pragma unroll
  for (int i = 0; i < Groups; ++i) *reinterpret_cast<int4*>(tile + swz(row, chunk0 + i)) = v[i];
}

// A fragments for two m16 tiles (rows m0.. m0+31) at k16 step kk.
__device__ __forceinline__ void load_a_frags(const __nv_bfloat16* A, int m0, int kk, int lane,
                                             unsigned a[2][4]) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int row = m0 + 16 * mt + (lane & 15);
    const int chunk = kk * 2 + (lane >> 4);
    ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], smem_addr(A + swz(row, chunk)));
  }
}

// B fragments for two n8 tiles (weight rows n0..n0+15) at k16 step kk.
__device__ __forceinline__ void load_b_frags(const __nv_bfloat16* W, int n0, int kk, int lane,
                                             unsigned b[2][2]) {
  const int row = n0 + (lane & 7) + ((lane >> 4) << 3);
  const int chunk = kk * 2 + ((lane >> 3) & 1);
  ldmatrix_x4(b[0][0], b[0][1], b[1][0], b[1][1], smem_addr(W + swz(row, chunk)));
}

// Gather `Rows` activation rows (16-byte chunks via cp.async, zero-filled past `n`).
template <int Rows = kTileM, int Threads = kThreads>
__device__ __forceinline__ void load_a_tile(__nv_bfloat16* A, const __nv_bfloat16* const* rows,
                                            int k0, int n) {
  static_assert((Rows * kChunksPerRow) % Threads == 0);
#pragma unroll
  for (int i = 0; i < (Rows * kChunksPerRow) / Threads; ++i) {
    const int id = threadIdx.x + i * Threads;
    const int r = id / kChunksPerRow, c = id % kChunksPerRow;
    const __nv_bfloat16* src = rows[r] + k0 + c * 8;
    cp_async_zfill<16, Cache::cg>(A + swz(r, c), src, r < n ? 16 : 0);
  }
}

// Stage one byte segment of each of `Rows` weight rows into shared memory rows of `Stride` bytes:
// 4-byte cp.async from the 4-byte aligned address at or below each segment start (row_ptr[r] +
// seg_off), so the segment sits at offset (start & 3) of its smem row. Bytes at or past `limit`
// are zero-filled (never read from global memory). The stride is an odd number of words so that
// consecutive rows start in different banks.
constexpr int raw_stride4(int bytes) {
  const int words = (bytes + 3 + 3) / 4;
  return (words | 1) * 4;
}

template <int Rows, int Stride, int Threads = kThreads>
__device__ __forceinline__ void stage_raw4(uint8_t* raw, const uint8_t* const* row_ptr,
                                           int64_t seg_off, const uint8_t* limit) {
  static_assert(Stride % 4 == 0);
  constexpr int kWords = Stride / 4;
  for (int id = threadIdx.x; id < Rows * kWords; id += Threads) {
    const int r = id / kWords, wd = id % kWords;
    const uint8_t* start = row_ptr[r] + seg_off;
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(start) & ~uintptr_t{3}) + wd * 4;
    const int64_t left = limit - src;
    const int bytes = left >= 4 ? 4 : (left > 0 ? int(left) : 0);
    cp_async_zfill<4>(raw + r * Stride + wd * 4, src, bytes);
  }
}

__device__ __forceinline__ int raw_misalign4(const uint8_t* row, int64_t seg_off) {
  return int((reinterpret_cast<uintptr_t>(row) + uintptr_t(seg_off)) & 3);
}

template <MoeFormat F>
struct GateUpCfg {
  static constexpr int kRow = kRowBytes<F, kHidden>;
  static constexpr int kSegElems = 256;                           // one raw stage
  static constexpr int kSegBytes = kRow / (kHidden / kSegElems);  // bytes of 256 elements
  static constexpr int kRawStride = raw_stride4(kSegBytes);
  static constexpr int kStepsPerSeg = kSegElems / kBK;            // 4
  static constexpr int kAElems = kTileM * kBK;                    // 2 buffers
  static constexpr int kWElems = 2 * kTileN * kBK;                // single buffer
  static constexpr int kRawBytes = 2 * kTileN * kRawStride;       // single buffer
  static constexpr int kSmem = 2 * kAElems * 2 + kWElems * 2 + kRawBytes;
};

// ------------------------------------------------------------------------------- gate/up
// CTA = (64-token tile of one expert, 64 intermediate rows): gate and up rows are computed
// together so the epilogue can form h. Shared memory stays under ~48 KiB so that two CTAs share
// an SM (one CTA's barriers and load stalls overlap the other's MMAs). Per 64-wide K step ks:
//   A(ks+1) (token rows) is gathered with cp.async; W(ks+1) is dequantized from the staged raw
//   weight bytes into registers while the MMAs of step ks run and stored to the single BF16 W
//   buffer after a barrier; the raw bytes of the next 256 elements are staged into the single raw
//   buffer right after the last dequant of the current segment.
template <MoeFormat F>
__global__ void __launch_bounds__(kThreads, 2)
grouped_gate_up_kernel(const uint8_t* const* __restrict__ expert_blob, int64_t gate_off,
                       int64_t up_off, int64_t down_off, const int32_t* __restrict__ meta,
                       const int4* __restrict__ tiles, const int32_t* __restrict__ pair_token,
                       const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ h) {
  using C = GateUpCfg<F>;
  const int tile_idx = blockIdx.x / kGuRowTiles, rt = blockIdx.x % kGuRowTiles;
  if (tile_idx >= meta[0]) return;
  constexpr int kSteps = kHidden / kBK;  // 40
  extern __shared__ __align__(128) uint8_t smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [2][64][64]
  __nv_bfloat16* Ws = As + 2 * C::kAElems;                           // [128][64]
  uint8_t* Raw = reinterpret_cast<uint8_t*>(Ws + C::kWElems);        // [128][kRawStride]
  __shared__ const __nv_bfloat16* s_rows[kTileM];
  __shared__ const uint8_t* s_wrow[2 * kTileN];

  const int4 tile = tiles[tile_idx];
  const int pb = tile.x, n = tile.y - tile.x;
  const uint8_t* blob = expert_blob[tile.w];
  if (threadIdx.x < kTileM) {
    const int r = threadIdx.x;
    s_rows[r] = x + std::size_t(pair_token[pb + (r < n ? r : 0)]) * kHidden;
  }
  if (threadIdx.x < 2 * kTileN) {
    const int r = threadIdx.x;
    s_wrow[r] = blob + (r < kTileN ? gate_off : up_off) + int64_t(rt * kTileN + (r & (kTileN - 1))) * C::kRow;
  }
  const uint8_t* limit = blob + down_off;  // gate | up end where down begins
  __syncthreads();

  // Dequant assignment: thread -> weight row (0..63 gate, 64..127 up), 4 groups (32 elements).
  const int wrow = threadIdx.x >> 1;
  const int wgrp = (threadIdx.x & 1) * 4;
  const uint8_t* my_row = s_wrow[wrow];
  auto dequant_regs = [&](int k, int4 (&v)[4]) {
    const int seg = k / C::kStepsPerSeg;
    const uint8_t* src = Raw + wrow * C::kRawStride + raw_misalign4(my_row, int64_t(seg) * C::kSegBytes);
    dequant_groups_pinned<F, 4>(src, (k % C::kStepsPerSeg) * kChunksPerRow + wgrp, v);
  };

  stage_raw4<2 * kTileN, C::kRawStride>(Raw, s_wrow, 0, limit);
  load_a_tile(As, s_rows, 0, n);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  {
    int4 v[4];
    dequant_regs(0, v);
    store_groups<4>(Ws, wrow, wgrp, v);
  }

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const bool active = wm * 32 < n;
  float acc[2][4][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll 1
  for (int ks = 0; ks < kSteps; ++ks) {
    // A(ks) and any raw stage issued since the previous barrier have landed; the barrier also
    // publishes W(ks) and frees A buffer (ks+1)&1.
    cp_wait<0>();
    __syncthreads();
    if (ks + 1 < kSteps) {
      load_a_tile(As + ((ks + 1) & 1) * C::kAElems, s_rows, (ks + 1) * kBK, n);
      cp_commit();
    }

    int4 wnext[4];
    if (ks + 1 < kSteps) dequant_regs(ks + 1, wnext);

    if (active) {
      const __nv_bfloat16* A = As + (ks & 1) * C::kAElems;
#pragma unroll
      for (int kk = 0; kk < kBK / 16; ++kk) {
        unsigned a[2][4], bg[2][2], bu[2][2];
        load_a_frags(A, wm * 32, kk, lane, a);
        load_b_frags(Ws, wn * 16, kk, lane, bg);
        load_b_frags(Ws, kTileN + wn * 16, kk, lane, bu);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3], a[mt][0],
                     a[mt][1], a[mt][2], a[mt][3], bg[nt][0], bg[nt][1]);
            mma_bf16(acc[mt][2 + nt][0], acc[mt][2 + nt][1], acc[mt][2 + nt][2],
                     acc[mt][2 + nt][3], a[mt][0], a[mt][1], a[mt][2], a[mt][3], bu[nt][0],
                     bu[nt][1]);
          }
        }
      }
    }
    if (ks + 1 < kSteps) {
      __syncthreads();  // every warp is done reading W(ks) (and the raw segment, if finished)
      store_groups<4>(Ws, wrow, wgrp, wnext);
      const int k1 = ks + 1;
      if (k1 % C::kStepsPerSeg == C::kStepsPerSeg - 1 && k1 + 1 < kSteps) {
        // W(k1) was the last dequant of its segment: stage the next one (waited at the next top).
        stage_raw4<2 * kTileN, C::kRawStride>(Raw, s_wrow, int64_t(k1 / C::kStepsPerSeg + 1) * C::kSegBytes,
                                              limit);
        cp_commit();
      }
    }
  }

  // Epilogue: h = silu(g) * u, BF16, rows of valid tokens only.
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = wm * 32 + mt * 16 + (lane >> 2) + half * 8;
      if (tok >= n) continue;
      __nv_bfloat16* hrow = h + std::size_t(pb + tok) * kInter + rt * kTileN;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = wn * 16 + nt * 8 + (lane & 3) * 2;
        const float g0 = acc[mt][nt][half * 2 + 0], g1 = acc[mt][nt][half * 2 + 1];
        const float u0 = acc[mt][2 + nt][half * 2 + 0], u1 = acc[mt][2 + nt][half * 2 + 1];
        *reinterpret_cast<uint32_t*>(hrow + col) = pack_bf16x2(silu(g0) * u0, silu(g1) * u1);
      }
    }
  }
}

template <MoeFormat F>
struct DownCfg {
  static constexpr int kRow = kRowBytes<F, kInter>;
  static constexpr int kRawStride = raw_stride4(kRow);  // whole 640-element row staged once
  static constexpr int kAElems = kTileM * kBK;          // 2 buffers
  static constexpr int kWElems = kTileN * kBK;          // single buffer
  static constexpr int kSmem = 2 * kAElems * 2 + kWElems * 2 + kTileN * kRawStride;
};

// ---------------------------------------------------------------------------------- down
// CTA = (64-token tile of one expert, 64 output rows), two CTAs per SM. The 64 weight rows
// (K = 640) are staged raw once in the prologue; per 64-wide K step, A(ks+1) is gathered with
// cp.async, W(ks+1) is dequantized into registers ahead of the MMAs of step ks and stored to the
// single BF16 W buffer after a barrier. The epilogue atomically adds w * out into y[token].
template <MoeFormat F>
__global__ void __launch_bounds__(kThreads, 2)
grouped_down_kernel(const uint8_t* const* __restrict__ expert_blob, int64_t down_off,
                    const int32_t* __restrict__ meta, const int4* __restrict__ tiles,
                    const int32_t* __restrict__ pair_token, const float* __restrict__ pair_weight,
                    const __nv_bfloat16* __restrict__ h,
                    unsigned long long* __restrict__ y) {
  using C = DownCfg<F>;
  const int tile_idx = blockIdx.x / kDownRowTiles, rt = blockIdx.x % kDownRowTiles;
  if (tile_idx >= meta[0]) return;
  constexpr int kSteps = kInter / kBK;  // 10
  extern __shared__ __align__(128) uint8_t smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [2][64][64]
  __nv_bfloat16* Ws = As + 2 * C::kAElems;                           // [64][64]
  uint8_t* Raw = reinterpret_cast<uint8_t*>(Ws + C::kWElems);        // [64][kRawStride]
  __shared__ const __nv_bfloat16* s_rows[kTileM];
  __shared__ const uint8_t* s_wrow[kTileN];

  const int4 tile = tiles[tile_idx];
  const int pb = tile.x, n = tile.y - tile.x;
  const uint8_t* blob = expert_blob[tile.w];
  if (threadIdx.x < kTileM) {
    const int r = threadIdx.x;
    s_rows[r] = h + std::size_t(pb + (r < n ? r : 0)) * kInter;
    s_wrow[r] = blob + down_off + int64_t(rt * kTileN + r) * C::kRow;
  }
  const uint8_t* limit = blob + down_off + int64_t(kHidden) * C::kRow;
  __syncthreads();

  // Dequant assignment: thread -> weight row, 2 groups (16 elements) per step.
  const int wrow = threadIdx.x >> 2;
  const int wgrp = (threadIdx.x & 3) * 2;
  const uint8_t* my_raw = Raw + wrow * C::kRawStride + raw_misalign4(s_wrow[wrow], 0);

  stage_raw4<kTileN, C::kRawStride>(Raw, s_wrow, 0, limit);
  load_a_tile(As, s_rows, 0, n);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  {
    int4 v[2];
    dequant_groups_pinned<F, 2>(my_raw, wgrp, v);
    store_groups<2>(Ws, wrow, wgrp, v);
  }

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const bool active = wm * 32 < n;
  float acc[2][2][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll 1
  for (int ks = 0; ks < kSteps; ++ks) {
    cp_wait<0>();
    __syncthreads();
    if (ks + 1 < kSteps) {
      load_a_tile(As + ((ks + 1) & 1) * C::kAElems, s_rows, (ks + 1) * kBK, n);
      cp_commit();
    }
    int4 wnext[2];
    if (ks + 1 < kSteps) dequant_groups_pinned<F, 2>(my_raw, (ks + 1) * kChunksPerRow + wgrp, wnext);
    if (active) {
      const __nv_bfloat16* A = As + (ks & 1) * C::kAElems;
#pragma unroll
      for (int kk = 0; kk < kBK / 16; ++kk) {
        unsigned a[2][4], b[2][2];
        load_a_frags(A, wm * 32, kk, lane, a);
        load_b_frags(Ws, wn * 16, kk, lane, b);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
            mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3], a[mt][0],
                     a[mt][1], a[mt][2], a[mt][3], b[nt][0], b[nt][1]);
      }
    }
    if (ks + 1 < kSteps) {
      __syncthreads();  // every warp is done reading W(ks)
      store_groups<2>(Ws, wrow, wgrp, wnext);
    }
  }

#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = wm * 32 + mt * 16 + (lane >> 2) + half * 8;
      if (tok >= n) continue;
      const int p = pb + tok;
      const float w = pair_weight[p];
      unsigned long long* yrow = y + std::size_t(pair_token[p]) * kHidden + rt * kTileN;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = wn * 16 + nt * 8 + (lane & 3) * 2;
        // Fixed point (exact, order-independent integer sums): 2^32 scaling is exact in FP32.
        constexpr float kScale = float(1ull << kMoeYFixedShift);
        atomicAdd(yrow + col, static_cast<unsigned long long>(
                                  __float2ll_rn(w * acc[mt][nt][half * 2 + 0] * kScale)));
        atomicAdd(yrow + col + 1, static_cast<unsigned long long>(
                                      __float2ll_rn(w * acc[mt][nt][half * 2 + 1] * kScale)));
      }
    }
  }
}

int grid_tiles(int tokens) {
  return (tokens * dims::kTopK + kTileM - 1) / kTileM + dims::kExperts;
}

template <MoeFormat F>
cudaError_t launch_gate_up(const MoeExpertLayout& L, const uint8_t* const* expert_blob,
                           const __nv_bfloat16* x, int tokens,
                           const MoePrefillWorkspace& ws, cudaStream_t stream) {
  constexpr int kSmem = GateUpCfg<F>::kSmem;
  static const cudaError_t attr = cudaFuncSetAttribute(
      grouped_gate_up_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attr != cudaSuccess) return attr;
  grouped_gate_up_kernel<F><<<grid_tiles(tokens) * kGuRowTiles, kThreads, kSmem, stream>>>(
      expert_blob, L.gate_offset, L.up_offset, L.down_offset, ws.meta, ws.tiles,
      ws.pair_token, x, ws.h);
  return cudaGetLastError();
}

template <MoeFormat F>
cudaError_t launch_down(const MoeExpertLayout& L, const uint8_t* const* expert_blob, int tokens,
                        const MoePrefillWorkspace& ws,
                        cudaStream_t stream) {
  constexpr int kSmem = DownCfg<F>::kSmem;
  static const cudaError_t attr = cudaFuncSetAttribute(
      grouped_down_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attr != cudaSuccess) return attr;
  grouped_down_kernel<F><<<grid_tiles(tokens) * kDownRowTiles, kThreads, kSmem, stream>>>(
      expert_blob, L.down_offset, ws.meta, ws.tiles, ws.pair_token, ws.pair_weight,
      ws.h, reinterpret_cast<unsigned long long*>(ws.y_fixed));
  return cudaGetLastError();
}

}  // namespace

cudaError_t moe_grouped_gate_up(const MoeExpertLayout& layout, const uint8_t* const* expert_blob,
                                const __nv_bfloat16* x, int32_t tokens,
                                const MoePrefillWorkspace& ws, cudaStream_t stream) {
  if (tokens < 1 || tokens > ws.capacity || !expert_blob) return cudaErrorInvalidValue;
  const auto* eb = expert_blob;
  switch (layout.gate_up_format) {
    case MoeFormat::IQ2_S: return launch_gate_up<MoeFormat::IQ2_S>(layout, eb, x, tokens, ws, stream);
    case MoeFormat::IQ3_XXS: return launch_gate_up<MoeFormat::IQ3_XXS>(layout, eb, x, tokens, ws, stream);
    case MoeFormat::IQ3_S: return launch_gate_up<MoeFormat::IQ3_S>(layout, eb, x, tokens, ws, stream);
    case MoeFormat::IQ4_XS: return launch_gate_up<MoeFormat::IQ4_XS>(layout, eb, x, tokens, ws, stream);
    case MoeFormat::IQ4_NL: return launch_gate_up<MoeFormat::IQ4_NL>(layout, eb, x, tokens, ws, stream);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t moe_grouped_down(const MoeExpertLayout& layout, const uint8_t* const* expert_blob,
                             int32_t tokens, const MoePrefillWorkspace& ws, cudaStream_t stream) {
  if (tokens < 1 || tokens > ws.capacity || !expert_blob) return cudaErrorInvalidValue;
  switch (layout.down_format) {
    case MoeFormat::IQ4_NL: return launch_down<MoeFormat::IQ4_NL>(layout, expert_blob, tokens, ws, stream);
    case MoeFormat::Q2_0: return launch_down<MoeFormat::Q2_0>(layout, expert_blob, tokens, ws, stream);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace fngine::kernels
