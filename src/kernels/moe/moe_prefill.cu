// Prefill MoE routing (top-k, per-expert token lists, 64-token tiles, compacted cold-row publish)
// and the prefill combine with the CPU tier. Eager launches: `tokens` is a host value.
#include "kernels/moe/moe.h"
#include "kernels/moe/moe_common.cuh"

#include <cuda_bf16.h>

#include <algorithm>

namespace fngine::kernels {
namespace {

constexpr int kHidden = dims::kHidden;
constexpr int kTopK = dims::kTopK;
constexpr int kExperts = dims::kExperts;
constexpr int kRouteThreads = 256;
constexpr int kScanThreads = 1024;

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

int max_tiles_for(int32_t capacity) {
  return (capacity * kTopK + kMoeTileTokens - 1) / kMoeTileTokens + kExperts;
}

struct PrefillLayout {
  std::size_t meta, expert_count, expert_offset, expert_cursor, topk_expert, topk_weight,
      token_cold, row_of_token, pair_token, pair_weight, tiles, h, y, relay, total;
};

PrefillLayout prefill_layout(int32_t capacity) {
  const std::size_t c = std::size_t(capacity), pairs = c * kTopK;
  PrefillLayout l{};
  std::size_t off = 0;
  l.meta = off;          off = align256(off + 16 * sizeof(int32_t));
  l.expert_count = off;  off = align256(off + kExperts * sizeof(int32_t));
  l.expert_offset = off; off = align256(off + (kExperts + 1) * sizeof(int32_t));
  l.expert_cursor = off; off = align256(off + kExperts * sizeof(int32_t));
  l.topk_expert = off;   off = align256(off + pairs * sizeof(int32_t));
  l.topk_weight = off;   off = align256(off + pairs * sizeof(float));
  l.token_cold = off;    off = align256(off + c * sizeof(int32_t));
  l.row_of_token = off;  off = align256(off + c * sizeof(int32_t));
  l.pair_token = off;    off = align256(off + pairs * sizeof(int32_t));
  l.pair_weight = off;   off = align256(off + pairs * sizeof(float));
  l.tiles = off;         off = align256(off + std::size_t(max_tiles_for(capacity)) * sizeof(int4));
  l.h = off;             off = align256(off + pairs * dims::kExpertInter * sizeof(__nv_bfloat16));
  l.y = off;             off = align256(off + c * kHidden * sizeof(long long));
  l.relay = off;         off = align256(off + std::size_t(kExchangeLayers) * 2 * sizeof(uint32_t));
  l.total = off;
  return l;
}

// Warp per token: top-10, hot counts per expert, cold count per token.
__global__ void __launch_bounds__(kRouteThreads)
route_topk_kernel(const float* __restrict__ logits, const uint8_t* const* __restrict__ expert_blob, int T,
                  int32_t* __restrict__ topk_expert, float* __restrict__ topk_weight,
                  int32_t* __restrict__ expert_count, int32_t* __restrict__ token_cold) {
  const int t = blockIdx.x * (kRouteThreads / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= T) return;
  int ids[kTopK];
  float w[kTopK];
  float lv[moe_detail::kPerLane];
  moe_detail::warp_load_logits(logits + std::size_t(t) * dims::kRouterRows, lv);
  moe_detail::warp_topk10(lv, ids, w);
  bool cold = false;
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
    topk_expert[t * kTopK + lane] = id;
    topk_weight[t * kTopK + lane] = wt;
    if (expert_blob[id] != nullptr) atomicAdd(&expert_count[id], 1);
    else cold = true;
  }
  const unsigned m = __ballot_sync(0xffffffffu, cold);
  if (lane == 0) token_cold[t] = __popc(m);
}

// Single CTA: expert offsets, tile list, compacted cold rows.
__global__ void __launch_bounds__(kScanThreads)
route_scan_kernel(int T, int32_t* __restrict__ meta,
                  const int32_t* __restrict__ expert_count, int32_t* __restrict__ expert_offset,
                  int32_t* __restrict__ expert_cursor, int4* __restrict__ tiles,
                  const int32_t* __restrict__ token_cold, int32_t* __restrict__ row_of_token,
                  PrefillExchangeView view) {
  __shared__ int s_scan[32];
  const int i = threadIdx.x;
  const int c = i < kExperts ? expert_count[i] : 0;
  const int nt = (c + kMoeTileTokens - 1) / kMoeTileTokens;
  int total_pairs = 0, total_tiles = 0;
  const int off = moe_detail::block_exclusive_scan(c, s_scan, &total_pairs);
  const int tile0 = moe_detail::block_exclusive_scan(nt, s_scan, &total_tiles);
  if (i < kExperts) {
    expert_offset[i] = off;
    expert_cursor[i] = off;
    for (int j = 0; j < nt; ++j) {
      tiles[tile0 + j] = make_int4(off + j * kMoeTileTokens,
                                   off + min((j + 1) * kMoeTileTokens, c), 0, i);
    }
  }
  int rows = 0, cold = 0;
  for (int base = 0; base < T; base += kScanThreads) {
    const int t = base + i;
    const int nc = t < T ? token_cold[t] : 0;
    int chunk_rows = 0, chunk_cold = 0;
    const int r = moe_detail::block_exclusive_scan(nc > 0 ? 1 : 0, s_scan, &chunk_rows);
    moe_detail::block_exclusive_scan(nc, s_scan, &chunk_cold);
    if (t < T) {
      row_of_token[t] = nc > 0 ? rows + r : -1;
      if (nc > 0 && view.row_token) view.row_token[rows + r] = t;
    }
    rows += chunk_rows;
    cold += chunk_cold;
  }
  if (i == 0) {
    expert_offset[kExperts] = total_pairs;
    meta[0] = total_tiles;
    meta[1] = total_pairs;
    meta[2] = cold;
    meta[3] = rows;
    if (view.header) {
      view.header[0] = T;
      view.header[1] = rows;
      view.header[2] = cold;
    }
  }
}

// Warp per token: scatter hot pairs into expert lists; publish cold lists and x rows.
__global__ void __launch_bounds__(kRouteThreads)
route_scatter_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* const* __restrict__ expert_blob,
                     int T, const int32_t* __restrict__ topk_expert,
                     const float* __restrict__ topk_weight, int32_t* __restrict__ expert_cursor,
                     int32_t* __restrict__ pair_token, float* __restrict__ pair_weight,
                     const int32_t* __restrict__ token_cold,
                     const int32_t* __restrict__ row_of_token, PrefillExchangeView view) {
  const int t = blockIdx.x * (kRouteThreads / 32) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= T) return;
  int e = 0;
  bool hot = true;
  float w = 0.f;
  if (lane < kTopK) {
    e = topk_expert[t * kTopK + lane];
    w = topk_weight[t * kTopK + lane];
    hot = expert_blob[e] != nullptr;
    if (hot) {
      const int p = atomicAdd(&expert_cursor[e], 1);
      pair_token[p] = t;
      pair_weight[p] = w;
    }
  }
  const bool cold = lane < kTopK && !hot;
  const unsigned m = __ballot_sync(0xffffffffu, cold);
  const int row = row_of_token[t];
  if (row < 0 || !view.cold_expert) return;
  if (cold) {
    const int rank = __popc(m & ((1u << lane) - 1u));
    view.cold_expert[row * kTopK + rank] = e;
    view.cold_weight[row * kTopK + rank] = w;
  }
  if (lane == 0) view.cold_count[row] = token_cold[t];
  constexpr int kChunks = kHidden * 2 / 16;
  const int4* src = reinterpret_cast<const int4*>(x + std::size_t(t) * kHidden);
  int4* dst = reinterpret_cast<int4*>(view.x + std::size_t(row) * kHidden);
  for (int ch = lane; ch < kChunks; ch += 32) dst[ch] = src[ch];
  __threadfence_system();
}

__global__ void route_release_kernel(ExchangeControl* control, int32_t layer,
                                     const uint32_t* __restrict__ seq) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    exchange_store_release_sys(&control->ready[layer].value, *seq);
  }
}

constexpr int kCombineThreads = 256;
constexpr int kVec = kHidden / 4;

// Only CTA (0, 0) polls done[layer] in host memory; the others wait on its device-memory relay
// (exchange_wait_sys_relayed): a chunk launches 3 CTAs per token.
__global__ void __launch_bounds__(kCombineThreads)
prefill_combine_kernel(const int32_t* __restrict__ row_of_token,
                       const long long* __restrict__ y_hot, const float* __restrict__ y_shared,
                       ExchangeControl* control, PrefillExchangeView view, int32_t layer,
                       const uint32_t* __restrict__ seq, const float* __restrict__ inj,
                       float* __restrict__ X, float* __restrict__ y_out, uint32_t* relay,
                       long long timeout_cycles) {
  const int t = blockIdx.y;
  const int i = blockIdx.x * kCombineThreads + threadIdx.x;
  __shared__ int s_row;
  if (threadIdx.x == 0) {
    int row = -1;
    // The prefill exchange region is shared by all layers: the CPU pool releases done[layer] for
    // every chunk (also without cold pairs) once it has consumed the header, so always wait.
    if (control) {
      if (exchange_wait_sys_relayed(&control->done[layer].value, *seq,
                                    blockIdx.x == 0 && blockIdx.y == 0, relay + 2 * layer,
                                    timeout_cycles)) {
        row = row_of_token[t];
      } else {
        exchange_store_release_sys(&control->error.value,
                                   kExchangeErrorMoeTimeout | uint32_t(layer + 1));
      }
    }
    s_row = row;
  }
  __syncthreads();
  if (i >= kVec) return;
  const longlong2* yh = reinterpret_cast<const longlong2*>(y_hot + std::size_t(t) * kHidden) + 2 * i;
  const longlong2 y01 = yh[0], y23 = yh[1];
  constexpr float kInv = 1.0f / float(1ull << kMoeYFixedShift);
  float4 y = make_float4(__ll2float_rn(y01.x) * kInv, __ll2float_rn(y01.y) * kInv,
                         __ll2float_rn(y23.x) * kInv, __ll2float_rn(y23.y) * kInv);
  const float4 s = reinterpret_cast<const float4*>(y_shared + std::size_t(t) * kHidden)[i];
  y.x += s.x; y.y += s.y; y.z += s.z; y.w += s.w;
  if (s_row >= 0) {
    const float4 c = exchange_load_relaxed_sys(
        reinterpret_cast<const float4*>(view.y + std::size_t(s_row) * kHidden) + i);
    y.x += c.x; y.y += c.y; y.z += c.z; y.w += c.w;
  }
  if (y_out) reinterpret_cast<float4*>(y_out + std::size_t(t) * kHidden)[i] = y;
#pragma unroll
  for (int st = 0; st < dims::kStreams; ++st) {
    const float g = inj[t * dims::kStreams + st];
    float4* xp = reinterpret_cast<float4*>(X + (std::size_t(t) * dims::kStreams + st) * kHidden) + i;
    float4 xv = *xp;
    xv.x += g * y.x; xv.y += g * y.y; xv.z += g * y.z; xv.w += g * y.w;
    *xp = xv;
  }
}

// Abliterated variant of the prefill combine: one CTA per token so the dot product of the routed
// sum with the refusal direction is a block reduction. Only the routed experts (GPU fixed-point
// sum + CPU rows) are transformed; the shared expert is added afterwards (see combine_ablate_kernel).
__global__ void __launch_bounds__(kVec)
prefill_combine_ablate_kernel(const int32_t* __restrict__ row_of_token,
                              const long long* __restrict__ y_hot, const float* __restrict__ y_shared,
                              ExchangeControl* control, PrefillExchangeView view, int32_t layer,
                              const uint32_t* __restrict__ seq, const float* __restrict__ inj,
                              float* __restrict__ X, float* __restrict__ y_out, uint32_t* relay,
                              const float* __restrict__ direction, float scale,
                              long long timeout_cycles) {
  const int t = blockIdx.y;
  const int i = threadIdx.x;  // blockDim.x == kVec
  __shared__ int s_row;
  if (threadIdx.x == 0) {
    int row = -1;
    if (control) {
      if (exchange_wait_sys_relayed(&control->done[layer].value, *seq, blockIdx.y == 0,
                                    relay + 2 * layer, timeout_cycles)) {
        row = row_of_token[t];
      } else {
        exchange_store_release_sys(&control->error.value,
                                   kExchangeErrorMoeTimeout | uint32_t(layer + 1));
      }
    }
    s_row = row;
  }
  __syncthreads();
  const longlong2* yh = reinterpret_cast<const longlong2*>(y_hot + std::size_t(t) * kHidden) + 2 * i;
  const longlong2 y01 = yh[0], y23 = yh[1];
  constexpr float kInv = 1.0f / float(1ull << kMoeYFixedShift);
  float4 y = make_float4(__ll2float_rn(y01.x) * kInv, __ll2float_rn(y01.y) * kInv,
                         __ll2float_rn(y23.x) * kInv, __ll2float_rn(y23.y) * kInv);
  if (s_row >= 0) {
    const float4 c = exchange_load_relaxed_sys(
        reinterpret_cast<const float4*>(view.y + std::size_t(s_row) * kHidden) + i);
    y.x += c.x; y.y += c.y; y.z += c.z; y.w += c.w;
  }
  __shared__ float s_red[32];
  const float4 r = reinterpret_cast<const float4*>(direction)[i];
  const float dot = moe_detail::block_reduce_sum(y.x * r.x + y.y * r.y + y.z * r.z + y.w * r.w, s_red);
  const float k = -scale * dot;
  y.x += k * r.x; y.y += k * r.y; y.z += k * r.z; y.w += k * r.w;
  const float4 s = reinterpret_cast<const float4*>(y_shared + std::size_t(t) * kHidden)[i];
  y.x += s.x; y.y += s.y; y.z += s.z; y.w += s.w;

  if (y_out) reinterpret_cast<float4*>(y_out + std::size_t(t) * kHidden)[i] = y;
#pragma unroll
  for (int st = 0; st < dims::kStreams; ++st) {
    const float g = inj[t * dims::kStreams + st];
    float4* xp = reinterpret_cast<float4*>(X + (std::size_t(t) * dims::kStreams + st) * kHidden) + i;
    float4 xv = *xp;
    xv.x += g * y.x; xv.y += g * y.y; xv.z += g * y.z; xv.w += g * y.w;
    *xp = xv;
  }
}

}  // namespace

std::size_t moe_prefill_workspace_bytes(int32_t capacity) {
  return prefill_layout(capacity).total;
}

MoePrefillWorkspace make_moe_prefill_workspace(void* base, int32_t capacity) {
  const PrefillLayout l = prefill_layout(capacity);
  auto* b = static_cast<uint8_t*>(base);
  MoePrefillWorkspace ws;
  ws.capacity = capacity;
  ws.max_tiles = max_tiles_for(capacity);
  ws.meta = reinterpret_cast<int32_t*>(b + l.meta);
  ws.expert_count = reinterpret_cast<int32_t*>(b + l.expert_count);
  ws.expert_offset = reinterpret_cast<int32_t*>(b + l.expert_offset);
  ws.expert_cursor = reinterpret_cast<int32_t*>(b + l.expert_cursor);
  ws.topk_expert = reinterpret_cast<int32_t*>(b + l.topk_expert);
  ws.topk_weight = reinterpret_cast<float*>(b + l.topk_weight);
  ws.token_cold = reinterpret_cast<int32_t*>(b + l.token_cold);
  ws.row_of_token = reinterpret_cast<int32_t*>(b + l.row_of_token);
  ws.pair_token = reinterpret_cast<int32_t*>(b + l.pair_token);
  ws.pair_weight = reinterpret_cast<float*>(b + l.pair_weight);
  ws.tiles = reinterpret_cast<int4*>(b + l.tiles);
  ws.h = reinterpret_cast<__nv_bfloat16*>(b + l.h);
  ws.y_fixed = reinterpret_cast<long long*>(b + l.y);
  ws.relay = reinterpret_cast<uint32_t*>(b + l.relay);
  return ws;
}

cudaError_t moe_route(const float* router_logits, const __nv_bfloat16* x,
                      const uint8_t* const* expert_blob,
                      int32_t layer, const uint32_t* seq, ExchangeControl* control,
                      const PrefillExchangeView& view, int32_t tokens,
                      const MoePrefillWorkspace& ws, cudaStream_t stream) {
  if (tokens < 1 || tokens > ws.capacity || !expert_blob) return cudaErrorInvalidValue;
  if (control && (layer < 0 || layer >= kExchangeLayers || !seq)) return cudaErrorInvalidValue;
  if (view.header && view.capacity < tokens) return cudaErrorInvalidValue;
  cudaError_t err = cudaMemsetAsync(ws.expert_count, 0, kExperts * sizeof(int32_t), stream);
  if (err != cudaSuccess) return err;
  err = cudaMemsetAsync(ws.y_fixed, 0, std::size_t(tokens) * kHidden * sizeof(long long), stream);
  if (err != cudaSuccess) return err;
  const int blocks = (tokens + kRouteThreads / 32 - 1) / (kRouteThreads / 32);
  route_topk_kernel<<<blocks, kRouteThreads, 0, stream>>>(router_logits, expert_blob, tokens,
                                                          ws.topk_expert, ws.topk_weight,
                                                          ws.expert_count, ws.token_cold);
  route_scan_kernel<<<1, kScanThreads, 0, stream>>>(tokens, ws.meta, ws.expert_count,
                                                    ws.expert_offset, ws.expert_cursor,
                                                    ws.tiles, ws.token_cold, ws.row_of_token,
                                                    view);
  route_scatter_kernel<<<blocks, kRouteThreads, 0, stream>>>(
      x, expert_blob, tokens, ws.topk_expert, ws.topk_weight, ws.expert_cursor, ws.pair_token,
      ws.pair_weight, ws.token_cold, ws.row_of_token, view);
  if (control) route_release_kernel<<<1, 32, 0, stream>>>(control, layer, seq);
  return cudaGetLastError();
}

cudaError_t moe_prefill_combine(const MoePrefillWorkspace& ws, const float* y_shared,
                                ExchangeControl* control, const PrefillExchangeView& view,
                                int32_t layer, const uint32_t* seq, const float* inj, float* X,
                                float* y_out, int32_t tokens, long long timeout_cycles,
                                const float* ablate_direction, float ablate_scale,
                                cudaStream_t stream) {
  if (tokens < 1 || tokens > ws.capacity) return cudaErrorInvalidValue;
  if (control && (layer < 0 || layer >= kExchangeLayers || !seq || !view.y))
    return cudaErrorInvalidValue;
  if (ablate_direction) {
    prefill_combine_ablate_kernel<<<dim3(1, tokens), kVec, 0, stream>>>(
        ws.row_of_token, ws.y_fixed, y_shared, control, view, layer, seq, inj, X, y_out, ws.relay,
        ablate_direction, ablate_scale, timeout_cycles);
    return cudaGetLastError();
  }
  const dim3 grid((kVec + kCombineThreads - 1) / kCombineThreads, tokens);
  prefill_combine_kernel<<<grid, kCombineThreads, 0, stream>>>(
      ws.row_of_token, ws.y_fixed, y_shared, control, view, layer, seq, inj, X, y_out, ws.relay,
      timeout_cycles);
  return cudaGetLastError();
}

}  // namespace fngine::kernels
