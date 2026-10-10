// GDN prefill recurrence: an FP32-accurate chunked (WY) form over the leading T - T%64 tokens and
// the per-token recurrent kernel for the tail.
//
// The stage structure follows cinference's chunked Gated DeltaNet (prepare WY -> state passing ->
// output, cinference:src/ops/linear_attention/gated_delta_net/chunked @ 383e5db, see NOTICE) and
// the math of HF torch_chunk_gated_delta_rule. Unlike cinference (BF16 intermediates + TF32 MMA)
// every intermediate and every multiply-add here is FP32 (SIMT FMA), so the chunked path matches
// the FP64 recurrence to FP32 accuracy.
//
// Per chunk of 64 tokens and value head h (Q/K head h/3), with G_t = sum_{s<=t} g_s:
//   L[i][j]   = beta_i e^{G_i-G_j} (k_i . k_j)          (i > j)
//   Aqk[i][j] = e^{G_i-G_j} (q_i . k_j)                 (i >= j)
//   (I + L) [U | W] = [beta v | beta e^{G} k]           (forward substitution)
//   v_new     = U - W S^T
//   o_t       = scale * ((q_t e^{G_t}) S^T + sum_{s<=t} Aqk[t][s] v_new_s)
//   S         = S e^{G_63} + sum_t v_new_t (k_t e^{G_63-G_t})^T
// with S stored [v][k] (fngine/cinference layout) and q, k L2-normalized in FP32.

#include "kernels/gdn/gdn.h"
#include "kernels/gdn/layout.h"
#include "kernels/gdn/recurrent.cuh"

#include "core/device.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace gdn {

// ---------------------------------------------------------------------------------------------
// Direct per-token form for the prefill tail and short prefills (any width, eager). q/k are the
// raw BF16 columns of the conv output and are L2-normalized in-kernel.

struct DirectParams {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    std::int64_t qk_ld;
    const __nv_bfloat16* v;
    std::int64_t v_ld;
    const float* g;
    const float* beta;
    const float* state_read;
    float* state_write;
    __nv_bfloat16* out;
    std::int32_t width;
};

__device__ __forceinline__ void load_normalized_lane(float (&key)[kQkPerLane],
                                                     const __nv_bfloat16* base,
                                                     std::uint32_t dqk_base, int lane) {
    const RawQkLane raw = load_raw_qk_lane(base, dqk_base);
    const float scale   = head_l2_scale(lane_sumsq(raw.value), lane);
#pragma unroll
    for (int c = 0; c < kQkPerLane; ++c) { key[c] = raw.value[c] * scale; }
}

__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    gdn_direct_kernel(const __grid_constant__ DirectParams p) {
    const int lane                 = static_cast<int>(threadIdx.x);
    const int warp                 = static_cast<int>(threadIdx.y);
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t qk_head    = value_head / kGroup;
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);
    const std::int64_t head_offset = static_cast<std::int64_t>(value_head) * kStateDim * kStateDim;
    const int owned_row            = (lane >> 3) & (kDvPerWarp - 1);

    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, p.state_read + head_offset, dv_base, dqk_base);

    for (std::int32_t token = 0; token < p.width; ++token) {
        float key[kQkPerLane];
        load_normalized_lane(key, p.k + token * p.qk_ld + qk_head * kStateDim, dqk_base, lane);
        const std::int64_t gate = static_cast<std::int64_t>(token) * kValueHeadCount + value_head;
        const float alpha       = expf(p.g[gate]);
        const float beta        = p.beta[gate];
        const float v_r = __bfloat162float(p.v[token * p.v_ld + value_head * kStateDim + dv_base +
                                               owned_row]);

        float partial[kDvPerWarp];
        row_partials(state, key, partial);
        const float owned_delta = gdn_delta(beta, v_r, alpha, gdn_row_sums(partial, lane));
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            const float delta = __shfl_sync(kFullWarpMask, owned_delta, r * 8);
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                state[r][c] = gdn_update(alpha, state[r][c], delta, key[c]);
            }
        }

        float query[kQkPerLane];
        load_normalized_lane(query, p.q + token * p.qk_ld + qk_head * kStateDim, dqk_base, lane);
        row_partials(state, query, partial);
        const float attn_val = gdn_row_sums(partial, lane);
        if ((lane & 7) == 0) {
            p.out[static_cast<std::int64_t>(token) * gdn_dims::kValueDim +
                  static_cast<std::int64_t>(value_head) * kStateDim + dv_base + owned_row] =
                __float2bfloat16(attn_val * kOutputScale);
        }
    }
    store_state_tile(state, p.state_write + head_offset, dv_base, dqk_base);
}

} // namespace gdn

namespace {

using namespace gdn_dims;

constexpr int kBT      = 64;        // chunk tokens
constexpr int kD       = 128;       // head dim
constexpr int kPad     = kD + 4;    // padded FP32 row stride of [64][128] smem tiles
constexpr int kApad    = kBT + 4;   // padded row stride of [64][64] smem tiles
constexpr int kThreads = 256;
constexpr int kVTile   = 16;        // value rows per scan CTA
constexpr std::size_t kAlign = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("gdn_prefill: ") + message); }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

struct ChunkBuffers {
    float* qn;     // [Tf][16][128] L2-normalized q
    float* kn;     // [Tf][16][128] L2-normalized k
    float* U;      // [Tf][48][128]
    float* W;      // [Tf][48][128]
    float* aqk;    // [C][48][64][64]
    float* eg;     // [Tf][48]  e^{G_t}
    float* ekd;    // [Tf][48]  e^{G_63 - G_t}
    float* dlast;  // [C][48]   e^{G_63}
};

struct PrefillLayout {
    gdn::LayoutRegion qn, kn, U, W, aqk, eg, ekd, dlast;
    std::size_t total = 0;
};

PrefillLayout plan_prefill(std::int32_t tokens) {
    PrefillLayout layout{};
    const std::size_t full = static_cast<std::size_t>(tokens / kChunk * kChunk);
    if (full == 0) { return layout; }
    const std::size_t chunks = full / kChunk;
    gdn::LayoutBuilder b;
    layout.qn    = b.add(full * kKeyDim * sizeof(float), kAlign, "GDN normalized q");
    layout.kn    = b.add(full * kKeyDim * sizeof(float), kAlign, "GDN normalized k");
    layout.U     = b.add(full * kValueDim * sizeof(float), kAlign, "GDN U");
    layout.W     = b.add(full * kValueDim * sizeof(float), kAlign, "GDN W");
    layout.aqk   = b.add(chunks * kValueHeads * kBT * kBT * sizeof(float), kAlign, "GDN Aqk");
    layout.eg    = b.add(full * kValueHeads * sizeof(float), kAlign, "GDN e^G");
    layout.ekd   = b.add(full * kValueHeads * sizeof(float), kAlign, "GDN key decay");
    layout.dlast = b.add(chunks * kValueHeads * sizeof(float), kAlign, "GDN chunk decay");
    layout.total = b.finish(kAlign, "GDN prefill workspace");
    return layout;
}

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ float4 scale4(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}
// acc + a . b over 4 lanes, in order.
__device__ __forceinline__ float dot4(float4 a, float4 b, float acc) {
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    return fmaf(a.w, b.w, acc);
}

// One warp per 128-wide head row; rows [0, Tf*16) are q heads, [Tf*16, 2*Tf*16) are k heads.
// The scale is formed as in the recurrent kernels; the normalized values stay FP32.
__global__ void __launch_bounds__(256)
    l2norm_qk_kernel(const __nv_bfloat16* __restrict__ conv_out, float* __restrict__ qn,
                     float* __restrict__ kn, std::int64_t head_rows) {
    const int lane         = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * 8 + threadIdx.x / kWarpSize;
    if (row >= 2 * head_rows) { return; }
    const bool is_key     = row >= head_rows;
    const std::int64_t r  = is_key ? row - head_rows : row;
    const std::int64_t t  = r / kQkHeads;
    const std::int64_t h  = r - t * kQkHeads;
    const std::uint32_t d = static_cast<std::uint32_t>(lane * 4);
    const gdn::RawQkLane raw = gdn::load_raw_qk_lane(
        conv_out + t * kConvChannels + (is_key ? kKeyDim : 0) + h * kHeadDim, d);
    const float scale = gdn::head_l2_scale(gdn::lane_sumsq(raw.value), lane);
    st4((is_key ? kn : qn) + r * kHeadDim + d,
        make_float4(raw.value[0] * scale, raw.value[1] * scale, raw.value[2] * scale,
                    raw.value[3] * scale));
}

// Stage 1, one CTA per (chunk, value head): decays, L, Aqk and the WY solve for U and W.
constexpr int kPrepareSmemFloats = 2 * kBT * kPad + kBT * kBT + 3 * kBT;

__global__ void __launch_bounds__(kThreads)
    chunk_prepare_kernel(const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                         const float* __restrict__ beta, ChunkBuffers ws) {
    extern __shared__ float4 smem4[];
    float* const Ks   = reinterpret_cast<float*>(smem4);  // [64][kPad]
    float* const Qs   = Ks + kBT * kPad;                    // [64][kPad]
    float* const L    = Qs + kBT * kPad;                    // [64][64]
    float* const G    = L + kBT * kBT;                      // [64]
    float* const Bt   = G + kBT;                            // [64] beta
    float* const BtEg = Bt + kBT;                           // [64] beta e^{G}

    const int tid          = static_cast<int>(threadIdx.x);
    const int chunk        = static_cast<int>(blockIdx.x);
    const int h            = static_cast<int>(blockIdx.y);
    const int qh           = h / (kValueHeads / kQkHeads);
    const std::int64_t t0  = static_cast<std::int64_t>(chunk) * kBT;

    for (int i = tid; i < kBT * (kD / 4); i += kThreads) {
        const int r  = i / (kD / 4);
        const int c4 = (i % (kD / 4)) * 4;
        const std::int64_t src = ((t0 + r) * kQkHeads + qh) * kD + c4;
        st4(Ks + r * kPad + c4, ld4(ws.kn + src));
        st4(Qs + r * kPad + c4, ld4(ws.qn + src));
    }
    if (tid < kBT) {
        Bt[tid] = beta[(t0 + tid) * kValueHeads + h];
        G[tid]  = g[(t0 + tid) * kValueHeads + h];
    }
    __syncthreads();
    if (tid == 0) {
        float running = 0.0f;
        for (int t = 0; t < kBT; ++t) {
            running += G[t];
            G[t] = running;
        }
    }
    __syncthreads();
    if (tid < kBT) {
        const float eg = expf(G[tid]);
        BtEg[tid]      = Bt[tid] * eg;
        ws.eg[(t0 + tid) * kValueHeads + h]  = eg;
        ws.ekd[(t0 + tid) * kValueHeads + h] = expf(G[kBT - 1] - G[tid]);
        if (tid == 0) { ws.dlast[chunk * kValueHeads + h] = expf(G[kBT - 1]); }
    }

    // K K^T and Q K^T: thread (ti, tj) owns rows ti + 16a and columns tj + 16b.
    {
        const int ti = tid / 16;
        const int tj = tid % 16;
        float kk[4][4] = {};
        float qk[4][4] = {};
        for (int c = 0; c < kD; c += 4) {
            float4 ki[4], qi[4], kj[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                ki[a] = ld4(Ks + (ti + 16 * a) * kPad + c);
                qi[a] = ld4(Qs + (ti + 16 * a) * kPad + c);
                kj[a] = ld4(Ks + (tj + 16 * a) * kPad + c);
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    kk[a][b] = dot4(ki[a], kj[b], kk[a][b]);
                    qk[a][b] = dot4(qi[a], kj[b], qk[a][b]);
                }
            }
        }
        float* const aqk = ws.aqk + (static_cast<std::int64_t>(chunk) * kValueHeads + h) * kBT * kBT;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int i       = ti + 16 * a;
                const int j       = tj + 16 * b;
                // Select rather than multiply: e^{G_i - G_j} may overflow above the diagonal.
                const float decay = i >= j ? expf(G[i] - G[j]) : 0.0f;
                L[i * kBT + j]    = i > j ? Bt[i] * kk[a][b] * decay : 0.0f;
                aqk[i * kBT + j]  = i >= j ? qk[a][b] * decay : 0.0f;
            }
        }
    }
    __syncthreads();

    // Forward substitution, one column per thread: columns 0..127 solve U, 128..255 solve W.
    const bool is_u = tid < kD;
    const int col   = tid & (kD - 1);
    const __nv_bfloat16* vcol = v + t0 * kConvChannels + static_cast<std::int64_t>(h) * kD + col;
    float x[kBT];
#pragma unroll
    for (int i = 0; i < kBT; ++i) {
        float rhs = is_u ? Bt[i] * __bfloat162float(vcol[static_cast<std::int64_t>(i) * kConvChannels])
                         : BtEg[i] * Ks[i * kPad + col];
#pragma unroll
        for (int j = 0; j < i; ++j) { rhs = fmaf(-L[i * kBT + j], x[j], rhs); }
        x[i] = rhs;
    }
    float* const dst = (is_u ? ws.U : ws.W) + (t0 * kValueHeads + h) * kD + col;
#pragma unroll
    for (int i = 0; i < kBT; ++i) { dst[static_cast<std::int64_t>(i) * kValueDim] = x[i]; }
}

// Stage 2, one CTA per (value head, 16-row value tile), sequential over chunks: v_new, outputs and
// the state update. The state tile stays in shared memory for the whole sequence.
constexpr int kScanSmemFloats = kVTile * kPad + kBT * kPad + kBT * kApad + kBT * kVTile + 2 * kBT;

__global__ void __launch_bounds__(kThreads)
    chunk_scan_kernel(const float* state_in, float* state_out, __nv_bfloat16* __restrict__ out,
                      ChunkBuffers ws, int chunks) {
    extern __shared__ float4 smem4[];
    float* const S   = reinterpret_cast<float*>(smem4);  // [16][kPad] state rows v0..v0+15
    float* const buf = S + kVTile * kPad;                  // [64][kPad] W, then Q e^G, then K decay
    float* const A   = buf + kBT * kPad;                   // [64][kApad]
    float* const Vn  = A + kBT * kApad;                    // [64][16]
    float* const eg  = Vn + kBT * kVTile;                  // [64]
    float* const ekd = eg + kBT;                           // [64]

    const int tid  = static_cast<int>(threadIdx.x);
    const int h    = static_cast<int>(blockIdx.x);
    const int qh   = h / (kValueHeads / kQkHeads);
    const int v0   = static_cast<int>(blockIdx.y) * kVTile;
    const std::int64_t state_base = (static_cast<std::int64_t>(h) * kD + v0) * kD;

    for (int i = tid; i < kVTile * (kD / 4); i += kThreads) {
        const int r  = i / (kD / 4);
        const int c4 = (i % (kD / 4)) * 4;
        st4(S + r * kPad + c4, ld4(state_in + state_base + r * kD + c4));
    }

    // Phases a/b: thread (t, vg) owns token t and value rows vg + 4r (phase c has its own tiling).
    const int t  = tid / 4;
    const int vg = tid % 4;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const std::int64_t t0 = static_cast<std::int64_t>(chunk) * kBT;
        for (int i = tid; i < kBT * (kD / 4); i += kThreads) {
            const int r  = i / (kD / 4);
            const int c4 = (i % (kD / 4)) * 4;
            st4(buf + r * kPad + c4, ld4(ws.W + ((t0 + r) * kValueHeads + h) * kD + c4));
        }
        if (tid < kBT) {
            eg[tid]  = ws.eg[(t0 + tid) * kValueHeads + h];
            ekd[tid] = ws.ekd[(t0 + tid) * kValueHeads + h];
        }
        float acc[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            acc[r] = ws.U[((t0 + t) * kValueHeads + h) * kD + v0 + vg + 4 * r];
        }
        __syncthreads();

        // (a) v_new = U - W S^T.
        for (int c = 0; c < kD; c += 4) {
            const float4 w = ld4(buf + t * kPad + c);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float4 s = ld4(S + (vg + 4 * r) * kPad + c);
                acc[r] = fmaf(-w.x, s.x, acc[r]);
                acc[r] = fmaf(-w.y, s.y, acc[r]);
                acc[r] = fmaf(-w.z, s.z, acc[r]);
                acc[r] = fmaf(-w.w, s.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { Vn[t * kVTile + vg + 4 * r] = acc[r]; }
        __syncthreads();

        // (b) outputs from the pre-chunk state and the intra-chunk term.
        for (int i = tid; i < kBT * (kD / 4); i += kThreads) {
            const int r  = i / (kD / 4);
            const int c4 = (i % (kD / 4)) * 4;
            st4(buf + r * kPad + c4,
                scale4(ld4(ws.qn + ((t0 + r) * kQkHeads + qh) * kD + c4), eg[r]));
        }
        const float* const aqk = ws.aqk + (static_cast<std::int64_t>(chunk) * kValueHeads + h) * kBT * kBT;
        for (int i = tid; i < kBT * (kBT / 4); i += kThreads) {
            const int r  = i / (kBT / 4);
            const int c4 = (i % (kBT / 4)) * 4;
            st4(A + r * kApad + c4, ld4(aqk + r * kBT + c4));
        }
        __syncthreads();
        float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = 0; c < kD; c += 4) {
            const float4 q = ld4(buf + t * kPad + c);
#pragma unroll
            for (int r = 0; r < 4; ++r) { o[r] = dot4(q, ld4(S + (vg + 4 * r) * kPad + c), o[r]); }
        }
        for (int s = 0; s <= t; s += 4) {
            const float4 a = ld4(A + t * kApad + s);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int vr = vg + 4 * r;
                o[r] = fmaf(a.x, Vn[(s + 0) * kVTile + vr], o[r]);
                o[r] = fmaf(a.y, Vn[(s + 1) * kVTile + vr], o[r]);
                o[r] = fmaf(a.z, Vn[(s + 2) * kVTile + vr], o[r]);
                o[r] = fmaf(a.w, Vn[(s + 3) * kVTile + vr], o[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            out[(t0 + t) * kValueDim + static_cast<std::int64_t>(h) * kD + v0 + vg + 4 * r] =
                __float2bfloat16_rn(o[r] * gdn::kOutputScale);
        }
        __syncthreads();

        // (c) S = S e^{G_63} + sum_t v_new_t (k_t e^{G_63 - G_t})^T.
        for (int i = tid; i < kBT * (kD / 4); i += kThreads) {
            const int r  = i / (kD / 4);
            const int c4 = (i % (kD / 4)) * 4;
            st4(buf + r * kPad + c4,
                scale4(ld4(ws.kn + ((t0 + r) * kQkHeads + qh) * kD + c4), ekd[r]));
        }
        const float dl = ws.dlast[chunk * kValueHeads + h];
        __syncthreads();
        {
            // Thread (vq, kq, half): state rows 4vq..4vq+3, key columns 4kq..4kq+3, tokens
            // [32 half, 32 half + 32). The upper half's partial sums go through A (free after
            // phase b) and the lower half adds them after its own tokens.
            const int vq   = tid / 64;
            const int kq   = (tid / 2) % 32;
            const int half = tid % 2;
            float4 st[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                st[r] = half == 0 ? scale4(ld4(S + (4 * vq + r) * kPad + 4 * kq), dl)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll 4
            for (int i = 0; i < kBT / 2; ++i) {
                const int tt    = half * (kBT / 2) + i;
                const float4 kd = ld4(buf + tt * kPad + 4 * kq);
                const float4 vn = ld4(Vn + tt * kVTile + 4 * vq);
                const float vr[4] = {vn.x, vn.y, vn.z, vn.w};
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    st[r].x = fmaf(kd.x, vr[r], st[r].x);
                    st[r].y = fmaf(kd.y, vr[r], st[r].y);
                    st[r].z = fmaf(kd.z, vr[r], st[r].z);
                    st[r].w = fmaf(kd.w, vr[r], st[r].w);
                }
            }
            float* const partial = A;  // [16][128]
            if (half == 1) {
#pragma unroll
                for (int r = 0; r < 4; ++r) { st4(partial + (4 * vq + r) * kD + 4 * kq, st[r]); }
            }
            __syncthreads();
            if (half == 0) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float4 p = ld4(partial + (4 * vq + r) * kD + 4 * kq);
                    st4(S + (4 * vq + r) * kPad + 4 * kq,
                        make_float4(st[r].x + p.x, st[r].y + p.y, st[r].z + p.z, st[r].w + p.w));
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < kVTile * (kD / 4); i += kThreads) {
        const int r  = i / (kD / 4);
        const int c4 = (i % (kD / 4)) * 4;
        st4(state_out + state_base + r * kD + c4, ld4(S + r * kPad + c4));
    }
}

} // namespace

std::size_t gdn_prefill_workspace_bytes(std::int32_t tokens) {
    require(tokens >= 1, "tokens must be positive");
    return plan_prefill(tokens).total;
}

void gdn_prefill(const __nv_bfloat16* conv_out, const float* g, const float* beta,
                 const float* state_in, float* state_out, __nv_bfloat16* core,
                 std::int32_t tokens, void* workspace, std::size_t workspace_bytes,
                 cudaStream_t stream) {
    require(tokens >= 1, "tokens must be positive");
    require(aligned(conv_out, 16) && aligned(g, 4) && aligned(beta, 4) && aligned(state_in, 16) &&
                aligned(state_out, 16) && aligned(core, 16),
            "invalid tensor pointer");
    const PrefillLayout layout = plan_prefill(tokens);
    const std::int32_t full    = tokens / kChunk * kChunk;
    const std::int32_t tail    = tokens - full;

    if (full > 0) {
        require(workspace_bytes >= layout.total, "workspace too small");
        ChunkBuffers ws{};
        ws.qn    = static_cast<float*>(layout.qn.bind(workspace, workspace_bytes));
        ws.kn    = static_cast<float*>(layout.kn.bind(workspace, workspace_bytes));
        ws.U     = static_cast<float*>(layout.U.bind(workspace, workspace_bytes));
        ws.W     = static_cast<float*>(layout.W.bind(workspace, workspace_bytes));
        ws.aqk   = static_cast<float*>(layout.aqk.bind(workspace, workspace_bytes));
        ws.eg    = static_cast<float*>(layout.eg.bind(workspace, workspace_bytes));
        ws.ekd   = static_cast<float*>(layout.ekd.bind(workspace, workspace_bytes));
        ws.dlast = static_cast<float*>(layout.dlast.bind(workspace, workspace_bytes));
        const int chunks = full / kChunk;

        const std::int64_t head_rows = static_cast<std::int64_t>(full) * kQkHeads;
        l2norm_qk_kernel<<<static_cast<unsigned>((2 * head_rows + 7) / 8), 256, 0, stream>>>(
            conv_out, ws.qn, ws.kn, head_rows);
        CUDA_CHECK(cudaGetLastError());

        constexpr int prepare_smem = kPrepareSmemFloats * static_cast<int>(sizeof(float));
        CUDA_CHECK(cudaFuncSetAttribute(chunk_prepare_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, prepare_smem));
        // Without a full carveout the driver may size the L1/shared split for one CTA per SM.
        CUDA_CHECK(cudaFuncSetAttribute(chunk_prepare_kernel,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        chunk_prepare_kernel<<<dim3(chunks, kValueHeads), kThreads, prepare_smem, stream>>>(
            conv_out + 2 * kKeyDim, g, beta, ws);
        CUDA_CHECK(cudaGetLastError());

        constexpr int scan_smem = kScanSmemFloats * static_cast<int>(sizeof(float));
        CUDA_CHECK(cudaFuncSetAttribute(chunk_scan_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, scan_smem));
        CUDA_CHECK(cudaFuncSetAttribute(chunk_scan_kernel,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        chunk_scan_kernel<<<dim3(kValueHeads, kHeadDim / kVTile), kThreads, scan_smem, stream>>>(
            state_in, state_out, core, ws, chunks);
        CUDA_CHECK(cudaGetLastError());
    }

    if (tail > 0) {
        gdn::DirectParams p{};
        const __nv_bfloat16* rows = conv_out + static_cast<std::int64_t>(full) * kConvChannels;
        p.q           = rows;
        p.k           = rows + kKeyDim;
        p.qk_ld       = kConvChannels;
        p.v           = rows + 2 * kKeyDim;
        p.v_ld        = kConvChannels;
        p.g           = g + static_cast<std::int64_t>(full) * kValueHeads;
        p.beta        = beta + static_cast<std::int64_t>(full) * kValueHeads;
        // After full chunks the running state lives in state_out.
        p.state_read  = full > 0 ? state_out : state_in;
        p.state_write = state_out;
        p.out         = core + static_cast<std::int64_t>(full) * kValueDim;
        p.width       = tail;
        gdn::gdn_direct_kernel<<<dim3(kValueHeads, 1, kHeadDim / gdn::kBlockDv),
                                 dim3(kWarpSize, gdn::kNumWarps), 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace fngine::kernels
