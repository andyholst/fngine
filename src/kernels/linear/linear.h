#pragma once

// Dense projections of Qwen3.8-Flash-Next: row-scaled FP8 (E4M3 codes + BF16 row scales) and BF16
// weights, each fused with the epilogue its consumer needs.
//
// Closed registry. Every call names (N, K, epilogue); an unregistered triple throws
// std::invalid_argument. FP8 (pack format fp8_e4m3_row):
//   (16384, 2560) StoreBf16    gdn.qkvz            out bf16 [T][16384]
//   (13312, 2560) StoreBf16    qsa.qkv             out bf16 [T][13312]
//   ( 2560, 6144) GrWrite      gdn.out / qsa.o     X[t][s][n] += inj[t][s] * acc[t][n]
//   ( 1280, 2560) SwiGlu       moe.shared_gate_up  h bf16 [T][640] = silu(acc[j]) * acc[640 + j]
//   ( 2560,  640) SharedScale  moe.shared_down     y f32 [T][2560] = sigmoid(gate_logit[t]) * acc
//   (  320,10240) SiluQuarter  gr down             r bf16 [T][384] = silu(acc / 4); cols 320..383
//                                                  are never written (the caller zeroes them once)
//   (10240,  384) GrMix        gr up (permuted)    mix bf16 [T][2560], see GrMix below
//   (10240, 2560) StoreBf16    ple.key
//   ( 2560, 2560) StoreBf16    ple.value, mtp.fc_embedding, mtp.fc_hidden
//   (248320,2560) StoreF32     lm_head             logits f32 [T][248320]
// BF16 (Bf16Weight): every FP8 entry above (the `--dense bf16` validation copies) plus
//   (  513, 2560) StoreF32     moe.router (+ shared gate logit in col 512)
//   (   96, 2560) StoreF32     gdn.ab
//   (  640, 2560) StoreBf16    qsa.idx_qk
//
// Layouts are token-major: x is BF16 [T][K]; outputs as listed (leading dimension = row count of
// the listed output, 384 for SiluQuarter). Every output element is written exactly once by one
// thread after the complete K reduction, so all paths are deterministic and GrWrite's
// read-modify-write of X never races.
//
// GrMix: output row group 4d..4d+3 of the permuted GR up projection belongs to dim d:
//   mix[t][d] = bf16(0.25 * sum_s sigmoid(acc[t][4d + s]) * Xn[t][s*2560 + d]),  Xn bf16 [T][10240].
//
// Paths (FP8). A16 keeps BF16 activations and widens the E4M3 weights exactly: T = 1 GEMV, T <= 16
// CUDA-core SIMT or tensor-core K-split MMA, T > 16 a tiled BF16 tensor-core GEMM (FP32
// accumulation, FP32-staged epilogue). A8 quantizes each token to E4M3 with a per-token FP32
// scale (max|x| / 448) into the caller workspace, then runs the E4M3 x E4M3 MMA (1-4% rms output
// noise per projection; about 1.3-2x faster than the A16 GEMM at T = 4096). Auto = A16 at every T
// (prefill accuracy: mean KLD 0.049 -> 0.034 on the 8192-token golden); A8 only when forced.
// lm_head has no A8 path.
// BF16 weights: GEMV for T <= 16, cuBLASLt (BF16 x BF16 -> FP32) plus an epilogue kernel above
// (StoreBf16 / StoreF32 are written by cuBLASLt directly).
//
// Determinism: every path reduces in a fixed order (no atomics), so repeated calls with the same
// T are bitwise identical. Different T may take different kernels (different FP32 summation
// order).
//
// Graphs. Kernels never allocate or synchronize and are capturable. `tokens` is a host scalar
// baked into the launch: decode/verify graphs are captured once per width, prefill is eager.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fngine::kernels {

struct Fp8RowWeight {
    const std::uint8_t* codes     = nullptr;  // E4M3FN [n][k], 16-byte aligned
    const __nv_bfloat16* scales   = nullptr;  // BF16 [n], 4-byte aligned
    std::int32_t n                = 0;
    std::int32_t k                = 0;
};

struct Bf16Weight {
    const __nv_bfloat16* data = nullptr;  // BF16 [n][k], 16-byte aligned
    std::int32_t n            = 0;
    std::int32_t k            = 0;
};

enum class LinearEpilogue : std::uint8_t {
    StoreBf16,
    StoreF32,
    GrWrite,
    SwiGlu,
    SharedScale,
    SiluQuarter,
    GrMix,
};

const char* linear_epilogue_name(LinearEpilogue epilogue);

// Output side of one projection. Fields not used by `kind` are ignored.
struct LinearEpilogueParams {
    LinearEpilogue kind = LinearEpilogue::StoreBf16;
    // StoreBf16 / SwiGlu / SiluQuarter / GrMix: __nv_bfloat16*; StoreF32 / SharedScale: float*;
    // GrWrite: the FP32 residual streams X [T][4][n] updated in place.
    void* out = nullptr;
    const float* inj = nullptr;            // GrWrite: FP32 [T][4]
    const float* gate_logit = nullptr;     // SharedScale: logit of token t at gate_logit[t * gate_stride]
    std::int32_t gate_stride = 0;
    const __nv_bfloat16* xn = nullptr;     // GrMix: BF16 [T][4 * n / 4] = [T][10240]
};

inline LinearEpilogueParams linear_store_bf16(__nv_bfloat16* out) {
    return {LinearEpilogue::StoreBf16, out};
}
inline LinearEpilogueParams linear_store_f32(float* out) { return {LinearEpilogue::StoreF32, out}; }
inline LinearEpilogueParams linear_gr_write(float* x, const float* inj) {
    return {LinearEpilogue::GrWrite, x, inj};
}
inline LinearEpilogueParams linear_swiglu(__nv_bfloat16* h) { return {LinearEpilogue::SwiGlu, h}; }
inline LinearEpilogueParams linear_shared_scale(float* y, const float* gate_logit,
                                                std::int32_t gate_stride) {
    return {LinearEpilogue::SharedScale, y, nullptr, gate_logit, gate_stride};
}
inline LinearEpilogueParams linear_silu_quarter(__nv_bfloat16* r) {
    return {LinearEpilogue::SiluQuarter, r};
}
inline LinearEpilogueParams linear_gr_mix(__nv_bfloat16* mix, const __nv_bfloat16* xn) {
    return {LinearEpilogue::GrMix, mix, nullptr, nullptr, 0, xn};
}

// ---------------------------------------------------------------------------------------------
// FP8 row-scaled weights.

enum class Fp8LinearPath : std::uint8_t {
    Auto,  // measured per-shape threshold
    A16,   // force BF16 activations (every registered shape)
    A8,    // force E4M3 activations (throws for lm_head, which has no A8 path)
};

// True when (n, k, epilogue) is in the FP8 registry.
bool linear_fp8_registered(std::int32_t n, std::int32_t k, LinearEpilogue epilogue);

// Smallest T at which Auto selects A8 for this shape (INT32_MAX when never).
std::int32_t linear_fp8_a8_min_tokens(std::int32_t n, std::int32_t k);

// True when Fp8LinearPath::A8 is available for this shape (every shape except lm_head).
bool linear_fp8_has_a8(std::int32_t n, std::int32_t k);

// Device workspace bytes needed by linear_fp8 for any T in [1, max_tokens] on any path
// (A8 activation codes [T][k] + per-token scales). Zero when no path needs workspace.
std::size_t linear_fp8_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens);

// out = epilogue(x [tokens][k] . W^T). workspace: device memory, 256-byte aligned, of at least
// linear_fp8_workspace_bytes(n, k, tokens) bytes (may be null when that is zero).
void linear_fp8(const __nv_bfloat16* x, std::int32_t tokens, const Fp8RowWeight& weight,
                const LinearEpilogueParams& epilogue, void* workspace, std::size_t workspace_bytes,
                cudaStream_t stream, Fp8LinearPath path = Fp8LinearPath::Auto);

// ---------------------------------------------------------------------------------------------
// BF16 weights.

// Owns the cuBLASLt handle used by linear_bf16 for T > 16. Create once per device/thread;
// not thread-safe (it caches algorithm choices per problem).
class CublasLtContext {
public:
    CublasLtContext();
    ~CublasLtContext();
    CublasLtContext(const CublasLtContext&)            = delete;
    CublasLtContext& operator=(const CublasLtContext&) = delete;

    struct Impl;
    Impl* impl() const { return impl_; }

private:
    Impl* impl_;
};

inline constexpr std::int32_t kLinearBf16GemvMaxTokens = 16;

bool linear_bf16_registered(std::int32_t n, std::int32_t k, LinearEpilogue epilogue);

// Device workspace for any T in [1, max_tokens]: cuBLASLt scratch plus, for epilogues other than
// StoreBf16/StoreF32, the FP32 [T][n] GEMM result. Zero when max_tokens <= 16.
std::size_t linear_bf16_workspace_bytes(std::int32_t n, std::int32_t k, LinearEpilogue epilogue,
                                        std::int32_t max_tokens);

// out = epilogue(x [tokens][k] . W^T). `cublas` may be null when tokens <= 16.
void linear_bf16(const __nv_bfloat16* x, std::int32_t tokens, const Bf16Weight& weight,
                 const LinearEpilogueParams& epilogue, CublasLtContext* cublas, void* workspace,
                 std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace fngine::kernels
