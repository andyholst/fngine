// BF16 projections: CUDA-core GEMV for T <= 16, cuBLASLt (BF16 x BF16 -> FP32 accumulate) plus
// an epilogue kernel above that. Serves the router / gdn.ab / idx_qk weights in every mode and
// all FP8-registered matrices in `--dense bf16` validation mode.

#include "core/device.h"
#include "kernels/common/math.h"
#include "kernels/linear/bf16/bf16_gemv.cuh"
#include "kernels/linear/linear.h"
#include "kernels/linear/linear_output.cuh"
#include "kernels/linear/linear_params.h"

#include <cublasLt.h>

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace fngine::kernels {

namespace {

void cublas_check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

constexpr std::size_t kCublasWorkspaceBytes = 32u << 20;

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

} // namespace

struct CublasLtContext::Impl {
    cublasLtHandle_t handle = nullptr;
    // (n, k, T, output type) -> chosen algorithm.
    std::map<std::tuple<int, int, int, int>, cublasLtMatmulAlgo_t> algos;
};

CublasLtContext::CublasLtContext() : impl_(new Impl) {
    try {
        cublas_check(cublasLtCreate(&impl_->handle), "create");
    } catch (...) {
        delete impl_;
        throw;
    }
}

CublasLtContext::~CublasLtContext() {
    if (impl_->handle) cublasLtDestroy(impl_->handle);
    delete impl_;
}

namespace {

using detail::EpilogueTraits;

// GEMV schedule: rows per warp, row warps, K-split warps.
struct GemvConfig {
    int rows, row_warps, ksplit;
};

using GemvFn = void (*)(const __nv_bfloat16*, std::int32_t, const Bf16Weight&,
                        const LinearEpilogueParams&, cudaStream_t);

template <int Tokens, int R, int RW, int KS, LinearEpilogue Kind>
void launch_gemv_tokens(const __nv_bfloat16* x, std::int32_t tokens, const Bf16Weight& w,
                        const LinearEpilogueParams& ep, cudaStream_t stream) {
    using Traits         = EpilogueTraits<Kind>;
    constexpr bool kPair = Traits::kPair;
    constexpr int kStoredPerCta = RW * (kPair ? R / 2 : R);
    using Rows = std::conditional_t<kPair, detail::Bf16PairRows<R / 2>, detail::Bf16IdentityRows>;
    Rows rows{};
    if constexpr (kPair) rows.intermediate = w.n / 2;
    const int stored = kPair ? w.n / 2 : w.n;
    const int blocks = div_up(stored, kStoredPerCta);
    CUDA_CHECK(pdl_launch(detail::bf16_gemv_kernel<Tokens, R, RW, KS, typename Traits::Output, Rows, kPair>,
                          dim3(blocks), dim3(RW * KS * 32), 0, stream, x, w.data, w.n, w.k, tokens,
                          Traits::make(ep, w.n), rows));
}

template <int R, int RW, int KS, LinearEpilogue Kind>
void launch_gemv(const __nv_bfloat16* x, std::int32_t tokens, const Bf16Weight& w,
                 const LinearEpilogueParams& ep, cudaStream_t stream) {
    if (tokens == 1) return launch_gemv_tokens<1, R, RW, KS, Kind>(x, tokens, w, ep, stream);
    if (tokens == 2) return launch_gemv_tokens<2, R, RW, KS, Kind>(x, tokens, w, ep, stream);
    if (tokens <= 4) return launch_gemv_tokens<4, R, RW, KS, Kind>(x, tokens, w, ep, stream);
    if (tokens <= 8) return launch_gemv_tokens<8, R, RW, KS, Kind>(x, tokens, w, ep, stream);
    return launch_gemv_tokens<16, R, RW, KS, Kind>(x, tokens, w, ep, stream);
}

template <LinearEpilogue Kind>
void launch_epilogue(const float* acc, std::int32_t tokens, std::int32_t n,
                     const LinearEpilogueParams& ep, cudaStream_t stream) {
    using Traits         = EpilogueTraits<Kind>;
    using Output         = typename Traits::Output;
    constexpr bool kPair = Traits::kPair;
    const int units      = kPair ? n / 2 : (detail::output_row_group<Output>() == 4 ? n / 4 : n);
    const dim3 grid(static_cast<unsigned>(div_up(units, 256)), static_cast<unsigned>(tokens));
    detail::bf16_epilogue_kernel<Output, kPair>
        <<<grid, 256, 0, stream>>>(acc, n, units, Traits::make(ep, n));
    CUDA_CHECK(cudaGetLastError());
}

using EpilogueFn = void (*)(const float*, std::int32_t, std::int32_t, const LinearEpilogueParams&,
                            cudaStream_t);

struct Bf16Entry {
    std::int32_t n, k;
    LinearEpilogue epilogue;
    GemvFn gemv;
    EpilogueFn finish;  // null: cuBLASLt writes the output directly
};

constexpr auto S = LinearEpilogue::StoreBf16;
constexpr auto F = LinearEpilogue::StoreF32;

// The GEMV schedules: wide-N shapes give each warp two rows; narrow-N shapes split K over warps.
const std::array kEntries{
    Bf16Entry{16384, 2560, S, launch_gemv<2, 4, 1, S>, nullptr},
    Bf16Entry{13312, 2560, S, launch_gemv<2, 4, 1, S>, nullptr},
    Bf16Entry{2560, 6144, LinearEpilogue::GrWrite, launch_gemv<2, 2, 2, LinearEpilogue::GrWrite>,
              launch_epilogue<LinearEpilogue::GrWrite>},
    Bf16Entry{1280, 2560, LinearEpilogue::SwiGlu, launch_gemv<2, 2, 2, LinearEpilogue::SwiGlu>,
              launch_epilogue<LinearEpilogue::SwiGlu>},
    Bf16Entry{2560, 640, LinearEpilogue::SharedScale,
              launch_gemv<2, 4, 1, LinearEpilogue::SharedScale>,
              launch_epilogue<LinearEpilogue::SharedScale>},
    Bf16Entry{320, 10240, LinearEpilogue::SiluQuarter,
              launch_gemv<2, 1, 8, LinearEpilogue::SiluQuarter>,
              launch_epilogue<LinearEpilogue::SiluQuarter>},
    Bf16Entry{10240, 384, LinearEpilogue::GrMix, launch_gemv<4, 4, 1, LinearEpilogue::GrMix>,
              launch_epilogue<LinearEpilogue::GrMix>},
    Bf16Entry{10240, 2560, S, launch_gemv<2, 4, 1, S>, nullptr},
    Bf16Entry{2560, 2560, S, launch_gemv<2, 2, 2, S>, nullptr},
    Bf16Entry{248320, 2560, F, launch_gemv<2, 4, 1, F>, nullptr},
    Bf16Entry{513, 2560, F, launch_gemv<1, 2, 4, F>, nullptr},
    Bf16Entry{96, 2560, F, launch_gemv<1, 2, 4, F>, nullptr},
    Bf16Entry{640, 2560, S, launch_gemv<1, 2, 4, S>, nullptr},
};

const Bf16Entry* find_entry(std::int32_t n, std::int32_t k) {
    for (const auto& e : kEntries)
        if (e.n == n && e.k == k) return &e;
    return nullptr;
}

const Bf16Entry& require_entry(std::int32_t n, std::int32_t k, LinearEpilogue epilogue) {
    const auto* e = find_entry(n, k);
    if (e == nullptr || e->epilogue != epilogue) {
        throw std::invalid_argument(std::string("linear_bf16: unregistered N=") +
                                    std::to_string(n) + " K=" + std::to_string(k) + " " +
                                    linear_epilogue_name(epilogue));
    }
    return *e;
}

bool aligned(const void* p, std::uintptr_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}

const cublasLtMatmulAlgo_t& choose_algo(CublasLtContext::Impl& ctx, cublasLtMatmulDesc_t op,
                                        cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
                                        cublasLtMatrixLayout_t d, int n, int k, int tokens,
                                        int out_type) {
    const auto key = std::make_tuple(n, k, tokens, out_type);
    auto it        = ctx.algos.find(key);
    if (it != ctx.algos.end()) return it->second;
    cublasLtMatmulPreference_t preference = nullptr;
    cublas_check(cublasLtMatmulPreferenceCreate(&preference), "preference create");
    const std::uint64_t workspace = kCublasWorkspaceBytes;
    cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace, sizeof(workspace)),
                 "preference workspace");
    // Deterministic results: no in-place (atomic) split-K reduction.
    const std::uint32_t reduction_mask =
        CUBLASLT_REDUCTION_SCHEME_MASK & ~static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_INPLACE);
    cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                      &reduction_mask, sizeof(reduction_mask)),
                 "preference reduction scheme");
    cublasLtMatmulHeuristicResult_t result{};
    int found               = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(ctx.handle, op, a, b, d, d,
                                                             preference, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(preference);
    cublas_check(st, "heuristic");
    if (found == 0) throw std::runtime_error("cuBLASLt: no algorithm for BF16 linear");
    return ctx.algos.emplace(key, result.algo).first->second;
}

// D[T][n] (row-major, i.e. column-major n x T) = x[T][k] . W[n][k]^T.
void cublas_gemm(CublasLtContext::Impl& ctx, const __nv_bfloat16* x, std::int32_t tokens,
                 const Bf16Weight& w, void* d, cudaDataType_t d_type, void* scratch,
                 cudaStream_t stream) {
    cublasLtMatmulDesc_t op = nullptr;
    cublas_check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, dl = nullptr;
    try {
        cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                                    sizeof(trans_a)),
                     "transa");
        cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                                    sizeof(trans_b)),
                     "transb");
        cublas_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, w.k, w.n, w.k), "layout a");
        cublas_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, w.k, tokens, w.k), "layout b");
        cublas_check(cublasLtMatrixLayoutCreate(&dl, d_type, w.n, tokens, w.n), "layout d");
        const auto& algo =
            choose_algo(ctx, op, a, b, dl, w.n, w.k, tokens, static_cast<int>(d_type));
        const float alpha = 1.0F;
        const float beta  = 0.0F;
        cublas_check(cublasLtMatmul(ctx.handle, op, &alpha, w.data, a, x, b, &beta, d, dl, d, dl,
                                    &algo, scratch, kCublasWorkspaceBytes, stream),
                     "matmul");
    } catch (...) {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (dl) cublasLtMatrixLayoutDestroy(dl);
        cublasLtMatmulDescDestroy(op);
        throw;
    }
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(dl);
    cublasLtMatmulDescDestroy(op);
}

} // namespace

bool linear_bf16_registered(std::int32_t n, std::int32_t k, LinearEpilogue epilogue) {
    const auto* e = find_entry(n, k);
    return e != nullptr && e->epilogue == epilogue;
}

std::size_t linear_bf16_workspace_bytes(std::int32_t n, std::int32_t k, LinearEpilogue epilogue,
                                        std::int32_t max_tokens) {
    if (max_tokens <= 0) throw std::invalid_argument("linear_bf16_workspace_bytes: max_tokens <= 0");
    const auto& e = require_entry(n, k, epilogue);
    if (max_tokens <= kLinearBf16GemvMaxTokens) return 0;
    std::size_t bytes = kCublasWorkspaceBytes;
    if (e.finish != nullptr) {
        bytes += align256(static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(n) *
                          sizeof(float));
    }
    return bytes;
}

void linear_bf16(const __nv_bfloat16* x, std::int32_t tokens, const Bf16Weight& weight,
                 const LinearEpilogueParams& epilogue, CublasLtContext* cublas, void* workspace,
                 std::size_t workspace_bytes, cudaStream_t stream) {
    const auto& e = require_entry(weight.n, weight.k, epilogue.kind);
    if (tokens <= 0) throw std::invalid_argument("linear_bf16: tokens must be positive");
    if (x == nullptr || weight.data == nullptr) {
        throw std::invalid_argument("linear_bf16: null input or weight");
    }
    if (!aligned(x, 16) || !aligned(weight.data, 16)) {
        throw std::invalid_argument("linear_bf16: misaligned input or weight");
    }
    detail::require_epilogue_pointers(epilogue);
    if (tokens <= kLinearBf16GemvMaxTokens) {
        e.gemv(x, tokens, weight, epilogue, stream);
        return;
    }
    if (cublas == nullptr) throw std::invalid_argument("linear_bf16: T > 16 needs a cuBLASLt context");
    const std::size_t need = linear_bf16_workspace_bytes(weight.n, weight.k, epilogue.kind, tokens);
    if (workspace == nullptr || workspace_bytes < need || !aligned(workspace, 256)) {
        throw std::invalid_argument("linear_bf16: workspace missing, too small or misaligned");
    }
    auto* scratch = static_cast<std::uint8_t*>(workspace);
    if (e.finish == nullptr) {
        const cudaDataType_t type =
            epilogue.kind == LinearEpilogue::StoreF32 ? CUDA_R_32F : CUDA_R_16BF;
        cublas_gemm(*cublas->impl(), x, tokens, weight, epilogue.out, type, scratch, stream);
        return;
    }
    auto* acc = reinterpret_cast<float*>(scratch + kCublasWorkspaceBytes);
    cublas_gemm(*cublas->impl(), x, tokens, weight, acc, CUDA_R_32F, scratch, stream);
    e.finish(acc, tokens, weight.n, epilogue, stream);
}

} // namespace fngine::kernels
