// Modified for fngine from cinference:src/ops/linear/fp8/fp8_dispatch.cpp @ 383e5db. See NOTICE.
// fngine: (N, K, epilogue) registry with measured A8 thresholds and forced-path selection.

#include "kernels/linear/fp8/fp8_a8.h"
#include "kernels/linear/fp8/fp8_shapes.h"
#include "kernels/linear/linear.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace fngine::kernels {
namespace {

using detail::Fp8ShapeEntry;

const std::array kShapes{&detail::kFp8N16384K2560, &detail::kFp8N13312K2560,
                         &detail::kFp8N2560K6144,  &detail::kFp8N1280K2560,
                         &detail::kFp8N2560K640,   &detail::kFp8N320K10240,
                         &detail::kFp8N10240K384,  &detail::kFp8N10240K2560,
                         &detail::kFp8N2560K2560,  &detail::kFp8N248320K2560,
                         &detail::kFp8N32768K2560};

const Fp8ShapeEntry* find_shape(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return shape;
    return nullptr;
}

const Fp8ShapeEntry& require_shape(std::int32_t n, std::int32_t k) {
    const auto* shape = find_shape(n, k);
    if (shape == nullptr) {
        throw std::invalid_argument("linear_fp8: unregistered shape N=" + std::to_string(n) +
                                    " K=" + std::to_string(k));
    }
    return *shape;
}

bool aligned(const void* p, std::uintptr_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}

} // namespace

const char* linear_epilogue_name(LinearEpilogue epilogue) {
    switch (epilogue) {
    case LinearEpilogue::StoreBf16: return "StoreBf16";
    case LinearEpilogue::StoreF32: return "StoreF32";
    case LinearEpilogue::GrWrite: return "GrWrite";
    case LinearEpilogue::SwiGlu: return "SwiGlu";
    case LinearEpilogue::SharedScale: return "SharedScale";
    case LinearEpilogue::SiluQuarter: return "SiluQuarter";
    case LinearEpilogue::GrMix: return "GrMix";
    }
    return "?";
}

bool linear_fp8_registered(std::int32_t n, std::int32_t k, LinearEpilogue epilogue) {
    const auto* shape = find_shape(n, k);
    return shape != nullptr && shape->epilogue == epilogue;
}

std::int32_t linear_fp8_a8_min_tokens(std::int32_t n, std::int32_t k) {
    const auto& shape = require_shape(n, k);
    return shape.a8 == nullptr ? std::numeric_limits<std::int32_t>::max() : shape.a8_min_tokens;
}

bool linear_fp8_has_a8(std::int32_t n, std::int32_t k) { return require_shape(n, k).a8 != nullptr; }

std::size_t linear_fp8_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    if (max_tokens <= 0) throw std::invalid_argument("linear_fp8_workspace_bytes: max_tokens <= 0");
    const auto& shape = require_shape(n, k);
    // Covers a forced A8 call at any T as well as the automatic choice.
    return shape.a8 == nullptr ? 0 : detail::fp8_a8_workspace_bytes(max_tokens, k);
}

void linear_fp8(const __nv_bfloat16* x, std::int32_t tokens, const Fp8RowWeight& weight,
                const LinearEpilogueParams& epilogue, void* workspace, std::size_t workspace_bytes,
                cudaStream_t stream, Fp8LinearPath path) {
    const auto& shape = require_shape(weight.n, weight.k);
    if (shape.epilogue != epilogue.kind) {
        throw std::invalid_argument(std::string("linear_fp8: N=") + std::to_string(weight.n) +
                                    " K=" + std::to_string(weight.k) + " is registered with " +
                                    linear_epilogue_name(shape.epilogue) + ", not " +
                                    linear_epilogue_name(epilogue.kind));
    }
    if (tokens <= 0) throw std::invalid_argument("linear_fp8: tokens must be positive");
    if (x == nullptr || weight.codes == nullptr || weight.scales == nullptr) {
        throw std::invalid_argument("linear_fp8: null input or weight");
    }
    if (!aligned(x, 16) || !aligned(weight.codes, 16) || !aligned(weight.scales, 4)) {
        throw std::invalid_argument("linear_fp8: misaligned input or weight");
    }
    detail::require_epilogue_pointers(epilogue);

    bool use_a8 = false;
    switch (path) {
    case Fp8LinearPath::Auto: use_a8 = shape.a8 != nullptr && tokens >= shape.a8_min_tokens; break;
    case Fp8LinearPath::A16: use_a8 = false; break;
    case Fp8LinearPath::A8:
        if (shape.a8 == nullptr) throw std::invalid_argument("linear_fp8: shape has no A8 path");
        use_a8 = true;
        break;
    }

    detail::Fp8Call call{x, tokens, weight.codes, weight.scales, &epilogue, {}, stream};
    if (use_a8) {
        call.a8 = detail::fp8_a8_workspace(workspace, workspace_bytes, tokens, weight.k);
        shape.a8(call);
    } else {
        shape.a16(call);
    }
}

} // namespace fngine::kernels
