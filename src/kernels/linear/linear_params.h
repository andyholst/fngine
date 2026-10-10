#pragma once

// Host-side helpers over LinearEpilogueParams shared by the FP8 and BF16 dispatchers.

#include "kernels/linear/linear.h"

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels::detail {

inline void require_epilogue_pointers(const LinearEpilogueParams& p) {
    if (p.out == nullptr) throw std::invalid_argument("linear: epilogue output is null");
    if (p.kind == LinearEpilogue::GrWrite && p.inj == nullptr)
        throw std::invalid_argument("linear: GrWrite requires inj");
    if (p.kind == LinearEpilogue::SharedScale && p.gate_logit == nullptr)
        throw std::invalid_argument("linear: SharedScale requires gate_logit");
    if (p.kind == LinearEpilogue::GrMix && p.xn == nullptr)
        throw std::invalid_argument("linear: GrMix requires xn");
}

// Epilogue parameters of the token range starting at `offset` (n = weight rows).
inline LinearEpilogueParams offset_epilogue_tokens(const LinearEpilogueParams& p,
                                                   std::int32_t offset, std::int32_t n) {
    LinearEpilogueParams r = p;
    const std::int64_t t   = offset;
    switch (p.kind) {
    case LinearEpilogue::StoreBf16: r.out = static_cast<__nv_bfloat16*>(p.out) + t * n; break;
    case LinearEpilogue::StoreF32: r.out = static_cast<float*>(p.out) + t * n; break;
    case LinearEpilogue::GrWrite:
        r.out = static_cast<float*>(p.out) + t * 4 * n;
        r.inj = p.inj + t * 4;
        break;
    case LinearEpilogue::SwiGlu: r.out = static_cast<__nv_bfloat16*>(p.out) + t * (n / 2); break;
    case LinearEpilogue::SharedScale:
        r.out        = static_cast<float*>(p.out) + t * n;
        r.gate_logit = p.gate_logit + t * p.gate_stride;
        break;
    case LinearEpilogue::SiluQuarter: r.out = static_cast<__nv_bfloat16*>(p.out) + t * 384; break;
    case LinearEpilogue::GrMix:
        r.out = static_cast<__nv_bfloat16*>(p.out) + t * (n / 4);
        r.xn  = p.xn + t * n;
        break;
    }
    return r;
}

} // namespace fngine::kernels::detail
