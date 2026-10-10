// [13312,2560] BF16-store projection (qsa.qkv); schedules in k2560_store.cuh.

#include "kernels/linear/fp8/shapes/k2560_store.cuh"

namespace fngine::kernels::detail {
const Fp8ShapeEntry kFp8N13312K2560{13312, 2560, k2560_store::kKind, k2560_store::launch_a16<13312>,
                                  k2560_store::launch_a8<13312, 32>, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
