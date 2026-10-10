// [10240,2560] BF16-store projection (ple.key); schedules in k2560_store.cuh.

#include "kernels/linear/fp8/shapes/k2560_store.cuh"

namespace fngine::kernels::detail {
const Fp8ShapeEntry kFp8N10240K2560{10240, 2560, k2560_store::kKind, k2560_store::launch_a16<10240>,
                                  k2560_store::launch_a8<10240, 32>, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
