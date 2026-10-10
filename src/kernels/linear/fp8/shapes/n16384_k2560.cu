// [16384,2560] BF16-store projection (gdn.qkvz); schedules in k2560_store.cuh.

#include "kernels/linear/fp8/shapes/k2560_store.cuh"

namespace fngine::kernels::detail {
const Fp8ShapeEntry kFp8N16384K2560{16384, 2560, k2560_store::kKind, k2560_store::launch_a16<16384>,
                                  k2560_store::launch_a8<16384, 32>, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
