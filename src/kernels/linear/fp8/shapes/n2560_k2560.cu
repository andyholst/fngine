// [2560,2560] BF16-store projection (ple.value, mtp.fc_embedding, mtp.fc_hidden); schedules in k2560_store.cuh.

#include "kernels/linear/fp8/shapes/k2560_store.cuh"

namespace fngine::kernels::detail {
const Fp8ShapeEntry kFp8N2560K2560{2560, 2560, k2560_store::kKind, k2560_store::launch_a16<2560>,
                                  k2560_store::launch_a8<2560, 128>, kFp8AutoNeverA8};
} // namespace fngine::kernels::detail
