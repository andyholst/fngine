// Host-only layout checks: no device allocations or GPU calls. The optional real-pack check
// must reject an unsupported context before runtime CUDA initialization or weight loading.
#include "common/test_util.h"
#include "model/runtime.h"
#include "model/state.h"

#include <limits>
#include <stdexcept>

using namespace fngine;

namespace {

template <class F>
bool rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    constexpr int32_t last_page = std::numeric_limits<int32_t>::max() - 63;
    for (const auto format : {kernels::QsaKvFormat::K8V8, kernels::QsaKvFormat::K8V4,
                              kernels::QsaKvFormat::BF16}) {
        const std::size_t per_token = model::kQsaCaches *
            (kernels::qsa_kv_bytes_per_token(format) + 128 * sizeof(__nv_bfloat16) / 4);
        const std::size_t fixed = model::ModelState::device_bytes({64, format}) - 64 * per_token;
        for (const int32_t requested : {1, 63, 64, 65, 131072, 180000, 200000, 262143, 262144,
                                       last_page}) {
            const int32_t capacity = (requested + 63) / 64 * 64;
            const model::StateOptions options{requested, format};
            const auto bytes = model::ModelState::device_bytes(options);
            FNGINE_CHECK(bytes == fixed + std::size_t(capacity) * per_token);
            std::size_t end = 0, position_bytes = 0;
            for (const auto& region : model::ModelState::position_regions(options)) {
                FNGINE_CHECK(region.offset % 256 == 0 && region.offset >= end);
                FNGINE_CHECK(region.bytes_per_token > 0);
                FNGINE_CHECK(region.bytes == std::size_t(capacity) * region.bytes_per_token);
                end = region.offset + region.bytes;
                FNGINE_CHECK(end <= bytes);
                position_bytes += region.bytes;
            }
            FNGINE_CHECK(position_bytes == std::size_t(capacity) * per_token);
        }
    }
    for (const int32_t invalid : {0, -1, std::numeric_limits<int32_t>::max()}) {
        FNGINE_CHECK(rejects([&] { (void)model::ModelState::device_bytes({invalid}); }));
        model::RuntimeOptions options;
        options.max_context = invalid;
        FNGINE_CHECK(rejects([&] { model::ModelRuntime runtime(options); }));
    }

    if (const char* path = std::getenv("FNGINE_TEST_PACK"); path && *path) {
        const auto pack = io::Pack::open(path);
        const auto config = model::ModelConfig::from_pack(pack);
        FNGINE_CHECK(config.max_positions > 0 && config.max_positions < last_page);
        model::RuntimeOptions options;
        options.pack = path;
        options.max_context = config.max_positions + 1;
        FNGINE_CHECK(rejects([&] { model::ModelRuntime runtime(options); }));
    }
    std::printf("PASS context capacity layout and position bounds\n");
    return 0;
}
