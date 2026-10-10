// Snapshot selection rule (engine/session.h): largest position p with 0 < p <= min(lcp - 1,
// prompt_len - 1).
#include "engine/session.h"

#include <iostream>
#include <vector>

int main() {
    using fngine::engine::common_prefix;
    using fngine::engine::select_snapshot;
    int failures    = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::cerr << "FAIL: " << what << '\n';
            ++failures;
        }
    };
    const std::vector<std::int32_t> positions{4096, 8192, 100, 300};
    check(select_snapshot(positions, 9000, 9500) == std::size_t{1}, "largest usable position");
    check(select_snapshot(positions, 8192, 9500) == std::size_t{0},
          "position equal to the common prefix is not usable (token p must match)");
    check(select_snapshot(positions, 8193, 9500) == std::size_t{1}, "p = lcp - 1 is usable");
    check(select_snapshot(positions, 9000, 8192) == std::size_t{0},
          "at least one prompt token must be prefilled");
    check(!select_snapshot(positions, 100, 9000).has_value(), "nothing at or below 99");
    check(!select_snapshot(positions, 0, 10).has_value(), "no common prefix");
    check(!select_snapshot({}, 50, 50).has_value(), "no snapshots");

    const std::vector<std::int32_t> a{1, 2, 3, 4}, b{1, 2, 9}, c{1, 2, 3, 4, 5};
    check(common_prefix(a, b) == 2 && common_prefix(a, c) == 4 && common_prefix(a, {}) == 0,
          "common prefix");
    if (failures == 0) std::cout << "session selection tests passed\n";
    return failures == 0 ? 0 : 1;
}
