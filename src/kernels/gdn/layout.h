// Modified for fngine from cinference:src/core/layout.h @ 383e5db. See NOTICE.
#pragma once

// Checked offset builder for caller-owned backing allocations. The GDN storage planners (state
// pool, replay records, chunked workspace) describe their regions with it, so the byte total a
// caller allocates and every later binding share exactly the same offsets.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fngine::kernels::gdn {

struct LayoutRegion {
    std::size_t offset    = 0;
    std::size_t bytes     = 0;
    std::size_t alignment = 1;

    // Returns the region's address inside `backing` (which must hold `backing_bytes`).
    [[nodiscard]] void* bind(void* backing, std::size_t backing_bytes) const {
        if (backing == nullptr) { throw std::invalid_argument("layout backing is null"); }
        const auto base = reinterpret_cast<std::uintptr_t>(backing);
        if ((base + offset) % alignment != 0) {
            throw std::invalid_argument("layout backing does not satisfy region alignment");
        }
        if (offset > backing_bytes || bytes > backing_bytes - offset) {
            throw std::out_of_range("layout region exceeds its backing allocation");
        }
        return static_cast<unsigned char*>(backing) + offset;
    }
};

class LayoutBuilder {
public:
    [[nodiscard]] LayoutRegion add(std::size_t bytes, std::size_t alignment,
                                   std::string_view label) {
        if (bytes == 0) { throw std::invalid_argument(std::string(label) + " must not be empty"); }
        const std::size_t offset = align(cursor_, alignment, label);
        cursor_                  = checked_add(offset, bytes, label);
        return LayoutRegion{offset, bytes, alignment};
    }

    [[nodiscard]] std::size_t finish(std::size_t alignment  = 1,
                                     std::string_view label = "layout") const {
        return align(cursor_, alignment, label);
    }

private:
    static std::size_t checked_add(std::size_t a, std::size_t b, std::string_view label) {
        if (b > std::numeric_limits<std::size_t>::max() - a) {
            throw std::overflow_error(std::string(label) + " size overflow");
        }
        return a + b;
    }

    static std::size_t align(std::size_t value, std::size_t alignment, std::string_view label) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
        }
        const std::size_t mask = alignment - 1;
        return checked_add(value, mask, label) & ~mask;
    }

    std::size_t cursor_ = 0;
};

} // namespace fngine::kernels::gdn
