// Modified for fngine from cinference:src/artifact/file_io.h @ 383e5db. See NOTICE.
#pragma once

// O_DIRECT file reads, host fills and staged host->device uploads of byte ranges.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fngine::io {

inline constexpr std::size_t kDirectAlignment = 4096;

class IoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

constexpr std::uint64_t align_down(std::uint64_t v, std::uint64_t a) { return v / a * a; }
constexpr std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// A read-only file with a buffered and an O_DIRECT descriptor. Both reads are positional and
// thread-safe. When the filesystem rejects O_DIRECT (EINVAL, e.g. some tmpfs), direct reads
// fall back to the buffered descriptor with the same alignment contract.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] bool direct() const noexcept { return direct_fd_ != fd_; }

    // Buffered read of exactly destination.size() bytes; throws on EOF or error.
    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    // Direct read: offset, size and buffer address must be kDirectAlignment-aligned. Returns the
    // bytes read (short only at end of file).
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset, std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    int fd_              = -1;
    int direct_fd_       = -1;
    std::uint64_t bytes_ = 0;
};

// File byte range [offset, offset + bytes) -> destination pointer.
struct CopyRange {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
    void* destination    = nullptr;
};

struct TransferStats {
    std::uint64_t payload_bytes = 0;  // sum of requested range bytes
    std::uint64_t read_bytes    = 0;  // bytes read from the file, including alignment/gap padding
    double seconds              = 0;
    [[nodiscard]] double gb_per_s() const noexcept {
        return seconds > 0 ? double(payload_bytes) / seconds * 1e-9 : 0.0;
    }
};

// Uploads file ranges into device memory: sorted ranges are coalesced into 4096-aligned direct
// read spans (gaps < 4 KiB are read through), streamed through 4 x 64 MiB pinned staging slots
// (each slot read by `read_threads` parallel O_DIRECT reads) and copied with cudaMemcpyAsync on
// `stream`. Returns after the stream finished the copies. Ranges must not overlap.
TransferStats upload_regions(const InputFile& file, std::vector<CopyRange> ranges, cudaStream_t stream,
                             int read_threads = 2);

// Fills host memory from file ranges with `threads` parallel readers. Ranges whose offset, size
// and destination are 4096-aligned are read with O_DIRECT straight into the destination
// (adjacent file+destination ranges are merged); others go through an aligned bounce buffer.
TransferStats read_regions_to_host(const InputFile& file, std::vector<CopyRange> ranges, int threads = 4);

} // namespace fngine::io
