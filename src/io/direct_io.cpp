// Modified for fngine from cinference:src/artifact/file_io.cpp and src/artifact/materializer.cpp
// @ 383e5db. See NOTICE.
#include "io/direct_io.h"

#include "core/arena.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fngine::io {
namespace {

constexpr std::size_t kSlotBytes     = 64ull << 20;
constexpr std::size_t kSlotCount     = 4;
constexpr std::size_t kHostTaskBytes = 32ull << 20;
constexpr std::uint64_t kReadPartBytes = 8ull << 20;

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw IoError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw IoError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw IoError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                      cudaGetErrorString(status));
    }
}

// Runs fn(task) for task in [0, n) on up to `threads` threads; rethrows the first exception.
template <class Fn>
void parallel_tasks(std::size_t n, int threads, Fn&& fn) {
    const auto workers = static_cast<std::size_t>(std::max(1, threads));
    if (n <= 1 || workers == 1) {
        for (std::size_t i = 0; i < n; ++i) { fn(i, 0); }
        return;
    }
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto body = [&](std::size_t worker) {
        try {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) { fn(i, worker); }
        } catch (...) {
            std::lock_guard lock(error_mutex);
            if (!error) { error = std::current_exception(); }
            next.store(n);
        }
    };
    std::vector<std::thread> pool;
    const auto count = std::min(workers, n);
    for (std::size_t w = 1; w < count; ++w) { pool.emplace_back(body, w); }
    body(0);
    for (auto& t : pool) { t.join(); }
    if (error) { std::rethrow_exception(error); }
}

struct AlignedFree {
    void operator()(std::byte* p) const noexcept { std::free(p); }
};
using AlignedBuffer = std::unique_ptr<std::byte, AlignedFree>;

AlignedBuffer aligned_buffer(std::size_t bytes) {
    void* p = std::aligned_alloc(kDirectAlignment, align_up(bytes, kDirectAlignment));
    if (!p) { throw std::bad_alloc(); }
    return AlignedBuffer(static_cast<std::byte*>(p));
}

bool is_aligned(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kDirectAlignment == 0; }

void sort_and_check(std::vector<CopyRange>& ranges, const InputFile& file) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& r = ranges[i];
        if (r.offset > file.bytes() || r.bytes > file.bytes() - r.offset) {
            throw IoError(file.path().string() + ": range [" + std::to_string(r.offset) + ", +" +
                          std::to_string(r.bytes) + ") exceeds file length");
        }
        if (i && ranges[i - 1].offset + ranges[i - 1].bytes > r.offset) {
            throw IoError(file.path().string() + ": copy ranges overlap");
        }
    }
    std::erase_if(ranges, [](const CopyRange& r) { return r.bytes == 0; });
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create staging event");
    }
    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event) { (void)cudaEventDestroy(event); }
    }
    void wait() {
        if (pending) {
            check_cuda(cudaEventSynchronize(event), "wait for staging transfer");
            pending = false;
        }
    }
    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    cudaStream_t stream;
    bool pending = true;
    ~TransferCompletion() {
        if (pending) { (void)cudaStreamSynchronize(stream); }
    }
    void finish() {
        check_cuda(cudaStreamSynchronize(stream), "complete upload");
        pending = false;
    }
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

// Reads [offset, offset + need) of the file into dst (aligned, capacity >= align_up(need)) with
// up to `threads` parallel direct reads. Returns bytes read (>= need).
std::uint64_t read_direct_parallel(const InputFile& file, std::uint64_t offset, std::uint64_t need,
                                   std::byte* dst, int threads) {
    const std::uint64_t request = align_up(need, kDirectAlignment);
    const std::uint64_t part =
        align_up((request + std::uint64_t(std::max(1, threads)) - 1) / std::max(1, threads), kDirectAlignment);
    const std::size_t parts = static_cast<std::size_t>((request + part - 1) / part);
    std::vector<std::uint64_t> got(parts, 0);
    parallel_tasks(parts, threads, [&](std::size_t i, std::size_t) {
        const std::uint64_t begin = i * part;
        const std::uint64_t len   = std::min(part, request - begin);
        const std::uint64_t want  = std::min(len, need - std::min(need, begin));
        std::uint64_t done        = 0;
        while (done < want) {
            const auto r = file.read_direct(offset + begin + done, {dst + begin + done, static_cast<std::size_t>(len - done)});
            if (r == 0) { break; }
            done += r;
        }
        if (done < want) { throw IoError(file.path().string() + ": direct read ended before the required payload"); }
        got[i] = done;
    });
    std::uint64_t total = 0;
    for (auto g : got) { total += g; }
    return total;
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }
    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw IoError(path_.string() + ": expected a regular file");
    }
    bytes_     = static_cast<std::uint64_t>(status.st_size);
    direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct_fd_ < 0) {
        if (errno != EINVAL) {
            const auto error = errno;
            ::close(fd_);
            fd_   = -1;
            errno = error;
            fail(path_, "open direct");
        }
        direct_fd_ = fd_;
    }
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0 && direct_fd_ != fd_) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw IoError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ull << 20);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw IoError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kDirectAlignment || destination.size() % kDirectAlignment || !is_aligned(destination.data()) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw IoError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

TransferStats upload_regions(const InputFile& file, std::vector<CopyRange> ranges, cudaStream_t stream,
                             int read_threads) {
    sort_and_check(ranges, file);
    TransferStats stats;
    for (const auto& r : ranges) { stats.payload_bytes += r.bytes; }
    if (ranges.empty()) { return stats; }

    std::vector<ReadSpan> spans;
    for (const auto& range : ranges) {
        const auto begin = align_down(range.offset, kDirectAlignment);
        const auto end   = range.offset + range.bytes;
        if (spans.empty() || begin > align_up(spans.back().end, kDirectAlignment)) {
            spans.push_back({begin, end});
        } else {
            spans.back().end = std::max(spans.back().end, end);
        }
    }
    std::uint64_t aligned_bytes = 0;
    for (const auto& span : spans) { aligned_bytes += align_up(span.end - span.begin, kDirectAlignment); }
    const auto slot_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
    const auto slot_count =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    for (std::size_t i = 0; i < slot_count; ++i) { slots.push_back(std::make_unique<Slot>(slot_bytes)); }

    TransferCompletion completion{stream};
    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    std::uint64_t copied   = 0;
    const auto start       = std::chrono::steady_clock::now();
    // Each slot is packed with consecutive pieces of the spans (scattered small spans share a slot),
    // its pieces are read with parallel O_DIRECT reads, then the ranges they cover are copied out.
    struct Piece {
        std::uint64_t source, need, slot_offset;
    };
    std::vector<Piece> pieces, parts;
    std::size_t span_index = 0;
    std::uint64_t cursor   = spans.front().begin;
    while (span_index < spans.size()) {
        auto& slot = *slots[next_slot++ % slot_count];
        slot.wait();
        auto* staging = static_cast<std::byte*>(slot.buffer.data());
        pieces.clear();
        std::uint64_t used = 0;
        while (span_index < spans.size() && used < slot_bytes) {
            const auto& span = spans[span_index];
            const auto len   = std::min<std::uint64_t>(span.end - cursor, slot_bytes - used);
            pieces.push_back({cursor, len, used});
            used += align_up(len, kDirectAlignment);
            cursor += len;
            if (cursor >= span.end && ++span_index < spans.size()) { cursor = spans[span_index].begin; }
        }
        parts.clear();
        for (const auto& piece : pieces) {
            for (std::uint64_t off = 0; off < piece.need; off += kReadPartBytes) {
                parts.push_back({piece.source + off, std::min(kReadPartBytes, piece.need - off), piece.slot_offset + off});
            }
        }
        std::atomic<std::uint64_t> received{0};
        parallel_tasks(parts.size(), read_threads, [&](std::size_t i, std::size_t) {
            const auto& part      = parts[i];
            const auto request    = align_up(part.need, kDirectAlignment);
            std::uint64_t done    = 0;
            while (done < part.need) {
                const auto r = file.read_direct(part.source + done,
                                                {staging + part.slot_offset + done, static_cast<std::size_t>(request - done)});
                if (r == 0) { throw IoError(file.path().string() + ": direct read ended before the required payload"); }
                done += r;
            }
            received += done;
        });
        stats.read_bytes += received.load();
        for (const auto& piece : pieces) {
            const auto chunk_end = piece.source + piece.need;
            while (next_range < ranges.size() && ranges[next_range].offset < chunk_end) {
                const auto& range    = ranges[next_range];
                const auto range_end = range.offset + range.bytes;
                const auto begin     = std::max(piece.source, range.offset);
                const auto end       = std::min(chunk_end, range_end);
                if (begin < end) {
                    check_cuda(cudaMemcpyAsync(static_cast<std::byte*>(range.destination) + (begin - range.offset),
                                               staging + piece.slot_offset + (begin - piece.source),
                                               static_cast<std::size_t>(end - begin), cudaMemcpyHostToDevice, stream),
                               "upload bytes");
                    copied += end - begin;
                }
                if (range_end > chunk_end) { break; }
                ++next_range;
            }
        }
        check_cuda(cudaEventRecord(slot.event, stream), "record staging completion");
        slot.pending = true;
    }
    for (const auto& slot : slots) { slot->wait(); }
    completion.finish();
    if (copied != stats.payload_bytes || next_range != ranges.size()) { throw IoError("incomplete device upload"); }
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}

TransferStats read_regions_to_host(const InputFile& file, std::vector<CopyRange> ranges, int threads) {
    sort_and_check(ranges, file);
    TransferStats stats;
    for (const auto& r : ranges) { stats.payload_bytes += r.bytes; }
    if (ranges.empty()) { return stats; }

    // Merge ranges contiguous in both the file and the destination, then cut into tasks.
    std::vector<CopyRange> merged;
    for (const auto& r : ranges) {
        if (!merged.empty()) {
            auto& last = merged.back();
            if (last.offset + last.bytes == r.offset &&
                static_cast<std::byte*>(last.destination) + last.bytes == static_cast<std::byte*>(r.destination)) {
                last.bytes += r.bytes;
                continue;
            }
        }
        merged.push_back(r);
    }
    std::vector<CopyRange> tasks;
    for (const auto& r : merged) {
        for (std::uint64_t done = 0; done < r.bytes;) {
            // Keep task boundaries aligned (relative to an aligned start) so direct reads stay valid.
            std::uint64_t len = std::min<std::uint64_t>(kHostTaskBytes, r.bytes - done);
            tasks.push_back({r.offset + done, len, static_cast<std::byte*>(r.destination) + done});
            done += len;
        }
    }

    const int workers = std::max(1, threads);
    std::vector<AlignedBuffer> bounce(static_cast<std::size_t>(workers));
    std::atomic<std::uint64_t> read_bytes{0};
    const auto start = std::chrono::steady_clock::now();
    parallel_tasks(tasks.size(), workers, [&](std::size_t i, std::size_t worker) {
        const auto& t   = tasks[i];
        auto* dst       = static_cast<std::byte*>(t.destination);
        const bool fits = t.offset % kDirectAlignment == 0 && is_aligned(dst) &&
                          (t.bytes % kDirectAlignment == 0);
        if (fits) {
            // Fault the destination in from this thread first: a direct read into untouched anonymous
            // (THP) memory faults it inside the kernel one page at a time, ~3x slower than
            // parallel first-touch followed by the read.
            for (std::uint64_t p = 0; p < t.bytes; p += kDirectAlignment) {
                reinterpret_cast<volatile std::byte*>(dst)[p] = std::byte{0};
            }
            std::uint64_t done = 0;
            while (done < t.bytes) {
                const auto r = file.read_direct(t.offset + done, {dst + done, static_cast<std::size_t>(t.bytes - done)});
                if (r == 0) { throw IoError(file.path().string() + ": unexpected EOF"); }
                done += r;
            }
            read_bytes += done;
            return;
        }
        // Bounce: read the enclosing aligned span, copy the requested bytes out.
        const auto begin = align_down(t.offset, kDirectAlignment);
        const auto need  = t.offset + t.bytes - begin;
        auto& buf        = bounce[worker];
        if (!buf) { buf = aligned_buffer(kHostTaskBytes + 2 * kDirectAlignment); }
        const auto got = read_direct_parallel(file, begin, need, buf.get(), 1);
        read_bytes += got;
        std::memcpy(dst, buf.get() + (t.offset - begin), static_cast<std::size_t>(t.bytes));
    });
    stats.read_bytes = read_bytes.load();
    stats.seconds    = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}

} // namespace fngine::io
