#include "core/host_memory.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/mman.h>

namespace fngine {
namespace {

std::size_t align_huge(std::size_t bytes) {
    return (bytes + kHugePageBytes - 1) / kHugePageBytes * kHugePageBytes;
}

} // namespace

HostRegion::HostRegion(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("HostRegion size must be nonzero"); }
    const std::size_t size = align_huge(bytes);
    // Over-map by one huge page so the region can start on a 2 MiB boundary, then trim.
    const std::size_t span = size + kHugePageBytes;
    void* raw = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                       -1, 0);
    if (raw == MAP_FAILED) {
        throw std::runtime_error("HostRegion: mmap of " + std::to_string(span) +
                                 " bytes failed: " + std::strerror(errno));
    }
    const auto base    = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (base + kHugePageBytes - 1) & ~(std::uintptr_t(kHugePageBytes) - 1);
    const std::size_t head = aligned - base;
    const std::size_t tail = span - head - size;
    if (head) { ::munmap(raw, head); }
    if (tail) { ::munmap(reinterpret_cast<void*>(aligned + size), tail); }
    data_ = reinterpret_cast<std::byte*>(aligned);
    size_ = size;
    if (::madvise(data_, size_, MADV_HUGEPAGE) != 0) {
        spdlog::warn("HostRegion: madvise(MADV_HUGEPAGE) failed: {}", std::strerror(errno));
    }
}

HostRegion::~HostRegion() { release(); }

HostRegion::HostRegion(HostRegion&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
      registered_(std::exchange(other.registered_, false)),
      device_(std::exchange(other.device_, nullptr)), register_seconds_(other.register_seconds_) {}

HostRegion& HostRegion::operator=(HostRegion&& other) noexcept {
    if (this != &other) {
        release();
        data_             = std::exchange(other.data_, nullptr);
        size_             = std::exchange(other.size_, 0);
        registered_       = std::exchange(other.registered_, false);
        device_           = std::exchange(other.device_, nullptr);
        register_seconds_ = other.register_seconds_;
    }
    return *this;
}

bool HostRegion::register_with_cuda() {
    if (registered_ || !data_) { return registered_; }
    const auto start = std::chrono::steady_clock::now();
    cudaError_t err  = cudaHostRegister(data_, size_, cudaHostRegisterPortable | cudaHostRegisterMapped);
    register_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        spdlog::warn("HostRegion: cudaHostRegister of {:.2f} GiB failed ({}: {}); the region stays "
                     "pageable; asynchronous transfers require pinned staging",
                     double(size_) / double(1ull << 30), cudaGetErrorName(err), cudaGetErrorString(err));
        return false;
    }
    void* dev = nullptr;
    err       = cudaHostGetDevicePointer(&dev, data_, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaHostUnregister(data_);
        spdlog::warn("HostRegion: cudaHostGetDevicePointer failed ({}); region stays CPU-only",
                     cudaGetErrorString(err));
        return false;
    }
    registered_ = true;
    device_     = dev;
    return true;
}

void HostRegion::unregister() noexcept {
    if (registered_) {
        const cudaError_t err = cudaHostUnregister(data_);
        if (err != cudaSuccess) {
            spdlog::error("HostRegion: cudaHostUnregister failed: {}", cudaGetErrorString(err));
        }
        registered_ = false;
        device_     = nullptr;
    }
}

void HostRegion::release() noexcept {
    unregister();
    if (data_) { ::munmap(data_, size_); }
    data_ = nullptr;
    size_ = 0;
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("MappedHostBuffer size must be nonzero"); }
    void* ptr       = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaHostAlloc(mapped) of ") + std::to_string(bytes) +
                                 " bytes failed: " + cudaGetErrorString(err));
    }
    void* dev = nullptr;
    err       = cudaHostGetDevicePointer(&dev, ptr, 0);
    if (err != cudaSuccess) {
        (void)cudaFreeHost(ptr);
        throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                 cudaGetErrorString(err));
    }
    std::memset(ptr, 0, bytes);
    data_   = ptr;
    device_ = dev;
    size_   = bytes;
}

MappedHostBuffer::~MappedHostBuffer() { release(); }

MappedHostBuffer::MappedHostBuffer(MappedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), device_(std::exchange(other.device_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedHostBuffer& MappedHostBuffer::operator=(MappedHostBuffer&& other) noexcept {
    if (this != &other) {
        release();
        data_   = std::exchange(other.data_, nullptr);
        device_ = std::exchange(other.device_, nullptr);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void MappedHostBuffer::release() noexcept {
    if (data_) {
        const cudaError_t err = cudaFreeHost(data_);
        if (err != cudaSuccess) {
            spdlog::error("MappedHostBuffer: cudaFreeHost failed: {}", cudaGetErrorString(err));
        }
    }
    data_   = nullptr;
    device_ = nullptr;
    size_   = 0;
}

std::uint64_t mem_available_bytes() {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (in >> key >> value) {
        std::getline(in, unit);
        if (key == "MemAvailable:") { return value * 1024; }
    }
    return 0;
}

} // namespace fngine
