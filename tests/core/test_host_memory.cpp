// HostRegion / MappedHostBuffer and io/direct_io: host fills and device uploads of mixed aligned and
// unaligned ranges (including the unaligned end of file) must reproduce the file bytes exactly.
#include "core/host_memory.h"
#include "io/direct_io.h"

#include "common/test_util.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;
using fngine::io::CopyRange;

namespace {

std::uint8_t pattern(std::uint64_t i) { return std::uint8_t((i * 2654435761ull) >> 13); }

bool equal_file(const std::byte* got, std::uint64_t offset, std::uint64_t bytes) {
    for (std::uint64_t i = 0; i < bytes; ++i) {
        if (std::uint8_t(got[i]) != pattern(offset + i)) {
            std::fprintf(stderr, "mismatch at file offset %llu\n", (unsigned long long)(offset + i));
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    // HostRegion: 2 MiB-rounded, 2 MiB-aligned, writable.
    {
        fngine::HostRegion region((5ull << 20) + 1);
        FNGINE_CHECK(region.size() == 6ull << 20);
        FNGINE_CHECK(reinterpret_cast<std::uintptr_t>(region.data()) % fngine::kHugePageBytes == 0);
        std::memset(region.data(), 0xAB, region.size());
        FNGINE_CHECK(std::uint8_t(region.data()[region.size() - 1]) == 0xAB);
        FNGINE_CHECK(!region.registered() && region.device_ptr() == nullptr);
        fngine::HostRegion moved = std::move(region);
        FNGINE_CHECK(region.data() == nullptr && moved.size() == 6ull << 20);
    }
    FNGINE_CHECK(fngine::mem_available_bytes() > (1ull << 30));

    // Test file in the working directory (a real filesystem; O_DIRECT where supported).
    const fs::path path = fs::current_path() / ("fngine-direct-io-" + std::to_string(::getpid()) + ".bin");
    const std::uint64_t file_bytes = (70ull << 20) + 1234;  // more than one 64 MiB staging slot, unaligned tail
    {
        std::vector<char> buf(file_bytes);
        for (std::uint64_t i = 0; i < file_bytes; ++i) { buf[i] = char(pattern(i)); }
        std::ofstream(path, std::ios::binary).write(buf.data(), std::streamsize(file_bytes));
    }
    struct Spec {
        std::uint64_t offset, bytes, dst;
    };
    // Aligned, adjacent aligned (mergeable), unaligned head, unaligned size, gap < 4 KiB, large,
    // and the unaligned end of the file.
    const std::vector<Spec> specs = {
        {0, 8192, 0},
        {8192, 4096, 8192},
        {12288 + 17, 1000, 20000},
        {40960, 33ull << 20, 1ull << 20},
        {(40ull << 20) + 100, 5000, 40ull << 20},
        {(40ull << 20) + 6000, 300, (41ull << 20)},
        {(50ull << 20), (19ull << 20), (42ull << 20)},
        {file_bytes - 1234 - 4096, 4096 + 1234, (63ull << 20)},
    };
    {
        fngine::io::InputFile file(path);
        FNGINE_CHECK(file.bytes() == file_bytes);
        std::printf("direct io: %s\n", file.direct() ? "O_DIRECT" : "buffered fallback");
        fngine::HostRegion host(64ull << 20);
        std::vector<CopyRange> ranges;
        for (const auto& s : specs) { ranges.push_back({s.offset, s.bytes, host.data() + s.dst}); }
        const auto stats = fngine::io::read_regions_to_host(file, ranges, 4);
        for (const auto& s : specs) { FNGINE_CHECK(equal_file(host.data() + s.dst, s.offset, s.bytes)); }
        std::printf("host fill: %llu payload bytes, %llu read\n", (unsigned long long)stats.payload_bytes,
                    (unsigned long long)stats.read_bytes);

        bool overlap_rejected = false;
        try {
            fngine::io::read_regions_to_host(file, {{0, 100, host.data()}, {50, 100, host.data() + 200}}, 1);
        } catch (const fngine::io::IoError&) { overlap_rejected = true; }
        FNGINE_CHECK(overlap_rejected);
        bool bounds_rejected = false;
        try {
            fngine::io::read_regions_to_host(file, {{file_bytes - 10, 11, host.data()}}, 1);
        } catch (const fngine::io::IoError&) { bounds_rejected = true; }
        FNGINE_CHECK(bounds_rejected);

        if (fngine::test::skip_without_gpu()) {
            fs::remove(path);
            std::printf("PASS host memory (CPU only)\n");
            return 0;
        }

        // Registration exposes a device-visible alias of the host bytes.
        FNGINE_CHECK(host.register_with_cuda());
        FNGINE_CHECK(host.device_ptr() != nullptr);
        std::vector<std::byte> back(8192);
        FNGINE_TEST_CUDA(cudaMemcpy(back.data(), host.device_ptr(), back.size(), cudaMemcpyDeviceToHost));
        FNGINE_CHECK(equal_file(back.data(), 0, back.size()));

        // Device upload of the same ranges through the staging slots.
        fngine::test::DeviceArray<std::byte> dev(64ull << 20);
        std::vector<CopyRange> dranges;
        for (const auto& s : specs) { dranges.push_back({s.offset, s.bytes, dev.ptr + s.dst}); }
        cudaStream_t stream;
        FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto up = fngine::io::upload_regions(file, dranges, stream, 4);
        std::vector<std::byte> got(64ull << 20);
        FNGINE_TEST_CUDA(cudaMemcpy(got.data(), dev.ptr, got.size(), cudaMemcpyDeviceToHost));
        for (const auto& s : specs) { FNGINE_CHECK(equal_file(got.data() + s.dst, s.offset, s.bytes)); }
        std::printf("upload: %llu payload bytes, %llu read, %.2f GB/s\n", (unsigned long long)up.payload_bytes,
                    (unsigned long long)up.read_bytes, up.gb_per_s());
        FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
    }
    fs::remove(path);

    // MappedHostBuffer: zeroed, device-visible.
    {
        fngine::MappedHostBuffer buf(1 << 16);
        FNGINE_CHECK(buf.device_ptr() != nullptr);
        auto* p = static_cast<std::uint32_t*>(buf.data());
        FNGINE_CHECK(p[0] == 0 && p[(1 << 14) - 1] == 0);
        for (int i = 0; i < (1 << 14); ++i) { p[i] = std::uint32_t(i * 7); }
        std::vector<std::uint32_t> back(1 << 14);
        FNGINE_TEST_CUDA(cudaMemcpy(back.data(), buf.device_ptr(), 1 << 16, cudaMemcpyDeviceToHost));
        for (int i = 0; i < (1 << 14); ++i) { FNGINE_CHECK(back[std::size_t(i)] == std::uint32_t(i * 7)); }
    }
    std::printf("PASS host memory\n");
    return 0;
}
