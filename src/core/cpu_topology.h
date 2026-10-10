#pragma once

// CPU topology from sysfs and the fngine thread placement derived from it.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fngine {

struct PhysicalCore {
    int primary = -1;           // lowest online logical CPU of the core (worker threads use it)
    std::vector<int> siblings;  // all logical CPUs of the core, ascending
};

// One L3 domain (a CCD on Zen). Ordered by the lowest logical CPU it contains.
struct Ccd {
    int l3_id              = -1;  // cache/index3/id (or die_id when no L3 is reported)
    std::uint64_t l3_bytes = 0;
    std::vector<PhysicalCore> cores;  // ascending primary CPU
    std::vector<int> cpus;            // every online logical CPU, ascending
};

// Placement policy: the engine thread runs on the first physical core of CCD0, the I/O thread on
// the first physical core of CCD1 (with one CCD: the second core of CCD0). Such a core lends an
// online SMT sibling to the thread and keeps its primary thread as a CPU expert worker; without a
// spare sibling the core is reserved for the thread. Every other physical core's primary thread is
// a worker; the remaining SMT siblings stay idle (the expert tier is DRAM- and decode-bound).
// On the 9900X3D: engine 12, I/O 18, workers 0-11.
struct CpuTopology {
    std::vector<Ccd> ccds;
    int engine_cpu = 0;
    int io_cpu     = 0;
    std::vector<int> worker_cpus;

    // Reads <root>/cpu<N>/{online,topology/*,cache/index*/...}. Throws when no CPU is found.
    static CpuTopology detect(const std::filesystem::path& root = "/sys/devices/system/cpu");

    [[nodiscard]] int physical_cores() const noexcept;
    [[nodiscard]] std::string summary() const;
};

// Parses a sysfs CPU list ("0-5,12-17").
std::vector<int> parse_cpu_list(const std::string& text);

// Pins the calling thread to one logical CPU. Returns false (errno preserved) on failure.
bool pin_current_thread(int cpu);

} // namespace fngine
