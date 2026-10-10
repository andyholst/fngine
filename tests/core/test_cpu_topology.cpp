// CpuTopology: synthetic sysfs trees (two CCDs with SMT, one CCD, offline CPUs) and this machine.
#include "core/cpu_topology.h"

#include "common/test_util.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

void put(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text << "\n";
}

struct FakeCpu {
    int cpu;
    std::string siblings;
    int l3_id;
    std::string l3_size;
    bool online = true;
};

fs::path make_tree(const std::string& name, const std::vector<FakeCpu>& cpus) {
    const fs::path root = fs::temp_directory_path() / ("fngine-topo-" + std::to_string(::getpid()) + "-" + name);
    fs::remove_all(root);
    for (const auto& c : cpus) {
        const fs::path dir = root / ("cpu" + std::to_string(c.cpu));
        if (c.cpu != 0) { put(dir / "online", c.online ? "1" : "0"); }
        if (!c.online) { continue; }
        put(dir / "topology" / "thread_siblings_list", c.siblings);
        put(dir / "topology" / "core_id", std::to_string(c.cpu));
        put(dir / "cache" / "index0" / "level", "1");
        put(dir / "cache" / "index3" / "level", "3");
        put(dir / "cache" / "index3" / "id", std::to_string(c.l3_id));
        put(dir / "cache" / "index3" / "size", c.l3_size);
    }
    put(root / "online", "0-63");  // a non-cpu entry the scanner must ignore
    fs::create_directories(root / "cpufreq");
    return root;
}

bool same(const std::vector<int>& a, const std::vector<int>& b) { return a == b; }

} // namespace

int main() {
    using fngine::CpuTopology;
    FNGINE_CHECK(same(fngine::parse_cpu_list("0-5,12-17"), {0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17}));
    FNGINE_CHECK(same(fngine::parse_cpu_list("6,18"), {6, 18}));

    // 9900X3D shape: CCD0 = 0-5 + 12-17 (96 MiB L3), CCD1 = 6-11 + 18-23 (32 MiB).
    {
        std::vector<FakeCpu> cpus;
        for (int c = 0; c < 24; ++c) {
            const int core = c % 12;
            const int ccd  = core / 6;
            cpus.push_back({c, std::to_string(core) + "," + std::to_string(core + 12), ccd, ccd ? "32768K" : "98304K"});
        }
        const auto root = make_tree("x3d", cpus);
        const auto topo = CpuTopology::detect(root);
        std::printf("x3d: %s\n", topo.summary().c_str());
        FNGINE_CHECK(topo.ccds.size() == 2);
        FNGINE_CHECK(topo.ccds[0].l3_bytes == 96ull << 20 && topo.ccds[1].l3_bytes == 32ull << 20);
        FNGINE_CHECK(topo.ccds[0].cores.size() == 6 && topo.ccds[0].cpus.size() == 12);
        FNGINE_CHECK(topo.physical_cores() == 12);
        FNGINE_CHECK(topo.engine_cpu == 12 && topo.io_cpu == 18);
        FNGINE_CHECK(same(topo.worker_cpus, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
        fs::remove_all(root);
    }
    // One L3 domain, no SMT, one CPU offline: engine = first core, I/O = second, rest workers.
    {
        std::vector<FakeCpu> cpus = {{0, "0", 0, "16M"}, {1, "1", 0, "16M"}, {2, "2", 0, "16M", false},
                                     {3, "3", 0, "16M"}, {4, "4", 0, "16M"}};
        const auto root = make_tree("one", cpus);
        const auto topo = CpuTopology::detect(root);
        std::printf("one ccd: %s\n", topo.summary().c_str());
        FNGINE_CHECK(topo.ccds.size() == 1 && topo.ccds[0].l3_bytes == 16ull << 20);
        FNGINE_CHECK(topo.engine_cpu == 0 && topo.io_cpu == 1);
        FNGINE_CHECK(same(topo.worker_cpus, {3, 4}));
        fs::remove_all(root);
    }
    // SMT siblings listed with an offline primary: the online sibling becomes the core's primary.
    // CCD0's core lends its sibling 2 to the engine thread and keeps worker 0; CCD1's only online
    // CPU is reserved for the I/O thread.
    {
        std::vector<FakeCpu> cpus = {{0, "0,2", 0, "8M"}, {1, "1,3", 1, "8M", false}, {2, "0,2", 0, "8M"},
                                     {3, "1,3", 1, "8M"}};
        const auto root = make_tree("smt", cpus);
        const auto topo = CpuTopology::detect(root);
        std::printf("smt: %s\n", topo.summary().c_str());
        FNGINE_CHECK(topo.ccds.size() == 2);
        FNGINE_CHECK(topo.ccds[0].cores.size() == 1 && topo.ccds[1].cores.size() == 1);
        FNGINE_CHECK(topo.engine_cpu == 2 && topo.io_cpu == 3);
        FNGINE_CHECK(same(topo.worker_cpus, {0}));
        fs::remove_all(root);
    }

    // This machine.
    const auto topo = CpuTopology::detect();
    std::printf("host: %s\n", topo.summary().c_str());
    FNGINE_CHECK(!topo.ccds.empty() && !topo.worker_cpus.empty());

    if (topo.ccds.size() == 2 && topo.ccds[0].cores.size() == 6 && topo.ccds[1].cores.size() == 6 &&
        topo.ccds[0].cpus.front() == 0 && topo.ccds[1].cpus.front() == 6) {
        FNGINE_CHECK(topo.engine_cpu == 12 && topo.io_cpu == 18);
        FNGINE_CHECK(same(topo.worker_cpus, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    }
    FNGINE_CHECK(fngine::pin_current_thread(topo.engine_cpu));
    FNGINE_CHECK(sched_getcpu() == topo.engine_cpu);
    std::printf("PASS cpu topology\n");
    return 0;
}