#include "core/cpu_topology.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>

namespace fngine {
namespace {

namespace fs = std::filesystem;

bool read_text(const fs::path& path, std::string& out) {
    std::ifstream in(path);
    if (!in) { return false; }
    std::getline(in, out);
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) { out.pop_back(); }
    return true;
}

bool read_int(const fs::path& path, long long& out) {
    std::string text;
    if (!read_text(path, text) || text.empty()) { return false; }
    try {
        out = std::stoll(text);
    } catch (const std::exception&) { return false; }
    return true;
}

std::uint64_t parse_size(const std::string& text) {
    if (text.empty()) { return 0; }
    std::size_t used = 0;
    const auto value = std::stoull(text, &used);
    std::uint64_t mult = 1;
    if (used < text.size()) {
        switch (std::toupper(static_cast<unsigned char>(text[used]))) {
        case 'K': mult = 1ull << 10; break;
        case 'M': mult = 1ull << 20; break;
        case 'G': mult = 1ull << 30; break;
        default: break;
        }
    }
    return value * mult;
}

struct CpuEntry {
    int cpu       = -1;
    long long l3  = -1;
    std::uint64_t l3_bytes = 0;
    std::vector<int> siblings;
};

} // namespace

std::vector<int> parse_cpu_list(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) { continue; }
        const auto dash = part.find('-');
        const int lo    = std::stoi(part.substr(0, dash));
        const int hi    = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
        for (int c = lo; c <= hi; ++c) { cpus.push_back(c); }
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

CpuTopology CpuTopology::detect(const fs::path& root) {
    std::vector<CpuEntry> entries;
    std::error_code ec;
    for (const auto& dir : fs::directory_iterator(root, ec)) {
        const auto name = dir.path().filename().string();
        if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
            !std::all_of(name.begin() + 3, name.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
            continue;
        }
        CpuEntry e;
        e.cpu = std::stoi(name.substr(3));
        long long online = 1;
        if (read_int(dir.path() / "online", online) && online == 0) { continue; }
        const auto topo = dir.path() / "topology";
        if (!fs::exists(topo)) { continue; }
        std::string siblings;
        if (read_text(topo / "thread_siblings_list", siblings)) { e.siblings = parse_cpu_list(siblings); }
        if (e.siblings.empty()) { e.siblings = {e.cpu}; }
        // L3 domain: the cache index whose level is 3.
        for (int index = 0; index < 8; ++index) {
            const auto cache = dir.path() / "cache" / ("index" + std::to_string(index));
            long long level = 0;
            if (!read_int(cache / "level", level)) { continue; }
            if (level != 3) { continue; }
            long long id = -1;
            if (!read_int(cache / "id", id)) {
                std::string shared;
                if (read_text(cache / "shared_cpu_list", shared)) {
                    const auto list = parse_cpu_list(shared);
                    if (!list.empty()) { id = list.front(); }
                }
            }
            std::string size;
            if (read_text(cache / "size", size)) { e.l3_bytes = parse_size(size); }
            e.l3 = id;
            break;
        }
        if (e.l3 < 0) {
            long long die = 0, package = 0;
            read_int(topo / "die_id", die);
            read_int(topo / "physical_package_id", package);
            e.l3 = package * 4096 + die;
        }
        entries.push_back(std::move(e));
    }
    if (entries.empty()) { throw std::runtime_error("cpu topology: no online CPUs under " + root.string()); }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.cpu < b.cpu; });

    std::map<long long, std::size_t> by_l3;
    CpuTopology topo;
    for (const auto& e : entries) {
        auto [it, inserted] = by_l3.try_emplace(e.l3, topo.ccds.size());
        if (inserted) {
            Ccd ccd;
            ccd.l3_id    = static_cast<int>(e.l3);
            ccd.l3_bytes = e.l3_bytes;
            topo.ccds.push_back(std::move(ccd));
        }
        Ccd& ccd = topo.ccds[it->second];
        ccd.cpus.push_back(e.cpu);
        // Entries are ascending, so the first CPU seen of a core is its lowest online sibling.
        if (std::none_of(ccd.cores.begin(), ccd.cores.end(), [&](const PhysicalCore& c) {
                return std::find(c.siblings.begin(), c.siblings.end(), e.cpu) != c.siblings.end();
            })) {
            PhysicalCore core;
            core.primary  = e.cpu;
            core.siblings = e.siblings;
            ccd.cores.push_back(std::move(core));
        }
    }
    // CCDs were created in ascending order of their lowest CPU because entries are sorted.
    // The engine and I/O threads take the first physical core of CCD0 / CCD1 (one CCD: its first
    // two cores). A core with an online SMT sibling lends that sibling to the thread and keeps its
    // primary thread as an expert worker; a core without one is reserved for the thread.
    std::vector<int> reserved;  // primaries that host no worker
    auto place = [&](const Ccd& ccd, const PhysicalCore& core) {
        for (int s : core.siblings) {
            if (s != core.primary && std::find(ccd.cpus.begin(), ccd.cpus.end(), s) != ccd.cpus.end()) return s;
        }
        reserved.push_back(core.primary);
        return core.primary;
    };
    if (topo.ccds.size() >= 2) {
        topo.engine_cpu = place(topo.ccds[0], topo.ccds[0].cores.front());
        topo.io_cpu     = place(topo.ccds[1], topo.ccds[1].cores.front());
    } else {
        const Ccd& ccd  = topo.ccds[0];
        topo.engine_cpu = place(ccd, ccd.cores[0]);
        topo.io_cpu     = ccd.cores.size() > 1 ? place(ccd, ccd.cores[1]) : topo.engine_cpu;
    }
    for (const auto& ccd : topo.ccds) {
        for (const auto& core : ccd.cores) {
            if (std::find(reserved.begin(), reserved.end(), core.primary) == reserved.end()) {
                topo.worker_cpus.push_back(core.primary);
            }
        }
    }
    if (topo.worker_cpus.empty()) { topo.worker_cpus.push_back(topo.ccds.back().cores.back().primary); }
    std::sort(topo.worker_cpus.begin(), topo.worker_cpus.end());
    return topo;
}

int CpuTopology::physical_cores() const noexcept {
    int n = 0;
    for (const auto& ccd : ccds) { n += static_cast<int>(ccd.cores.size()); }
    return n;
}

std::string CpuTopology::summary() const {
    std::ostringstream os;
    for (std::size_t i = 0; i < ccds.size(); ++i) {
        const auto& ccd = ccds[i];
        os << "ccd" << i << ": L3 " << (ccd.l3_bytes >> 20) << " MiB, " << ccd.cores.size()
           << " cores, " << ccd.cpus.size() << " cpus; ";
    }
    os << "engine cpu " << engine_cpu << ", io cpu " << io_cpu << ", workers [";
    for (std::size_t i = 0; i < worker_cpus.size(); ++i) { os << (i ? "," : "") << worker_cpus[i]; }
    os << "]";
    return os.str();
}

bool pin_current_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

} // namespace fngine
