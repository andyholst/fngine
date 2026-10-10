#include "model/placement.h"

#include "core/model_dims.h"
#include "core/device.h"
#include "model/routing_counts.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <limits>
#include <cstring>
#include <utility>

namespace fngine::model {
namespace {

using namespace fngine::dims;
namespace fs = std::filesystem;

constexpr double kGiB                   = double(1ull << 30);
constexpr std::uint64_t kLowVramWarning = 20ull << 30;

std::vector<std::uint64_t> read_counts(const fs::path& path) {
    const std::uint64_t want = std::uint64_t(kLayers) * kExperts * sizeof(std::uint64_t);
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec || size != want) {
        throw io::PackError("routing stats " + path.string() + " must be " + std::to_string(want) + " bytes (u64 [48][512])");
    }
    std::vector<std::uint64_t> counts(std::size_t(kLayers) * kExperts);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(counts.data()), std::streamsize(want))) {
        throw io::PackError("cannot read routing stats " + path.string());
    }
    return counts;
}

// Elastic slots in the appended state's position regions. A region's first page is never
// usable (the first token writes it). Per-layer counts follow the rank order of the non-resident
// pairs within the regions' capacity; each layer's slots then form runs, the layers holding the
// best candidates first, at the tops of the regions that the context reaches last (best experts
// topmost). Picks that no longer fit (alignment between runs) stay on the CPU. Returns the score
// of the placed experts.
double carve_elastic(PlacementPlan& plan, const std::vector<io::ExpertLayout>& layers,
                     const std::vector<std::int32_t>& order, const std::vector<bool>& resident,
                     const PlacementOptions& options) {
    struct Region {
        std::uint64_t begin, top, bytes_per_token;
    };
    std::uint64_t min_stride = std::numeric_limits<std::uint64_t>::max();
    for (int l = 0; l < options.expert_layers; ++l) min_stride = std::min(min_stride, layers[std::size_t(l)].blob_stride);
    std::vector<Region> regions;
    std::uint64_t capacity = 0;
    for (const PositionRegion& r : options.state_regions) {
        const std::uint64_t first_page = 64 * r.bytes_per_token;
        if (r.bytes_per_token == 0 || r.bytes < first_page + min_stride) continue;
        const std::uint64_t begin = plan.state_offset + r.offset;
        regions.push_back({begin, begin + r.bytes, r.bytes_per_token});
        capacity += r.bytes - first_page;
    }
    if (regions.empty()) return 0;

    std::array<std::vector<std::int32_t>, kLayers> picks{};  // per layer, best first
    std::array<std::size_t, kLayers> first{};                // rank of the layer's best pick
    std::uint64_t used = 0;
    std::size_t rank = 0;
    for (const auto pair : order) {
        if (resident[std::size_t(pair)]) continue;
        const int l = pair / kExperts;
        const auto stride = layers[std::size_t(l)].blob_stride;
        if (used + stride > capacity) break;
        used += stride;
        if (picks[std::size_t(l)].empty()) first[std::size_t(l)] = rank;
        picks[std::size_t(l)].push_back(pair % kExperts);
        ++rank;
    }
    std::vector<int> layer_order;
    for (int l = 0; l < options.expert_layers; ++l)
        if (!picks[std::size_t(l)].empty()) layer_order.push_back(l);
    std::sort(layer_order.begin(), layer_order.end(),
              [&](int a, int b) { return first[std::size_t(a)] < first[std::size_t(b)]; });

    double hit = 0;
    for (const int l : layer_order) {
        const std::uint64_t stride = layers[std::size_t(l)].blob_stride;
        const std::uint64_t base = plan.layer_offset[std::size_t(l)];
        const auto& want = picks[std::size_t(l)];
        std::size_t next = 0;
        // Highest slot of this layer below a region's top that leaves the region's first page.
        const auto top_slot = [&](const Region& r) -> std::uint64_t {
            const std::uint64_t a = base + (r.top - stride - base) / stride * stride;
            return a >= r.begin + 64 * r.bytes_per_token ? a : 0;
        };
        while (next < want.size()) {
            Region* best = nullptr;
            std::uint64_t best_tokens = 0;
            for (Region& r : regions) {
                const std::uint64_t a = r.top >= base + stride ? top_slot(r) : 0;
                if (!a) continue;
                const std::uint64_t tokens = (a - r.begin) / (64 * r.bytes_per_token) * 64;
                if (!best || tokens > best_tokens) best = &r, best_tokens = tokens;
            }
            if (!best) break;
            for (std::uint64_t a = top_slot(*best); a && next < want.size();) {
                const auto tokens = std::int32_t(std::min<std::uint64_t>(
                    (a - best->begin) / (64 * best->bytes_per_token) * 64, std::uint64_t(INT32_MAX)));
                const std::int32_t e = want[next++];
                const auto slot = std::int32_t((a - base) / stride);
                plan.elastic.push_back({l, slot, tokens, e});
                plan.slot_of[std::size_t(l) * kExperts + std::size_t(e)] = slot;
                plan.elastic_bytes += stride;
                hit += double(plan.score[std::size_t(l) * kExperts + std::size_t(e)]);
                best->top = a;
                a = best->top >= base + stride ? top_slot(*best) : 0;
            }
        }
    }
    return hit;
}

} // namespace

PlacementPlan plan_placement(const io::Pack& pack, const PlacementOptions& options) {
    if (options.expert_layers < 0 || options.expert_layers > kLayers) {
        throw std::invalid_argument("placement: expert_layers must be in [0, 48]");
    }
    if (options.bounce_buffer_bytes == 0 || options.bounce_buffer_bytes > (64ull << 20))
        throw std::invalid_argument("placement: bounce buffer size must be in (0, 64 MiB]");
    PlacementPlan plan;
    plan.expert_layers = options.expert_layers;

    std::size_t free = 0, total = 0;
    const cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("placement: cudaMemGetInfo failed: ") + cudaGetErrorString(err));
    }
    plan.free_vram = free;
    if (std::uint64_t(free) + options.preloaded_vram_bytes < kLowVramWarning) {
        spdlog::warn("only {:.1f} GiB of VRAM was free at start: another GPU process (e.g. Strata on :8080) holds "
                     "memory, so fewer experts fit on the GPU",
                     double(std::uint64_t(free) + options.preloaded_vram_bytes) / kGiB);
    }
    const std::uint64_t reserve = options.vram_reserve_mib << 20;
    const std::uint64_t needed  = options.fixed_vram_bytes + reserve;
    plan.budget                 = free > needed ? free - needed : 0;
    if (options.vram_cap_bytes) { plan.budget = std::min(plan.budget, *options.vram_cap_bytes); }

    // Score: decode routing frequency decides residency, because decode speed depends on the
    // resident share of decode traffic and prompt text routes differently from the model's own
    // output. Decode evidence (the generation profile shipped with the pack plus the user's
    // persisted decode routing stats, pooled by count) carries weight 1 - kPrefillWeight. The
    // prefill calibration profile keeps kPrefillWeight: it covers routing the generation prompts
    // miss, e.g. continuing raw documents (offline on held-out counts: 0.05 -> 0.2 trades 0.5
    // points of chat-decode hit rate for 1.8 points on document continuation).
    constexpr double kPrefillWeight = 0.2;
    plan.score.assign(std::size_t(kLayers) * kExperts, 0);
    auto optional_counts = [&](const fs::path& explicit_path, const char* name) {
        fs::path path = explicit_path;
        if (path.empty()) {
            path = pack.dir() / name;
            if (!fs::exists(path)) return std::vector<std::uint64_t>{};
        }
        return read_counts(path);
    };
    const std::vector<std::uint64_t> stats   = optional_counts(options.routing_stats, "routing-stats.bin");
    const std::vector<std::uint64_t> decoded = optional_counts(options.decode_profile, "decode-profile.bin");
    std::vector<double> decode(std::size_t(kLayers) * kExperts, 0.0);
    double decode_total = 0;
    for (const auto* src : {&stats, &decoded}) {
        for (std::size_t i = 0; i < src->size(); ++i) {
            decode[i] += double((*src)[i]);
            decode_total += double((*src)[i]);
        }
    }
    const auto saturated = [](std::uint64_t a, std::uint64_t b) {
        return b > std::numeric_limits<std::uint64_t>::max() - a ? std::numeric_limits<std::uint64_t>::max() : a + b;
    };
    for (auto c : stats) plan.stats_total = saturated(plan.stats_total, c);
    for (auto c : decoded) plan.decode_profile_total = saturated(plan.decode_profile_total, c);
    double prefill_total = 0;
    if (pack.has_profile())
        for (auto c : pack.profile()) prefill_total += double(c);
    if (decode_total > 0 || prefill_total > 0) {
        const double wd = decode_total > 0 ? (prefill_total > 0 ? 1.0 - kPrefillWeight : 1.0) : 0.0;
        const double wp = prefill_total > 0 ? 1.0 - wd : 0.0;
        constexpr double kScale = double(1ull << 52);  // frequencies as exact integers
        for (std::size_t i = 0; i < plan.score.size(); ++i) {
            double f = 0;
            if (wd > 0) f += wd * decode[i] / decode_total;
            if (wp > 0) f += wp * double(pack.profile()[i]) / prefill_total;
            plan.score[i] = std::uint64_t(f * kScale);
        }
        plan.scored = true;
        spdlog::info("expert scores: {} decode-profile pairs + {} routing-stats pairs (decode), {} prefill "
                     "profile pairs (weight {:.2f})",
                     plan.decode_profile_total, plan.stats_total, std::uint64_t(prefill_total), wp);
    }

    // Greedy fill by descending score; ties (and the unscored case) in (layer, expert) order.
    const auto& layers = pack.expert_layers();
    std::vector<std::int32_t> order(std::size_t(options.expert_layers) * kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int32_t a, std::int32_t b) { return plan.score[std::size_t(a)] > plan.score[std::size_t(b)]; });
    std::vector<bool> resident(std::size_t(kLayers) * kExperts, false);
    std::uint64_t remaining = plan.budget;
    // One spare slot per loaded layer for in-session swaps, reserved before the greedy fill.
    plan.spare_slot.fill(-1);
    if (options.spare_slots) {
        std::uint64_t spare_bytes = 0;
        for (int l = 0; l < options.expert_layers; ++l) { spare_bytes += layers[std::size_t(l)].blob_stride; }
        if (spare_bytes > 0 && spare_bytes <= remaining) {
            plan.spares = true;
            remaining -= spare_bytes;
        } else if (spare_bytes > 0) {
            spdlog::warn("expert pool budget ({:.2f} GiB) cannot hold one spare slot per layer: swaps disabled",
                         double(plan.budget) / kGiB);
        }
    }
    // Sparse slot indexes let each layer retain base + slot * stride addressing while its
    // lowest-ranked slots live together at the end of the allocation. At most one stride of
    // padding per layer is needed between those tail groups.
    if (options.loan_bytes) {
        std::uint64_t padding = 0;
        for (int l = 0; l < options.expert_layers; ++l) padding += layers[std::size_t(l)].blob_stride;
        remaining = remaining > padding ? remaining - padding : 0;
    }
    for (const auto pair : order) {
        const auto stride = layers[std::size_t(pair / kExperts)].blob_stride;
        if (stride <= remaining) {
            resident[std::size_t(pair)] = true;
            remaining -= stride;
        }
    }
    std::vector<bool> loaned(resident.size(), false);
    std::uint64_t loan_payload = 0;
    if (options.loan_bytes) {
        for (auto it = order.rbegin(); it != order.rend() && loan_payload < options.loan_bytes; ++it) {
            const auto i = std::size_t(*it);
            if (!resident[i]) continue;
            loaned[i] = true;
            ++plan.loan_slots[i / kExperts];
            loan_payload += layers[i / kExperts].blob_stride;
        }
        if (loan_payload < options.loan_bytes) {
            // Caller replans with a static workspace before allocating anything.
            std::fill(loaned.begin(), loaned.end(), false);
            plan.loan_slots.fill(0);
            loan_payload = 0;
        }
    }

    plan.slot_of.assign(std::size_t(kLayers) * kExperts, -1);
    std::uint64_t host_bytes = 0;
    double hit = 0, mass = 0;
    for (int l = 0; l < options.expert_layers; ++l) {
        const auto stride = layers[std::size_t(l)].blob_stride;
        for (int e = 0; e < kExperts; ++e) {
            const auto i = std::size_t(l) * kExperts + std::size_t(e);
            mass += double(plan.score[i]);
            if (resident[i]) {
                plan.slot_of[i] = plan.gpu_slots[std::size_t(l)]++;
                hit += double(plan.score[i]);
                plan.vram_bytes += stride;
            } else {
                ++plan.cpu_slots[std::size_t(l)];
                host_bytes += stride;
            }
        }
        if (plan.spares) {
            plan.spare_slot[std::size_t(l)] = plan.gpu_slots[std::size_t(l)];
            plan.vram_bytes += stride;
        }
        plan.gpu_experts += plan.gpu_slots[std::size_t(l)];
        plan.cpu_experts += plan.cpu_slots[std::size_t(l)];
    }
    const auto loaded       = plan.gpu_experts + plan.cpu_experts;
    plan.predicted_hit_rate = plan.scored && mass > 0 ? hit / mass
                              : loaded            ? double(plan.gpu_experts) / double(loaded)
                                                  : 0.0;
    // First the ordinary per-layer arrays (including spares), then all loan slots. No second
    // device allocation backs the loan: prefill overwrites these exact expert bytes.
    std::uint64_t offset = 0;
    for (int l = 0; l < options.expert_layers; ++l) {
        const auto li = std::size_t(l);
        const auto stride = layers[li].blob_stride;
        plan.layer_offset[li] = offset;
        int slot = 0;
        for (int e = 0; e < kExperts; ++e) {
            const auto i = li * kExperts + std::size_t(e);
            if (resident[i] && !loaned[i]) plan.slot_of[i] = slot++;
        }
        if (plan.spares) plan.spare_slot[li] = slot++;
        offset += std::uint64_t(slot) * stride;
    }
    plan.loan_offset = offset;
    plan.loan_host_offset = host_bytes;
    for (int l = 0; l < options.expert_layers; ++l) {
        const auto li = std::size_t(l);
        if (!plan.loan_slots[li]) continue;
        const auto stride = layers[li].blob_stride;
        // align_up in direct_io assumes power-of-two alignment; strides are not powers of two.
        const auto delta = offset - plan.layer_offset[li];
        offset += (stride - delta % stride) % stride;
        int slot = int((offset - plan.layer_offset[li]) / stride);
        plan.loan_first_slot[li] = slot;
        for (int e = 0; e < kExperts; ++e) {
            const auto i = li * kExperts + std::size_t(e);
            if (loaned[i]) plan.slot_of[i] = slot++;
        }
        offset += std::uint64_t(plan.loan_slots[li]) * stride;
    }
    plan.loan_bytes = loan_payload ? offset - plan.loan_offset : 0;
    plan.vram_bytes = offset;
    host_bytes += loan_payload;
    plan.host_bytes = host_bytes ? io::align_up(host_bytes, kHugePageBytes) : 0;
    // The state follows the expert pool in the same allocation; with spares, elastic slots in its
    // position regions take the next-ranked experts (they keep their RAM blobs as backups).
    if (options.state_bytes) {
        plan.state_offset = (offset + 255) / 256 * 256;
        plan.state_bytes  = options.state_bytes;
        if (plan.spares) {
            hit += carve_elastic(plan, layers, order, resident, options);
            if (plan.scored && mass > 0) plan.predicted_hit_rate = hit / mass;
        }
    }

    // Startup host check.
    plan.mem_available            = mem_available_bytes();
    const std::uint64_t other_host = options.other_host_bytes +
                                    (plan.host_bytes ? 2 * options.bounce_buffer_bytes : 0);
    const std::uint64_t required = plan.host_bytes + other_host;
    const std::uint64_t allowance = plan.mem_available > kHostHeadroomBytes ? plan.mem_available - kHostHeadroomBytes : 0;
    if (required > allowance) {
        throw HostMemoryError(fmt::format(
            "not enough host memory: the RAM expert tier ({:.2f} GiB) plus other host buffers ({:.2f} GiB) need "
            "{:.2f} GiB, but only {:.2f} GiB is available (MemAvailable {:.2f} GiB minus 6 GiB headroom). Free "
            "memory, give the GPU more room for experts (stop other GPU processes, lower --ctx), or reduce "
            "--ngram-cache-mib",
            double(plan.host_bytes) / kGiB, double(other_host) / kGiB, double(required) / kGiB,
            double(allowance) / kGiB, double(plan.mem_available) / kGiB));
    }
    return plan;
}

ExpertPlacement ExpertPlacement::build(const io::Pack& pack, cudaStream_t stream, const PlacementOptions& options) {
    return build(pack, plan_placement(pack, options), stream, options);
}

ExpertPlacement ExpertPlacement::build(const io::Pack& pack, PlacementPlan plan, cudaStream_t stream,
                                       const PlacementOptions& options) {
    ExpertPlacement p;
    p.plan_           = std::move(plan);
    const auto& plan_ = p.plan_;
    const auto& layers = pack.expert_layers();

    p.pool_     = DeviceBuffer(static_cast<std::size_t>(plan_.state_bytes ? plan_.state_offset + plan_.state_bytes
                                                                          : plan_.vram_bytes));
    if (plan_.state_bytes) {
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(p.pool_.p) + plan_.state_offset, 0, plan_.state_bytes,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    p.ram_tier_ = plan_.host_bytes ? HostRegion(static_cast<std::size_t>(plan_.host_bytes)) : HostRegion();
    p.host_slot_of_.assign(std::size_t(kLayers) * kExperts, nullptr);
    p.backup_of_.assign(std::size_t(kLayers) * kExperts, nullptr);
    for (const auto& s : plan_.elastic) p.elastic_[std::size_t(s.layer)].push_back({s.slot, s.tokens, false});
    for (auto& v : p.elastic_)
        std::sort(v.begin(), v.end(), [](const Elastic& a, const Elastic& b) { return a.slot < b.slot; });
    p.update_next_retire();

    std::vector<io::CopyRange> gpu_ranges, host_ranges;
    std::uint64_t host_off = 0, loan_host_off = plan_.loan_host_offset;
    for (int l = 0; l < kLayers; ++l) {
        const auto& layout                 = layers[std::size_t(l)];
        p.blob_stride_[std::size_t(l)]     = layout.blob_stride;
        if (l >= plan_.expert_layers) { continue; }
        auto* base = static_cast<std::uint8_t*>(p.pool_.p) + plan_.layer_offset[std::size_t(l)];
        const bool has_slots = plan_.gpu_slots[std::size_t(l)] > 0 || plan_.spare_slot[std::size_t(l)] >= 0;
        p.slot_base_[std::size_t(l)] = has_slots ? base : nullptr;
        for (int e = 0; e < kExperts; ++e) {
            const auto i    = std::size_t(l) * kExperts + std::size_t(e);
            const auto slot = plan_.slot_of[i];
            if (slot >= 0) gpu_ranges.push_back({layout.blob_offset(e), layout.blob_stride, base + std::uint64_t(slot) * layout.blob_stride});
            if (slot >= 0 && p.loan_slot(l, slot)) {
                auto* backup = p.ram_tier_.data() + loan_host_off;
                p.backup_of_[i] = backup;
                host_ranges.push_back({layout.blob_offset(e), layout.blob_stride, backup});
                loan_host_off += layout.blob_stride;
            } else if (slot < 0 || p.elastic_slot(l, slot)) {
                // CPU experts, and elastic experts whose RAM blob is their backup.
                auto* dst = p.ram_tier_.data() + host_off;
                if (slot < 0) p.host_slot_of_[i] = dst;
                else p.backup_of_[i] = dst;
                host_ranges.push_back({layout.blob_offset(e), layout.blob_stride, dst});
                host_off += layout.blob_stride;
            }
        }
    }

    io::InputFile file(pack.file_path(io::PackFile::Experts));
    p.stats_.gpu_upload = io::upload_regions(file, std::move(gpu_ranges), stream);
    p.stats_.host_fill  = io::read_regions_to_host(file, std::move(host_ranges), options.read_threads);
    if (p.ram_tier_.size()) {
        if (options.register_ram) p.ram_tier_.register_with_cuda();
        p.stats_.host_register_seconds = p.ram_tier_.register_seconds();
    }
    if (p.ram_tier_.size() && !p.ram_tier_.registered()) {
        p.bounce_ = std::make_unique<BounceState>();
        p.bounce_->bytes = options.bounce_buffer_bytes;
        for (std::size_t i = 0; i < 2; ++i) {
            p.bounce_->buffers[i] = std::make_unique<PinnedHostBuffer>(options.bounce_buffer_bytes);
            CUDA_CHECK(cudaEventCreateWithFlags(&p.bounce_->done[i], cudaEventDisableTiming));
        }
        spdlog::warn("expert RAM is not registered: streaming, swaps and loan restores use two {:.2f} MiB "
                     "pinned bounce buffers", double(options.bounce_buffer_bytes) / double(1ull << 20));
    }

    p.slot_of_dev_ = DeviceBuffer(plan_.slot_of.size() * sizeof(std::int32_t));
    p.slot_of_dev_.copy_from_host(plan_.slot_of.data(), p.slot_of_dev_.bytes);
    p.slot_base_dev_ = DeviceBuffer(sizeof(p.slot_base_));
    p.slot_base_dev_.copy_from_host(p.slot_base_.data(), sizeof(p.slot_base_));

    spdlog::info("expert placement: {}", p.summary());
    return p;
}

const std::uint8_t* ExpertPlacement::device_blob(int layer, int expert) const noexcept {
    const auto slot = slot_of(layer, expert);
    if (slot < 0) { return nullptr; }
    return slot_base_[std::size_t(layer)] + std::uint64_t(slot) * blob_stride_[std::size_t(layer)];
}

const ExpertPlacement::Elastic* ExpertPlacement::elastic_at(int layer, int32_t slot) const noexcept {
    const auto& v = elastic_[std::size_t(layer)];
    const auto it = std::lower_bound(v.begin(), v.end(), slot, [](const Elastic& s, int32_t x) { return s.slot < x; });
    return it != v.end() && it->slot == slot ? &*it : nullptr;
}

bool ExpertPlacement::elastic_slot(int layer, int32_t slot) const noexcept { return elastic_at(layer, slot); }

bool ExpertPlacement::backed_slot(int layer, int32_t slot) const noexcept {
    return slot >= 0 && (loan_slot(layer, slot) || elastic_slot(layer, slot));
}

std::byte* ExpertPlacement::assign_spare_to(int layer, int expert) {
    const auto i     = std::size_t(layer) * kExperts + std::size_t(expert);
    const auto spare = plan_.spare_slot[std::size_t(layer)];
    if (borrowed_ || spare < 0 || plan_.slot_of[i] >= 0 || !host_slot_of_[i]) {
        throw std::logic_error("assign_spare_to: expert is not CPU-resident or the layer has no spare slot");
    }
    slot_writes_.emplace_back(std::int32_t(i), spare);
    auto* ram                            = static_cast<std::byte*>(const_cast<void*>(host_slot_of_[i]));
    plan_.slot_of[i]                     = spare;
    host_slot_of_[i]                     = nullptr;
    plan_.spare_slot[std::size_t(layer)] = -1;
    return ram;
}

void ExpertPlacement::release_to_ram(int layer, int expert, std::byte* ram_blob) {
    const auto i    = std::size_t(layer) * kExperts + std::size_t(expert);
    const auto slot = plan_.slot_of[i];
    if (borrowed_ || slot < 0 || backed_slot(layer, slot) || plan_.spare_slot[std::size_t(layer)] >= 0 || !ram_blob) {
        throw std::logic_error("release_to_ram: expert is not in a regular slot or the layer already has a spare");
    }
    slot_writes_.emplace_back(std::int32_t(i), -1);
    host_slot_of_[i]                     = ram_blob;
    plan_.slot_of[i]                     = -1;
    plan_.spare_slot[std::size_t(layer)] = slot;
}

std::int32_t ExpertPlacement::drop(int layer, int expert) {
    const auto i    = std::size_t(layer) * kExperts + std::size_t(expert);
    const auto slot = plan_.slot_of[i];
    if (borrowed_ || !backed_slot(layer, slot) || !backup_of_[i])
        throw std::logic_error("drop: expert is not in a backed slot");
    slot_writes_.emplace_back(std::int32_t(i), -1);
    host_slot_of_[i] = std::exchange(backup_of_[i], nullptr);
    plan_.slot_of[i] = -1;
    return slot;
}

std::int32_t ExpertPlacement::claim_free_slot(int layer) {
    auto& free = free_slots_[std::size_t(layer)];
    if (borrowed_ || free.empty()) return -1;
    const std::int32_t slot = free.back();
    free.pop_back();
    return slot;
}

void ExpertPlacement::fill(int layer, int expert, std::int32_t slot) {
    const auto i = std::size_t(layer) * kExperts + std::size_t(expert);
    if (borrowed_ || !backed_slot(layer, slot) || plan_.slot_of[i] >= 0 || !host_slot_of_[i])
        throw std::logic_error("fill: expert is not CPU-resident or the slot is not backed");
    if (const Elastic* s = elastic_at(layer, slot); s && s->retired) throw std::logic_error("fill: elastic slot is retired");
    slot_writes_.emplace_back(std::int32_t(i), slot);
    backup_of_[i]    = static_cast<std::byte*>(const_cast<void*>(std::exchange(host_slot_of_[i], nullptr)));
    plan_.slot_of[i] = slot;
}

bool ExpertPlacement::retires(int layer, std::int32_t slot, std::int32_t tokens) const noexcept {
    const Elastic* s = elastic_at(layer, slot);
    return s && !s->retired && tokens > s->tokens;
}

bool ExpertPlacement::vacate(std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= next_retire_) return false;
    bool changed = false;
    for (int l = 0; l < plan_.expert_layers; ++l) {
        auto& free = free_slots_[std::size_t(l)];
        for (Elastic& s : elastic_[std::size_t(l)]) {
            if (s.retired || tokens <= s.tokens) continue;
            for (int e = 0; e < kExperts; ++e) {
                const auto i = std::size_t(l) * kExperts + std::size_t(e);
                if (plan_.slot_of[i] != s.slot) continue;
                slot_writes_.emplace_back(std::int32_t(i), -1);
                host_slot_of_[i] = std::exchange(backup_of_[i], nullptr);
                plan_.slot_of[i] = -1;
                changed = true;
            }
            free.erase(std::remove(free.begin(), free.end(), s.slot), free.end());
            s.retired = true;
        }
    }
    update_next_retire();
    if (!changed) return false;
    if (borrowed_) {
        // The device table holds the lent map until end_loan copies the whole host table.
        CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_.p, plan_.slot_of.data(), slot_of_dev_.bytes, cudaMemcpyHostToDevice,
                                   stream));
        slot_writes_.clear();
    } else {
        publish_slot_writes(stream);
    }
    return true;
}

void ExpertPlacement::revalidate(std::int32_t tokens) {
    for (int l = 0; l < plan_.expert_layers; ++l) {
        for (Elastic& s : elastic_[std::size_t(l)]) {
            if (!s.retired || tokens > s.tokens) continue;
            s.retired = false;
            free_slots_[std::size_t(l)].push_back(s.slot);
        }
    }
    update_next_retire();
}

void ExpertPlacement::update_next_retire() noexcept {
    next_retire_ = std::numeric_limits<std::int32_t>::max();
    for (const auto& slots : elastic_)
        for (const Elastic& s : slots)
            if (!s.retired) next_retire_ = std::min(next_retire_, s.tokens);
}

std::int32_t ExpertPlacement::elastic_resident() const noexcept {
    std::int32_t n = 0;
    for (int l = 0; l < plan_.expert_layers; ++l)
        for (int e = 0; e < kExperts; ++e) {
            const auto slot = plan_.slot_of[std::size_t(l) * kExperts + std::size_t(e)];
            n += slot >= 0 && elastic_slot(l, slot);
        }
    return n;
}

void ExpertPlacement::publish_slot_writes(cudaStream_t stream) {
    for (std::size_t first = 0; first < slot_writes_.size(); first += SlotWrites::kMax) {
        SlotWrites batch;
        batch.n = int32_t(std::min<std::size_t>(SlotWrites::kMax, slot_writes_.size() - first));
        for (int32_t k = 0; k < batch.n; ++k) {
            batch.index[k] = slot_writes_[first + std::size_t(k)].first;
            batch.value[k] = slot_writes_[first + std::size_t(k)].second;
        }
        const cudaError_t err = set_slot_entries(static_cast<std::int32_t*>(slot_of_dev_.p), batch, stream);
        if (err != cudaSuccess) throw std::runtime_error(std::string("slot table update: ") + cudaGetErrorString(err));
    }
    slot_writes_.clear();
}

void ExpertPlacement::begin_loan(cudaStream_t stream) {
    if (borrowed_ || !plan_.loan_bytes) throw std::logic_error("prefill loan is unavailable or already active");
    lent_.clear();
    for (int l = 0; l < plan_.expert_layers; ++l) {
        if (!plan_.loan_slots[std::size_t(l)]) continue;
        for (int e = 0; e < kExperts; ++e) {
            const auto i    = std::size_t(l) * kExperts + std::size_t(e);
            const auto slot = plan_.slot_of[i];
            if (slot < 0 || !loan_slot(l, slot)) continue;
            if (!backup_of_[i]) throw std::logic_error("begin_loan: expert in a loan slot without a RAM backup");
            host_slot_of_[i] = backup_of_[i];
            plan_.slot_of[i] = -1;
            lent_.emplace_back(i, slot);
        }
    }
    // The full table copy supersedes recorded slot writes.
    const auto err = cudaMemcpyAsync(slot_of_dev_.p, plan_.slot_of.data(), slot_of_dev_.bytes,
                                    cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    slot_writes_.clear();
    borrowed_ = true;
}

void ExpertPlacement::end_loan(cudaStream_t stream, std::span<const DeviceRange> written) {
    if (!borrowed_) return;
    // Every prefill write to the loan precedes the copies that overwrite it.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const auto& [i, slot] : lent_) {
        const int layer          = int(i / kExperts);
        const std::uint8_t* blob = slot_address(layer, slot);
        const auto bytes         = std::size_t(blob_stride(layer));
        const bool overwritten   = std::any_of(written.begin(), written.end(), [&](const DeviceRange& r) {
            const auto* begin = static_cast<const std::uint8_t*>(r.data);
            return blob < begin + r.bytes && begin < blob + bytes;
        });
        if (overwritten) enqueue_upload(slot_address(layer, slot), backup_of_[i], bytes, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const auto& [i, slot] : lent_) {
        plan_.slot_of[i] = slot;
        host_slot_of_[i] = nullptr;
    }
    lent_.clear();
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_.p, plan_.slot_of.data(), slot_of_dev_.bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    borrowed_ = false;
}

ExpertPlacement::BounceState::~BounceState() {
    for (std::size_t i = 0; i < 2; ++i) {
        if (pending[i]) cudaEventSynchronize(done[i]);
        if (done[i]) cudaEventDestroy(done[i]);
    }
}

void ExpertPlacement::enqueue_upload(void* device_dst, const void* host_src, std::size_t bytes,
                                     cudaStream_t stream) const {
    if (!bounce_) {
        CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, bytes, cudaMemcpyHostToDevice, stream));
        return;
    }
    auto* dst = static_cast<std::byte*>(device_dst);
    auto* src = static_cast<const std::byte*>(host_src);
    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t i = bounce_->next++ % 2;
        if (bounce_->pending[i]) CUDA_CHECK(cudaEventSynchronize(bounce_->done[i]));
        const auto n = std::min(bounce_->bytes, bytes - offset);
        void* staging = bounce_->buffers[i]->data();
        std::memcpy(staging, src + offset, n);
        CUDA_CHECK(cudaMemcpyAsync(dst + offset, staging, n, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(bounce_->done[i], stream));
        bounce_->pending[i] = true;
        offset += n;
    }
}

void ExpertPlacement::enqueue_download(void* host_dst, const void* device_src, std::size_t bytes,
                                       cudaStream_t stream) const {
    if (!bounce_) {
        CUDA_CHECK(cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream));
        return;
    }
    auto* dst = static_cast<std::byte*>(host_dst);
    auto* src = static_cast<const std::byte*>(device_src);
    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t i = bounce_->next++ % 2;
        if (bounce_->pending[i]) CUDA_CHECK(cudaEventSynchronize(bounce_->done[i]));
        const auto n = std::min(bounce_->bytes, bytes - offset);
        void* staging = bounce_->buffers[i]->data();
        auto& copy = bounce_->copy_back[i];
        copy = {dst + offset, staging, n};
        CUDA_CHECK(cudaMemcpyAsync(staging, src + offset, n, cudaMemcpyDeviceToHost, stream));
        // Event completion must certify the RAM destination, not merely its staging buffer.
        // The heap-owned payload remains stable across ExpertPlacement moves and buffer reuse.
        CUDA_CHECK(cudaLaunchHostFunc(stream, [](void* arg) {
            const auto& c = *static_cast<const BounceState::CopyBack*>(arg);
            std::memcpy(c.dst, c.src, c.bytes);
        }, &copy));
        CUDA_CHECK(cudaEventRecord(bounce_->done[i], stream));
        bounce_->pending[i] = true;
        offset += n;
    }
}

std::string ExpertPlacement::summary() const {
    const auto& pl = plan_;
    return fmt::format(
        "{} layers; GPU {} experts + {} ({:.2f} GiB of {:.2f} GiB budget, {:.2f} GiB free) + {} elastic in the "
        "context tail ({:.2f} GiB), RAM {} experts ({:.2f} GiB, {}); predicted hit rate {:.1f}% ({}); GPU upload "
        "{:.2f} GB/s, RAM fill {:.2f} GB/s in {:.2f} s, register {:.2f} s",
        pl.expert_layers, pl.gpu_experts, pl.spares ? "1 spare slot per layer" : "no spare slots",
        double(pl.vram_bytes) / kGiB, double(pl.budget) / kGiB, double(pl.free_vram) / kGiB, pl.elastic.size(),
        double(pl.elastic_bytes) / kGiB, pl.cpu_experts, double(pl.host_bytes) / kGiB,
        ram_tier_.size() == 0 ? "empty" : ram_tier_.registered() ? "registered" : "NOT registered",
        100.0 * pl.predicted_hit_rate,
        (pl.stats_total || pl.decode_profile_total) ? "of decode routing" : pl.scored ? "prefill profile" : "no profile: uniform",
        stats_.gpu_upload.gb_per_s(), stats_.host_fill.gb_per_s(), stats_.host_fill.seconds,
        stats_.host_register_seconds);
}

} // namespace fngine::model
