#pragma once

// Routed-expert placement: the hottest (layer, expert) pairs by profile go into per-layer VRAM
// slot arrays, the complement into one registered RAM-tier HostRegion.
#include "core/arena.h"
#include "core/host_memory.h"
#include "io/direct_io.h"
#include "io/pack.h"
#include "model/state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fngine::model {

// Thrown by the startup host-memory check. The CLI maps it to exit code 2.
class HostMemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline constexpr std::uint64_t kHostHeadroomBytes = 6ull << 30;

// Device bytes [data, data + bytes).
struct DeviceRange {
    const void* data   = nullptr;
    std::uint64_t bytes = 0;
};

struct PlacementOptions {
    // Device bytes the engine still has to allocate after placement (KV, state, workspaces, ...).
    std::uint64_t fixed_vram_bytes = 0;
    std::uint64_t vram_reserve_mib = 512;
    // Optional hard cap on the expert pool (tests, or sharing the GPU).
    std::optional<std::uint64_t> vram_cap_bytes;
    // Device bytes the caller already holds (dense weights) — only used for the low-VRAM warning.
    std::uint64_t preloaded_vram_bytes = 0;
    // Host bytes the engine still has to allocate (n-gram cache, snapshots, exchange buffers),
    // added to the RAM tier for the startup check against MemAvailable - 6 GiB.
    std::uint64_t other_host_bytes = 0;
    // Decode routing counts u64 [48][512] persisted from generation (decode steps only); empty =
    // <pack>/routing-stats.bin when it exists.
    std::filesystem::path routing_stats;
    // Decode routing profile u64 [48][512] collected by generating with representative chat
    // prompts; empty = <pack>/decode-profile.bin when it exists.
    std::filesystem::path decode_profile;
    // Only the first `expert_layers` layers get experts (others: no GPU slot, no RAM copy).
    // Diagnostic/test knob; the engine always uses 48.
    int expert_layers = 48;
    int read_threads  = 4;  // parallel O_DIRECT readers filling the RAM tier
    // Reserve one spare GPU slot per layer (budget permitting) for in-session expert swaps.
    bool spare_slots = true;
    // Prefill scratch + streaming ring lent from a contiguous tail of the expert pool.
    std::uint64_t loan_bytes = 0;
    // Device state (ModelState) appended to the pool in one allocation and zero-filled before any
    // upload. With spare slots, the bytes of its position regions beyond the current context hold
    // elastic expert slots (ranked after the regular residents).
    std::uint64_t state_bytes = 0;
    std::vector<PositionRegion> state_regions;
    // Registration failure keeps transfers alive through two pinned bounce buffers. Tests can
    // force pageable RAM and smaller buffers to cover segmented transfer/reuse boundaries.
    bool register_ram = true;
    std::size_t bounce_buffer_bytes = 64ull << 20;
};

// The decision, computed without allocating anything (cudaMemGetInfo + pack metadata only).
struct PlacementPlan {
    std::uint64_t free_vram       = 0;  // cudaMemGetInfo free when planned
    std::uint64_t budget          = 0;  // bytes available to the expert pool
    std::uint64_t vram_bytes      = 0;  // bytes of GPU-resident blobs (sum of slot arrays)
    std::uint64_t host_bytes      = 0;  // RAM tier bytes (2 MiB rounded)
    std::uint64_t mem_available   = 0;  // /proc/meminfo MemAvailable at plan time
    int expert_layers             = 48;
    std::int64_t gpu_experts      = 0;
    std::int64_t cpu_experts      = 0;
    double predicted_hit_rate     = 0;  // sum score(resident) / sum score
    bool scored                   = false;  // false: no profile/stats, deterministic (layer, expert) order
    std::uint64_t stats_total     = 0;      // routing-stats.bin pairs added to the score (0: none)
    std::uint64_t decode_profile_total = 0; // decode-profile.bin pairs (0: none)
    bool spares                   = false;  // one spare slot per loaded layer is reserved
    // Host mirror of the device table [48*512]: GPU slot or -1. Expert swaps keep it current.
    std::vector<std::int32_t> slot_of;
    std::array<std::int32_t, 48> gpu_slots{};  // GPU-resident experts per layer (constant under swaps)
    std::array<std::int32_t, 48> cpu_slots{};  // RAM-tier blobs per layer (constant under swaps)
    std::array<std::int32_t, 48> spare_slot{};  // the layer's free slot index, -1 = none
    std::vector<std::uint64_t> score;      // [48*512]
    std::array<std::uint64_t, 48> layer_offset{};
    std::array<std::int32_t, 48> loan_first_slot{};
    std::array<std::int32_t, 48> loan_slots{};
    std::uint64_t loan_offset = 0;
    std::uint64_t loan_bytes = 0;
    std::uint64_t loan_host_offset = 0;
    std::uint64_t state_offset = 0;  // the appended state within the pool allocation
    std::uint64_t state_bytes = 0;
    // Elastic slots: (pair, slot) of their initial experts and the context they allow.
    struct ElasticSlot {
        std::int32_t layer = 0, slot = 0, tokens = 0;
        std::int32_t expert = -1;
    };
    std::vector<ElasticSlot> elastic;
    std::uint64_t elastic_bytes = 0;  // stride bytes of the elastic slots
};

// Plans placement. Throws HostMemoryError when the RAM tier plus other_host_bytes exceeds
// MemAvailable - 6 GiB, and io::PackError on malformed stats.
PlacementPlan plan_placement(const io::Pack& pack, const PlacementOptions& options);

struct PlacementLoadStats {
    io::TransferStats gpu_upload;
    io::TransferStats host_fill;
    double host_register_seconds = 0;
};

class ExpertPlacement {
public:
    // Plans (see plan_placement), allocates the slot arrays and the RAM tier, uploads GPU-resident
    // blobs on `stream`, fills the RAM tier with O_DIRECT reads and registers it with CUDA.
    static ExpertPlacement build(const io::Pack& pack, cudaStream_t stream, const PlacementOptions& options);
    static ExpertPlacement build(const io::Pack& pack, PlacementPlan plan, cudaStream_t stream,
                                 const PlacementOptions& options);

    ExpertPlacement()                                  = default;
    ExpertPlacement(ExpertPlacement&&) noexcept        = default;
    ExpertPlacement& operator=(ExpertPlacement&&) noexcept = default;

    [[nodiscard]] const PlacementPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const PlacementLoadStats& stats() const noexcept { return stats_; }

    // Per layer: device base of the slot array (slot s at base + s * blob_stride) and stride.
    [[nodiscard]] const std::uint8_t* slot_base(int layer) const noexcept { return slot_base_[std::size_t(layer)]; }
    [[nodiscard]] std::uint64_t blob_stride(int layer) const noexcept { return blob_stride_[std::size_t(layer)]; }
    // Device int32 [48][512]: GPU slot of (layer, expert), -1 = CPU-computed.
    [[nodiscard]] const std::int32_t* slot_of_device() const noexcept {
        return static_cast<const std::int32_t*>(slot_of_dev_.p);
    }
    // Device array [48] of slot_base pointers (for kernels indexing by layer).
    [[nodiscard]] const std::uint8_t* const* slot_base_device() const noexcept {
        return static_cast<const std::uint8_t* const*>(slot_base_dev_.p);
    }
    [[nodiscard]] std::int32_t slot_of(int layer, int expert) const noexcept {
        return plan_.slot_of[std::size_t(layer) * 512 + std::size_t(expert)];
    }
    // RAM-tier blob of a CPU expert (nullptr for GPU-resident or unloaded experts).
    [[nodiscard]] const std::byte* host_blob(int layer, int expert) const noexcept {
        return static_cast<const std::byte*>(host_slot_of_[std::size_t(layer) * 512 + std::size_t(expert)]);
    }
    // [48*512] RAM-tier blob pointers (the CPU expert pool reads this table every step).
    [[nodiscard]] const void* const* host_slot_table() const noexcept { return host_slot_of_.data(); }
    [[nodiscard]] const HostRegion& ram_tier() const noexcept { return ram_tier_; }
    // Device address of a GPU-resident blob (nullptr for CPU experts).
    [[nodiscard]] const std::uint8_t* device_blob(int layer, int expert) const noexcept;
    // Device address of a slot of the layer's array.
    [[nodiscard]] std::uint8_t* slot_address(int layer, int32_t slot) const noexcept {
        return const_cast<std::uint8_t*>(slot_base_[std::size_t(layer)]) +
               std::uint64_t(slot) * blob_stride_[std::size_t(layer)];
    }
    [[nodiscard]] std::int32_t spare_slot(int layer) const noexcept { return plan_.spare_slot[std::size_t(layer)]; }
    [[nodiscard]] bool loan_slot(int layer, int32_t slot) const noexcept {
        const auto l = std::size_t(layer);
        return slot >= plan_.loan_first_slot[l] && slot < plan_.loan_first_slot[l] + plan_.loan_slots[l];
    }
    [[nodiscard]] void* loan_data() const noexcept {
        return plan_.loan_bytes ? static_cast<std::uint8_t*>(pool_.p) + plan_.loan_offset : nullptr;
    }
    // Prompt storage loan: prefill scratch lives in the loan slots. begin_loan makes every expert
    // currently in a loan slot CPU-computed from its RAM backup. end_loan re-uploads the backups of
    // those whose slots overlap `written` (every device range prefill wrote in the loan) and
    // returns all of them to their slots. Only between fully idle steps.
    [[nodiscard]] bool borrowed() const noexcept { return borrowed_; }
    void begin_loan(cudaStream_t stream);
    void end_loan(cudaStream_t stream, std::span<const DeviceRange> written);

    // Backed slots (loan and elastic slots) hold experts whose RAM blob stays allocated as their
    // backup, so they leave the GPU without a download.
    [[nodiscard]] bool backed_slot(int layer, int32_t slot) const noexcept;
    [[nodiscard]] bool elastic_slot(int layer, int32_t slot) const noexcept;

    // Slot-ownership changes for expert swaps. Call only between steps (no graph, prefill chunk
    // or CPU pool step running). The host tables change immediately; the device table changes are
    // recorded and must be issued by publish_slot_writes before the next step.
    // CPU expert `e` takes the layer's spare (a regular slot whose bytes the caller uploaded);
    // returns e's RAM blob, which no expert uses any more.
    std::byte* assign_spare_to(int layer, int expert);
    // GPU expert `e` in a regular slot becomes CPU-computed from `ram_blob` (holding its bytes);
    // its slot becomes the layer's spare.
    void release_to_ram(int layer, int expert, std::byte* ram_blob);
    // GPU expert `e` in a backed slot becomes CPU-computed from its backup; returns the slot, now
    // empty and reserved for the caller's fill.
    std::int32_t drop(int layer, int expert);
    // Reserves an empty backed slot of the layer (-1: none) for a fill.
    std::int32_t claim_free_slot(int layer);
    [[nodiscard]] std::size_t free_slots(int layer) const noexcept { return free_slots_[std::size_t(layer)].size(); }
    // CPU expert `e` takes the reserved backed slot (bytes uploaded by the caller); its RAM blob
    // becomes its backup.
    void fill(int layer, int expert, std::int32_t slot);
    // Issues the recorded device slot-table changes on `stream` (the compute stream), in batches.
    void publish_slot_writes(cudaStream_t stream);
    // Ordered on stream, including the pageable D2H copy-back. Caller keeps source/destination
    // ownership stable until its completion event; no CPU pool may read a recycled RAM blob.
    void enqueue_upload(void* device_dst, const void* host_src, std::size_t bytes, cudaStream_t stream) const;
    void enqueue_download(void* host_dst, const void* device_src, std::size_t bytes, cudaStream_t stream) const;

    // Elastic slots live in the context tail of the device state (state_memory()); kernels read
    // the context only at written positions. Before the sequence may write `tokens` positions:
    // experts in slots overlapping them fall back to their backups and those slots retire (slot
    // writes published on `stream`). The caller first cancels copies into retiring slots it
    // reserved (retires()). Returns true when a slot map changed.
    bool vacate(std::int32_t tokens, cudaStream_t stream);
    // Whether vacate(tokens) retires `slot` of `layer`.
    [[nodiscard]] bool retires(int layer, std::int32_t slot, std::int32_t tokens) const noexcept;
    // The sequence was truncated to `tokens` positions: retired slots beyond them are free again.
    void revalidate(std::int32_t tokens);
    // Device state appended to the pool (nullptr when none).
    [[nodiscard]] void* state_memory() const noexcept {
        return plan_.state_bytes ? static_cast<std::uint8_t*>(pool_.p) + plan_.state_offset : nullptr;
    }
    // Experts in elastic slots now.
    [[nodiscard]] std::int32_t elastic_resident() const noexcept;

    [[nodiscard]] std::string summary() const;

private:
    PlacementPlan plan_;
    PlacementLoadStats stats_;
    DeviceBuffer pool_;
    DeviceBuffer slot_of_dev_;
    DeviceBuffer slot_base_dev_;
    std::array<const std::uint8_t*, 48> slot_base_{};
    std::array<std::uint64_t, 48> blob_stride_{};
    HostRegion ram_tier_;
    std::vector<const void*> host_slot_of_;
    // RAM backup of each GPU expert in a backed slot ([48*512], null otherwise).
    std::vector<std::byte*> backup_of_;
    // Elastic slots: usable while the sequence holds at most `tokens` positions.
    struct Elastic {
        std::int32_t slot = -1, tokens = 0;
        bool retired = false;
    };
    std::array<std::vector<Elastic>, 48> elastic_;  // per layer, ascending slot
    [[nodiscard]] const Elastic* elastic_at(int layer, std::int32_t slot) const noexcept;
    void update_next_retire() noexcept;
    std::int32_t next_retire_ = std::numeric_limits<std::int32_t>::max();  // least `tokens` of a live slot
    // Empty, unreserved backed slots per layer.
    std::array<std::vector<std::int32_t>, 48> free_slots_;
    // Experts lent by begin_loan: (pair index, slot).
    std::vector<std::pair<std::size_t, std::int32_t>> lent_;
    // Device slot-table writes (pair index, slot) awaiting publish_slot_writes.
    std::vector<std::pair<std::int32_t, std::int32_t>> slot_writes_;
    bool borrowed_ = false;
    struct BounceState {
        struct CopyBack { void* dst; const void* src; std::size_t bytes; };
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> buffers;
        std::array<cudaEvent_t, 2> done{};
        std::array<bool, 2> pending{};
        std::array<CopyBack, 2> copy_back{};
        std::size_t bytes = 0;
        unsigned next = 0;
        ~BounceState();
    };
    std::unique_ptr<BounceState> bounce_;
};

} // namespace fngine::model
