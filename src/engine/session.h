#pragma once

// Session snapshots for prompt-prefix reuse across requests (single sequence).
//
// The engine tracks the token sequence `seq` whose state the runtime holds: every prompt token
// plus every generated token, including the not-yet-processed pending anchor (the MTP KV of the
// last processed position was built with it). A snapshot taken at position p holds the
// non-position-addressed state after seq[0..p); the position-addressed KV/indexer caches stay
// valid for positions < p while seq[0..p] is unchanged.
//
// Invariant: every stored snapshot matches the current `seq` prefix. Whenever the sequence is
// rewound to position p (restore or reset), snapshots above p are dropped.
//
// Reuse rule for a new prompt with longest common prefix L against `seq`: restore the snapshot
// with the largest position p <= min(L - 1, prompt_len - 1) (strictly below L: token p must be
// unchanged because the MTP KV at p - 1 was built with it; at least one prompt token must be
// prefilled to produce logits).

#include "core/arena.h"
#include "io/ngram_store.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace fngine::model {
class ModelRuntime;
}

namespace fngine::engine {

// Minimum distance between snapshots taken at prefill chunk boundaries.
inline constexpr std::int32_t kSnapshotInterval = 4096;

// Longest common prefix of two token sequences.
[[nodiscard]] std::size_t common_prefix(std::span<const std::int32_t> a,
                                        std::span<const std::int32_t> b) noexcept;

// Index of the snapshot to restore (largest position <= min(lcp - 1, prompt_len - 1), position
// > 0), or nullopt.
[[nodiscard]] std::optional<std::size_t> select_snapshot(std::span<const std::int32_t> positions,
                                                         std::size_t lcp, std::size_t prompt_len);

class SessionStore {
public:
    explicit SessionStore(std::uint32_t slots) : slots_(slots) {}

    struct Entry {
        std::int32_t position = 0;
        io::NgramHistory history{};
        std::uint64_t last_use = 0;
        std::unique_ptr<PinnedHostBuffer> buffer;
    };

    // Restores into `rt` the best snapshot for a prompt (see the reuse rule); returns the restored
    // position or 0 when none applies (the caller then resets the sequence).
    std::int32_t restore_best(model::ModelRuntime& rt, std::size_t lcp, std::size_t prompt_len);

    // Snapshots the runtime at rt.position() (> 0). Replaces a snapshot at the same position;
    // otherwise uses a free slot (pinned buffer allocated lazily) or evicts the least recently
    // used one. Returns false when no pinned buffer could be allocated.
    bool take(model::ModelRuntime& rt);

    // Drops snapshots with position > `position` (their state is past a divergence point).
    void drop_above(std::int32_t position);
    void clear() { drop_above(0); }

    [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return static_cast<std::uint32_t>(entries_.size());
    }
    [[nodiscard]] std::optional<std::int32_t> newest_position() const noexcept;
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;

private:
    std::uint32_t slots_;
    std::uint64_t clock_ = 0;
    std::vector<Entry> entries_;
    std::vector<std::unique_ptr<PinnedHostBuffer>> free_;
};

} // namespace fngine::engine
