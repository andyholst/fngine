#include "engine/session.h"

#include "model/runtime.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>

namespace fngine::engine {

std::size_t common_prefix(std::span<const std::int32_t> a,
                          std::span<const std::int32_t> b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i       = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

std::optional<std::size_t> select_snapshot(std::span<const std::int32_t> positions,
                                           std::size_t lcp, std::size_t prompt_len) {
    if (lcp == 0 || prompt_len == 0) return std::nullopt;
    const std::size_t limit = std::min(lcp - 1, prompt_len - 1);
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const std::int32_t p = positions[i];
        if (p <= 0 || static_cast<std::size_t>(p) > limit) continue;
        if (!best || p > positions[*best]) best = i;
    }
    return best;
}

std::int32_t SessionStore::restore_best(model::ModelRuntime& rt, std::size_t lcp,
                                        std::size_t prompt_len) {
    std::vector<std::int32_t> positions;
    positions.reserve(entries_.size());
    for (const Entry& e : entries_) positions.push_back(e.position);
    const auto index = select_snapshot(positions, lcp, prompt_len);
    if (!index) return 0;
    Entry& entry   = entries_[*index];
    entry.last_use = ++clock_;
    rt.restore_from(entry.buffer->data(), entry.position, entry.history);
    return entry.position;
}

bool SessionStore::take(model::ModelRuntime& rt) {
    if (slots_ == 0) return false;
    const std::int32_t position = rt.position();
    if (position <= 0) return false;
    Entry* target = nullptr;
    for (Entry& e : entries_) {
        if (e.position == position) target = &e;
    }
    if (target == nullptr) {
        if (entries_.size() < slots_) {
            std::unique_ptr<PinnedHostBuffer> buffer;
            if (!free_.empty()) {
                buffer = std::move(free_.back());
                free_.pop_back();
            } else {
                try {
                    buffer = std::make_unique<PinnedHostBuffer>(rt.snapshot_bytes());
                } catch (const std::exception& error) {
                    spdlog::warn("session snapshot buffer allocation failed: {}", error.what());
                    return false;
                }
            }
            entries_.push_back(Entry{.buffer = std::move(buffer)});
            target = &entries_.back();
        } else {
            target = &*std::min_element(entries_.begin(), entries_.end(),
                                        [](const Entry& a, const Entry& b) {
                                            return a.last_use < b.last_use;
                                        });
        }
    }
    rt.snapshot_to(target->buffer->data());
    target->position = position;
    target->history  = rt.ngram_history();
    target->last_use = ++clock_;
    return true;
}

void SessionStore::drop_above(std::int32_t position) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->position > position) {
            free_.push_back(std::move(it->buffer));
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

std::optional<std::int32_t> SessionStore::newest_position() const noexcept {
    std::optional<std::int32_t> best;
    for (const Entry& e : entries_) {
        if (!best || e.position > *best) best = e.position;
    }
    return best;
}

std::size_t SessionStore::pinned_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const Entry& e : entries_) bytes += e.buffer->size();
    for (const auto& b : free_) bytes += b->size();
    return bytes;
}

} // namespace fngine::engine
