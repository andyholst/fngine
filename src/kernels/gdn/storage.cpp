// Modified for fngine from cinference:src/core/linear_attention_state.cpp and
// cinference:src/core/gdn_replay_records.cpp @ 383e5db. See NOTICE.
//
// fngine keeps one fixed geometry (36 layers, 48 value heads, 10240 conv channels, record width
// 16, a single sequence), so the pool and record planners reduce to fixed region lists.

#include "kernels/gdn/gdn.h"
#include "kernels/gdn/layout.h"

#include <cstdint>
#include <stdexcept>

namespace fngine::kernels {
namespace {

using namespace gdn_dims;

constexpr std::size_t kAlign = 256;

struct PoolLayout {
    gdn::LayoutRegion recurrent;
    gdn::LayoutRegion conv;
    std::size_t total = 0;
};

PoolLayout plan_pool(std::int32_t slots) {
    if (slots <= 0) { throw std::invalid_argument("GDN state pool: slot count must be positive"); }
    gdn::LayoutBuilder builder;
    PoolLayout layout;
    layout.recurrent = builder.add(static_cast<std::size_t>(slots) * gdn_state_slot_recurrent_bytes(),
                                   kAlign, "GDN recurrent state");
    layout.conv  = builder.add(static_cast<std::size_t>(slots) * gdn_state_slot_conv_bytes(), kAlign,
                               "GDN conv state");
    layout.total = builder.finish(kAlign, "GDN state pool");
    return layout;
}

struct RecordLayout {
    gdn::LayoutRegion conv;
    gdn::LayoutRegion key;
    gdn::LayoutRegion value;
    gdn::LayoutRegion gate;
    std::size_t total = 0;
};

RecordLayout plan_records() {
    constexpr std::size_t columns = static_cast<std::size_t>(kLayers) * kMaxRecordWidth;
    gdn::LayoutBuilder builder;
    RecordLayout layout;
    layout.conv  = builder.add(columns * kConvChannels * sizeof(__nv_bfloat16), kAlign,
                               "GDN conv records");
    layout.key   = builder.add(columns * kKeyDim * sizeof(__nv_bfloat16), kAlign, "GDN key records");
    layout.value = builder.add(columns * kValueDim * sizeof(__nv_bfloat16), kAlign,
                               "GDN value records");
    layout.gate  = builder.add(columns * kValueHeads * 2 * sizeof(float), kAlign, "GDN gate records");
    layout.total = builder.finish(kAlign, "GDN records");
    return layout;
}

void require_slot_layer(const GdnStatePool& pool, std::int32_t slot, std::int32_t layer) {
    if (slot < 0 || slot >= pool.slots) { throw std::out_of_range("GDN state slot out of range"); }
    if (layer < 0 || layer >= kLayers) { throw std::out_of_range("GDN layer out of range"); }
}

} // namespace

std::size_t gdn_state_slot_recurrent_bytes() {
    return static_cast<std::size_t>(kLayers) * kRecurrentLayerElems * sizeof(float);
}

std::size_t gdn_state_slot_conv_bytes() {
    return static_cast<std::size_t>(kLayers) * kConvLayerElems * sizeof(__nv_bfloat16);
}

std::size_t gdn_state_pool_bytes(std::int32_t slots) { return plan_pool(slots).total; }

GdnStatePool gdn_state_pool_bind(void* backing, std::size_t backing_bytes, std::int32_t slots) {
    const PoolLayout layout = plan_pool(slots);
    GdnStatePool pool;
    pool.recurrent = static_cast<float*>(layout.recurrent.bind(backing, backing_bytes));
    pool.conv      = static_cast<__nv_bfloat16*>(layout.conv.bind(backing, backing_bytes));
    pool.slots     = slots;
    return pool;
}

float* gdn_recurrent_state(const GdnStatePool& pool, std::int32_t slot, std::int32_t layer) {
    require_slot_layer(pool, slot, layer);
    return pool.recurrent +
           (static_cast<std::size_t>(slot) * kLayers + static_cast<std::size_t>(layer)) *
               kRecurrentLayerElems;
}

__nv_bfloat16* gdn_conv_state(const GdnStatePool& pool, std::int32_t slot, std::int32_t layer) {
    require_slot_layer(pool, slot, layer);
    return pool.conv +
           (static_cast<std::size_t>(slot) * kLayers + static_cast<std::size_t>(layer)) *
               kConvLayerElems;
}

std::size_t gdn_records_bytes() { return plan_records().total; }

GdnRecords gdn_records_bind(void* backing, std::size_t backing_bytes) {
    const RecordLayout layout = plan_records();
    GdnRecords records;
    records.conv  = static_cast<__nv_bfloat16*>(layout.conv.bind(backing, backing_bytes));
    records.key   = static_cast<__nv_bfloat16*>(layout.key.bind(backing, backing_bytes));
    records.value = static_cast<__nv_bfloat16*>(layout.value.bind(backing, backing_bytes));
    records.gate  = static_cast<float*>(layout.gate.bind(backing, backing_bytes));
    return records;
}

} // namespace fngine::kernels
