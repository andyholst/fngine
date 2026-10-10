#include "model/state.h"

#include "core/device.h"
#include "core/model_dims.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fngine::model {
namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

int32_t round_capacity(int32_t tokens) {
  if (tokens <= 0) throw std::invalid_argument("max_context must be positive");
  const int32_t page = kernels::kQsaKvPageTokens;
  if (tokens > std::numeric_limits<int32_t>::max() - (page - 1))
    throw std::invalid_argument("max_context exceeds the page-aligned position range");
  return (tokens + page - 1) / page * page;
}

struct Layout {
  std::size_t gdn_pool = 0, gdn_records = 0, gdn_control = 0;
  std::array<std::size_t, kQsaCaches> kv{}, indexer{};
  std::size_t ple = 0, total = 0;
  std::size_t gdn_pool_bytes = 0, kv_bytes = 0, indexer_bytes = 0;
};

Layout layout(const StateOptions& o) {
  const int32_t cap = round_capacity(o.max_context);
  Layout l;
  std::size_t off = 0;
  l.gdn_pool_bytes = kernels::gdn_state_pool_bytes(1);
  l.gdn_pool = off;    off = align_up(off + l.gdn_pool_bytes);
  l.gdn_records = off; off = align_up(off + kernels::gdn_records_bytes());
  l.gdn_control = off; off = align_up(off + sizeof(kernels::GdnStepControl));
  l.kv_bytes = kernels::qsa_kv_layer_bytes(o.kv_format, cap);
  l.indexer_bytes = kernels::indexer_state_bytes(cap);
  for (int i = 0; i < kQsaCaches; ++i) {
    l.kv[std::size_t(i)] = off;      off = align_up(off + l.kv_bytes);
    l.indexer[std::size_t(i)] = off; off = align_up(off + l.indexer_bytes);
  }
  l.ple = off; off = align_up(off + std::size_t(dims::kPleConvState) * dims::kStreamHidden * 2);
  l.total = off;
  return l;
}

}  // namespace

std::size_t ModelState::device_bytes(const StateOptions& options) { return layout(options).total; }

std::vector<PositionRegion> ModelState::position_regions(const StateOptions& options) {
  const Layout l = layout(options);
  const int32_t cap = round_capacity(options.max_context);
  const auto planes = kernels::qsa_kv_planes(options.kv_format, cap);
  std::vector<PositionRegion> regions;
  for (int i = 0; i < kQsaCaches; ++i) {
    for (const auto& p : planes)
      if (p.bytes) regions.push_back({l.kv[std::size_t(i)] + p.offset, p.bytes, p.bytes / std::size_t(cap)});
    // Block keys: one 128-wide bf16 row per 4 tokens, after the 16-entry raw-key ring.
    const std::size_t keys = std::size_t(cap / 4) * 128 * 2;
    regions.push_back({l.indexer[std::size_t(i)] + align_up(16 * 128 * 2), keys, keys / std::size_t(cap)});
  }
  return regions;
}

ModelState::ModelState(const StateOptions& options, void* memory)
    : options_(options), capacity_(round_capacity(options.max_context)) {
  const Layout l = layout(options);
  bytes_ = l.total;
  if (memory) {
    base_ = static_cast<unsigned char*>(memory);
  } else {
    buffer_ = DeviceBuffer(l.total);
    buffer_.fill(0);
    base_ = static_cast<unsigned char*>(buffer_.p);
  }
  auto* base = base_;
  gdn_pool_bytes_ = l.gdn_pool_bytes;
  gdn_pool_ = kernels::gdn_state_pool_bind(base + l.gdn_pool, l.gdn_pool_bytes, 1);
  gdn_records_ = kernels::gdn_records_bind(base + l.gdn_records, kernels::gdn_records_bytes());
  gdn_control_ = reinterpret_cast<kernels::GdnStepControl*>(base + l.gdn_control);
  for (int i = 0; i < kQsaCaches; ++i) {
    QsaLayerState& q = qsa_[std::size_t(i)];
    q.kv = kernels::qsa_kv_cache_view(options.kv_format, base + l.kv[std::size_t(i)], capacity_);
    auto* ix = base + l.indexer[std::size_t(i)];
    q.ring = reinterpret_cast<__nv_bfloat16*>(ix);
    q.block_keys = reinterpret_cast<__nv_bfloat16*>(ix + align_up(16 * 128 * 2));
    if (align_up(16 * 128 * 2) + std::size_t(capacity_ / 4) * 128 * 2 > l.indexer_bytes)
      throw std::logic_error("indexer state layout does not fit indexer_state_bytes");
  }
  ple_state_ = reinterpret_cast<__nv_bfloat16*>(base + l.ple);
}

void ModelState::reset(cudaStream_t stream) {
  std::size_t at = 0;
  for (const PositionRegion& r : position_regions(options_)) {
    if (r.offset > at) CUDA_CHECK(cudaMemsetAsync(base_ + at, 0, r.offset - at, stream));
    at = std::max(at, r.offset + r.bytes);
  }
  if (bytes_ > at) CUDA_CHECK(cudaMemsetAsync(base_ + at, 0, bytes_ - at, stream));
  const kernels::GdnStepControl control{0, 0, 0, 0};
  CUDA_CHECK(cudaMemcpyAsync(gdn_control_, &control, sizeof(control), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

float* ModelState::gdn_recurrent(int32_t gdn_layer) const {
  return kernels::gdn_recurrent_state(gdn_pool_, 0, gdn_layer);
}

__nv_bfloat16* ModelState::gdn_conv(int32_t gdn_layer) const {
  return kernels::gdn_conv_state(gdn_pool_, 0, gdn_layer);
}

std::array<ModelState::Region, 3 + kQsaCaches> ModelState::snapshot_regions() const {
  std::array<Region, 3 + kQsaCaches> r{};
  r[0] = {gdn_pool_.recurrent, kernels::gdn_state_slot_recurrent_bytes()};
  r[1] = {gdn_pool_.conv, kernels::gdn_state_slot_conv_bytes()};
  r[2] = {ple_state_, std::size_t(dims::kPleConvState) * dims::kStreamHidden * 2};
  for (int i = 0; i < kQsaCaches; ++i) r[std::size_t(3 + i)] = {qsa_[std::size_t(i)].ring, 16 * 128 * 2};
  return r;
}

}  // namespace fngine::model
