#include "core/exchange.h"

#include <stdexcept>

namespace fngine {
namespace {

constexpr std::size_t align64(std::size_t v) { return (v + 63) & ~std::size_t{63}; }

struct PrefillLayout {
  std::size_t header, row_token, cold_count, cold_expert, cold_weight, x, y, total;
};

PrefillLayout prefill_layout(int32_t capacity) {
  if (capacity <= 0) throw std::invalid_argument("prefill exchange capacity must be positive");
  const std::size_t c = static_cast<std::size_t>(capacity);
  PrefillLayout l{};
  std::size_t off = 0;
  l.header = off;      off = align64(off + 16 * sizeof(int32_t));
  l.row_token = off;   off = align64(off + c * sizeof(int32_t));
  l.cold_count = off;  off = align64(off + c * sizeof(int32_t));
  l.cold_expert = off; off = align64(off + c * dims::kTopK * sizeof(int32_t));
  l.cold_weight = off; off = align64(off + c * dims::kTopK * sizeof(float));
  l.x = off;           off = align64(off + c * dims::kHidden * sizeof(__nv_bfloat16));
  l.y = off;           off = align64(off + c * dims::kHidden * sizeof(float));
  l.total = off;
  return l;
}

}  // namespace

std::size_t prefill_exchange_bytes(int32_t capacity) { return prefill_layout(capacity).total; }

PrefillExchangeView make_prefill_exchange_view(void* base, int32_t capacity) {
  if (reinterpret_cast<std::uintptr_t>(base) % 64 != 0)
    throw std::invalid_argument("prefill exchange base must be 64-byte aligned");
  const PrefillLayout l = prefill_layout(capacity);
  auto* b = static_cast<unsigned char*>(base);
  PrefillExchangeView v;
  v.capacity = capacity;
  v.header = reinterpret_cast<int32_t*>(b + l.header);
  v.row_token = reinterpret_cast<int32_t*>(b + l.row_token);
  v.cold_count = reinterpret_cast<int32_t*>(b + l.cold_count);
  v.cold_expert = reinterpret_cast<int32_t*>(b + l.cold_expert);
  v.cold_weight = reinterpret_cast<float*>(b + l.cold_weight);
  v.x = reinterpret_cast<__nv_bfloat16*>(b + l.x);
  v.y = reinterpret_cast<float*>(b + l.y);
  return v;
}

}  // namespace fngine
