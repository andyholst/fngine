#pragma once

// Persistent per-sequence device state: GDN recurrent/conv state and verify records, the QSA
// paged KV caches and indexer state (12 main layers + the MTP layer), and the PLE conv state.
#include "core/arena.h"
#include "kernels/gdn/gdn.h"
#include "kernels/qsa/qsa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fngine::model {

// QSA cache index: main QSA layers 0..11 (layer / 4), MTP = 12.
inline constexpr int32_t kQsaCaches = 13;
inline constexpr int32_t kMtpCache = 12;

struct StateOptions {
  int32_t max_context = 131072;  // tokens (rounded up to whole 64-token pages)
  kernels::QsaKvFormat kv_format = kernels::QsaKvFormat::K8V8;
};

struct QsaLayerState {
  kernels::QsaKvCache kv;
  __nv_bfloat16* ring = nullptr;        // [16][128]
  __nv_bfloat16* block_keys = nullptr;  // [capacity / 4][128]
};

// A position-major plane of the state: tokens < n occupy its first
// ceil(n / 64) * 64 * bytes_per_token bytes.
struct PositionRegion {
  std::size_t offset = 0;  // from the start of the state allocation
  std::size_t bytes = 0;
  std::size_t bytes_per_token = 0;
};

class ModelState {
 public:
  static std::size_t device_bytes(const StateOptions& options);
  // Every KV plane and indexer block-key array (address order).
  static std::vector<PositionRegion> position_regions(const StateOptions& options);

  // `memory`: device_bytes(options) zero-filled, 256-byte aligned device bytes owned by the caller;
  // null allocates them.
  explicit ModelState(const StateOptions& options, void* memory = nullptr);
  ModelState(const ModelState&) = delete;
  ModelState& operator=(const ModelState&) = delete;

  // New sequence: zeroes everything outside position_regions (those are written before they
  // are read, and their tails may hold elastic expert slots). GDN step control: source =
  // destination = slot 0.
  void reset(cudaStream_t stream);

  [[nodiscard]] int32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] int32_t max_blocks() const noexcept { return capacity_ / 4; }
  [[nodiscard]] kernels::QsaKvFormat kv_format() const noexcept { return options_.kv_format; }

  // GDN: one state slot (the verify fold commits in place, source == destination).
  [[nodiscard]] const kernels::GdnStatePool& gdn_pool() const noexcept { return gdn_pool_; }
  [[nodiscard]] const kernels::GdnRecords& gdn_records() const noexcept { return gdn_records_; }
  [[nodiscard]] kernels::GdnStepControl* gdn_control() const noexcept { return gdn_control_; }
  [[nodiscard]] float* gdn_recurrent(int32_t gdn_layer) const;
  [[nodiscard]] __nv_bfloat16* gdn_conv(int32_t gdn_layer) const;

  [[nodiscard]] const QsaLayerState& qsa(int32_t cache) const { return qsa_.at(std::size_t(cache)); }

  // PLE conv state bf16 [9][10240] (oldest first).
  [[nodiscard]] __nv_bfloat16* ple_conv_state() const noexcept { return ple_state_; }

  // Raw device regions for session snapshots: (pointer, bytes) of everything that is not
  // position-addressed (GDN state image, PLE state, indexer rings).
  struct Region {
    void* data;
    std::size_t bytes;
  };
  [[nodiscard]] std::array<Region, 3 + kQsaCaches> snapshot_regions() const;

 private:
  StateOptions options_;
  int32_t capacity_ = 0;
  DeviceBuffer buffer_;  // when the state owns its memory
  unsigned char* base_ = nullptr;
  std::size_t bytes_ = 0;
  kernels::GdnStatePool gdn_pool_{};
  kernels::GdnRecords gdn_records_{};
  kernels::GdnStepControl* gdn_control_ = nullptr;
  std::array<QsaLayerState, kQsaCaches> qsa_{};
  __nv_bfloat16* ple_state_ = nullptr;
  std::size_t gdn_pool_bytes_ = 0;
};

}  // namespace fngine::model
