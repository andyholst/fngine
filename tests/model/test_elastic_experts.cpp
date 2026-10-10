// Elastic expert slots in the context tail of the device state (real pack, FNGINE_TEST_PACK):
// placement fills them with the next-ranked experts; growing the context retires the slots it
// reaches (their experts fall back to their RAM backups) while the other slots keep their
// experts through context writes; swaps take experts out of elastic slots without a download; a
// truncated sequence frees the slots for fills; a fill in flight into a slot the context takes is
// cancelled and finished before the context may write it.
#include "common/test_util.h"
#include "io/direct_io.h"
#include "model/adaptive_experts.h"
#include "model/state.h"

#include <algorithm>
#include <cstring>

using namespace fngine;

int main() {
  const auto dir = test::env_or_skip("FNGINE_TEST_PACK");
  if (dir.empty() || test::skip_without_gpu()) return test::kSkip;
  const auto pack = io::Pack::open(dir);
  cudaStream_t stream;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    model::StateOptions so;
    so.max_context = 16384;
    model::PlacementOptions po;
    po.expert_layers = 3;
    po.vram_cap_bytes = 192ull << 20;
    po.state_bytes = model::ModelState::device_bytes(so);
    po.state_regions = model::ModelState::position_regions(so);
    auto placement = model::ExpertPlacement::build(pack, stream, po);
    const auto plan = placement.plan();  // the initial occupants
    FNGINE_CHECK(plan.spares && plan.elastic.size() >= 20);
    const auto* state = static_cast<const uint8_t*>(placement.state_memory());

    io::InputFile file(pack.file_path(io::PackFile::Experts));
    const auto file_blob = [&](int l, int e) {
      const auto& layout = pack.expert_layers()[size_t(l)];
      std::vector<std::byte> b(layout.blob_bytes);
      file.read_exact(layout.blob_offset(e), b);
      return b;
    };
    const auto slot_bytes = [&](int l, int32_t slot, size_t bytes) {
      std::vector<std::byte> b(bytes);
      FNGINE_TEST_CUDA(cudaMemcpy(b.data(), placement.slot_address(l, slot), bytes, cudaMemcpyDeviceToHost));
      return b;
    };
    const auto on_gpu = [&](int l, int e) {
      const int32_t slot = placement.slot_of(l, e);
      return slot >= 0 && !placement.host_blob(l, e) &&
             slot_bytes(l, slot, pack.expert_layers()[size_t(l)].blob_bytes) == file_blob(l, e);
    };
    const auto on_cpu = [&](int l, int e) {
      const auto want = file_blob(l, e);
      const std::byte* ram = placement.host_blob(l, e);
      return placement.slot_of(l, e) < 0 && ram && std::memcmp(ram, want.data(), want.size()) == 0;
    };
    const auto maps_agree = [&] {
      std::vector<int32_t> device(48 * 512);
      // Slot-map writes are queued on this nonblocking stream, not on the default stream.
      FNGINE_TEST_CUDA(cudaMemcpyAsync(device.data(), placement.slot_of_device(), device.size() * 4,
                                       cudaMemcpyDeviceToHost, stream));
      FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      return device == placement.plan().slot_of;
    };

    // Every slot lies past the first page of one position region, below its end, without overlap,
    // and allows the context its offset covers.
    std::vector<std::pair<const uint8_t*, const uint8_t*>> spans;
    for (const auto& s : plan.elastic) {
      const uint64_t stride = placement.blob_stride(s.layer);
      const auto* a = placement.slot_address(s.layer, s.slot);
      const auto region = std::find_if(po.state_regions.begin(), po.state_regions.end(), [&](const auto& r) {
        return a >= state + r.offset && a < state + r.offset + r.bytes;
      });
      FNGINE_CHECK(region != po.state_regions.end());
      const uint64_t at = uint64_t(a - (state + region->offset));
      FNGINE_CHECK(at >= 64 * region->bytes_per_token && at + stride <= region->bytes);
      FNGINE_CHECK(uint64_t(s.tokens) == at / (64 * region->bytes_per_token) * 64 && s.tokens > 0);
      FNGINE_CHECK(on_gpu(s.layer, s.expert) && placement.elastic_slot(s.layer, s.slot));
      spans.emplace_back(a, a + stride);
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i) FNGINE_CHECK(spans[i - 1].second <= spans[i].first);
    FNGINE_CHECK(placement.elastic_resident() == int32_t(plan.elastic.size()) && maps_agree());

    test::DeviceArray<uint32_t> counts(48 * 512);
    model::ExpertSwapper swapper(placement, stream, counts.ptr, {});
    FNGINE_CHECK(swapper.usable());

    // A swap whose victim sits in an elastic slot: no download, b takes that slot.
    const auto& victim = plan.elastic.front();
    int b = -1;
    for (int e = 0; e < 512 && b < 0; ++e)
      if (placement.host_blob(victim.layer, e)) b = e;
    swapper.queue(victim.layer, b, victim.expert);
    swapper.drain();
    FNGINE_CHECK(placement.slot_of(victim.layer, b) == victim.slot && on_gpu(victim.layer, b));
    FNGINE_CHECK(on_cpu(victim.layer, victim.expert) && maps_agree());

    // Grow the context to the median threshold.
    std::vector<int32_t> thresholds;
    for (const auto& s : plan.elastic) thresholds.push_back(s.tokens);
    std::sort(thresholds.begin(), thresholds.end());
    const int32_t grow = thresholds[thresholds.size() / 2];
    FNGINE_CHECK(grow > thresholds.front() && swapper.vacate(grow));
    std::vector<std::pair<int, int>> survivors;
    for (const auto& s : plan.elastic) {
      const int e = s.layer == victim.layer && s.slot == victim.slot ? b : s.expert;
      if (s.tokens >= grow) {
        FNGINE_CHECK(on_gpu(s.layer, e));
        survivors.emplace_back(s.layer, e);
      } else {
        FNGINE_CHECK(on_cpu(s.layer, e));
      }
    }
    FNGINE_CHECK(!survivors.empty() && maps_agree());
    // The context writes its prefix of every region; the surviving experts are untouched.
    const int32_t pages = (grow + 63) / 64;
    for (const auto& r : po.state_regions)
      FNGINE_TEST_CUDA(cudaMemset(const_cast<uint8_t*>(state) + r.offset, 0xa5,
                                  std::min<uint64_t>(r.bytes, uint64_t(pages) * 64 * r.bytes_per_token)));
    for (const auto& [l, e] : survivors) FNGINE_CHECK(on_gpu(l, e));

    // The sequence is truncated: the retired slots take fills again.
    placement.revalidate(0);
    int fill_layer = -1, filled = -1;
    for (int l = 0; l < po.expert_layers && fill_layer < 0; ++l) {
      if (!placement.free_slots(l)) continue;
      for (int e = 0; e < 512; ++e)
        if (placement.host_blob(l, e)) { fill_layer = l, filled = e; break; }
    }
    FNGINE_CHECK(fill_layer >= 0);
    swapper.queue(fill_layer, filled, -1);
    swapper.drain();
    FNGINE_CHECK(placement.elastic_slot(fill_layer, placement.slot_of(fill_layer, filled)) &&
                 on_gpu(fill_layer, filled) && maps_agree());

    // A fill whose upload is in flight when the context takes every slot is cancelled: the
    // expert stays on the CPU, and the upload has landed before the context writes the slot.
    int late = -1;
    for (int e = 0; e < 512 && late < 0; ++e)
      if (placement.host_blob(fill_layer, e)) late = e;
    swapper.queue(fill_layer, late, -1);
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    swapper.boundary();
    swapper.start();
    FNGINE_CHECK(swapper.vacate(so.max_context));
    FNGINE_CHECK(swapper.idle() && on_cpu(fill_layer, late) && on_cpu(fill_layer, filled));
    FNGINE_CHECK(placement.elastic_resident() == 0 && maps_agree());
    for (const auto& r : po.state_regions)
      FNGINE_TEST_CUDA(cudaMemset(const_cast<uint8_t*>(state) + r.offset, 0x5a, r.bytes));
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    for (const auto& s : plan.elastic) {
      if (s.layer != fill_layer) continue;
      const auto bytes = slot_bytes(s.layer, s.slot, placement.blob_stride(s.layer));
      FNGINE_CHECK(std::all_of(bytes.begin(), bytes.end(), [](std::byte x) { return x == std::byte{0x5a}; }));
    }
    std::printf("PASS elastic experts: %zu slots, %zu kept at %d tokens\n", plan.elastic.size(), survivors.size(), grow);
  }
  FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
  return 0;
}
