// Real-pack prompt storage: overwrite a contiguous expert loan (all of it, then only the
// workspace part), stream through every ring buffer across repeated/changing chunk widths, restore
// bytes/maps from the written ranges, and swap experts into and out of loan slots before a later
// loan. No full-model inference; run separately from other large GPU tests.
#include "common/test_util.h"
#include "io/direct_io.h"
#include "model/adaptive_experts.h"
#include "model/prefill_stream.h"
#include "model/workspace.h"

#include <algorithm>
#include <array>
#include <tuple>

using namespace fngine;

int main(int argc, char** argv) {
  const bool pageable = argc == 2 && std::string(argv[1]) == "--pageable";
  const auto dir = test::env_or_skip("FNGINE_TEST_PACK");
  if (dir.empty() || test::skip_without_gpu()) return test::kSkip;
  const auto pack = io::Pack::open(dir);
  cudaStream_t main;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking));
  {
    model::WorkspaceOptions wo;
    wo.max_tokens = 64;
    wo.max_blocks = 64;
    const auto workspace_bytes = model::WorkspaceBuffer::device_bytes(wo);
    const auto ring_bytes = model::PrefillStream::ring_bytes(pack, true);
    model::PlacementOptions po;
    po.expert_layers = 5;  // layers 2-4 reuse the ring buffers of layers 0-2
    po.vram_cap_bytes = 1536ull << 20;
    po.loan_bytes = workspace_bytes + ring_bytes;
    po.register_ram = !pageable;
    if (pageable) po.bounce_buffer_bytes = 64 * 1024;  // several segments per expert, frequent slot reuse
    auto placement = model::ExpertPlacement::build(pack, main, po);
    if (pageable) FNGINE_CHECK(!placement.ram_tier().registered());
    FNGINE_CHECK(placement.plan().loan_bytes >= po.loan_bytes);
    FNGINE_CHECK(placement.plan().vram_bytes <= placement.plan().budget);
    const auto original = placement.plan().slot_of;
    std::vector<std::tuple<int, int, int>> loans;
    uint64_t backing_bytes = 0;
    for (int l = 0; l < po.expert_layers; ++l)
      for (int e = 0; e < 512; ++e)
        if (placement.loan_slot(l, placement.slot_of(l, e))) {
          loans.emplace_back(l, e, placement.slot_of(l, e));
          backing_bytes += placement.blob_stride(l);
        }
    FNGINE_CHECK(backing_bytes >= po.loan_bytes);
    FNGINE_CHECK(placement.plan().host_bytes >= placement.plan().loan_host_offset + backing_bytes);

    model::PrefillStreamOptions so;
    so.ring_memory = static_cast<uint8_t*>(placement.loan_data()) + workspace_bytes;
    so.ring_memory_bytes = ring_bytes;
    model::PrefillStream stream(pack, placement, {}, so);
    FNGINE_CHECK(stream.streaming());  // registration failure must not disable the split path
    auto workspace = model::WorkspaceBuffer::view(wo, placement.loan_data(), workspace_bytes);
    io::InputFile file(pack.file_path(io::PackFile::Experts));
    PinnedHostBuffer samples(48 * 512 * 128);
    std::vector<std::byte> expected(48 * 512 * 128);
    std::vector<int32_t> device_map(48 * 512);
    for (int prompt = 0; prompt < 2; ++prompt) {
      placement.begin_loan(main);
      FNGINE_TEST_CUDA(cudaStreamSynchronize(main));
      for (const auto& [l, e, slot] : loans) {
        FNGINE_CHECK(placement.slot_of(l, e) == -1 && placement.host_blob(l, e));
        const auto* address = placement.slot_address(l, slot);
        const auto* begin = static_cast<const uint8_t*>(placement.loan_data());
        FNGINE_CHECK(address >= begin && address + placement.blob_stride(l) <= begin + placement.plan().loan_bytes);
      }
      stream.take_ring_written();
      const uint64_t scribbled = prompt == 0 ? placement.plan().loan_bytes : workspace_bytes;
      FNGINE_TEST_CUDA(cudaMemsetAsync(placement.loan_data(), 0xa5, scribbled, main));
      model::WorkspaceBuffer::initialize(workspace, main);
      FNGINE_TEST_CUDA(cudaStreamSynchronize(main));
      stream.invalidate();
      for (int T : {64, 3, 64}) {
        stream.begin_chunk(T, main);
        // Fetch tables once; no host synchronization may serialize the per-layer ring reuse.
        std::array<const uint8_t*, 48 * 512> tables{};
        FNGINE_TEST_CUDA(cudaMemcpy(tables.data(), stream.table(0), sizeof(tables), cudaMemcpyDeviceToHost));
        std::vector<size_t> sampled;
        for (int l = 0; l < 48; ++l) {
          stream.prefetch(l);
          stream.wait_ready(l, main);
          for (int e = 0; e < 512; ++e) {
            const auto* blob = tables[size_t(l) * 512 + size_t(e)];
            if (!blob || blob == placement.device_blob(l, e)) continue;
            FNGINE_CHECK(placement.host_blob(l, e));
            const auto bytes = pack.expert_layers()[size_t(l)].blob_bytes;
            const size_t offset = (size_t(l) * 512 + size_t(e)) * 128;
            auto* dst = static_cast<std::byte*>(samples.data()) + offset;
            const auto* src = placement.host_blob(l, e);
            std::memcpy(expected.data() + offset, src, 64);
            std::memcpy(expected.data() + offset + 64, src + bytes - 64, 64);
            FNGINE_TEST_CUDA(cudaMemcpyAsync(dst, blob, 64, cudaMemcpyDeviceToHost, main));
            FNGINE_TEST_CUDA(cudaMemcpyAsync(dst + 64, blob + bytes - 64, 64, cudaMemcpyDeviceToHost, main));
            sampled.push_back(offset);
          }
          stream.release(l, main);
        }
        FNGINE_TEST_CUDA(cudaStreamSynchronize(main));
        stream.synchronize();
        for (size_t offset : sampled)
          FNGINE_CHECK(std::memcmp(static_cast<const std::byte*>(samples.data()) + offset,
                                   expected.data() + offset, 128) == 0);
        std::printf("prompt%d T%d checked %zu streamed expert endpoints\n", prompt, T, sampled.size());
      }
      // Only the written ranges are restored; every loan expert must still match the file.
      std::vector<model::DeviceRange> written = stream.take_ring_written();
      FNGINE_CHECK(!written.empty());
      written.push_back({placement.loan_data(), scribbled});
      placement.end_loan(main, written);
      FNGINE_CHECK(!placement.borrowed());
      stream.invalidate();
      FNGINE_TEST_CUDA(cudaMemcpy(device_map.data(), placement.slot_of_device(), device_map.size() * 4,
                                  cudaMemcpyDeviceToHost));
      FNGINE_CHECK(device_map == original && placement.plan().slot_of == original);
      for (const auto& [l, e, slot] : loans) {
        const auto& layout = pack.expert_layers()[size_t(l)];
        std::vector<std::byte> want(layout.blob_bytes), got(layout.blob_bytes);
        file.read_exact(layout.blob_offset(e), want);
        FNGINE_TEST_CUDA(cudaMemcpy(got.data(), placement.device_blob(l, e), got.size(), cudaMemcpyDeviceToHost));
        FNGINE_CHECK(want == got && !placement.host_blob(l, e));
      }
    }
    // Per layer: a loan expert falls back to its backup and a CPU expert fills its slot (that
    // expert's RAM blob becomes its backup), and a regular resident is downloaded for a CPU expert
    // taking the spare. The next loan must lend and restore exactly the new loan-slot occupants.
    test::DeviceArray<uint32_t> routing_counts(48 * 512);
    model::ExpertSwapper swapper(placement, main, routing_counts.ptr, {});
    FNGINE_CHECK(swapper.usable());
    struct Moved {
      int layer, entered, left, evicted;
      int32_t slot;
    };
    std::vector<Moved> moved;
    for (int l = 0; l < po.expert_layers; ++l) {
      const auto loan = std::find_if(loans.begin(), loans.end(), [l](const auto& x) { return std::get<0>(x) == l; });
      if (loan == loans.end()) continue;
      const auto [ll, e, slot] = *loan;
      std::vector<int> cpu, regular;
      for (int x = 0; x < 512; ++x) {
        if (placement.host_blob(l, x)) cpu.push_back(x);
        else if (!placement.loan_slot(l, placement.slot_of(l, x)) && placement.slot_of(l, x) >= 0) regular.push_back(x);
      }
      if (cpu.size() < 2 || regular.empty()) continue;
      swapper.queue(l, cpu[0], e);           // cpu[0] fills e's loan slot
      swapper.queue(l, cpu[1], regular[0]);  // cpu[1] takes the regular spare
      moved.push_back({l, cpu[0], e, regular[0], slot});
    }
    FNGINE_CHECK(!moved.empty());
    swapper.drain();
    std::vector<std::tuple<int, int, int>> occupants;
    for (int l = 0; l < po.expert_layers; ++l)
      for (int e = 0; e < 512; ++e)
        if (placement.slot_of(l, e) >= 0 && placement.loan_slot(l, placement.slot_of(l, e)))
          occupants.emplace_back(l, e, placement.slot_of(l, e));
    for (const auto& m : moved) {
      FNGINE_CHECK(placement.slot_of(m.layer, m.entered) == m.slot);
      FNGINE_CHECK(placement.slot_of(m.layer, m.left) < 0 && placement.slot_of(m.layer, m.evicted) < 0);
    }
    placement.begin_loan(main);
    for (const auto& [l, e, slot] : occupants) FNGINE_CHECK(placement.slot_of(l, e) == -1 && placement.host_blob(l, e));
    FNGINE_TEST_CUDA(cudaMemsetAsync(placement.loan_data(), 0x5a, placement.plan().loan_bytes, main));
    const model::DeviceRange whole{placement.loan_data(), placement.plan().loan_bytes};
    placement.end_loan(main, {&whole, 1});
    const auto file_blob = [&](int l, int e) {
      const auto& layout = pack.expert_layers()[size_t(l)];
      std::vector<std::byte> want(layout.blob_bytes);
      file.read_exact(layout.blob_offset(e), want);
      return want;
    };
    for (const auto& [l, e, slot] : occupants) {
      const auto want = file_blob(l, e);
      std::vector<std::byte> got(want.size());
      FNGINE_TEST_CUDA(cudaMemcpy(got.data(), placement.device_blob(l, e), got.size(), cudaMemcpyDeviceToHost));
      FNGINE_CHECK(placement.slot_of(l, e) == slot && !placement.host_blob(l, e) && got == want);
    }
    for (const auto& m : moved) {
      for (int e : {m.left, m.evicted}) {
        const auto want = file_blob(m.layer, e);
        FNGINE_CHECK(std::memcmp(placement.host_blob(m.layer, e), want.data(), want.size()) == 0);
      }
    }
  }
  FNGINE_TEST_CUDA(cudaStreamDestroy(main));
  std::puts("PASS prefill streaming loan restore and ring buffer reuse");
}
