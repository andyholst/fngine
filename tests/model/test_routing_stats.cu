// Routing counters and their persistence: decode counting inside moe_select_publish (in a
// captured graph), modular deltas in RoutingStats::collect, and the decayed routing-stats.bin.
#include "kernels/moe/moe.h"
#include "model/adaptive_experts.h"

#include "common/test_util.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

using namespace fngine;
namespace fs = std::filesystem;

namespace {

constexpr std::size_t kPairs = std::size_t(dims::kLayers) * dims::kExperts;

std::vector<uint64_t> read_file(const fs::path& p) {
  std::vector<uint64_t> v(kPairs);
  std::ifstream in(p, std::ios::binary);
  FNGINE_CHECK(bool(in.read(reinterpret_cast<char*>(v.data()), std::streamsize(kPairs * 8))));
  return v;
}

void write_file(const fs::path& p, const std::vector<uint64_t>& v) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  FNGINE_CHECK(bool(out.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 8))));
  out.close();
  FNGINE_CHECK(bool(out));
}

}  // namespace

int main() {
  if (test::skip_without_gpu()) return test::kSkip;
  cudaStream_t s;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  test::DeviceArray<uint32_t> counts(kPairs);
  // This nonblocking stream does not inherit ordering from default-stream H2D copies.
  FNGINE_TEST_CUDA(cudaMemsetAsync(counts.ptr, 0, kPairs * 4, s));

  // Decode router logits for T = 3 tokens of layer 5: token t routes to (t*37 + k*11) % 512.
  constexpr int kT = 3;
  std::vector<float> logits(std::size_t(kT) * dims::kRouterRows, -4.0f);
  std::vector<uint64_t> want(kPairs, 0);
  for (int t = 0; t < kT; ++t)
    for (int k = 0; k < dims::kTopK; ++k) {
      const int e = (t * 37 + k * 11) % dims::kExperts;
      logits[std::size_t(t) * dims::kRouterRows + e] = 8.0f - 0.5f * float(k);
      want[5 * 512 + std::size_t(e)] += 1;
    }
  test::DeviceArray<float> dlogits(logits);
  test::DeviceArray<__nv_bfloat16> dx(std::size_t(kT) * dims::kHidden);
  test::DeviceArray<uint8_t> wsmem(kernels::moe_decode_workspace_bytes());
  const kernels::MoeDecodeWorkspace ws = kernels::make_moe_decode_workspace(wsmem.ptr);
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // DeviceArray fills ran on the legacy stream

  // Captured into a graph and replayed twice (as the decode step graphs do); no exchange, every
  // expert hot (identity slots). Only the T routed tokens count.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  FNGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  FNGINE_TEST_CUDA(kernels::moe_select_publish(dlogits.ptr, dx.ptr, nullptr, 0, nullptr, nullptr, kT,
                                               ws, s, counts.ptr + 5 * dims::kExperts));
  FNGINE_TEST_CUDA(cudaStreamEndCapture(s, &graph));
  FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
  FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
  for (auto& w : want) w *= 2;

  const fs::path file = fs::temp_directory_path() / ("fngine-routing-" + std::to_string(::getpid()) + ".bin");
  fs::remove(file);
  model::RoutingStats stats(file, counts.ptr);
  const std::vector<uint64_t> d1 = stats.collect(s);
  FNGINE_CHECK(d1 == want);
  FNGINE_CHECK(stats.pending_total() == 2 * 30);
  FNGINE_CHECK(stats.flush());
  FNGINE_CHECK(read_file(file) == want);
  FNGINE_CHECK(stats.pending_total() == 0);
  const auto& no_delta = stats.collect(s);
  FNGINE_CHECK(std::all_of(no_delta.begin(), no_delta.end(), [](uint64_t x) { return x == 0; }));

  // Counter wraparound: start the device cell near 2^32; the delta stays exact.
  uint32_t near = 0xFFFFFFF0u;
  FNGINE_TEST_CUDA(cudaMemcpyAsync(counts.ptr + 3, &near, 4, cudaMemcpyHostToDevice, s));
  FNGINE_CHECK(stats.collect(s)[3] == uint64_t(near));  // absorb the ordered jump
  FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));  // layer 5 again (+30 pairs)
  // 40 more pairs of expert 3, layer 0, carry the counter across 2^32.
  const uint32_t wrapped = near + 40u;
  FNGINE_TEST_CUDA(cudaMemcpyAsync(counts.ptr + 3, &wrapped, 4, cudaMemcpyHostToDevice, s));
  const std::vector<uint64_t> d2 = stats.collect(s);
  FNGINE_CHECK(d2[3] == 40);
  uint64_t d2_total = 0;
  for (auto v : d2) d2_total += v;
  FNGINE_CHECK(d2_total == 70);
  // Second flush: stored = stored * 0.5 + new, minus the absorbed jump (pending includes it).
  const std::vector<uint64_t> pending_before = [&] {
    std::vector<uint64_t> p(kPairs, 0);
    for (std::size_t i = 0; i < kPairs; ++i) p[i] = d2[i];
    p[3] += uint64_t(0xFFFFFFF0u);  // the jump collected above is pending too
    return p;
  }();
  FNGINE_CHECK(stats.flush());
  const std::vector<uint64_t> stored = read_file(file);
  for (std::size_t i = 0; i < kPairs; ++i) FNGINE_CHECK(stored[i] == want[i] / 2 + pending_before[i]);
  FNGINE_CHECK(!fs::exists(fs::path(file.string() + ".part")));

  // A flush without new counts leaves the file untouched (no decay without activity).
  FNGINE_CHECK(stats.flush());
  FNGINE_CHECK(read_file(file) == stored);

  // Simulate a new engine instance: fresh device counters merge into the persisted file. Counts
  // above 2^53 must retain their low bits when halved; double-based decay silently loses them.
  std::vector<uint64_t> large = stored;
  large[11] = (uint64_t(1) << 63) + 3;
  write_file(file, large);
  FNGINE_TEST_CUDA(cudaMemsetAsync(counts.ptr, 0, kPairs * 4, s));
  const uint32_t one = 1;
  FNGINE_TEST_CUDA(cudaMemcpyAsync(counts.ptr + 11, &one, 4, cudaMemcpyHostToDevice, s));
  model::RoutingStats restarted(file, counts.ptr);
  FNGINE_CHECK(restarted.collect(s)[11] == 1);
  FNGINE_CHECK(restarted.flush());
  std::vector<uint64_t> merged = large;
  for (auto& v : merged) v /= 2;
  ++merged[11];
  FNGINE_CHECK(read_file(file) == merged);

  // A failed temporary-file write preserves the old file and every newly collected count.
  FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
  const std::vector<uint64_t> failed_delta = restarted.collect(s);
  const fs::path part(file.string() + ".part");
  fs::create_directory(part);
  write_file(part / "blocker", {});
  FNGINE_CHECK(!restarted.flush());
  FNGINE_CHECK(restarted.pending_total() == 30);
  FNGINE_CHECK(read_file(file) == merged);
  fs::remove_all(part);
  FNGINE_CHECK(restarted.flush());
  for (std::size_t i = 0; i < kPairs; ++i) merged[i] = merged[i] / 2 + failed_delta[i];
  FNGINE_CHECK(read_file(file) == merged);
  FNGINE_CHECK(restarted.pending_total() == 0);

  // An unreadable/truncated previous file must not be silently reset, losing historic counts.
  FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
  restarted.collect(s);
  write_file(file, {123});
  FNGINE_CHECK(!restarted.flush());
  FNGINE_CHECK(restarted.pending_total() == 30);
  write_file(file, merged);
  FNGINE_CHECK(restarted.flush());
  for (std::size_t i = 0; i < kPairs; ++i) merged[i] = merged[i] / 2 + failed_delta[i];
  FNGINE_CHECK(read_file(file) == merged);

  fs::remove(file);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
  std::printf("PASS routing stats\n");
  return 0;
}
