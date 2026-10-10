// Exchange layout invariants and a graph-captured GPU->CPU->GPU round trip on host-mapped memory.
#include "common/test_util.h"
#include "core/decode_graph.h"
#include "core/exchange.h"

#include <atomic>
#include <thread>

using namespace fngine;

__global__ void publish_and_wait(ExchangeControl* control, const uint32_t* seq, float* y_out,
                                 const MoeDecodeResult* result) {
  if (threadIdx.x != 0) return;
  const uint32_t s = *seq;
  __threadfence_system();
  exchange_store_release_sys(&control->ready[3].value, s);
  if (!exchange_wait_sys(&control->done[3].value, s, kExchangeTimeoutCycles)) {
    exchange_store_release_sys(&control->error.value, kExchangeErrorMoeTimeout | 4u);
    return;
  }
  *y_out = exchange_load_relaxed_sys(&result[3].y[1][7]);
}

int main() {
  static_assert(offsetof(ExchangeControl, done) % 64 == 0);
  static_assert(sizeof(MoeDecodePublish) % 64 == 0);
  static_assert(offsetof(DecodeExchange, publish) % 64 == 0);
  {
    std::vector<unsigned char> buf(prefill_exchange_bytes(4096) + 64);
    void* base = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t{63});
    PrefillExchangeView v = make_prefill_exchange_view(base, 4096);
    FNGINE_CHECK(reinterpret_cast<unsigned char*>(v.y + 4096 * 2560) <=
                 static_cast<unsigned char*>(base) + prefill_exchange_bytes(4096));
    FNGINE_CHECK(reinterpret_cast<uintptr_t>(v.x) % 64 == 0);
  }
  if (test::skip_without_gpu()) return test::kSkip;

  DecodeExchange* ex = nullptr;
  FNGINE_TEST_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&ex), sizeof(DecodeExchange),
                                 cudaHostAllocMapped | cudaHostAllocPortable));
  std::memset(ex, 0, sizeof(DecodeExchange));
  test::DeviceArray<uint32_t> seq(1);
  test::DeviceArray<float> y(1);

  cudaStream_t stream;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DecodeGraphDefinition def;
  def.capture(stream, [&] {
    publish_and_wait<<<1, 32, 0, stream>>>(&ex->control, seq.get(), y.get(), ex->result);
  });
  DecodeGraphExecutable exe;
  exe.instantiate(def);

  std::atomic<bool> stop{false};
  std::thread cpu([&] {
    uint32_t seen = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      uint32_t r = host_load_acquire(ex->control.ready[3].value);
      if (r != seen) {
        seen = r;
        ex->result[3].y[1][7] = float(r) * 0.5f;
        host_store_release(ex->control.done[3].value, r);
      }
    }
  });
  for (uint32_t s = 1; s <= 100; ++s) {
    FNGINE_TEST_CUDA(cudaMemcpyAsync(seq.get(), &s, 4, cudaMemcpyHostToDevice, stream));
    exe.launch(stream);
    float got = 0;
    FNGINE_TEST_CUDA(cudaMemcpyAsync(&got, y.get(), 4, cudaMemcpyDeviceToHost, stream));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    FNGINE_CHECK(ex->control.error.value == 0);
    FNGINE_CHECK(got == float(s) * 0.5f);
  }
  stop = true;
  cpu.join();
  std::printf("PASS exchange round trips\n");
  return 0;
}
