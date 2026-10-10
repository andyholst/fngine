// Q8_K / Q8_0 SIMD quantizers vs ggml reference: bit-identical output.
#include "cpu/cpu_test_util.h"
#include "cpu/q8_cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace fngine;
using namespace fngine::test;

namespace {

// Block contents exercising: plain normals, magnitude ties of opposite sign (the reference keeps
// the FIRST max-magnitude element), all-zero blocks, a single nonzero, huge/tiny scales and
// values landing exactly on .5 rounding boundaries.
std::vector<float> make_input(int k, int pattern, Rng& rng) {
  std::vector<float> x(std::size_t(k), 0.0f);
  for (int i = 0; i < k; ++i) {
    switch (pattern) {
      case 0: x[i] = float(rng.normal()); break;
      case 1: x[i] = float(int(rng.uniform(-4, 5))) * 0.25f; break;  // many ties incl. +/-1
      case 2: x[i] = 0.0f; break;
      case 3: x[i] = (i % 97 == 5) ? float(rng.normal()) : 0.0f; break;
      case 4: x[i] = float(rng.normal() * 1e-30); break;
      case 5: x[i] = float(rng.normal() * 3e4); break;
      case 6: x[i] = (float(int(rng.uniform(-254, 255))) + 0.5f) / 254.0f * 127.0f; break;
      default: x[i] = float(rng.uniform(-1, 1)) * (i % 32 == 0 ? -1.0f : 0.5f); break;
    }
  }
  if (pattern == 1 && k >= 2) {  // put -max first in some blocks
    x[0] = -1.0f;
    x[1] = 1.0f;
  }
  return x;
}

}  // namespace

int main() {
  Rng rng(1234);
  int failures = 0;
  long long blocks_k = 0, blocks_0 = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int pattern = trial % 8;
    const int k = 2560;
    std::vector<float> xf = make_input(k, pattern, rng);
    std::vector<uint16_t> xb(static_cast<std::size_t>(k));
    std::vector<float> xbf(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
      xb[i] = bf16_bits(xf[i]);
      xbf[i] = bf16_bits_to_float(xb[i]);
    }
    const int nk = k / QK_K, n0 = k / QK8_0;
    // Q8_K from BF16 and from FP32
    for (int src = 0; src < 2; ++src) {
      const float* ref_in = src == 0 ? xbf.data() : xf.data();
      std::vector<block_q8_K> ref(static_cast<std::size_t>(nk)), got(static_cast<std::size_t>(nk));
      std::memset(ref.data(), 0, ref.size() * sizeof(block_q8_K));
      std::memset(got.data(), 0x5a, got.size() * sizeof(block_q8_K));
      quantize_row_q8_K_ref(ref_in, ref.data(), k);
      if (src == 0) cpu::quantize_row_q8_K_bf16(xb.data(), got.data(), k);
      else cpu::quantize_row_q8_K_f32(xf.data(), got.data(), k);
      for (int b = 0; b < nk; ++b, ++blocks_k) {
        if (std::memcmp(&ref[b], &got[b], sizeof(block_q8_K)) != 0) {
          if (failures++ < 10)
            std::printf("FAIL q8_K src=%d trial=%d pattern=%d block=%d d ref=%g got=%g\n", src,
                        trial, pattern, b, ref[b].d, got[b].d);
        }
      }
    }
    // Q8_0 from BF16 and from FP32
    for (int src = 0; src < 2; ++src) {
      const float* ref_in = src == 0 ? xbf.data() : xf.data();
      std::vector<block_q8_0> ref(static_cast<std::size_t>(n0)), got(static_cast<std::size_t>(n0));
      std::memset(ref.data(), 0, ref.size() * sizeof(block_q8_0));
      std::memset(got.data(), 0x5a, got.size() * sizeof(block_q8_0));
      quantize_row_q8_0_ref(ref_in, ref.data(), k);
      if (src == 0) cpu::quantize_row_q8_0_bf16(xb.data(), got.data(), k);
      else cpu::quantize_row_q8_0_f32(xf.data(), got.data(), k);
      for (int b = 0; b < n0; ++b, ++blocks_0) {
        if (std::memcmp(&ref[b], &got[b], sizeof(block_q8_0)) != 0) {
          if (failures++ < 10)
            std::printf("FAIL q8_0 src=%d trial=%d pattern=%d block=%d\n", src, trial, pattern, b);
        }
      }
    }
  }
  std::printf("%s q8 quantizers: %lld Q8_K blocks, %lld Q8_0 blocks, %d mismatches\n",
              failures ? "FAIL" : "PASS", blocks_k, blocks_0, failures);
  return failures ? 1 : 0;
}
