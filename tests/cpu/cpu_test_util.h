#pragma once

// Helpers shared by the tests in tests/cpu: pack expert access, reference dequantization,
// synthetic IQ blocks and a host-memory exchange that stands in for the GPU side.
#include "common/test_util.h"
#include "core/exchange.h"
#include "cpu/iq_cpu.h"
#include "ggml-iq/ggml-iq.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fngine::test {

using cpu::ActFormat;
using cpu::WeightFormat;

// FP32 values of one weight row via ggml's reference dequantizers.
inline void dequant_row(WeightFormat fmt, const void* row, int k, float* out) {
  switch (fmt) {
    case WeightFormat::IQ2_S: dequantize_row_iq2_s(static_cast<const block_iq2_s*>(row), out, k); break;
    case WeightFormat::IQ3_XXS: dequantize_row_iq3_xxs(static_cast<const block_iq3_xxs*>(row), out, k); break;
    case WeightFormat::IQ3_S: dequantize_row_iq3_s(static_cast<const block_iq3_s*>(row), out, k); break;
    case WeightFormat::IQ4_XS: dequantize_row_iq4_xs(static_cast<const block_iq4_xs*>(row), out, k); break;
    case WeightFormat::IQ4_NL: dequantize_row_iq4_nl(static_cast<const block_iq4_nl*>(row), out, k); break;
    case WeightFormat::Q2_0: dequantize_row_q2_0(static_cast<const block_q2_0*>(row), out, k); break;
  }
}

// Reference quantization of FP32 values into the activation format, and its dequantization.
inline std::vector<uint8_t> ref_quant_act(ActFormat act, const float* x, int k) {
  std::vector<uint8_t> out(cpu::act_row_bytes(act, k), 0);
  if (act == ActFormat::Q8_K) quantize_row_q8_K_ref(x, reinterpret_cast<block_q8_K*>(out.data()), k);
  else quantize_row_q8_0_ref(x, reinterpret_cast<block_q8_0*>(out.data()), k);
  return out;
}
inline void dequant_act(ActFormat act, const void* q, int k, float* out) {
  if (act == ActFormat::Q8_K) dequantize_row_q8_K(static_cast<const block_q8_K*>(q), out, k);
  else dequantize_row_q8_0(static_cast<const block_q8_0*>(q), out, k);
}

inline uint16_t bf16_bits(float f) {
  const __nv_bfloat16 b = float_to_bf16(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
inline float bf16_bits_to_float(uint16_t u) {
  const uint32_t w = uint32_t(u) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// Random bytes for `nblk` weight blocks with a finite, moderate FP16 block scale.
inline std::vector<uint8_t> random_blocks(WeightFormat fmt, int nblk, Rng& rng) {
  const cpu::FormatInfo fi = cpu::format_info(fmt);
  std::vector<uint8_t> out(std::size_t(nblk) * fi.block_bytes);
  for (auto& b : out) b = uint8_t(rng.next());
  for (int i = 0; i < nblk; ++i) {
    const float d = float(rng.uniform(0.002, 0.02)) * (rng.uniform() < 0.5 ? -1.0f : 1.0f);
    const uint16_t h = ggml_compute_fp32_to_fp16(d);
    std::memcpy(out.data() + std::size_t(i) * fi.block_bytes, &h, 2);  // `d` is the first field
  }
  return out;
}

struct PackExperts {
  struct Layer {
    WeightFormat gu = WeightFormat::IQ2_S, down = WeightFormat::IQ4_NL;
    std::size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0, blob_bytes = 0, blob_stride = 0,
                offset = 0;
  };
  std::string dir;
  std::vector<Layer> layers;
  Layer mtp;
  int fd_experts = -1, fd_mtp = -1;

  PackExperts() = default;
  PackExperts(const PackExperts&) = delete;
  PackExperts& operator=(const PackExperts&) = delete;
  ~PackExperts() {
    if (fd_experts >= 0) ::close(fd_experts);
    if (fd_mtp >= 0) ::close(fd_mtp);
  }

  static Layer parse_layer(const nlohmann::json& j) {
    Layer l;
    if (!cpu::parse_weight_format(j.at("gu_type").get<std::string>(), &l.gu) ||
        !cpu::parse_weight_format(j.at("down_type").get<std::string>(), &l.down))
      throw std::runtime_error("unsupported expert format in manifest");
    l.gate_bytes = j.at("gate_bytes").get<std::size_t>();
    l.up_bytes = j.at("up_bytes").get<std::size_t>();
    l.down_bytes = j.at("down_bytes").get<std::size_t>();
    l.blob_bytes = j.at("blob_bytes").get<std::size_t>();
    l.blob_stride = j.at("blob_stride").get<std::size_t>();
    l.offset = j.at("offset").get<std::size_t>();
    return l;
  }

  void open(const std::string& pack_dir) {
    dir = pack_dir;
    std::ifstream f(dir + "/manifest.json");
    if (!f) throw std::runtime_error("cannot open " + dir + "/manifest.json");
    const nlohmann::json m = nlohmann::json::parse(f);
    for (const auto& j : m.at("experts").at("layers")) layers.push_back(parse_layer(j));
    mtp = parse_layer(m.at("mtp_experts"));
    const auto& files = m.at("files");
    fd_experts = ::open((dir + "/" + files.at("experts").get<std::string>()).c_str(), O_RDONLY);
    fd_mtp = ::open((dir + "/" + files.at("mtp_experts").get<std::string>()).c_str(), O_RDONLY);
    if (fd_experts < 0 || fd_mtp < 0) throw std::runtime_error("cannot open expert files");
  }

  const Layer& layer(int l) const { return l < 0 ? mtp : layers.at(std::size_t(l)); }

  // Blob of (layer, expert); layer -1 = MTP. 64-byte aligned buffer of blob_bytes.
  void read_blob(int l, int e, uint8_t* dst) const {
    const Layer& L = layer(l);
    const std::size_t off = L.offset + std::size_t(e) * L.blob_stride;
    std::size_t done = 0;
    while (done < L.blob_bytes) {
      const ssize_t r = ::pread(l < 0 ? fd_mtp : fd_experts, dst + done, L.blob_bytes - done,
                                off_t(off + done));
      if (r <= 0) throw std::runtime_error("short read of expert blob");
      done += std::size_t(r);
    }
  }
};

// Aligned, zeroed host allocation.
struct HostBuf {
  uint8_t* p = nullptr;
  std::size_t bytes = 0;
  explicit HostBuf(std::size_t n, std::size_t align = 4096) : bytes(n) {
    p = static_cast<uint8_t*>(std::aligned_alloc(align, (n + align - 1) / align * align));
    FNGINE_CHECK(p != nullptr);
    std::memset(p, 0, n);
  }
  ~HostBuf() { std::free(p); }
  HostBuf(const HostBuf&) = delete;
  HostBuf& operator=(const HostBuf&) = delete;
};

// Plain host-memory exchange; the test thread plays the GPU.
struct HostExchange {
  HostBuf dec_mem{sizeof(DecodeExchange)};
  HostBuf pre_mem;
  DecodeExchange* dec = reinterpret_cast<DecodeExchange*>(dec_mem.p);
  PrefillExchangeView pre;
  explicit HostExchange(int prefill_capacity)
      : pre_mem(prefill_exchange_bytes(prefill_capacity), 64),
        pre(make_prefill_exchange_view(pre_mem.p, prefill_capacity)) {}

  void release_all(uint32_t seq) {
    for (auto& c : dec->control.ready) host_store_release(c.value, seq);
  }
  void wait_done(int l, uint32_t seq) const {
    while (host_load_acquire(dec->control.done[l].value) != seq) _mm_pause();
  }
  // Prefill step played layer by layer like the GPU: fill(l) writes the shared region (header
  // included), then ready[l] is released and done[l] awaited (the CPU acks every layer).
  template <class Fill>
  void prefill_step(uint32_t seq, Fill&& fill) {
    for (int l = 0; l < kExchangeLayers; ++l) {
      fill(l);
      host_store_release(dec->control.ready[l].value, seq);
      wait_done(l, seq);
    }
  }
};

}  // namespace fngine::test
