#pragma once

// Shared helpers for fngine tests: skip codes, deterministic data, BF16/FP8 conversion,
// and tolerance comparison against FP64 oracles.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fngine::test {

inline constexpr int kSkip = 77;

#define FNGINE_TEST_CUDA(expr)                                                            \
  do {                                                                                    \
    cudaError_t e_ = (expr);                                                              \
    if (e_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,           \
                   cudaGetErrorString(e_));                                               \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

#define FNGINE_CHECK(cond)                                                                \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);       \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// Returns true (after printing a skip line) when no CUDA device is usable.
inline bool skip_without_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip: no CUDA device\n");
    return true;
  }
  return false;
}

// Returns the value of an environment variable or an empty string (and prints a skip line).
inline std::string env_or_skip(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) {
    std::printf("skip: %s is not set\n", name);
    return {};
  }
  return v;
}

// Deterministic splitmix64-based generator.
struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed) {}
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, 1).
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  double normal() {
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

inline float bf16_to_float(__nv_bfloat16 v) {
  uint16_t bits;
  std::memcpy(&bits, &v, 2);
  uint32_t w = uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// Round-to-nearest-even FP32 -> BF16 (NaN preserved).
inline __nv_bfloat16 float_to_bf16(float f) {
  uint32_t w;
  std::memcpy(&w, &f, 4);
  uint16_t bits;
  if ((w & 0x7fffffffu) > 0x7f800000u) {
    bits = uint16_t((w >> 16) | 0x40);
  } else {
    bits = uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
  }
  __nv_bfloat16 v;
  std::memcpy(&v, &bits, 2);
  return v;
}

inline float round_bf16(float f) { return bf16_to_float(float_to_bf16(f)); }

// E4M3FN code -> float.
inline float e4m3_to_float(uint8_t code) {
  const int sign = code >> 7, exp = (code >> 3) & 0xF, man = code & 7;
  float v;
  if (exp == 0xF && man == 7) return NAN;
  if (exp == 0) v = std::ldexp(float(man) / 8.0f, -6);
  else v = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
  return sign ? -v : v;
}

struct ErrorStats {
  double max_abs = 0, ref_rms = 0, max_rel_to_rms = 0;
  size_t worst = 0;
};

// max |got - ref| and rms(ref); passes when max_abs <= tol * rms(ref).
inline ErrorStats compare(const float* got, const double* ref, size_t n) {
  ErrorStats s;
  double sq = 0;
  for (size_t i = 0; i < n; ++i) sq += ref[i] * ref[i];
  s.ref_rms = std::sqrt(sq / double(n ? n : 1));
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(got[i]) - ref[i]);
    if (!(d <= s.max_abs)) {
      s.max_abs = d;
      s.worst = i;
    }
  }
  s.max_rel_to_rms = s.ref_rms > 0 ? s.max_abs / s.ref_rms : s.max_abs;
  return s;
}

inline bool report(const char* what, const ErrorStats& s, double tol_rel_rms) {
  const bool ok = s.max_rel_to_rms <= tol_rel_rms;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e) worst=%zu\n",
              ok ? "PASS" : "FAIL", what, s.max_abs, s.ref_rms, s.max_rel_to_rms, tol_rel_rms,
              s.worst);
  return ok;
}

// RAII device buffer for tests. Construction and upload complete their default-stream writes
// before returning, so consumers on nonblocking streams never race initialization.
template <class T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t count = 0;
  DeviceArray() = default;
  explicit DeviceArray(size_t n) : count(n) {
    FNGINE_TEST_CUDA(cudaMalloc(&ptr, n * sizeof(T) + 16));
    FNGINE_TEST_CUDA(cudaMemset(ptr, 0, n * sizeof(T) + 16));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  explicit DeviceArray(const std::vector<T>& h) : DeviceArray(h.size()) { upload(h); }
  ~DeviceArray() {
    if (ptr) cudaFree(ptr);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; }
  void upload(const std::vector<T>& h) {
    FNGINE_TEST_CUDA(cudaMemcpy(ptr, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    FNGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  std::vector<T> download() const {
    std::vector<T> h(count);
    FNGINE_TEST_CUDA(cudaMemcpy(h.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  T* get() const { return ptr; }
};

}  // namespace fngine::test
