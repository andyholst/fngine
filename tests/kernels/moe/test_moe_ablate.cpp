// Abliteration in the MoE combine kernels: only the routed-expert sum is transformed along a
// refusal direction (y_routed -= scale * r * (r . y_routed)), the shared expert is added
// untouched (its weights already carry the edit), for both the decode/verify combine and the
// prefill combine; the residual stream receives exactly inj * y. Real abliterations use
// scale != 1 (non-idempotent), so transforming the shared part too would be a visible error.
// Passing no direction must leave the previous behaviour bit-exact (that path is unchanged).
// Synthetic workspaces; no pack and no experts needed.
#include "moe_test_common.h"

#include <cmath>
#include <vector>

using namespace fngine;
using namespace fngine::test;
using namespace fngine::test::moe;

namespace {

constexpr int kHidden = dims::kHidden;
constexpr int kStreams = dims::kStreams;

// Deterministic pseudo-random unit direction.
std::vector<float> make_direction(uint64_t seed, double norm_target) {
  Rng rng(seed);
  std::vector<float> r(kHidden);
  for (auto& v : r) v = float(rng.normal());
  double n = 0;
  for (float v : r) n += double(v) * double(v);
  n = std::sqrt(n);
  for (auto& v : r) v = float(double(v) / n * norm_target);
  return r;
}

std::vector<float> make_dense(int n, uint64_t seed, double scale = 1.0) {
  Rng rng(seed);
  std::vector<float> v(static_cast<std::size_t>(n));
  for (auto& x : v) x = float(rng.normal() * scale);
  return v;
}

// Oracle: transform rows of `y` (T x 2560) along unit r: y - scale * r * (r . y).
std::vector<float> project(const std::vector<float>& y, const std::vector<float>& r, int T,
                           float scale) {
  std::vector<float> out(y.size());
  for (int t = 0; t < T; ++t) {
    double dot = 0;
    for (int d = 0; d < kHidden; ++d) dot += double(r[d]) * double(y[size_t(t) * kHidden + d]);
    for (int d = 0; d < kHidden; ++d)
      out[size_t(t) * kHidden + d] =
          float(double(y[size_t(t) * kHidden + d]) - double(scale) * dot * double(r[d]));
  }
  return out;
}

// Oracle for the combine: transform the routed part (total - shared), then add shared back.
std::vector<float> combine_oracle(const std::vector<float>& y_total, const std::vector<float>& y_shared,
                                  const std::vector<float>& r, int T, float scale) {
  std::vector<float> routed(y_total.size());
  for (size_t i = 0; i < routed.size(); ++i) routed[i] = float(double(y_total[i]) - double(y_shared[i]));
  std::vector<float> out = project(routed, r, T, scale);
  for (size_t i = 0; i < out.size(); ++i) out[i] = float(double(out[i]) + double(y_shared[i]));
  return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
  return m;
}

int expect(bool cond, const char* what) {
  std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
  return cond ? 0 : 1;
}

// ---------------------------------------------------------------- decode/verify combine

int test_decode_combine(double scale) {
  constexpr int T = 5;
  int fails = 0;

  std::vector<float> direction = make_direction(11, 1.0);  // unit vector
  std::vector<float> y_shared = make_dense(T * kHidden, 12, 0.5);
  std::vector<float> inj = make_dense(T * kStreams, 13, 1.5);
  std::vector<float> X0 = make_dense(T * kStreams * kHidden, 14, 0.25);
  std::vector<float> pairs = make_dense(kernels::kMoeDecodeMaxPairs * kHidden, 15, 0.35);

  DeviceArray<uint8_t> ws_mem(kernels::moe_decode_workspace_bytes());
  const auto ws = kernels::make_moe_decode_workspace(ws_mem.ptr);

  // Work list: token t owns t+1 hot pairs (slot order), no cold pairs.
  kernels::MoeDecodeWork work{};
  int slot = 0;
  for (int t = 0; t < T; ++t) {
    work.token_pair_count[t] = t + 1;
    for (int j = 0; j <= t; ++j) work.token_pairs[t][j] = slot++;
  }
  work.tokens = T;
  work.n_items = slot;
  FNGINE_TEST_CUDA(cudaMemcpy(ws.work, &work, sizeof(work), cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(ws.partial, pairs.data(), pairs.size() * 4, cudaMemcpyHostToDevice));

  DeviceArray<float> d_ysh(y_shared.size());
  DeviceArray<float> d_inj(inj.size());
  DeviceArray<float> d_dir(direction.size());
  DeviceArray<float> d_X(X0.size());
  DeviceArray<float> d_Xref(X0.size());
  DeviceArray<float> d_y(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(d_ysh.ptr, y_shared.data(), y_shared.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_inj.ptr, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_dir.ptr, direction.data(), direction.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_X.ptr, X0.data(), X0.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // pageable H2D may still be in flight
  FNGINE_TEST_CUDA(cudaMemcpy(d_Xref.ptr, X0.data(), X0.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // pageable H2D may still be in flight

  cudaStream_t s;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, d_ysh.ptr, nullptr, 0, nullptr, d_inj.ptr, d_X.ptr,
                                        d_y.ptr, T, kernels::kMoeExchangeTimeoutCycles, nullptr,
                                        1.0f, s));
  FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
  std::vector<float> y_plain(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(y_plain.data(), d_y.ptr, y_plain.size() * 4, cudaMemcpyDeviceToHost));

  // Same call with the direction: y must be exactly the projected plain sum, X must match inj * y.
  FNGINE_TEST_CUDA(cudaMemcpy(d_X.ptr, X0.data(), X0.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // pageable H2D may still be in flight
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, d_ysh.ptr, nullptr, 0, nullptr, d_inj.ptr, d_X.ptr,
                                        d_y.ptr, T, kernels::kMoeExchangeTimeoutCycles,
                                        d_dir.ptr, scale, s));
  FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
  std::vector<float> y_got(size_t(T) * kHidden), X_got(X0.size());
  FNGINE_TEST_CUDA(cudaMemcpy(y_got.data(), d_y.ptr, y_got.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(X_got.data(), d_X.ptr, X_got.size() * 4, cudaMemcpyDeviceToHost));

  const std::vector<float> y_want = combine_oracle(y_plain, y_shared, direction, T, scale);
  double tol = 1e-4;
  for (double v : y_want) tol = std::max(tol, 2e-6 * std::fabs(v));
  const double dy = max_abs_diff(y_got, y_want);
  fails += expect(dy <= tol, "decode combine: routed sum transformed, shared added untouched");
  if (dy > tol) std::printf("  dy=%.3e tol=%.3e\n", dy, tol);

  // Along r, the routed component must be scaled by exactly (1 - scale) while the shared
  // component is unchanged: r.(y_after - shared) == (1 - scale) * r.(y_before - shared).
  double worst = 0, y_norm = 0;
  for (int t = 0; t < T; ++t) {
    double before = 0, after = 0;
    for (int d = 0; d < kHidden; ++d) {
      const size_t k = size_t(t) * kHidden + d;
      const double r = direction[d];
      before += r * (double(y_plain[k]) - double(y_shared[k]));
      after += r * (double(y_got[k]) - double(y_shared[k]));
      y_norm += double(y_got[k]) * double(y_got[k]);
    }
    worst = std::max(worst, std::fabs(after - (1.0 - double(scale)) * before));
  }
  const double rel = worst / std::sqrt(y_norm);
  fails += expect(rel < 1e-5, "decode combine: routed component along r scaled by (1 - scale)");
  if (rel >= 1e-5) std::printf("  relative error %.3e\n", rel);

  std::vector<float> X_want = X0;
  for (int t = 0; t < T; ++t)
    for (int st = 0; st < kStreams; ++st)
      for (int d = 0; d < kHidden; ++d) {
        const size_t i = (size_t(t) * kStreams + st) * kHidden + d;
        X_want[i] = float(double(X0[i]) +
                          double(inj[size_t(t) * kStreams + st]) * double(y_want[size_t(t) * kHidden + d]));
      }
  const double dx = max_abs_diff(X_got, X_want);
  fails += expect(dx <= tol, "decode combine: residual streams receive inj * projected y");
  if (dx > tol) std::printf("  dx=%.3e tol=%.3e\n", dx, tol);

  // Without a direction the plain path must be untouched (regression guard).
  FNGINE_TEST_CUDA(kernels::moe_combine(ws, d_ysh.ptr, nullptr, 0, nullptr, d_inj.ptr, d_Xref.ptr,
                                        d_y.ptr, T, kernels::kMoeExchangeTimeoutCycles, nullptr,
                                        1.0f, s));
  FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
  std::vector<float> y_ref(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(y_ref.data(), d_y.ptr, y_ref.size() * 4, cudaMemcpyDeviceToHost));
  fails += expect(max_abs_diff(y_ref, y_plain) == 0.0, "decode combine: null direction is bit-exact");

  FNGINE_TEST_CUDA(cudaStreamDestroy(s));
  return fails;
}

// ---------------------------------------------------------------- prefill combine

int test_prefill_combine(double scale) {
  constexpr int T = 37;  // not a multiple of the combine tile
  int fails = 0;

  std::vector<float> direction = make_direction(21, 1.0);
  std::vector<float> y_shared = make_dense(T * kHidden, 22, 0.5);
  std::vector<float> inj = make_dense(T * kStreams, 23, 1.5);
  std::vector<float> X0 = make_dense(T * kStreams * kHidden, 24, 0.25);
  std::vector<double> y_hot_f(size_t(T) * kHidden);
  {
    Rng rng(25);
    for (auto& v : y_hot_f) v = rng.normal() * 0.3;
  }
  // The prefill sum is fixed point; the oracle starts from the same quantized values.
  std::vector<long long> y_fixed(y_hot_f.size());
  for (size_t i = 0; i < y_hot_f.size(); ++i)
    y_fixed[i] = static_cast<long long>(std::llround(y_hot_f[i] * double(1ull << kernels::kMoeYFixedShift)));

  const int capacity = 64;
  DeviceArray<uint8_t> ws_mem(kernels::moe_prefill_workspace_bytes(capacity));
  const auto ws = kernels::make_moe_prefill_workspace(ws_mem.ptr, capacity);
  std::vector<int32_t> row_of_token(T, -1);
  FNGINE_TEST_CUDA(cudaMemcpy(ws.y_fixed, y_fixed.data(), y_fixed.size() * 8, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(ws.row_of_token, row_of_token.data(), T * 4, cudaMemcpyHostToDevice));

  DeviceArray<float> d_ysh(y_shared.size()), d_inj(inj.size()), d_dir(direction.size());
  DeviceArray<float> d_X(X0.size()), d_y(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(d_ysh.ptr, y_shared.data(), y_shared.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_inj.ptr, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_dir.ptr, direction.data(), direction.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaMemcpy(d_X.ptr, X0.data(), X0.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // pageable H2D may still be in flight

  cudaStream_t s;
  FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  FNGINE_TEST_CUDA(kernels::moe_prefill_combine(ws, d_ysh.ptr, nullptr, PrefillExchangeView{}, 0,
                                                nullptr, d_inj.ptr, d_X.ptr, d_y.ptr, T,
                                                kernels::kMoeExchangeTimeoutCycles, nullptr, 1.0f, s));
  FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
  std::vector<float> y_plain(size_t(T) * kHidden);
  FNGINE_TEST_CUDA(cudaMemcpy(y_plain.data(), d_y.ptr, y_plain.size() * 4, cudaMemcpyDeviceToHost));

  FNGINE_TEST_CUDA(cudaMemcpy(d_X.ptr, X0.data(), X0.size() * 4, cudaMemcpyHostToDevice));
  FNGINE_TEST_CUDA(cudaDeviceSynchronize());  // pageable H2D may still be in flight
  FNGINE_TEST_CUDA(kernels::moe_prefill_combine(ws, d_ysh.ptr, nullptr, PrefillExchangeView{}, 0,
                                                nullptr, d_inj.ptr, d_X.ptr, d_y.ptr, T,
                                                kernels::kMoeExchangeTimeoutCycles, d_dir.ptr,
                                                scale, s));
  FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
  std::vector<float> y_got(size_t(T) * kHidden), X_got(X0.size());
  FNGINE_TEST_CUDA(cudaMemcpy(y_got.data(), d_y.ptr, y_got.size() * 4, cudaMemcpyDeviceToHost));
  FNGINE_TEST_CUDA(cudaMemcpy(X_got.data(), d_X.ptr, X_got.size() * 4, cudaMemcpyDeviceToHost));

  const std::vector<float> y_want = combine_oracle(y_plain, y_shared, direction, T, scale);
  double tol = 1e-4;
  for (double v : y_want) tol = std::max(tol, 2e-6 * std::fabs(v));
  const double dy = max_abs_diff(y_got, y_want);
  fails += expect(dy <= tol, "prefill combine: routed sum transformed, shared added untouched");
  if (dy > tol) std::printf("  dy=%.3e tol=%.3e\n", dy, tol);

  std::vector<float> X_want = X0;
  for (int t = 0; t < T; ++t)
    for (int st = 0; st < kStreams; ++st)
      for (int d = 0; d < kHidden; ++d) {
        const size_t i = (size_t(t) * kStreams + st) * kHidden + d;
        X_want[i] = float(double(X0[i]) +
                          double(inj[size_t(t) * kStreams + st]) * double(y_want[size_t(t) * kHidden + d]));
      }
  const double dx = max_abs_diff(X_got, X_want);
  fails += expect(dx <= tol, "prefill combine: residual streams receive inj * projected y");
  if (dx > tol) std::printf("  dx=%.3e tol=%.3e\n", dx, tol);

  FNGINE_TEST_CUDA(cudaStreamDestroy(s));
  return fails;
}

// The projection is idempotent at scale 1: projecting an already projected vector changes nothing.
int test_idempotent() {
  constexpr int T = 3;
  std::vector<float> direction = make_direction(31, 1.0);
  std::vector<float> y = make_dense(T * kHidden, 32, 1.0);
  const std::vector<float> once = project(y, direction, T, 1.0f);
  const std::vector<float> twice = project(once, direction, T, 1.0f);
  const double d = max_abs_diff(once, twice);
  return expect(d < 1e-4, "projection is idempotent at scale 1");
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  int fails = 0;
  fails += test_decode_combine(1.0);
  fails += test_decode_combine(0.5);
  fails += test_decode_combine(1.955);  // the scale fitted on the released abliteration
  fails += test_prefill_combine(1.0);
  fails += test_prefill_combine(1.955);
  fails += test_idempotent();
  std::printf("%s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
  return fails == 0 ? 0 : 1;
}
