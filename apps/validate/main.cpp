// fngine-validate: teacher-forced prefill of the saved golden prompts (G0/G1/G2) through the
// engine's forward pass, compared with the Hugging Face reference logits (top-1 agreement, mean
// KLD on positions with saved full logits) and, for G0, per-layer residual-stream cosine.
// --decode-width W instead teacher-forces the prompts through the decode/verify step graphs.
#include "core/device.h"
#include "core/model_dims.h"
#include "fngine/types.h"
#include "kernels/head/head.h"
#include "model/runtime.h"
#include "model/step_graph.h"
#include "safetensors.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <memory>
#include <span>
#include <vector>

using namespace fngine;
using validate::SafeTensors;
using validate::StTensor;

namespace {

struct Args {
  std::string pack = "/mnt/data/fngine/pack";
  std::string ref = "/mnt/data/fngine/ref";
  std::vector<std::string> golden = {"g0", "g1", "g2"};
  bool dense_bf16 = false;
  std::string fp8_path = "auto";
  bool kv_bf16 = false;
  std::string kv = "k8v8";
  std::string ref_dtype = "bf16";  // planned HF BF16 reference; FP32 is an explicit diagnostic
  bool dump_layers = false;
  int32_t chunk = 4096;
  int cpu_workers = 0;  // 0: every worker core of the CPU placement policy
  std::optional<uint64_t> expert_vram_mib;
  std::string json_out;
  bool repeat_diff = false;  // run each golden twice and report per-layer run-to-run differences
  bool timing = false;       // prefill each golden without hooks and report tokens/s
  int32_t swaps_per_chunk = 0;  // --swap-every-chunk: random GPU<->RAM expert swaps before every chunk
  int32_t synthetic_tokens = 0;  // timing mode: prefill N tokens (the first golden's ids repeated)
  bool prefill_stream = true;
  bool prefill_borrow = true;
  // Path-vs-path comparison: save our logits (argmax of every position, full rows of every 32nd
  // position) / compare against such a file (top-1 agreement and KLD(saved || ours)).
  std::string save_logits;
  std::string compare_logits;
  // Decode-path mode: width-W verify steps (1 = plain decode) fed the golden tokens.
  int32_t decode_width = 0;
  bool adaptive = true;  // --no-adaptive: fixed placement for path-vs-path comparisons
  bool ablation = true;  // --no-ablation: ignore an abliterated pack's ablation section
  int64_t limit = 0;        // --limit N: only the first N golden positions (debugging)
  int64_t save_every = 32;  // --save-every N: full saved/compared logit rows every N positions
  // --generate N: prefill the (limited) golden prompt, then greedy-generate N tokens through the
  // step graphs; --mtp drafts min(W - 1, 3) MTP tokens per step (W = --decode-width).
  int32_t generate = 0;
  bool mtp = false;
  std::string dump_decode_counts;  // --dump-decode-counts FILE: u64 [48][512] decode routing counts
};

[[noreturn]] void usage(int code) {
  std::printf(
      "usage: fngine-validate [--pack DIR] [--ref DIR] [--golden g0,g1,g2] [--dense fp8|bf16]\n"
      "                       [--kv k8v8|k8v4|bf16] [--ref-dtype bf16|fp32] [--dump-layers]\n"
      "                       [--prefill-chunk N] [--cpu-workers N] [--expert-vram-mib N] [--json FILE]\n"
      "                       [--no-borrow --swap-every-chunk [--swaps-per-chunk N]]\n"
      "                       [--timing [--synthetic-tokens N]] [--no-prefill-stream] [--no-borrow]\n"
      "                       [--save-logits FILE | --compare-logits FILE] [--decode-width 1..9]\n"
      "                       [--no-adaptive] [--no-ablation] [--limit N] [--save-every N]\n");
  std::exit(code);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(2);
      return argv[++i];
    };
    if (k == "--pack") a.pack = val();
    else if (k == "--ref") a.ref = val();
    else if (k == "--golden") {
      a.golden.clear();
      std::string s = val();
      for (std::size_t p = 0; p < s.size();) {
        std::size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        a.golden.push_back(s.substr(p, q - p));
        p = q + 1;
      }
    } else if (k == "--dense") a.dense_bf16 = val() == "bf16";
    else if (k == "--fp8-path") a.fp8_path = val();
    else if (k == "--kv") {
      a.kv = val();
      a.kv_bf16 = a.kv == "bf16";
    }
    else if (k == "--ref-dtype") a.ref_dtype = val();
    else if (k == "--dump-layers") a.dump_layers = true;
    else if (k == "--prefill-chunk") a.chunk = std::stoi(val());
    else if (k == "--cpu-workers") a.cpu_workers = std::stoi(val());
    else if (k == "--expert-vram-mib") a.expert_vram_mib = std::stoull(val());
    else if (k == "--json") a.json_out = val();
    else if (k == "--repeat-diff") a.repeat_diff = true;
    else if (k == "--timing") a.timing = true;
    else if (k == "--swap-every-chunk") a.swaps_per_chunk = std::max(a.swaps_per_chunk, 32);
    else if (k == "--swaps-per-chunk") a.swaps_per_chunk = std::stoi(val());
    else if (k == "--synthetic-tokens") a.synthetic_tokens = std::stoi(val());
    else if (k == "--no-prefill-stream") a.prefill_stream = false;
    else if (k == "--no-borrow") a.prefill_borrow = false;
    else if (k == "--save-logits") a.save_logits = val();
    else if (k == "--compare-logits") a.compare_logits = val();
    else if (k == "--decode-width") a.decode_width = std::stoi(val());
    else if (k == "--no-adaptive") a.adaptive = false;
    else if (k == "--no-ablation") a.ablation = false;
    else if (k == "--limit") a.limit = std::stoll(val());
    else if (k == "--save-every") a.save_every = std::stoll(val());
    else if (k == "--generate") a.generate = std::stoi(val());
    else if (k == "--mtp") a.mtp = true;
    else if (k == "--dump-decode-counts") a.dump_decode_counts = val();
    else if (k == "--help" || k == "-h") usage(0);
    else {
      std::fprintf(stderr, "unknown argument %s\n", k.c_str());
      usage(2);
    }
  }
  if (a.limit < 0 || a.save_every < 1 || a.generate < 0) usage(2);
  if ((a.generate > 0 || a.mtp) && a.decode_width <= 0) {
    std::fprintf(stderr, "--generate/--mtp need --decode-width\n");
    usage(2);
  }
  if (a.ref_dtype != "bf16" && a.ref_dtype != "fp32") usage(2);
  if (a.kv != "bf16" && a.kv != "k8v4" && a.kv != "k8v8") usage(2);
  if (a.fp8_path != "auto" && a.fp8_path != "a16" && a.fp8_path != "a8") usage(2);
  if (a.synthetic_tokens < 0 || (!a.save_logits.empty() && !a.compare_logits.empty())) usage(2);
  if (a.swaps_per_chunk < 0) usage(2);
  if (a.swaps_per_chunk > 0 && a.prefill_borrow) {
    std::fprintf(stderr, "forced swaps require --no-borrow: prefill scratch occupies loaned expert slots\n");
    usage(2);
  }
  if ((!a.save_logits.empty() || !a.compare_logits.empty()) && a.golden.size() != 1) {
    std::fprintf(stderr, "--save-logits/--compare-logits need exactly one --golden\n");
    usage(2);
  }
  if (a.decode_width < 0 || a.decode_width > dims::kMaxGraphWidth) usage(2);
  if (a.decode_width > 0 && (a.repeat_diff || a.dump_layers || a.swaps_per_chunk > 0)) {
    std::fprintf(stderr, "--decode-width cannot be combined with repeat, layer dumps or swaps\n");
    usage(2);
  }
  return a;
}

// log-softmax normalizer of a row (FP64).
double logsumexp(const float* x, int n) {
  double m = -INFINITY;
  for (int i = 0; i < n; ++i) m = std::max(m, double(x[i]));
  double s = 0;
  for (int i = 0; i < n; ++i) s += std::exp(double(x[i]) - m);
  return m + std::log(s);
}

// KL(p_ref || p_ours) in nats.
double kld(const std::vector<float>& ref, const float* ours, int n) {
  const double lr = logsumexp(ref.data(), n), lo = logsumexp(ours, n);
  double k = 0;
  for (int i = 0; i < n; ++i) {
    const double a = double(ref[std::size_t(i)]) - lr;
    k += std::exp(a) * (a - (double(ours[i]) - lo));
  }
  return std::max(0.0, k);
}

int argmax_row(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[best]) best = i;
  return best;
}

struct Result {
  std::string name;
  int64_t positions = 0, top1_match = 0, kld_rows = 0;
  double kld_sum = 0, kld_max = 0;
  double seconds = 0;
  std::vector<double> layer_cos;      // global cosine per stream index (0 = embedding)
  // Per 256-position window: positions, top-1 matches, KLD rows, KLD sum.
  std::vector<std::array<double, 4>> windows;
  std::vector<double> layer_cos_min;  // min per-position cosine
  // --compare-logits: agreement with the saved run (top-1 over all positions, KLD on every 32nd).
  int64_t pair_positions = 0, pair_top1 = 0, pair_kld_rows = 0;
  double pair_kld_sum = 0, pair_kld_max = 0;
  [[nodiscard]] double top1() const { return positions ? double(top1_match) / double(positions) : 0; }
  [[nodiscard]] double mean_kld() const { return kld_rows ? kld_sum / double(kld_rows) : 0; }
};

// Saved-logits file: int64 T, int32 argmax[T], float rows[(T + E - 1) / E][vocab] (positions 0, E,
// ...), E = --save-every (default 32; compare with the cadence the file was saved with).
struct SavedLogits {
  int64_t T = 0;
  std::vector<int32_t> argmax;
  std::vector<float> rows;
};

Result run_golden(model::ModelRuntime& rt, model::StepRunner* runner, const Args& args,
                  const std::string& name) {
  const std::string suffix = args.ref_dtype == "fp32" ? "_fp32" : "";
  const SafeTensors g(args.ref + "/" + name + suffix + ".safetensors");
  const StTensor& ids_t = g.at("ids");
  const int64_t T = args.limit > 0 ? std::min<int64_t>(args.limit, ids_t.numel()) : ids_t.numel();
  const int64_t kSavedRowEvery = args.save_every;
  std::vector<int32_t> ids(static_cast<std::size_t>(T));
  for (int64_t i = 0; i < T; ++i) ids[std::size_t(i)] = validate::st_int(ids_t, i);

  // Reference: top-1 per position and full logits per position (when saved).
  const bool full_all = g.has("logits");
  std::vector<int32_t> ref_top1(static_cast<std::size_t>(T));
  std::map<int64_t, int64_t> full_row;  // position -> row in the full-logit tensor
  const StTensor* full = nullptr;
  constexpr int V = dims::kVocab;
  if (full_all) {
    full = &g.at("logits");
    for (int64_t p = 0; p < T; ++p) full_row[p] = p;
  } else {
    full = &g.at("full_logits");
    const StTensor& fp = g.at("full_positions");
    for (int64_t r = 0; r < fp.numel(); ++r) full_row[validate::st_int(fp, r)] = r;
    const StTensor& top = g.at("top32_ids");
    const int64_t K = top.shape[1];
    for (int64_t p = 0; p < T; ++p) ref_top1[std::size_t(p)] = validate::st_int(top, p * K);
  }
  std::vector<float> row(static_cast<std::size_t>(V));
  auto load_ref_row = [&](int64_t r) {
    for (int i = 0; i < V; ++i) row[std::size_t(i)] = validate::st_float(*full, r * V + i);
  };
  if (full_all) {
    for (int64_t p = 0; p < T; ++p) {
      load_ref_row(p);
      ref_top1[std::size_t(p)] = argmax_row(row.data(), V);
    }
  }

  // Optional per-layer streams (G0).
  std::unique_ptr<SafeTensors> streams_file;
  const StTensor* streams = nullptr;
  Result res;
  res.name = name;
  if (args.dump_layers && name == "g0") {
    const std::string p = args.ref + "/g0_streams" + suffix + ".safetensors";
    streams_file = std::make_unique<SafeTensors>(p);
    streams = &streams_file->at("streams");
    res.layer_cos.assign(dims::kLayers + 1, 0.0);
    res.layer_cos_min.assign(dims::kLayers + 1, 1.0);
  }
  std::vector<double> dot(dims::kLayers + 1, 0), n_ours(dims::kLayers + 1, 0), n_ref(dims::kLayers + 1, 0);

  model::PrefillHooks hooks;
  std::vector<float> host_logits;
  std::vector<float> xs;
  SavedLogits saved;
  if (!args.compare_logits.empty()) {
    std::ifstream in(args.compare_logits, std::ios::binary);
    in.read(reinterpret_cast<char*>(&saved.T), sizeof(saved.T));
    if (!in || saved.T != T) throw std::runtime_error("--compare-logits: missing file or token count mismatch");
    saved.argmax.resize(std::size_t(T));
    saved.rows.resize(std::size_t((T + kSavedRowEvery - 1) / kSavedRowEvery) * V);
    in.read(reinterpret_cast<char*>(saved.argmax.data()), std::streamsize(saved.argmax.size() * 4));
    in.read(reinterpret_cast<char*>(saved.rows.data()), std::streamsize(saved.rows.size() * 4));
    if (!in) throw std::runtime_error("--compare-logits: truncated file");
  } else if (!args.save_logits.empty()) {
    saved.T = T;
    saved.argmax.resize(std::size_t(T));
    saved.rows.resize(std::size_t((T + kSavedRowEvery - 1) / kSavedRowEvery) * V);
  }
  std::vector<float> saved_row;
  hooks.on_logits = [&](int32_t pos0, int32_t rows, const float* dev_logits) {
    host_logits.resize(std::size_t(rows) * V);
    CUDA_CHECK(cudaMemcpy(host_logits.data(), dev_logits, host_logits.size() * 4, cudaMemcpyDeviceToHost));
    for (int32_t r = 0; r < rows; ++r) {
      const int64_t p = pos0 + r;
      const float* ours = host_logits.data() + std::size_t(r) * V;
      const std::size_t w = std::size_t(p / 256);
      if (res.windows.size() <= w) res.windows.resize(w + 1, {0, 0, 0, 0});
      res.positions++;
      res.windows[w][0] += 1;
      const int am = argmax_row(ours, V);
      if (am == ref_top1[std::size_t(p)]) {
        res.top1_match++;
        res.windows[w][1] += 1;
      }
      const bool row_saved = p % kSavedRowEvery == 0;
      float* srow = saved.rows.empty() ? nullptr : saved.rows.data() + std::size_t(p / kSavedRowEvery) * V;
      if (!args.save_logits.empty()) {
        saved.argmax[std::size_t(p)] = am;
        if (row_saved) std::copy(ours, ours + V, srow);
      } else if (!args.compare_logits.empty()) {
        res.pair_positions++;
        res.pair_top1 += saved.argmax[std::size_t(p)] == am;
        if (row_saved) {
          saved_row.assign(srow, srow + V);
          const double k = kld(saved_row, ours, V);
          res.pair_kld_sum += k;
          res.pair_kld_max = std::max(res.pair_kld_max, k);
          res.pair_kld_rows++;
        }
      }
      auto it = full_row.find(p);
      if (it != full_row.end()) {
        load_ref_row(it->second);
        const double k = kld(row, ours, V);
        res.kld_sum += k;
        res.kld_max = std::max(res.kld_max, k);
        res.kld_rows++;
        res.windows[w][2] += 1;
        res.windows[w][3] += k;
      }
    }
  };
  if (streams) {
    hooks.after_layer = [&](int32_t layer, int32_t pos0, int32_t rows) {
      const int idx = layer + 1;
      xs.resize(std::size_t(rows) * dims::kStreamHidden);
      CUDA_CHECK(cudaMemcpy(xs.data(), rt.ws().X, xs.size() * 4, cudaMemcpyDeviceToHost));
      const int64_t TT = streams->shape[1];
      for (int32_t r = 0; r < rows; ++r) {
        const int64_t p = pos0 + r;
        double d = 0, a = 0, b = 0;
        for (int j = 0; j < dims::kStreamHidden; ++j) {
          const double o = xs[std::size_t(r) * dims::kStreamHidden + std::size_t(j)];
          const double f = validate::st_float(*streams, (int64_t(idx) * TT + p) * dims::kStreamHidden + j);
          d += o * f; a += o * o; b += f * f;
        }
        dot[std::size_t(idx)] += d; n_ours[std::size_t(idx)] += a; n_ref[std::size_t(idx)] += b;
        const double c = d / std::sqrt(std::max(a * b, 1e-300));
        res.layer_cos_min[std::size_t(idx)] = std::min(res.layer_cos_min[std::size_t(idx)], c);
      }
    };
  }
  // Random expert swaps before the first chunk and between chunks (slot maps change only there).
  uint64_t swap_seed = 0x5eed0000u + std::hash<std::string>{}(name);
  if (args.swaps_per_chunk > 0) {
    hooks.after_chunk = [&](int32_t) { rt.force_random_swaps(args.swaps_per_chunk, swap_seed++); };
  }

  rt.reset_sequence();
  if (args.swaps_per_chunk > 0) rt.force_random_swaps(args.swaps_per_chunk, swap_seed++);
  const auto t0 = std::chrono::steady_clock::now();
  if (runner) {
    // Position 0 is prefilled (not compared); every later position is computed by a verify step
    // whose inputs are golden tokens. Only committed rows are recorded: a rejected suffix is
    // recomputed by the next step at the same positions.
    rt.prefill(std::span<const int32_t>(ids.data(), 1));
    for (int64_t p = 1; p < T;) {
      const int64_t width = std::min<int64_t>(args.decode_width, T - p);
      const model::StepResult r =
          runner->step(ids[std::size_t(p)], std::span<const int32_t>(ids.data() + p + 1, std::size_t(width - 1)));
      hooks.on_logits(int32_t(p), r.accepted + 1, rt.ws().logits);
      p += r.accepted + 1;
    }
  } else {
    rt.prefill(ids, &hooks);
  }
  if (args.swaps_per_chunk > 0) std::printf("%s: %llu expert swaps completed so far\n", name.c_str(),
                                            (unsigned long long)rt.expert_swaps());
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (streams) {
    for (int i = 0; i <= dims::kLayers; ++i)
      res.layer_cos[std::size_t(i)] =
          dot[std::size_t(i)] / std::sqrt(std::max(n_ours[std::size_t(i)] * n_ref[std::size_t(i)], 1e-300));
  }
  if (!args.save_logits.empty()) {
    std::ofstream out(args.save_logits, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&saved.T), sizeof(saved.T));
    out.write(reinterpret_cast<const char*>(saved.argmax.data()), std::streamsize(saved.argmax.size() * 4));
    out.write(reinterpret_cast<const char*>(saved.rows.data()), std::streamsize(saved.rows.size() * 4));
    if (!out) throw std::runtime_error("--save-logits: write failed");
  }
  return res;
}

// Runs the golden prompt twice and prints, per layer, the largest run-to-run difference of the
// residual streams (every 16th position), relative to the stream RMS. Deterministic kernels give
// zero (prefill MoE atomics give ~1e-7); anything large means a race or an uninitialized read.
void repeat_diff(model::ModelRuntime& rt, const Args& args, const std::string& name) {
  const std::string suffix = args.ref_dtype == "fp32" ? "_fp32" : "";
  const SafeTensors g(args.ref + "/" + name + suffix + ".safetensors");
  const StTensor& ids_t = g.at("ids");
  std::vector<int32_t> ids(static_cast<std::size_t>(ids_t.numel()));
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = validate::st_int(ids_t, int64_t(i));
  constexpr int kStride = 16;
  std::vector<std::vector<float>> first(dims::kLayers + 1);
  std::vector<float> xs;
  for (int run = 0; run < 2; ++run) {
    std::vector<double> maxdiff(dims::kLayers + 1, 0), rms(dims::kLayers + 1, 0);
    std::vector<int64_t> cnt(dims::kLayers + 1, 0);
    model::PrefillHooks hooks;
    hooks.after_layer = [&](int32_t layer, int32_t pos0, int32_t rows) {
      const std::size_t idx = std::size_t(layer + 1);
      xs.resize(std::size_t(rows) * dims::kStreamHidden);
      CUDA_CHECK(cudaMemcpy(xs.data(), rt.ws().X, xs.size() * 4, cudaMemcpyDeviceToHost));
      for (int32_t r = 0; r < rows; ++r) {
        if ((pos0 + r) % kStride != 0) continue;
        const float* x = xs.data() + std::size_t(r) * dims::kStreamHidden;
        if (run == 0) {
          first[idx].insert(first[idx].end(), x, x + dims::kStreamHidden);
        } else {
          const float* a = first[idx].data() + std::size_t(cnt[idx]) * dims::kStreamHidden;
          for (int j = 0; j < dims::kStreamHidden; ++j) {
            maxdiff[idx] = std::max(maxdiff[idx], double(std::fabs(a[j] - x[j])));
            rms[idx] += double(a[j]) * a[j];
          }
        }
        cnt[idx]++;
      }
    };
    rt.reset_sequence();
    rt.prefill(ids, &hooks);
    if (run == 1) {
      std::printf("%s run-to-run max |diff| / rms per stream index:\n", name.c_str());
      for (int i = 0; i <= dims::kLayers; ++i) {
        const double r = std::sqrt(rms[std::size_t(i)] / double(std::max<int64_t>(1, cnt[std::size_t(i)]) * dims::kStreamHidden));
        std::printf("  %2d: %.3e%s", i, maxdiff[std::size_t(i)] / std::max(r, 1e-30), (i % 6 == 5) ? "\n" : "");
      }
      std::printf("\n");
    }
  }
}

// Decode-path timing: teacher-forced golden tokens through the width-W step graphs without reading
// logits, so every run (and every build) decodes the same token stream with the same routing.
void decode_timing(model::ModelRuntime& rt, model::StepRunner& runner, const Args& args,
                   const std::string& name, nlohmann::json& out) {
  const SafeTensors g(args.ref + "/" + name + ".safetensors");
  const StTensor& ids_t = g.at("ids");
  const int64_t T = args.limit > 0 ? std::min<int64_t>(args.limit, ids_t.numel()) : ids_t.numel();
  std::vector<int32_t> ids(static_cast<std::size_t>(T));
  for (int64_t i = 0; i < T; ++i) ids[std::size_t(i)] = validate::st_int(ids_t, i);
  for (int rep = 0; rep < 2; ++rep) {
    rt.reset_sequence();
    rt.prefill(std::span<const int32_t>(ids.data(), 1));
    rt.pool().reset_timings();
    int64_t steps = 0;
    double step_seconds = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t p = 1; p < T;) {
      const int64_t width = std::min<int64_t>(args.decode_width, T - p);
      const model::StepResult r =
          runner.step(ids[std::size_t(p)], std::span<const int32_t>(ids.data() + p + 1, std::size_t(width - 1)));
      step_seconds += r.seconds;
      ++steps;
      p += r.accepted + 1;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    uint64_t comp = 0, layers = 0, pairs = 0, experts = 0;
    for (const auto& lt : rt.pool().timings()) {
      comp += lt.compute_ns;
      layers += lt.steps;
      pairs += lt.pairs;
      experts += lt.experts;
    }
    const double positions = double(T - 1);
    const double cpu_us = layers ? 1e-3 * double(comp) / double(layers) : 0.0;
    std::printf("%s decode W=%d rep %d: %.0f positions in %.3f s = %.2f tok/s; %lld steps, %.3f ms/step "
                "(%.2f tok/step); CPU tier %.1f us per computed layer (%.2f pairs, %.2f experts, "
                "%.1f computed layers/step)\n",
                name.c_str(), args.decode_width, rep, positions, s, positions / s, (long long)steps,
                1e3 * step_seconds / double(steps), positions / double(steps), cpu_us,
                layers ? double(pairs) / double(layers) : 0.0, layers ? double(experts) / double(layers) : 0.0,
                double(layers) / double(steps));
    out["decode_timing"][name].push_back({{"repetition", rep}, {"width", args.decode_width},
        {"positions", positions}, {"seconds", s}, {"tokens_per_second", positions / s},
        {"steps", steps}, {"ms_per_step", 1e3 * step_seconds / double(steps)},
        {"cpu_us_per_computed_layer", cpu_us}});
  }
}

// Greedy generation timing: prefill the golden prompt (limited by --limit), then generate
// --generate tokens; with --mtp each step verifies min(W - 1, 3) MTP drafts (k = 0 still runs the
// MTP drafter, which measures its overhead).
void generate_timing(model::ModelRuntime& rt, model::StepRunner& runner, const Args& args,
                     const std::string& name, nlohmann::json& out, std::vector<uint64_t>* decode_counts) {
  const SafeTensors g(args.ref + "/" + name + ".safetensors");
  const StTensor& ids_t = g.at("ids");
  const int64_t P = args.limit > 0 ? std::min<int64_t>(args.limit, ids_t.numel()) : ids_t.numel();
  std::vector<int32_t> ids(static_cast<std::size_t>(P));
  for (int64_t i = 0; i < P; ++i) ids[std::size_t(i)] = validate::st_int(ids_t, i);
  const int k = runner.mtp() ? std::min(args.decode_width - 1, int(model::kMtpDrafts)) : 0;
  for (int rep = 0; rep < 2; ++rep) {
    rt.reset_sequence();
    rt.prefill(ids);
    model::StepResult r = runner.first_token(int32_t(P));
    int32_t anchor = r.licensed[0];
    std::array<int32_t, model::kMtpDrafts> drafts{};
    std::copy(std::begin(r.mtp_drafts), std::begin(r.mtp_drafts) + model::kMtpDrafts, drafts.begin());
    int64_t produced = 1, steps = 0, accepted = 0;
    std::vector<uint32_t> before;
    if (decode_counts && rep == 0) before = rt.routing_counters();
    rt.pool().reset_timings();
    const auto t0 = std::chrono::steady_clock::now();
    while (produced < args.generate && rt.position() + k + 2 < rt.options().max_context) {
      r = runner.step(anchor, std::span<const int32_t>(drafts.data(), std::size_t(std::min(k, r.mtp_count))));
      produced += r.count;
      accepted += r.accepted;
      ++steps;
      anchor = r.licensed[r.count - 1];
      std::copy(std::begin(r.mtp_drafts), std::begin(r.mtp_drafts) + model::kMtpDrafts, drafts.begin());
    }
    if (decode_counts && rep == 0) {
      const std::vector<uint32_t> after = rt.routing_counters();
      for (std::size_t i = 0; i < after.size(); ++i) (*decode_counts)[i] += uint32_t(after[i] - before[i]);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gen = double(produced - 1);
    uint64_t comp = 0, layers = 0, pairs = 0;
    for (const auto& lt : rt.pool().timings()) {
      comp += lt.compute_ns;
      layers += lt.steps;
      pairs += lt.pairs;
    }
    std::printf("CPU tier %.1f us x %.1f layers/step, %.2f pairs/layer | ",
                layers ? 1e-3 * double(comp) / double(layers) : 0.0, double(layers) / double(steps),
                layers ? double(pairs) / double(layers) : 0.0);
    std::printf("%s generate prompt %lld, mtp=%d k=%d rep %d: %.0f tokens in %.3f s = %.2f tok/s; %lld steps, "
                "%.3f ms/step, %.3f tok/step (draft acceptance %.3f)\n",
                name.c_str(), (long long)P, runner.mtp() ? 1 : 0, k, rep, gen, s, gen / s,
                (long long)steps, 1e3 * s / double(steps), gen / double(steps),
                k ? double(accepted) / double(steps * k) : 0.0);
    out["generate"][name].push_back({{"repetition", rep}, {"mtp", runner.mtp()}, {"k", k},
        {"tokens", gen}, {"seconds", s}, {"tokens_per_second", gen / s}, {"steps", steps}});
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  try {
    model::RuntimeOptions ro;
    ro.pack = args.pack;
    ro.dense_bf16 = args.dense_bf16;
    ro.fp8_path = args.fp8_path == "a16" ? kernels::Fp8LinearPath::A16
                  : args.fp8_path == "a8" ? kernels::Fp8LinearPath::A8
                                          : kernels::Fp8LinearPath::Auto;
    ro.kv = args.kv == "bf16"   ? kernels::QsaKvFormat::BF16
            : args.kv == "k8v4" ? kernels::QsaKvFormat::K8V4
                                : kernels::QsaKvFormat::K8V8;
    ro.prefill_chunk = args.chunk;
    ro.cpu_workers = args.cpu_workers;
    ro.mtp = args.mtp;
    ro.logit_rows = 256;
    ro.max_context = std::max<int32_t>(8192, args.synthetic_tokens) + 64;
    ro.prefill_stream = args.prefill_stream;
    ro.prefill_borrow = args.prefill_borrow;
    ro.adaptive_experts = args.adaptive;
    ro.ablation = args.ablation;
    if (args.expert_vram_mib) ro.expert_vram_cap = *args.expert_vram_mib << 20;
    model::ModelRuntime rt(ro);
    std::unique_ptr<model::StepRunner> runner;
    if (args.decode_width > 0)
      runner = std::make_unique<model::StepRunner>(rt, /*use_graphs=*/true, /*mtp=*/args.mtp);

    const bool bf16_mode = args.dense_bf16 && args.kv_bf16;
    const double top1_min = bf16_mode ? 0.985 : 0.96;
    const double kld_max = bf16_mode ? 0.01 : 0.03;
    bool ok = true;
    nlohmann::json out = nlohmann::json::object();
    out["mode"] = {{"dense", args.dense_bf16 ? "bf16" : "fp8"}, {"kv", args.kv},
                   {"ref_dtype", args.ref_dtype}, {"prefill_stream", args.prefill_stream},
                   {"prefill_borrow", args.prefill_borrow}, {"decode_width", args.decode_width},
                   {"ablation", args.ablation && rt.pack().has_ablation()}};
    std::vector<uint64_t> decode_counts;
    if (!args.dump_decode_counts.empty()) decode_counts.assign(std::size_t(dims::kLayers) * dims::kExperts, 0);
    for (const std::string& name : args.golden) {
      if (args.generate > 0) {
        generate_timing(rt, *runner, args, name, out, decode_counts.empty() ? nullptr : &decode_counts);
        continue;
      }
      if (args.timing && runner) {
        decode_timing(rt, *runner, args, name, out);
        continue;
      }
      if (args.timing) {
        const SafeTensors g(args.ref + "/" + name + ".safetensors");
        const StTensor& ids_t = g.at("ids");
        std::vector<int32_t> ids(static_cast<std::size_t>(ids_t.numel()));
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = validate::st_int(ids_t, int64_t(i));
        std::string label = name;
        if (args.synthetic_tokens > 0) {  // the golden's ids repeated to N tokens
          const std::size_t n0 = ids.size();
          ids.resize(std::size_t(args.synthetic_tokens));
          for (std::size_t i = n0; i < ids.size(); ++i) ids[i] = ids[i % n0];
          label = "synthetic(" + name + ")";
        }
        for (int rep = 0; rep < 2; ++rep) {
          rt.reset_sequence();
          rt.pool().reset_timings();
          const auto t0 = std::chrono::steady_clock::now();
          rt.prefill(ids);
          const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          uint64_t wait = 0, comp = 0, pairs = 0, experts = 0;
          for (const auto& lt : rt.pool().timings()) {
            wait += lt.wait_ns; comp += lt.compute_ns; pairs += lt.pairs; experts += lt.experts;
          }
          std::printf("%s timing rep %d: %zu tokens in %.3f s = %.0f tok/s; CPU pool: compute %.3f s, wait %.3f s, %llu cold pairs, %llu cold expert-loads; streamed %d experts (%.2f GiB) per full chunk\n",
                      label.c_str(), rep, ids.size(), s, double(ids.size()) / s, comp * 1e-9, wait * 1e-9,
                      (unsigned long long)pairs, (unsigned long long)experts,
                      rt.prefill_stream().streamed_experts(),
                      double(rt.prefill_stream().streamed_bytes()) / double(1ull << 30));
          out["timing"][label].push_back({{"repetition", rep}, {"tokens", ids.size()},
              {"seconds", s}, {"tokens_per_second", double(ids.size()) / s},
              {"cpu_compute_seconds", comp * 1e-9}, {"cpu_wait_seconds", wait * 1e-9},
              {"cold_pairs", pairs}, {"cold_expert_loads", experts},
              {"streamed_experts", rt.prefill_stream().streamed_experts()}});
        }
        if (args.synthetic_tokens > 0) break;
        continue;
      }
      if (args.repeat_diff) {
        repeat_diff(rt, args, name);
        continue;
      }
      const Result r = run_golden(rt, runner.get(), args, name);
      const bool pass_top1 = r.top1() >= top1_min;
      const bool pass_kld = r.mean_kld() <= kld_max;
      std::printf("%s: %lld positions in %.1f s  top-1 %.4f (%s >= %.3f)  mean KLD %.5f over %lld rows (%s <= %.3f)  max KLD %.4f\n",
                  name.c_str(), (long long)r.positions, r.seconds, r.top1(), pass_top1 ? "PASS" : "FAIL",
                  top1_min, r.mean_kld(), (long long)r.kld_rows, pass_kld ? "PASS" : "FAIL", kld_max, r.kld_max);
      ok = ok && pass_top1 && pass_kld;
      const double pair_kld = r.pair_kld_rows ? r.pair_kld_sum / double(r.pair_kld_rows) : 0.0;
      const bool pass_pair = r.pair_kld_rows > 0 && pair_kld <= 1e-3;
      if (!args.compare_logits.empty()) {
        std::printf("  vs %s: top-1 agreement %.5f over %lld positions, mean KLD(saved || ours) %.3e over %lld rows (%s <= 1e-3), max %.3e\n",
                    args.compare_logits.c_str(),
                    r.pair_positions ? double(r.pair_top1) / double(r.pair_positions) : 0.0,
                    (long long)r.pair_positions, pair_kld, (long long)r.pair_kld_rows,
                    pass_pair ? "PASS" : "FAIL", r.pair_kld_max);
        ok = ok && pass_pair;
      }
      std::printf("  windows (256 pos): ");
      for (const auto& w : r.windows)
        std::printf("[top1 %.3f kld %.4f] ", w[0] > 0 ? w[1] / w[0] : 0.0, w[2] > 0 ? w[3] / w[2] : 0.0);
      std::printf("\n");
      nlohmann::json j = {{"positions", r.positions}, {"top1", r.top1()}, {"mean_kld", r.mean_kld()},
                          {"max_kld", r.kld_max}, {"kld_rows", r.kld_rows}, {"seconds", r.seconds}};
      if (!args.compare_logits.empty())
        j["comparison"] = {{"baseline", args.compare_logits},
            {"top1", double(r.pair_top1) / double(r.pair_positions)},
            {"mean_kld", pair_kld}, {"max_kld", r.pair_kld_max},
            {"kld_rows", r.pair_kld_rows}, {"pass", pass_pair}};
      if (!r.layer_cos.empty()) {
        double worst = 1;
        for (std::size_t i = 0; i < r.layer_cos.size(); ++i) {
          std::printf("  stream %2zu (%s): cosine %.6f  min/pos %.6f\n", i,
                      i == 0 ? "embedding" : ("after L" + std::to_string(i - 1)).c_str(), r.layer_cos[i],
                      r.layer_cos_min[i]);
          worst = std::min(worst, r.layer_cos[i]);
        }
        const bool pass_cos = !bf16_mode || worst >= 0.999;
        std::printf("  worst layer cosine %.6f (%s)\n", worst, bf16_mode ? (pass_cos ? "PASS >= 0.999" : "FAIL < 0.999") : "info");
        ok = ok && pass_cos;
        j["layer_cosine"] = r.layer_cos;
        j["layer_cosine_min"] = r.layer_cos_min;
      }
      out[name] = j;
    }
    if (!args.dump_decode_counts.empty()) {
      std::ofstream file(args.dump_decode_counts, std::ios::binary);
      file.write(reinterpret_cast<const char*>(decode_counts.data()),
                 std::streamsize(decode_counts.size() * sizeof(uint64_t)));
      if (!file) throw std::runtime_error("cannot write " + args.dump_decode_counts);
      uint64_t total = 0;
      for (uint64_t c : decode_counts) total += c;
      std::printf("decode routing counts: %llu routed pairs written to %s\n", (unsigned long long)total,
                  args.dump_decode_counts.c_str());
    }
    const bool diagnostic = args.timing || args.repeat_diff || args.generate > 0;
    out["status"] = diagnostic ? "diagnostic" : (ok ? "pass" : "fail");
    if (!args.json_out.empty()) {
      std::ofstream file(args.json_out);
      file << out.dump(1) << "\n";
      if (!file) throw std::runtime_error("cannot write validation JSON: " + args.json_out);
    }
    std::printf("%s\n", diagnostic ? "DIAGNOSTIC COMPLETE (accuracy not checked)"
                                  : (ok ? "VALIDATION PASS" : "VALIDATION FAIL"));
    return ok ? 0 : 1;
  } catch (const StartupError& e) {
    std::fprintf(stderr, "startup error: %s\n", e.what());
    return e.exit_code();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
