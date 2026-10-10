#pragma once

// Cost-model speculation controller (pure host logic).
//
// Each decode round verifies T = 1 + k tokens in one step: the next token plus k drafts from one
// source — MTP (up to 3 drafts produced by the previous step) or prompt lookup (up to 8 drafts).
// The controller picks the (source, k) maximizing expected committed tokens per second:
//
//   value(source, k) = (1 + Σ_{i=1..k} Π_{j=1..i} a_j) / L(1 + k)
//
// where a_j is the probability that draft j is accepted given drafts 1..j-1 were, and L(T) is the
// step latency at width T. k = 0 is plain decode (source None, T = 1). Exact ties (relative
// 1e-9) go to the smaller T; between sources at equal T, MTP wins.
//
// Estimates:
//  - L(T), T = 1..9: EMA (α = 0.1) of measured step seconds, initialized by the first measurement
//    of that width. Unmeasured widths use L(T) = L1 * (1 + s * (T - 1)), where L1 is the measured
//    L(1) if any, else the first measurement of any width scaled to T = 1 by the prior slope, else
//    10 ms; s is the slope of the widest measured width > 1 relative to L1 (clamped to
//    [0.05, 1]), else the prior 0.25 (a verify token costs far less than a decode step: dense
//    weights are shared and only its experts are added).
//  - MTP: per-position conditional acceptance EMAs a_1..a_3 (α = 0.1) starting at 0.7/0.6/0.5.
//    A round that offered n drafts and accepted m updates a_i for i ≤ min(n, m + 1) only
//    (position i is observed only when 1..i-1 were accepted): 1 for i ≤ m, 0 for i = m + 1.
//  - Prompt lookup: one per-token conditional acceptance EMA per window (8, 4, 2 — the
//    PromptLookup windows), starting at 0.8, updated the same way. A round's per-token
//    probability is PromptLookup's own estimate (exp(log_probability())) when the caller passes
//    one in (0, 1]; otherwise the EMA of the proposal's window, or the lowest window EMA when the
//    window is unknown.

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace fngine::engine {

enum class SpeculationMode { Auto, Mtp, Lookup, None };
enum class SpecSource { None, Mtp, Lookup };

// Parses "auto" | "mtp" | "lookup" | "none".
std::optional<SpeculationMode> parse_speculation_mode(std::string_view s);
const char* to_string(SpeculationMode m);
const char* to_string(SpecSource s);

struct SpecDecision {
  SpecSource source = SpecSource::None;
  int k = 0;              // drafts to verify; step width T = 1 + k
  double value = 0.0;     // expected committed tokens per second of the choice
  int width() const { return 1 + k; }
};

struct SpecSourceStats {
  uint64_t decisions = 0;        // rounds decide() chose this source (None: plain decode)
  uint64_t rounds = 0;           // outcomes recorded
  uint64_t drafts_offered = 0;
  uint64_t drafts_accepted = 0;
  // Committed tokens per recorded round (accepted drafts + the verified token).
  double tokens_per_round() const {
    return rounds ? double(drafts_accepted + rounds) / double(rounds) : 0.0;
  }
  double acceptance() const {
    return drafts_offered ? double(drafts_accepted) / double(drafts_offered) : 0.0;
  }
};

class SpeculationController {
 public:
  static constexpr int kMaxWidth = 9;
  static constexpr int kMaxMtpDrafts = 3;
  static constexpr int kMaxLookupDrafts = 8;
  static constexpr std::array<int, 3> kLookupWindows{8, 4, 2};
  static constexpr double kAlpha = 0.1;
  static constexpr double kDefaultStepSeconds = 0.010;
  static constexpr double kWidthSlope = 0.25;
  static constexpr double kMinWidthSlope = 0.05;
  static constexpr double kMaxWidthSlope = 1.0;

  explicit SpeculationController(SpeculationMode mode, int max_mtp = kMaxMtpDrafts,
                                 int max_lookup = kMaxLookupDrafts);

  // Chooses the source and draft count for the next round. `*_available` are the drafts the
  // sources can supply now; `lookup_token_prob` is PromptLookup's per-token probability
  // (exp(log_probability())), or ≤ 0 / NaN to use the window EMA of `lookup_window` (8, 4, 2;
  // 0 = unknown). Pinned modes only consider their own source (None → always plain decode).
  SpecDecision decide(int mtp_drafts_available, int lookup_drafts_available, float lookup_token_prob,
                      int lookup_window = 0);

  // Measured wall time of one verify/decode step of width T (1..9).
  void record_step(int T, double seconds);
  // Outcome of a round that verified `offered` drafts of which the first `accepted` matched.
  void record_mtp_outcome(int offered, int accepted);
  void record_lookup_outcome(int window, int offered, int accepted);
  // A plain decode round (k = 0) committed one token.
  void record_plain_round();

  // Model state.
  double step_latency(int T) const;       // L(T) as used by decide()
  bool step_measured(int T) const { return measured_[size_t(T - 1)]; }
  double mtp_acceptance(int position) const { return mtp_accept_[size_t(position - 1)]; }  // 1..3
  double lookup_acceptance(int window) const;                                             // 8/4/2
  double expected_value(SpecSource source, int k, double lookup_token_prob = 0.0) const;

  SpeculationMode mode() const { return mode_; }
  const SpecSourceStats& stats(SpecSource s) const { return stats_[size_t(s)]; }
  // Committed tokens per round over all recorded rounds.
  double tokens_per_round() const;

 private:
  static int window_index(int window);
  double lookup_prob(float token_prob, int window) const;

  SpeculationMode mode_;
  int max_mtp_;
  int max_lookup_;
  std::array<double, kMaxWidth> latency_{};
  std::array<bool, kMaxWidth> measured_{};
  double first_l1_ = 0.0;  // first measurement scaled to T = 1 (0 = none yet)
  std::array<double, kMaxMtpDrafts> mtp_accept_{0.7, 0.6, 0.5};
  std::array<double, kLookupWindows.size()> lookup_accept_{0.8, 0.8, 0.8};
  std::array<SpecSourceStats, 3> stats_{};
};

}  // namespace fngine::engine
