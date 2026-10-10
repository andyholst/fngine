#include "engine/speculation.h"

#include <algorithm>
#include <cmath>

#include "text/prompt_lookup.h"

namespace fngine::engine {

static_assert(SpeculationController::kLookupWindows[0] == int(text::PromptLookup::kWindows[0]) &&
                  SpeculationController::kLookupWindows[1] == int(text::PromptLookup::kWindows[1]) &&
                  SpeculationController::kLookupWindows[2] == int(text::PromptLookup::kWindows[2]),
              "lookup windows must match PromptLookup");
static_assert(1 + SpeculationController::kMaxLookupDrafts <= SpeculationController::kMaxWidth);

std::optional<SpeculationMode> parse_speculation_mode(std::string_view s) {
  if (s == "auto") return SpeculationMode::Auto;
  if (s == "mtp") return SpeculationMode::Mtp;
  if (s == "lookup") return SpeculationMode::Lookup;
  if (s == "none") return SpeculationMode::None;
  return std::nullopt;
}

const char* to_string(SpeculationMode m) {
  switch (m) {
    case SpeculationMode::Auto: return "auto";
    case SpeculationMode::Mtp: return "mtp";
    case SpeculationMode::Lookup: return "lookup";
    case SpeculationMode::None: return "none";
  }
  return "?";
}

const char* to_string(SpecSource s) {
  switch (s) {
    case SpecSource::None: return "none";
    case SpecSource::Mtp: return "mtp";
    case SpecSource::Lookup: return "lookup";
  }
  return "?";
}

SpeculationController::SpeculationController(SpeculationMode mode, int max_mtp, int max_lookup)
    : mode_(mode),
      max_mtp_(std::clamp(max_mtp, 0, kMaxMtpDrafts)),
      max_lookup_(std::clamp(max_lookup, 0, kMaxLookupDrafts)) {}

int SpeculationController::window_index(int window) {
  for (size_t i = 0; i < kLookupWindows.size(); ++i)
    if (kLookupWindows[i] == window) return int(i);
  return -1;
}

double SpeculationController::lookup_acceptance(int window) const {
  const int i = window_index(window);
  return i >= 0 ? lookup_accept_[size_t(i)] : *std::min_element(lookup_accept_.begin(), lookup_accept_.end());
}

double SpeculationController::lookup_prob(float token_prob, int window) const {
  if (token_prob > 0.0f && token_prob <= 1.0f) return double(token_prob);  // false for NaN
  return lookup_acceptance(window);
}

// Unmeasured widths extrapolate linearly from L(1): with the slope of the widest measured width
// when there is one (clamped), the prior slope otherwise.
double SpeculationController::step_latency(int T) const {
  T = std::clamp(T, 1, kMaxWidth);
  if (measured_[size_t(T - 1)]) return latency_[size_t(T - 1)];
  const double l1 = measured_[0] ? latency_[0] : (first_l1_ > 0.0 ? first_l1_ : kDefaultStepSeconds);
  double slope = kWidthSlope;
  for (int w = kMaxWidth; w >= 2; --w) {
    if (measured_[size_t(w - 1)]) {
      slope = std::clamp((latency_[size_t(w - 1)] / l1 - 1.0) / double(w - 1), kMinWidthSlope, kMaxWidthSlope);
      break;
    }
  }
  return l1 * (1.0 + slope * double(T - 1));
}

double SpeculationController::expected_value(SpecSource source, int k, double lookup_token_prob) const {
  double tokens = 1.0, run = 1.0;
  if (source == SpecSource::Mtp) {
    k = std::clamp(k, 0, kMaxMtpDrafts);
    for (int i = 0; i < k; ++i) tokens += (run *= mtp_accept_[size_t(i)]);
  } else if (source == SpecSource::Lookup) {
    k = std::clamp(k, 0, kMaxLookupDrafts);
    const double p = lookup_prob(float(lookup_token_prob), 0);
    for (int i = 0; i < k; ++i) tokens += (run *= p);
  } else {
    k = 0;
  }
  return tokens / step_latency(1 + k);
}

SpecDecision SpeculationController::decide(int mtp_drafts_available, int lookup_drafts_available,
                                           float lookup_token_prob, int lookup_window) {
  const bool use_mtp = mode_ == SpeculationMode::Auto || mode_ == SpeculationMode::Mtp;
  const bool use_lookup = mode_ == SpeculationMode::Auto || mode_ == SpeculationMode::Lookup;
  const int kmtp = use_mtp ? std::clamp(mtp_drafts_available, 0, max_mtp_) : 0;
  const int klook = use_lookup ? std::clamp(lookup_drafts_available, 0, max_lookup_) : 0;
  const double p = lookup_prob(lookup_token_prob, lookup_window);

  SpecDecision best{SpecSource::None, 0, 1.0 / step_latency(1)};
  // Walk widths in increasing order; a candidate must beat the best by a relative margin, so
  // ties keep the smaller T (and MTP over lookup at equal T).
  double mtp_tokens = 1.0, mtp_run = 1.0, look_tokens = 1.0, look_run = 1.0;
  for (int k = 1; k <= std::max(kmtp, klook); ++k) {
    const double L = step_latency(1 + k);
    if (k <= kmtp) {
      mtp_tokens += (mtp_run *= mtp_accept_[size_t(k - 1)]);
      const double v = mtp_tokens / L;
      if (v > best.value * (1.0 + 1e-9)) best = {SpecSource::Mtp, k, v};
    }
    if (k <= klook) {
      look_tokens += (look_run *= p);
      const double v = look_tokens / L;
      if (v > best.value * (1.0 + 1e-9)) best = {SpecSource::Lookup, k, v};
    }
  }
  ++stats_[size_t(best.source)].decisions;
  return best;
}

void SpeculationController::record_step(int T, double seconds) {
  if (T < 1 || T > kMaxWidth || !(seconds > 0.0) || !std::isfinite(seconds)) return;
  const size_t i = size_t(T - 1);
  if (first_l1_ <= 0.0) first_l1_ = seconds / (1.0 + kWidthSlope * double(T - 1));
  if (!measured_[i]) {
    latency_[i] = seconds;
    measured_[i] = true;
  } else {
    latency_[i] += kAlpha * (seconds - latency_[i]);
  }
}

namespace {

// Conditional acceptance update for a round that offered `offered` drafts and accepted the first
// `accepted`: position i (1-based) is observed only if 1..i-1 were accepted, so positions
// 1..min(offered, accepted + 1) update — 1 if accepted, 0 for the first rejection. Position i
// updates ema[i - 1], or ema[0] for every position when `shared` (one per-token estimate).
void update_conditional(double* ema, int positions, bool shared, int offered, int accepted) {
  const int observed = std::min(offered, accepted + 1);
  for (int i = 1; i <= observed; ++i) {
    if (!shared && i > positions) break;
    double& a = ema[shared ? 0 : i - 1];
    a += SpeculationController::kAlpha * ((i <= accepted ? 1.0 : 0.0) - a);
  }
}

}  // namespace

void SpeculationController::record_mtp_outcome(int offered, int accepted) {
  if (offered <= 0) return;
  accepted = std::clamp(accepted, 0, offered);
  update_conditional(mtp_accept_.data(), int(mtp_accept_.size()), false, offered, accepted);
  auto& s = stats_[size_t(SpecSource::Mtp)];
  ++s.rounds;
  s.drafts_offered += uint64_t(offered);
  s.drafts_accepted += uint64_t(accepted);
}

void SpeculationController::record_lookup_outcome(int window, int offered, int accepted) {
  if (offered <= 0) return;
  accepted = std::clamp(accepted, 0, offered);
  if (const int w = window_index(window); w >= 0)
    update_conditional(&lookup_accept_[size_t(w)], 1, true, offered, accepted);
  auto& s = stats_[size_t(SpecSource::Lookup)];
  ++s.rounds;
  s.drafts_offered += uint64_t(offered);
  s.drafts_accepted += uint64_t(accepted);
}

void SpeculationController::record_plain_round() { ++stats_[size_t(SpecSource::None)].rounds; }

double SpeculationController::tokens_per_round() const {
  uint64_t rounds = 0, tokens = 0;
  for (const auto& s : stats_) {
    rounds += s.rounds;
    tokens += s.rounds + s.drafts_accepted;
  }
  return rounds ? double(tokens) / double(rounds) : 0.0;
}

}  // namespace fngine::engine
