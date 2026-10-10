// Speculation controller: decision boundaries against hand-computed values, EMA updates,
// pinned modes and tie-breaking.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "engine/speculation.h"

using fngine::engine::SpecDecision;
using fngine::engine::SpecSource;
using fngine::engine::SpeculationController;
using fngine::engine::SpeculationMode;

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::printf("FAIL line %d: %s\n", line, what);
    ++g_failures;
  }
}
#define CHECK(cond) check((cond), #cond, __LINE__)

bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

bool is(const SpecDecision& d, SpecSource s, int k) { return d.source == s && d.k == k; }

void flat_latency(SpeculationController& c, double seconds) {
  for (int T = 1; T <= SpeculationController::kMaxWidth; ++T) c.record_step(T, seconds);
}

void test_priors() {
  // No measurements: L(T) = 10 ms * (1 + 0.25 (T-1)); MTP a = .7/.6/.5; lookup p = .8.
  SpeculationController c(SpeculationMode::Auto);
  CHECK(near(c.step_latency(1), 0.010));
  CHECK(near(c.step_latency(4), 0.0175));
  // MTP: k0 100, k1 1.7/.0125 = 136, k2 2.12/.015 = 141.3, k3 2.33/.0175 = 133.1 -> k = 2.
  CHECK(near(c.expected_value(SpecSource::Mtp, 1), 136.0));
  CHECK(near(c.expected_value(SpecSource::Mtp, 3), 2.33 / 0.0175));
  SpeculationController m(SpeculationMode::Mtp);
  const auto dm = m.decide(3, 8, 0.0f);
  CHECK(is(dm, SpecSource::Mtp, 2));
  CHECK(near(dm.value, 2.12 / 0.015));
  // Lookup p = .8: k3 (1 + .8 + .64 + .512) / .0175 = 168.7 beats k4 (3.3616 / .02 = 168.1) and
  // every MTP width -> auto picks lookup k = 3.
  const auto da = c.decide(3, 8, 0.0f);
  CHECK(is(da, SpecSource::Lookup, 3));
  CHECK(near(da.value, 2.952 / 0.0175));
  // No drafts available -> plain decode.
  CHECK(is(c.decide(0, 0, 0.0f), SpecSource::None, 0));
}

void test_extrapolation() {
  // Measured L(1) = 10 ms and L(3) = 14 ms -> slope .2 for the unmeasured widths.
  SpeculationController c(SpeculationMode::Mtp);
  c.record_step(1, 0.010);
  c.record_step(3, 0.014);
  CHECK(near(c.step_latency(2), 0.012));
  CHECK(near(c.step_latency(4), 0.016));
  // MTP: k1 1.7/.012 = 141.7, k2 2.12/.014 = 151.4, k3 2.33/.016 = 145.6 -> k = 2.
  CHECK(is(c.decide(3, 0, 0.0f), SpecSource::Mtp, 2));
  // A slope below the floor is clamped: L(9) = 10 ms * (1 + .05 * 8).
  SpeculationController f(SpeculationMode::Mtp);
  f.record_step(1, 0.010);
  f.record_step(2, 0.010);
  CHECK(near(f.step_latency(9), 0.014));
}

void test_latency_shapes() {
  {  // Flat latency: more drafts always help -> max k per source.
    SpeculationController c(SpeculationMode::Auto);
    flat_latency(c, 0.010);
    const auto d = c.decide(3, 0, 0.0f);
    CHECK(is(d, SpecSource::Mtp, 3));
    CHECK(near(d.value, 233.0));
    // Lookup p = .8 over 8 drafts: 1 + Σ .8^i (i=1..8) = 5.6576 tokens beats MTP's 2.33.
    const auto d2 = c.decide(3, 8, 0.0f);
    CHECK(is(d2, SpecSource::Lookup, 8));
    CHECK(near(d2.value, (1.0 - std::pow(0.8, 9)) / 0.2 / 0.010, 1e-9));
    // Fewer lookup drafts available than the maximum: capped.
    CHECK(is(c.decide(3, 5, 0.0f), SpecSource::Lookup, 5));
  }
  {  // Steep latency L(T) = 10 ms * T: k = 1 gives 1.7/.02 = 85 < 100 -> plain decode.
    SpeculationController c(SpeculationMode::Auto);
    for (int T = 1; T <= 9; ++T) c.record_step(T, 0.010 * T);
    // Repeated decisions must still maximize the measured value: no periodic forced draft.
    for (int round = 0; round < 64; ++round) {
      CHECK(is(c.decide(3, 8, 0.0f), SpecSource::None, 0));
      CHECK(is(c.decide(3, 8, 0.99f), SpecSource::None, 0));  // 1.99/.02 = 99.5 < 100
    }
  }
  {  // Boundary: L(2) = 17 ms -> MTP k1 = 1.7/.017 = 100 == k0 (tie -> T = 1); 16.9 ms -> k = 1.
    SpeculationController c(SpeculationMode::Mtp);
    c.record_step(1, 0.010);
    c.record_step(2, 0.017);
    c.record_step(3, 1.0);
    c.record_step(4, 1.0);
    CHECK(is(c.decide(3, 0, 0.0f), SpecSource::None, 0));
    SpeculationController c2(SpeculationMode::Mtp);
    c2.record_step(1, 0.010);
    c2.record_step(2, 0.0169);
    c2.record_step(3, 1.0);
    c2.record_step(4, 1.0);
    CHECK(is(c2.decide(3, 0, 0.0f), SpecSource::Mtp, 1));
  }
}

void test_ties() {
  {  // Same width, different sources: the higher per-token acceptance wins (MTP a1 = .7).
    SpeculationController c(SpeculationMode::Auto);
    flat_latency(c, 0.010);
    CHECK(is(c.decide(1, 1, 0.75f), SpecSource::Lookup, 1));
    CHECK(is(c.decide(1, 1, 0.65f), SpecSource::Mtp, 1));
  }
  {  // Across widths: lookup p = .5, L1 = L2 = 20 ms, L3 = 1.75/75 s -> k1 = 1.5/.02 = 75 = k2.
    SpeculationController c(SpeculationMode::Lookup);
    c.record_step(1, 0.020);
    c.record_step(2, 0.020);
    c.record_step(3, 1.75 / 75.0);
    for (int T = 4; T <= 9; ++T) c.record_step(T, 10.0);
    const auto d = c.decide(0, 8, 0.5f);
    CHECK(is(d, SpecSource::Lookup, 1));
    CHECK(near(d.value, 75.0));
  }
  {  // All drafts rejected for long: gains of ~1e-19 tokens are ties -> T = 1 at flat latency.
    SpeculationController c(SpeculationMode::Auto);
    flat_latency(c, 0.010);
    for (int i = 0; i < 400; ++i) {
      c.record_mtp_outcome(3, 0);
      c.record_lookup_outcome(2, 8, 0);
    }
    CHECK(c.mtp_acceptance(1) < 1e-15);  // .7 * .9^400
    CHECK(is(c.decide(3, 8, 0.0f, 2), SpecSource::None, 0));
  }
}

void test_pinned() {
  SpeculationController none(SpeculationMode::None);
  flat_latency(none, 0.010);
  CHECK(is(none.decide(3, 8, 0.99f), SpecSource::None, 0));

  SpeculationController mtp(SpeculationMode::Mtp);
  flat_latency(mtp, 0.010);
  CHECK(is(mtp.decide(3, 8, 0.99f), SpecSource::Mtp, 3));  // lookup would win in auto
  CHECK(is(mtp.decide(2, 8, 0.99f), SpecSource::Mtp, 2));
  CHECK(is(mtp.decide(0, 8, 0.99f), SpecSource::None, 0));

  SpeculationController look(SpeculationMode::Lookup);
  flat_latency(look, 0.010);
  CHECK(is(look.decide(3, 0, 0.0f), SpecSource::None, 0));
  CHECK(is(look.decide(3, 4, 0.0f), SpecSource::Lookup, 4));

  SpeculationController capped(SpeculationMode::Auto, 2, 3);  // max_mtp = 2, max_lookup = 3
  flat_latency(capped, 0.010);
  CHECK(is(capped.decide(3, 0, 0.0f), SpecSource::Mtp, 2));
  CHECK(is(capped.decide(0, 8, 0.0f), SpecSource::Lookup, 3));

  CHECK(fngine::engine::parse_speculation_mode("auto") == SpeculationMode::Auto);
  CHECK(fngine::engine::parse_speculation_mode("lookup") == SpeculationMode::Lookup);
  CHECK(!fngine::engine::parse_speculation_mode("tree").has_value());
}

void test_emas() {
  SpeculationController c(SpeculationMode::Auto);
  // MTP: offered 3, accepted 1 -> a1 .7 -> .73, a2 .6 -> .54, a3 unobserved.
  c.record_mtp_outcome(3, 1);
  CHECK(near(c.mtp_acceptance(1), 0.73));
  CHECK(near(c.mtp_acceptance(2), 0.54));
  CHECK(near(c.mtp_acceptance(3), 0.5));
  // Offered 2, all accepted -> a1 .757, a2 .586, a3 not offered.
  c.record_mtp_outcome(2, 2);
  CHECK(near(c.mtp_acceptance(1), 0.757));
  CHECK(near(c.mtp_acceptance(2), 0.586));
  CHECK(near(c.mtp_acceptance(3), 0.5));
  // First draft rejected -> only a1 updates.
  c.record_mtp_outcome(3, 0);
  CHECK(near(c.mtp_acceptance(1), 0.6813));
  CHECK(near(c.mtp_acceptance(2), 0.586));
  // All three accepted -> a3 .55.
  c.record_mtp_outcome(3, 3);
  CHECK(near(c.mtp_acceptance(3), 0.55));

  // Lookup window 4: offered 5, accepted 2 -> .8 -> .82 -> .838 -> .7542; others stay .8.
  c.record_lookup_outcome(4, 5, 2);
  CHECK(near(c.lookup_acceptance(4), 0.7542));
  CHECK(near(c.lookup_acceptance(8), 0.8));
  CHECK(near(c.lookup_acceptance(2), 0.8));
  CHECK(near(c.lookup_acceptance(0), 0.7542));  // unknown window -> lowest EMA

  // The window EMA drives decide() when no PromptLookup probability is given; a given
  // probability overrides it. Defaults L(T) = 10 ms (1 + .25 (T-1)).
  SpeculationController d(SpeculationMode::Lookup);
  for (int i = 0; i < 30; ++i) d.record_lookup_outcome(2, 1, 0);  // window 2 -> .8 * .9^30 ≈ .034
  CHECK(is(d.decide(0, 8, 0.0f, 2), SpecSource::None, 0));         // 1.034/.0125 = 82.7 < 100
  CHECK(is(d.decide(0, 8, 0.0f, 8), SpecSource::Lookup, 3));        // window 8 still .8
  CHECK(is(d.decide(0, 8, 0.2f, 8), SpecSource::None, 0));          // 1.2/.0125 = 96 < 100
  CHECK(is(d.decide(0, 8, std::nanf(""), 8), SpecSource::Lookup, 3));

  // Latency EMA: first measurement initializes, then α = .1.
  SpeculationController l(SpeculationMode::Auto);
  l.record_step(1, 0.020);
  CHECK(near(l.step_latency(1), 0.020));
  l.record_step(1, 0.030);
  CHECK(near(l.step_latency(1), 0.021));
  CHECK(near(l.step_latency(3), 0.021 * 1.5));  // unmeasured width from measured L(1)
  CHECK(l.step_measured(1) && !l.step_measured(3));
  l.record_step(1, -1.0);  // ignored
  CHECK(near(l.step_latency(1), 0.021));
  // First measurement at T = 4 seeds L(1) = .028 / 1.75 = .016; its slope is the prior's.
  SpeculationController l4(SpeculationMode::Auto);
  l4.record_step(4, 0.028);
  CHECK(near(l4.step_latency(1), 0.016));
  CHECK(near(l4.step_latency(2), 0.020));
  CHECK(near(l4.step_latency(4), 0.028));

  // Stats.
  SpeculationController s(SpeculationMode::Auto);
  s.decide(0, 0, 0.0f);
  s.record_plain_round();
  s.record_mtp_outcome(3, 2);
  s.record_mtp_outcome(3, 3);
  s.record_lookup_outcome(8, 8, 8);
  CHECK(s.stats(SpecSource::None).decisions == 1 && s.stats(SpecSource::None).rounds == 1);
  CHECK(s.stats(SpecSource::Mtp).rounds == 2 && s.stats(SpecSource::Mtp).drafts_accepted == 5);
  CHECK(near(s.stats(SpecSource::Mtp).tokens_per_round(), 3.5));
  CHECK(near(s.stats(SpecSource::Mtp).acceptance(), 5.0 / 6.0));
  CHECK(near(s.stats(SpecSource::Lookup).tokens_per_round(), 9.0));
  CHECK(near(s.tokens_per_round(), (1.0 + 3.0 + 4.0 + 9.0) / 4.0));
}

}  // namespace

int main() {
  test_priors();
  test_latency_shapes();
  test_extrapolation();
  test_ties();
  test_pinned();
  test_emas();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS speculation\n");
  return 0;
}
