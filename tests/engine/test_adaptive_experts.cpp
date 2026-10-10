// Adaptive expert swaps end to end on the real pack (FNGINE_TEST_PACK; skipped otherwise; needs
// ~25 GB VRAM / ~30 GB RAM): two 512-token greedy chat turns with and without swaps.
//
// Backend-specific output correctness across each map transition is covered by test_expert_swap;
// forced-swap teacher-forced quality is covered by fngine-validate. This test exercises the public
// engine across two actual chat turns and reports the on/off routing-mass hit rates.
#include "fngine/engine.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace fngine;

namespace {

constexpr int kSkip = 77;
constexpr std::uint32_t kTokens = 512;

EngineOptions engine_options(const char* pack, bool adaptive) {
    EngineOptions options;
    options.pack_path           = pack;
    options.max_context         = 8192;
    options.prefill_chunk       = 4096;
    options.speculative.backend = SpeculativeBackend::None;
    options.adaptive_experts    = adaptive;
    return options;
}

RequestOptions greedy(std::uint32_t tokens) {
    RequestOptions request;
    request.execution.requested_output_tokens    = tokens;
    request.execution.sampling.temperature       = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.allow_prefix_reuse         = false;
    request.stop.include_model_defaults          = false;
    return request;
}

PromptInput chat(const std::string& text) {
    PromptInput input;
    input.messages.push_back(ChatMessage{.role = ChatRole::User, .content = text});
    return input;
}

PromptInput followup(const std::string& first, const GenerationResult& reply, const std::string& next) {
    PromptInput input = chat(first);
    input.messages.push_back(ChatMessage{.role = ChatRole::Assistant, .content = reply.content,
                                         .reasoning_content = reply.reasoning});
    input.messages.push_back(ChatMessage{.role = ChatRole::User, .content = next});
    return input;
}

double interval_hit(const RuntimeStats& before, const RuntimeStats& after) {
    const double mass = after.decode_routing_mass - before.decode_routing_mass;
    return mass > 0 ? (after.decode_gpu_routing_mass - before.decode_gpu_routing_mass) / mass : 0.0;
}

} // namespace

int main() {
    const char* pack = std::getenv("FNGINE_TEST_PACK");
    if (pack == nullptr || *pack == '\0') {
        std::printf("skip: FNGINE_TEST_PACK is not set\n");
        return kSkip;
    }
    setenv("FNGINE_NO_ROUTING_STATS", "1", 1);  // do not change the pack's routing-stats.bin
    const std::string turn1 = "Write a C function that reverses a singly linked list.";
    const std::string turn2 = "Now write the same function for a doubly linked list, in C.";
    int failures = 0;

    GenerationResult plain_first;
    double plain_hit2 = 0;
    {
        Engine engine(engine_options(pack, false));
        plain_first = engine.generate(engine.prepare(chat(turn1)), greedy(kTokens));
        const RuntimeStats before = engine.runtime_stats();
        const auto plain_second = engine.generate(engine.prepare(followup(turn1, plain_first, turn2)), greedy(kTokens));
        if (plain_first.generated_token_ids.size() != kTokens || plain_second.generated_token_ids.size() != kTokens)
            ++failures;
        plain_hit2 = interval_hit(before, engine.runtime_stats());
        std::printf("ADAPTIVE_BENCH enabled=0 turn1_tok_s=%.4f turn2_tok_s=%.4f turn2_gpu_hit=%.6f\n",
                    (plain_first.generated_token_ids.size() - 1) / plain_first.timings.decode_seconds,
                    (plain_second.generated_token_ids.size() - 1) / plain_second.timings.decode_seconds,
                    plain_hit2);
        if (engine.runtime_stats().expert_swaps != 0) {
            std::printf("FAIL: --no-adaptive-experts performed %llu swaps\n",
                        (unsigned long long)engine.runtime_stats().expert_swaps);
            ++failures;
        }
    }
    std::printf("no swaps: turn-2 GPU hit %.1f%%\n", 100 * plain_hit2);

    {
        Engine engine(engine_options(pack, true));
        const RuntimeStats s0 = engine.runtime_stats();
        GenerationResult r1   = engine.generate(engine.prepare(chat(turn1)), greedy(kTokens));
        const RuntimeStats s1 = engine.runtime_stats();
        GenerationResult r2   = engine.generate(engine.prepare(followup(turn1, r1, turn2)), greedy(kTokens));
        const RuntimeStats s2 = engine.runtime_stats();
        std::printf("swaps: %llu during turn 1, %llu during turn 2; turn-1 GPU hit %.1f%%, turn-2 GPU hit "
                    "%.1f%%\n",
                    (unsigned long long)(s1.expert_swaps - s0.expert_swaps),
                    (unsigned long long)(s2.expert_swaps - s1.expert_swaps), 100 * interval_hit(s0, s1),
                    100 * interval_hit(s1, s2));
        std::printf("ADAPTIVE_BENCH enabled=1 turn1_tok_s=%.4f turn2_tok_s=%.4f turn2_gpu_hit=%.6f\n",
                    (r1.generated_token_ids.size() - 1) / r1.timings.decode_seconds,
                    (r2.generated_token_ids.size() - 1) / r2.timings.decode_seconds,
                    interval_hit(s1, s2));
        std::cout << "--- swapped turn 1 output ---\n" << r1.reasoning << r1.content << "\n---\n";
        if (s1.expert_swaps - s0.expert_swaps == 0) {
            std::printf("FAIL: no swaps during a 512-token generation\n");
            ++failures;
        }
        if (r1.generated_token_ids.size() != kTokens || r2.generated_token_ids.size() != kTokens) {
            std::printf("FAIL: expected %u generated tokens in each turn, got %zu / %zu\n",
                        kTokens, r1.generated_token_ids.size(), r2.generated_token_ids.size());
            ++failures;
        }
        if (s2.decode_routing_mass <= s1.decode_routing_mass ||
            s2.decode_gpu_routing_mass < s1.decode_gpu_routing_mass ||
            !std::isfinite(interval_hit(s1, s2)) || interval_hit(s1, s2) < 0.0 ||
            interval_hit(s1, s2) > 1.0) {
            std::printf("FAIL: invalid second-turn routing-mass accounting\n");
            ++failures;
        }
    }
    std::printf(failures ? "FAIL adaptive experts\n" : "PASS adaptive experts\n");
    return failures ? 1 : 0;
}
