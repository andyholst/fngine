// Engine smoke test on the real pack (FNGINE_TEST_PACK; skipped otherwise).
//
// Greedy outputs depend on rounding that differs between computation paths (CPU or GPU expert
// placement, decode Q8 versus prefill BF16 expert activations), so near-tied argmaxes can flip:
//  1. spec none vs spec auto, 32 greedy tokens of a chat prompt: reported; fails only when they
//     agree on fewer than 8 tokens.
//  2. session reuse: request 2 extends request 1's prompt + output. It must reuse the
//     generation-end snapshot (reused > 0, a cold run reuses 0) and produce the requested tokens.
//     Warm and cold outputs are only reported: the warm state of request 1's output is
//     decode-built (Q8 expert activations), the cold one prefill-built (BF16), and they
//     legitimately diverge at near ties, even at the first token. Exact restore equivalence is
//     checked deterministically by test_speculative_real (check 4).
#include "fngine/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace fngine;

namespace {

constexpr int kSkip = 77;

EngineOptions engine_options(const char* pack, SpeculativeBackend spec) {
    EngineOptions options;
    options.pack_path           = pack;
    options.max_context         = 8192;
    options.prefill_chunk       = 4096;
    options.speculative.backend = spec;
    return options;
}

RequestOptions greedy(std::uint32_t tokens, bool allow_reuse = true) {
    RequestOptions request;
    request.execution.requested_output_tokens    = tokens;
    request.execution.sampling.temperature       = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.allow_prefix_reuse         = allow_reuse;
    request.stop.include_model_defaults          = false;
    return request;
}

PromptInput chat(const std::string& text) {
    PromptInput input;
    input.messages.push_back(ChatMessage{.role = ChatRole::User, .content = text});
    return input;
}

void print_ids(const char* label, const std::vector<TokenId>& ids) {
    std::cout << label << " (" << ids.size() << "):";
    for (const TokenId id : ids) std::cout << ' ' << id;
    std::cout << '\n';
}

std::size_t agreement(const std::vector<TokenId>& a, const std::vector<TokenId>& b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    return i;
}

} // namespace

int main() {
    const char* pack = std::getenv("FNGINE_TEST_PACK");
    if (pack == nullptr || *pack == '\0') {
        std::printf("skip: FNGINE_TEST_PACK is not set\n");
        return kSkip;
    }
    int failures = 0;
    setenv("FNGINE_NO_ROUTING_STATS", "1", 1);  // do not change the pack's routing-stats.bin
    const std::string prompt_text = "Write a haiku about GPUs";

    std::vector<TokenId> plain;
    {
        Engine engine(engine_options(pack, SpeculativeBackend::None));
        GenerationResult r = engine.generate(engine.prepare(chat(prompt_text)), greedy(32));
        plain              = r.generated_token_ids;
        print_ids("spec none", plain);
        std::cout << "  reasoning: " << r.reasoning << "\n  decode " << r.timings.decode_seconds
                  << " s\n";
        if (plain.size() != 32) {
            std::cerr << "FAIL: expected 32 generated tokens\n";
            ++failures;
        }
    }

    Engine engine(engine_options(pack, SpeculativeBackend::Auto));
    {
        GenerationResult r = engine.generate(engine.prepare(chat(prompt_text)), greedy(32));
        print_ids("spec auto", r.generated_token_ids);
        const std::size_t agree = agreement(plain, r.generated_token_ids);
        std::cout << "  rounds " << r.speculative.rounds << " (mtp " << r.speculative.mtp_rounds
                  << ", lookup " << r.speculative.lookup_rounds << "), accepted "
                  << r.speculative.accepted_tokens << "/" << r.speculative.drafted_tokens
                  << "; agrees with spec none on " << agree << "/32 tokens\n";
        if (agree < 8) {
            std::cerr << "FAIL: spec auto diverges from spec none at token " << agree << '\n';
            ++failures;
        }
    }

    // Session reuse over raw token prompts (short enough to avoid the long-prompt QSA issue).
    const std::vector<TokenId> base = engine.tokenize_text(
        "The quick brown fox jumps over the lazy dog. GPUs execute thousands of threads in "
        "parallel, grouped into warps of thirty-two that share an instruction stream. Memory "
        "bandwidth, not arithmetic, usually limits inference. Continue this paragraph:");
    GenerationResult first = engine.generate(engine.prepare_tokens(base), greedy(32));
    std::vector<TokenId> extended = base;
    extended.insert(extended.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());
    const std::vector<TokenId> tail = engine.tokenize_text("\n\nNow summarize it in one line:");
    extended.insert(extended.end(), tail.begin(), tail.end());

    GenerationResult warm = engine.generate(engine.prepare_tokens(extended), greedy(24));
    GenerationResult cold =
        engine.generate(engine.prepare_tokens(extended), greedy(24, /*allow_reuse=*/false));
    std::cout << "session: prompt1 " << base.size() << " tokens, prompt2 " << extended.size()
              << " tokens, reused " << warm.reused_prompt_tokens << " (cold "
              << cold.reused_prompt_tokens << "), prefill warm " << warm.timings.prefill_seconds
              << " s vs cold " << cold.timings.prefill_seconds << " s\n";
    print_ids("warm", warm.generated_token_ids);
    print_ids("cold", cold.generated_token_ids);
    const std::size_t expected = static_cast<std::size_t>(base.size() + first.generated_token_ids.size() - 1);
    if (warm.reused_prompt_tokens != expected || cold.reused_prompt_tokens != 0) {
        std::cerr << "FAIL: expected the warm request to reuse " << expected
                  << " tokens and the cold request none\n";
        ++failures;
    }
    if (warm.generated_token_ids.size() != 24 || cold.generated_token_ids.size() != 24) {
        std::cerr << "FAIL: expected 24 generated tokens from the warm and cold requests\n";
        ++failures;
    }
    std::cout << "  warm agrees with cold on " << agreement(warm.generated_token_ids, cold.generated_token_ids)
              << " tokens\n";

    const RuntimeStats stats = engine.runtime_stats();
    std::cout << "stats: requests " << stats.requests << ", decode steps " << stats.decode_steps
              << ", gpu hit (routing mass) " << stats.gpu_hit_rate_by_routing_mass
              << ", gpu pairs " << stats.gpu_routed_pairs << "/" << stats.routed_pairs
              << ", cpu wait/layer " << stats.cpu_wait_us_per_layer << " us, ngram hit "
              << stats.ngram_cache_hit_rate << ", mtp a/r " << stats.accepted_tokens_per_round_mtp
              << ", lookup a/r " << stats.accepted_tokens_per_round_lookup << '\n';
    std::cout << (failures == 0 ? "engine tests passed\n" : "engine tests FAILED\n");
    return failures == 0 ? 0 : 1;
}
