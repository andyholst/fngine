// Modified for fngine from cinference:apps/cli/main.cpp @ 383e5db. See NOTICE.
// fngine changes: fngine Engine options/summaries; StartupError → exit code 2.
#include "options.h"

#include "fngine/engine.h"
#include "serve/app_support.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

std::string format_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds << " s";
    return out.str();
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << tokens / seconds << " tok/s";
    return out.str();
}

std::string format_percent(double fraction) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << 100.0 * fraction << " %";
    return out.str();
}

std::string format_gib(std::uint64_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1ULL << 30)
        << " GiB";
    return out.str();
}

std::string format_sampling(const fngine::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0F) { return "greedy (temperature 0)"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
           << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k
           << " min_p=" << sampling.min_p << " presence=" << sampling.presence_penalty
           << " freq=" << sampling.frequency_penalty << " seed=" << sampling.seed;
    return output.str();
}

std::string format_finish(fngine::FinishReason reason) {
    switch (reason) {
    case fngine::FinishReason::None:
        return "none";
    case fngine::FinishReason::OutputLimit:
        return "output-limit";
    case fngine::FinishReason::ContextCapacity:
        return "context-capacity";
    case fngine::FinishReason::StopToken:
        return "stop-token";
    case fngine::FinishReason::StopString:
        return "stop-string";
    case fngine::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(28) << label << value << '\n';
}

class StreamingSink final : public fngine::OutputSink {
public:
    void start(fngine::GenerationStart) override {}

    void progress(fngine::PromptProgress) override {}

    void timing(fngine::GenerationTimingObservation) override {}

    void publish(fngine::OutputDelta delta) override {
        std::ostream& output =
            delta.channel == fngine::OutputChannel::Reasoning ? std::cerr : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == fngine::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_in_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_in_newline_ = delta.text.back() == '\n'; }
        }
    }

    void finish_streams(bool successful = true) {
        if (finished_) { return; }
        finished_ = true;
        if ((successful && !content_seen_) || (content_seen_ && !content_ends_in_newline_)) {
            std::cout << '\n';
        }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_in_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_              = false;
    bool content_ends_in_newline_   = false;
    bool reasoning_seen_            = false;
    bool reasoning_ends_in_newline_ = false;
    bool finished_                  = false;
};

void print_summary(const fngine::GenerationResult& result,
                   const fngine::ResolvedSamplingParameters& sampling,
                   const fngine::MemorySummary& memory, const fngine::RuntimeStats& stats) {
    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const auto prefilled = result.prompt.prompt_tokens - result.reused_prompt_tokens;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    print_metric("reasoning tokens", std::to_string(result.reasoning_tokens));
    if (result.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.thinking.configured_budget));
        print_metric("thinking control", result.thinking.applied ? "applied" : "not applied");
    }
    print_metric("prepare", format_seconds(result.timings.prepare_seconds));
    print_metric("prefill", format_seconds(result.timings.prefill_seconds));
    print_metric("decode", format_seconds(result.timings.decode_seconds));
    print_metric("total", format_seconds(result.timings.total_seconds));
    print_metric("prefill speed",
                 format_rate(static_cast<double>(prefilled), result.timings.prefill_seconds));
    print_metric("decode speed",
                 format_rate(static_cast<double>(decoded), result.timings.decode_seconds));

    const fngine::SpeculativeStats& spec = result.speculative;
    print_metric("speculation", std::string(fngine::speculative_backend_name(spec.backend)));
    if (spec.rounds != 0) {
        std::ostringstream length;
        length << std::fixed << std::setprecision(2)
               << 1.0 + static_cast<double>(spec.accepted_tokens) /
                            static_cast<double>(spec.rounds)
               << " tok/round";
        print_metric("spec rounds (mtp/lookup)", std::to_string(spec.mtp_rounds) + " / " +
                                                     std::to_string(spec.lookup_rounds));
        print_metric("spec accepted/drafted", std::to_string(spec.accepted_tokens) + " / " +
                                                  std::to_string(spec.drafted_tokens));
        print_metric("spec tokens per round", length.str());
        if (!spec.accepted_per_position.empty()) {
            std::ostringstream positions;
            for (std::size_t i = 0; i < spec.accepted_per_position.size(); ++i) {
                if (i != 0) { positions << ','; }
                positions << spec.accepted_per_position[i];
            }
            print_metric("spec accepted by position", positions.str());
        }
    }

    print_metric("gpu experts", std::to_string(memory.gpu_experts) + " / " +
                                    std::to_string(memory.total_experts));
    print_metric("predicted gpu hit rate", format_percent(memory.predicted_gpu_hit_rate));
    print_metric("gpu hit rate (routing mass)", format_percent(stats.gpu_hit_rate_by_routing_mass));
    {
        std::ostringstream wait;
        wait << std::fixed << std::setprecision(1) << stats.cpu_wait_us_per_layer << " us";
        print_metric("cpu wait per layer", wait.str());
    }
    print_metric("ngram cache hit rate", format_percent(stats.ngram_cache_hit_rate));
    print_metric("dense weights", format_gib(memory.dense_weight_bytes));
    print_metric("state+kv+workspace", format_gib(memory.state_bytes));
    print_metric("expert pool (vram)", format_gib(memory.expert_pool_bytes));
    print_metric("host pinned", format_gib(memory.host_pinned_bytes));
}

} // namespace

int main(int argc, char** argv) {
    fngine::cli::Options cli;
    try {
        cli = fngine::cli::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << fngine::cli::usage_text(argv[0]);
        return 1;
    }
    if (cli.help_requested) {
        std::cout << fngine::cli::usage_text(argv[0]);
        return 0;
    }
    const auto logger = fngine::serve::make_logger("fngine", cli.log_level);

    try {
        fngine::PromptInput input =
            cli.messages_path.empty()
                ? fngine::serve::prompt_from_text(cli.prompt, cli.enable_thinking)
                : fngine::serve::prompt_from_messages_file(cli.messages_path,
                                                           cli.enable_thinking);
        if (cli.reasoning_effort) input.options.reasoning_effort = cli.reasoning_effort;

        fngine::RequestOptions request;
        request.execution.sampling                = cli.sampling;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.thinking.budget         = cli.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        fngine::Engine engine(cli.engine);
        const fngine::LoadSummary load = engine.load_summary();
        logger->info("loaded {} from {} in {:.1f} s", load.model_name, load.pack_path.string(),
                     load.load_seconds);
        engine.reset_memory_peaks();

        fngine::PreparedPrompt prompt = engine.prepare(std::move(input));
        StreamingSink sink;
        fngine::GenerationHandle generation = engine.submit(std::move(prompt), std::move(request),
                                                            fngine::OutputConsumerMode::Streaming);
        const fngine::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        fngine::GenerationResult result;
        try {
            result = generation.wait(&sink);
            sink.finish_streams();
        } catch (...) {
            sink.finish_streams(false);
            throw;
        }
        if (cli.print_token_ids) {
            std::cerr << "generated ids:";
            for (const fngine::TokenId id : result.generated_token_ids) std::cerr << ' ' << id;
            std::cerr << '\n';
        }
        if (cli.stats) {
            print_summary(result, sampling, engine.memory_summary(), engine.runtime_stats());
        }
        return 0;
    } catch (const fngine::StartupError& error) {
        logger->critical("startup failed: {}", error.what());
        return error.exit_code();
    } catch (const std::exception& error) {
        logger->error("{}", error.what());
        return 1;
    }
}
