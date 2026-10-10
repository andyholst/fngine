// Modified for fngine from cinference:apps/cli/options.cpp @ 383e5db. See NOTICE.
// fngine changes: --pack and shared engine/sampling flags, --stats, template reasoning efforts.
#include "options.h"

#include <limits>
#include <stdexcept>
#include <string_view>

namespace fngine::cli {
namespace {

std::uint32_t parse_u32(const char* text, const char* label, bool allow_zero = false) {
    const std::uint64_t value = serve::parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

ReasoningEffort parse_cli_reasoning_effort(std::string_view text) {
    // The Qwen3.8 template accepts xhigh/medium/low; none disables thinking.
    if (text == "xhigh") return ReasoningEffort::XHigh;
    if (text == "medium") return ReasoningEffort::Medium;
    if (text == "low") return ReasoningEffort::Low;
    if (text == "none") return ReasoningEffort::None;
    throw std::invalid_argument("invalid --reasoning-effort (expected xhigh|medium|low|none): " +
                                std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " --pack PATH (--prompt TEXT | --messages FILE) [options]\n"
           "\nRequest:\n"
           "  --prompt TEXT             one user message\n"
           "  --messages FILE           OpenAI-style messages JSON (array, or object with "
           "messages/tools)\n"
           "  --max-new N               output token limit (default 512)\n"
           "  --no-thinking             disable thinking\n"
           "  --reasoning-effort xhigh|medium|low|none\n"
           "  --thinking-budget N       cap model thinking tokens\n"
           "  --stop TEXT               content stop string (repeatable)\n"
           "  --reasoning-stop TEXT     reasoning stop string (repeatable)\n"
           "  --stop-token-id N         extra stop token (repeatable)\n"
           "  --greedy                  temperature 0 (exact argmax)\n"
           "  --raw-output              no reasoning/tool-call parsing; publish special tokens\n"
           "  --print-token-ids         print generated ids after the output\n"
           "  --stats                   print timing, memory and engine statistics\n"
           "  --log-level trace|debug|info|warning|error|critical|off\n"
           "\nEngine:\n" +
           serve::engine_flags_usage() + "\nSampling overrides:\n" + serve::sampling_flags_usage() +
           "\nStreams content to stdout and reasoning plus diagnostics to stderr.\n"
           "Exit codes: 0 ok, 1 error, 2 startup resource error (VRAM/RAM/files).\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const serve::RequireValue value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
            return options;
        }
        if (serve::parse_engine_flag(arg, value, options.engine) ||
            serve::parse_sampling_flag(arg, value, options.sampling)) {
            continue;
        }
        if (arg == "--prompt") {
            options.prompt = value("--prompt");
        } else if (arg == "--messages") {
            options.messages_path = value("--messages");
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value("--max-new"), "max-new");
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--stats") {
            options.stats = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value("--thinking-budget"), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_cli_reasoning_effort(value("--reasoning-effort"));
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value("--stop-token-id"), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg == "--stop" ? "--stop" : "--reasoning-stop");
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = serve::parse_log_level(value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    serve::validate_engine_options(options.engine);
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace fngine::cli
