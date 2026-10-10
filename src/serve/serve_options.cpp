// Modified for fngine from cinference:src/serve/serve_options.cpp @ 383e5db. See NOTICE.
// fngine changes: fngine engine flags, defaults port 8002 / model id qwen3.8-flash-next.
#include "serve/serve_options.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace fngine::serve {

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " --pack PATH [options]\n"
           "Serves the OpenAI Chat Completions API (/v1/chat/completions, /v1/models, /health).\n"
           "\nServer:\n"
           "  --host H                  bind address (default 127.0.0.1)\n"
           "  --port N                  port (default 8002)\n"
           "  --api-key KEY             require 'Authorization: Bearer KEY'\n"
           "  --model-id ID             public model id (default qwen3.8-flash-next)\n"
           "  --max-pending N           queued requests besides the active one (default 8)\n"
           "  --pending-timeout-ms N    queue wait limit (default 30000)\n"
           "  --max-request-mib N       request body limit (default 64)\n"
           "  --default-max-tokens N    output limit when the client omits one (default " +
           std::to_string(kDefaultMaxTokens) +
           ")\n"
           "  --default-thinking-budget N  cap model thinking tokens per request\n"
           "  --no-thinking             default enable_thinking=false\n"
           "  --no-preserve-thinking    default preserve_thinking=false\n"
           "  --log-stats-interval-ms N periodic engine stats log (default 5000, 0 = off)\n"
           "  --cors                    permissive CORS headers\n"
           "  --greedy                  force temperature 0 (exact argmax)\n"
           "  --log-level trace|debug|info|warning|error|critical|off\n"
           "\nEngine:\n" +
           engine_flags_usage() + "\nSampling overrides:\n" + sampling_flags_usage();
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const RequireValue value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
            return options;
        }
        if (parse_engine_flag(arg, value, options.engine) ||
            parse_sampling_flag(arg, value, options.sampling_overrides)) {
            continue;
        }
        if (arg == "--host") {
            options.host = value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id = value("--model-id");
            if (options.model_id.empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-pending") {
            options.engine.max_pending_requests =
                static_cast<std::uint32_t>(parse_nonnegative_int(value("--max-pending"),
                                                                 "max-pending"));
        } else if (arg == "--pending-timeout-ms") {
            options.engine.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib = parse_u64(value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > 4096) {
                throw std::invalid_argument("--max-request-mib must be in [1,4096]");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(value("--default-max-tokens"), "default-max-tokens");
            if (options.default_max_tokens == 0) {
                throw std::invalid_argument("--default-max-tokens must be positive");
            }
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--no-preserve-thinking") {
            options.preserve_thinking = false;
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = parse_log_level(value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.engine.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending must be positive");
    }
    if (options.engine.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    validate_engine_options(options.engine);
    return options;
}

} // namespace fngine::serve
