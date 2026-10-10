// Modified for fngine from cinference:src/serve/serve_options.h @ 383e5db. See NOTICE.
// fngine changes: engine flags shared via cli_common (pack, ctx, spec, CPU pool, ...); no media,
// context cache, concurrency, Responses store or request-log options.
#pragma once

#include "serve/cli_common.h"

#include <fngine/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fngine::serve {

// Protocol default when the client omits max_tokens. Engine independently clamps the request to
// its effective context capacity.
inline constexpr int kDefaultMaxTokens               = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes = 64ULL << 20;
inline constexpr std::string_view kDefaultModelId    = "qwen3.8-flash-next";

struct ServeOptions {
    bool help_requested = false;
    EngineOptions engine;
    std::string host = "127.0.0.1";
    int port         = 8002;
    std::string api_key; // empty => no auth
    std::string model_id = std::string(kDefaultModelId);
    std::uint32_t log_stats_interval_ms = 5000; // 0 disables periodic Engine stats logs
    std::size_t max_request_bytes       = kDefaultMaxRequestBytes;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // permissive CORS headers for browser UIs
    // Process-level overrides layered between model/mode defaults and request fields.
    SamplingOverrides sampling_overrides;
    bool greedy        = false; // --greedy: force temperature 0 (exact argmax)
    LogLevel log_level = LogLevel::Info;
    // Process argv for the server-start record; --api-key values are redacted.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string serve_usage_text(const char* argv0);

} // namespace fngine::serve
