// Modified for fngine from cinference:apps/cli/options.h @ 383e5db. See NOTICE.
// fngine changes: engine flags via serve/cli_common (pack, ctx, spec, CPU pool, kv, dense).
#pragma once

#include "fngine/types.h"
#include "serve/cli_common.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fngine::cli {

struct Options {
    bool help_requested = false;

    EngineOptions engine;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new = 512;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool stats           = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the model defaults and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy               = false;
    serve::LogLevel log_level = serve::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace fngine::cli
