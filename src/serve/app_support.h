// Helpers shared by the fngine applications: logger construction and prompt construction from
// CLI text or an OpenAI-style messages file.
#pragma once

#include "serve/cli_common.h"

#include <fngine/types.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace spdlog {
class logger;
}

namespace fngine::serve {

// Colored stderr logger (not registered globally).
[[nodiscard]] std::shared_ptr<spdlog::logger> make_logger(const std::string& name, LogLevel level);

// One user message.
[[nodiscard]] PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking);

// Messages file: either a JSON array of OpenAI chat messages or an object with "messages" and
// optional "tools", "chat_template_kwargs", "reasoning_effort". Parsed with the same rules as
// /v1/chat/completions. Throws std::invalid_argument on malformed input.
[[nodiscard]] PromptInput prompt_from_messages_file(const std::filesystem::path& path,
                                                    std::optional<bool> enable_thinking);

} // namespace fngine::serve
