// Modified for fngine from cinference:src/serve/request.h @ 383e5db. See NOTICE.
// fngine changes: text-only content, no cache boundaries, media sources or continuation mode.
#pragma once

#include <fngine/types.h>

// Internal, wire-format-independent representation of a generation request. The OpenAI chat
// schema maps into this value; translate.cpp then produces the public PromptInput and
// RequestOptions consumed by Engine.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fngine::serve {

// A structured API error mapped onto an error object + HTTP status.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens = 8192;
};

struct ContentPart {
    std::string text;
    std::string type_raw; // original wire "type" string for diagnostics
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ToolChoiceMode {
    Auto,
    None,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
};

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result, verified against the function identified by
    // tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    std::string reasoning_content; // assistant thinking carried across turns
};

// Sampling overrides that have an executable Engine meaning.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

struct GenerationRequest {
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    bool stop_strings_apply_to_reasoning = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

} // namespace fngine::serve
