// Modified for fngine from cinference:src/models/qwen3_5/frontend/chat_template.h @ 383e5db. See NOTICE.
// fngine changes: text-only messages (public fngine::ChatMessage), no media parts, cache markers,
// rewrite checkpoints, message boundaries or final-assistant continuation.
#pragma once

#include "text/jinja.h"

#include <fngine/types.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fngine::text {

inline constexpr std::string_view kCanonicalReasoningCloseSerialization = "\n</think>\n\n";

// Template parameters of the Qwen3.8-Flash-Next chat_template.jinja that the frontend sets.
// Typed options win; chat_template_kwargs_json may set them too but must not conflict.
struct ChatRenderOptions {
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    std::vector<std::string> tool_jsons;
};

struct RenderedChat {
    std::string text;
    // User-supplied regions: never parsed as added/special tokens.
    std::vector<text::ByteSpan> literal_spans;
    bool starts_in_reasoning = false;
};

// Throws RequestError(Cancelled|QueueTimeout) when the control has expired.
void check_preparation_control(const PreparationControl& control, std::string_view stage);

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate
    resolve(std::string_view source, std::string source_name = "chat_template.jinja",
            nlohmann::ordered_json special_tokens = nlohmann::ordered_json::object());
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options         = {},
                                      const PreparationControl& control = {}) const;

private:
    CompiledChatTemplate(text::JinjaTemplate compiled, nlohmann::ordered_json special_tokens)
        : compiled_(std::move(compiled)), special_tokens_(std::move(special_tokens)) {}

    text::JinjaTemplate compiled_;
    nlohmann::ordered_json special_tokens_;
};

} // namespace fngine::text
