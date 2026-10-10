// Modified for fngine from cinference:src/models/qwen3_5/frontend/prompt_layout.h @ 383e5db. See NOTICE.
// fngine changes: media placeholders, execution boundaries and source-region queries removed.
#pragma once

#include "text/jinja.h"

#include <fngine/types.h>

#include <cstddef>
#include <vector>

namespace fngine::text {

struct MessageLayout {
    ChatRole role             = ChatRole::User;
    std::size_t begin         = 0;
    std::size_t content_begin = 0;
    std::size_t content_end   = 0;
    std::size_t end           = 0;
    bool closed               = false;
};

struct PromptLayout {
    std::vector<MessageLayout> messages;
    // The final open assistant prefix opened <think> (template bytes) without closing it.
    bool starts_in_reasoning = false;
};

// ChatML block structure of a rendered prompt. Markers inside literal (user-supplied) spans are
// ignored.
[[nodiscard]] PromptLayout inspect_prompt_layout(const text::TemplateOutput& rendered);

} // namespace fngine::text
