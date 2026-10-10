// Modified for fngine from cinference:src/serve/translate.h @ 383e5db. See NOTICE.
// fngine changes: no media acquisition; prefix reuse is always requested (snapshot reuse).
#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "fngine/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

namespace fngine::serve {

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<fngine::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

fngine::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics);

// Public request options (output budget, thinking, stop policy, sampler). The sampler is
// resolved from the request's SamplingParams over the server overrides; --greedy forces exact
// argmax. An omitted seed becomes a fresh random seed per request.
fngine::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics);

} // namespace fngine::serve
