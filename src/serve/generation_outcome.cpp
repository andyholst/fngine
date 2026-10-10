// Engine-independent parts of the generation service. request_error_to_api_error and the
// outcome mapping are adapted from cinference:src/serve/generation_service.cpp @ 383e5db
// (Apache-2.0, see NOTICE).
#include "serve/generation_service.h"

#include <algorithm>

namespace fngine::serve {

ApiError request_error_to_api_error(const fngine::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case fngine::RequestErrorKind::InvalidRequest:
        error.status = 400;
        error.code   = "invalid_request";
        break;
    case fngine::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case fngine::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case fngine::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case fngine::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case fngine::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case fngine::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

GenerationOutcome make_generation_outcome(fngine::GenerationResult result,
                                          double prepare_seconds) {
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    GenerationMetrics& metrics = outcome.metrics;
    metrics.prepare_seconds    = prepare_seconds;
    metrics.ttft_seconds =
        prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    metrics.prefill_seconds         = result.timings.prefill_seconds;
    metrics.decode_seconds          = result.timings.decode_seconds;
    metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    metrics.total_seconds =
        prepare_seconds + std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    metrics.reused_prompt_tokens        = result.reused_prompt_tokens;
    metrics.speculative_backend         = result.speculative.backend;
    metrics.speculative_rounds          = result.speculative.rounds;
    metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    metrics.speculative_accepted_per_position = std::move(result.speculative.accepted_per_position);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    return outcome;
}

} // namespace fngine::serve
