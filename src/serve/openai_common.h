// Modified for fngine from cinference:src/serve/openai_common.h @ 383e5db. See NOTICE.
// fngine changes: prompt-cache policy and Responses identifiers removed.
#pragma once

// OpenAI wire objects shared by the Chat Completions HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace fngine::serve {

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();

} // namespace fngine::serve
