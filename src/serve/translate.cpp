// Modified for fngine from cinference:src/serve/translate.cpp @ 383e5db. See NOTICE.
// fngine changes: text-only message lowering, no cache markers/media/continuation.
#include "serve/translate.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fngine::serve {
namespace {

std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

fngine::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    fngine::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    if (request.min_p) { sampling.min_p = static_cast<float>(*request.min_p); }
    if (request.top_k) { sampling.top_k = static_cast<std::int32_t>(*request.top_k); }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature && (*sampling.temperature < 0.0F || *sampling.temperature > 2.0F)) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p && (*sampling.top_p < 0.0F || *sampling.top_p > 1.0F)) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    if (sampling.top_k && (*sampling.top_k < 0 || *sampling.top_k > 20)) {
        invalid_sampling("top_k must be in [0,20]", "top_k");
    }
    if (sampling.min_p && (*sampling.min_p < 0.0F || *sampling.min_p > 1.0F)) {
        invalid_sampling("min_p must be in [0,1]", "min_p");
    }
    if (sampling.presence_penalty &&
        (*sampling.presence_penalty < -2.0F || *sampling.presence_penalty > 2.0F)) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty &&
        (*sampling.frequency_penalty < -2.0F || *sampling.frequency_penalty > 2.0F)) {
        invalid_sampling("frequency_penalty must be in [-2,2]", "frequency_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

std::vector<const ToolDefinition*> effective_tools(const GenerationRequest& request) {
    std::vector<const ToolDefinition*> tools;
    if (!request.uses_tools()) { return tools; }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(&tool); }
    return tools;
}

std::string render_tool_definition(const ToolDefinition& tool) {
    using Json  = RequestJson;
    Json schema = Json::parse(tool.input_schema_json);
    Json function{{"name", tool.name}, {"parameters", std::move(schema)}, {"strict", false}};
    if (!tool.description.empty()) { function["description"] = tool.description; }
    return Json{{"type", "function"}, {"function", std::move(function)}}.dump();
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server) {
    using Json  = RequestJson;
    Json kwargs = request.chat_template_kwargs_json.empty()
                      ? Json::object()
                      : Json::parse(request.chat_template_kwargs_json);
    if (!kwargs.is_object())
        invalid_prompt_option("chat_template_kwargs must be an object", "chat_template_kwargs",
                              "invalid_template_option");
    auto merge_boolean = [&](const char* key, std::optional<bool> typed) {
        if (!kwargs.contains(key) || kwargs[key].is_null()) {
            kwargs.erase(key);
            return typed;
        }
        if (!kwargs[key].is_boolean())
            invalid_prompt_option(std::string(key) + " must be a boolean", key,
                                  "invalid_template_option");
        const bool nested = kwargs[key].get<bool>();
        if (typed && *typed != nested)
            invalid_prompt_option(std::string("conflicting ") + key + " values", key,
                                  "conflicting_template_option");
        kwargs.erase(key);
        return std::optional<bool>(nested);
    };
    auto thinking = merge_boolean("enable_thinking", request.enable_thinking);
    auto preserve = merge_boolean("preserve_thinking", request.preserve_thinking);
    auto effort   = request.reasoning_effort;
    if (kwargs.contains("reasoning_effort") && !kwargs["reasoning_effort"].is_null()) {
        if (!kwargs["reasoning_effort"].is_string())
            invalid_prompt_option("reasoning_effort must be a string", "reasoning_effort",
                                  "invalid_template_option");
        const auto nested =
            parse_reasoning_effort(kwargs["reasoning_effort"].get<std::string>());
        if (!nested)
            invalid_prompt_option("invalid reasoning_effort", "reasoning_effort",
                                  "invalid_template_option");
        if (effort && effort != nested)
            invalid_prompt_option("conflicting reasoning_effort values", "reasoning_effort",
                                  "conflicting_template_option");
        effort = nested;
    }
    kwargs.erase("reasoning_effort");
    ResolvedPromptSemantics result{
        .enable_thinking           = thinking ? thinking : server.enable_thinking,
        .preserve_thinking         = preserve ? preserve : server.preserve_thinking,
        .chat_template_kwargs_json = kwargs.dump(),
    };
    if (effort) {
        const bool enables = *effort != fngine::ReasoningEffort::None;
        if (thinking && *thinking != enables)
            invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                                  "reasoning_effort", "conflicting_template_option");
        // A request effort overrides the server's thinking default.
        result.enable_thinking  = enables;
        result.reasoning_effort = *effort;
    }
    return result;
}

fngine::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics) {
    fngine::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (std::size_t turn_index = 0; turn_index < request.messages.size(); ++turn_index) {
        const ChatTurn& turn = request.messages[turn_index];
        fngine::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(fngine::ToolCall{call.id, call.name, call.arguments_json});
        }

        // Text parts concatenate in order (OpenAI content arrays carry no separators).
        for (const ContentPart& part : turn.content) message.content += part.text;
        input.messages.push_back(std::move(message));
    }

    input.options.enable_thinking                  = semantics.enable_thinking;
    input.options.reasoning_effort                 = semantics.reasoning_effort;
    input.options.preserve_thinking                = semantics.preserve_thinking;
    input.options.chat_template_kwargs_json        = semantics.chat_template_kwargs_json;
    const std::vector<const ToolDefinition*> tools = effective_tools(request);
    input.options.tool_jsons.reserve(tools.size());
    for (const ToolDefinition* tool : tools) {
        input.options.tool_jsons.push_back(render_tool_definition(*tool));
    }
    return input;
}

fngine::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics) {
    fngine::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = true;
    if (semantics.enable_thinking != false) {
        options.execution.thinking.budget =
            request.thinking_budget ? request.thinking_budget : server.default_thinking_budget;
    }
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.output.tool_name_max_length = static_cast<std::uint32_t>(request.tool_name_max_length);
    options.stop.strings.reserve(request.stop_strings.size() *
                                 (request.stop_strings_apply_to_reasoning ? 2U : 1U));
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                fngine::StopString{.text              = stop,
                                   .channel           = fngine::OutputChannel::Content,
                                   .include_in_output = false});
            if (request.stop_strings_apply_to_reasoning) {
                options.stop.strings.push_back(
                    fngine::StopString{.text              = stop,
                                       .channel           = fngine::OutputChannel::Reasoning,
                                       .include_in_output = false});
            }
        }
    }
    return options;
}

} // namespace fngine::serve
