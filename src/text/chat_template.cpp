// Modified for fngine from cinference:src/models/qwen3_5/frontend/chat_template.cpp @ 383e5db. See NOTICE.
// fngine changes: text-only rendering; Developer lowered to the template's system role;
// reasoning_effort validated against the Qwen3.8 template set {xhigh, medium, low} (+ none).
#include "text/chat_template.h"

#include "text/prompt_layout.h"

#include <chrono>
#include <stdexcept>

namespace fngine::text {
namespace {

using Json = nlohmann::ordered_json;

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        // The Qwen3.8 template knows only system/user/assistant/tool.
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

bool instruction(ChatRole role) { return role == ChatRole::System || role == ChatRole::Developer; }

void merge_option(Json& context, std::string_view key, const Json& value) {
    const std::string name(key);
    if (context.contains(name) && !context[name].is_null() && context[name] != value) {
        throw std::invalid_argument("conflicting chat template parameter: " + name);
    }
    context[name] = value;
}

Json template_parameters(const ChatRenderOptions& options, const Json& special_tokens) {
    Json context;
    try {
        context = options.chat_template_kwargs_json.empty()
                      ? Json::object()
                      : Json::parse(options.chat_template_kwargs_json);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument(std::string("malformed chat_template_kwargs: ") + ex.what());
    }
    if (!context.is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
    for (const auto& item : context.items()) {
        const auto& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" ||
            key == "continue_final_message" || key == "bos_token" || key == "eos_token" ||
            key == "pad_token" || key == "unk_token" || key == "sep_token" || key == "cls_token" ||
            key == "mask_token" || key == "additional_special_tokens" ||
            special_tokens.contains(key)) {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    for (const char* name : {"enable_thinking", "preserve_thinking", "reasoning_effort"}) {
        if (context.contains(name) && context[name].is_null()) context.erase(name);
    }
    if (options.enable_thinking) merge_option(context, "enable_thinking", *options.enable_thinking);
    if (options.preserve_thinking)
        merge_option(context, "preserve_thinking", *options.preserve_thinking);
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) throw std::invalid_argument("invalid reasoning effort");
        merge_option(context, "reasoning_effort", name);
    }
    for (const char* name : {"enable_thinking", "preserve_thinking"}) {
        if (context.contains(name) && !context[name].is_boolean()) {
            throw std::invalid_argument(std::string(name) + " must be a boolean");
        }
    }
    if (context.contains("reasoning_effort")) {
        const auto& effort = context["reasoning_effort"];
        if (!effort.is_string() ||
            (effort != "none" && effort != "low" && effort != "medium" && effort != "xhigh")) {
            throw std::invalid_argument("unsupported reasoning_effort; supported values are "
                                        "xhigh (default), medium, low, and none");
        }
        const bool thinking = effort != "none";
        if (context.contains("enable_thinking") && context["enable_thinking"] != thinking) {
            throw std::invalid_argument("reasoning_effort conflicts with enable_thinking");
        }
        // 'none' is an explicit disable; the template itself only knows xhigh/medium/low.
        if (!thinking) {
            context["enable_thinking"] = false;
            context.erase("reasoning_effort");
        }
    }
    context.update(special_tokens);
    return context;
}

} // namespace

void check_preparation_control(const PreparationControl& control, std::string_view stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source, std::string source_name,
                                                   Json special_tokens) {
    return CompiledChatTemplate(text::JinjaTemplate(std::string(source), std::move(source_name)),
                                std::move(special_tokens));
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options,
                                          const PreparationControl& control) const {
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    check_preparation_control(control, "chat template");
    Json context                     = template_parameters(options, special_tokens_);
    context["messages"]              = Json::array();
    context["add_generation_prompt"] = options.add_generation_prompt;
    std::vector<text::TemplateInputRegion> regions;
    std::uint32_t next_tag = 1;
    const auto tag         = [&](std::string pointer) {
        regions.push_back({std::move(pointer), next_tag++});
    };
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (instruction(message.role) &&
            (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
             !message.tool_call_id.empty())) {
            throw std::invalid_argument("system and developer messages may contain only text");
        }
        Json value{{"role", role_name(message.role)}};
        const auto pointer = "/messages/" + std::to_string(i);
        value["content"]   = message.content;
        tag(pointer + "/content");
        if (!message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
            tag(pointer + "/reasoning_content");
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                Json arguments = Json::object();
                if (!call.arguments_json.empty()) {
                    try {
                        arguments = Json::parse(call.arguments_json);
                    } catch (const nlohmann::json::exception&) {
                        throw std::invalid_argument("tool call arguments must be valid JSON");
                    }
                }
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function", {{"name", call.name}, {"arguments", std::move(arguments)}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
    }
    if (!options.tool_jsons.empty()) {
        context["tools"] = Json::array();
        for (std::size_t i = 0; i < options.tool_jsons.size(); ++i) {
            try {
                context["tools"].push_back(Json::parse(options.tool_jsons[i]));
            } catch (const nlohmann::json::exception&) {
                throw std::invalid_argument("tool definition must be valid JSON");
            }
            tag("/tools/" + std::to_string(i));
        }
    }
    std::vector<std::string> control_variables;
    for (const auto& item : special_tokens_.items()) control_variables.push_back(item.key());
    text::TemplateRenderOptions execution{
        .checkpoint        = [&] { check_preparation_control(control, "chat template"); },
        .regions           = regions,
        .control_variables = control_variables};
    auto output       = compiled_.render(context, execution);
    const auto layout = inspect_prompt_layout(output);
    RenderedChat result;
    result.starts_in_reasoning = layout.starts_in_reasoning;
    result.text                = std::move(output.text);
    result.literal_spans       = std::move(output.literal_spans);
    return result;
}

} // namespace fngine::text
