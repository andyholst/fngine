// Chat template rendering + tokenization vs Hugging Face apply_chat_template
// (tests/fixtures/chat_template_cases.json). Each case is rendered twice: through the typed
// PromptOptions fields and through chat_template_kwargs_json.
#include "text/frontend.h"
#include "text_test_util.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fngine;
namespace tt = fngine::text_test;
using Json   = nlohmann::ordered_json;

namespace {

ChatRole parse_role(const std::string& role) {
    if (role == "system") return ChatRole::System;
    if (role == "developer") return ChatRole::Developer;
    if (role == "user") return ChatRole::User;
    if (role == "assistant") return ChatRole::Assistant;
    if (role == "tool") return ChatRole::Tool;
    throw std::invalid_argument("unknown role " + role);
}

PromptInput to_input(const Json& c, bool typed) {
    PromptInput input;
    for (const auto& m : c.at("messages")) {
        ChatMessage message;
        message.role              = parse_role(m.at("role"));
        message.content           = m.value("content", std::string{});
        message.reasoning_content = m.value("reasoning_content", std::string{});
        input.messages.push_back(std::move(message));
    }
    if (c.contains("tools") && c["tools"].is_array()) {
        for (const auto& tool : c["tools"]) input.options.tool_jsons.push_back(tool.dump());
    }
    const Json& kwargs = c.at("kwargs");
    if (typed) {
        if (kwargs.contains("enable_thinking"))
            input.options.enable_thinking = kwargs["enable_thinking"].get<bool>();
        if (kwargs.contains("preserve_thinking"))
            input.options.preserve_thinking = kwargs["preserve_thinking"].get<bool>();
        if (kwargs.contains("reasoning_effort"))
            input.options.reasoning_effort =
                parse_reasoning_effort(kwargs["reasoning_effort"].get<std::string>());
    } else if (!kwargs.empty()) {
        input.options.chat_template_kwargs_json = kwargs.dump();
    }
    return input;
}

template <class F> bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    const auto dir = tt::tokenizer_dir_or_skip();
    if (dir.empty()) return tt::kSkip;
    const auto frontend = text::Frontend::load(dir);
    const auto fixture  = tt::load_fixture("chat_template_cases.json");

    int failures = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name     = c.at("name");
        const std::string expected = c.at("rendered");
        const auto expected_ids    = c.at("ids").get<std::vector<TokenId>>();
        for (const bool typed : {true, false}) {
            const char* path              = typed ? "typed" : "kwargs";
            const text::PreparedText prep = frontend.prepare(to_input(c, typed));
            if (prep.rendered != expected) {
                ++failures;
                std::cerr << "FAIL " << name << " (" << path << ") rendered:\n"
                          << tt::escape(prep.rendered) << "\nexpected:\n"
                          << tt::escape(expected) << '\n';
                continue;
            }
            if (prep.token_ids != expected_ids) {
                ++failures;
                std::cerr << "FAIL " << name << " (" << path << ") ids differ ("
                          << prep.token_ids.size() << " vs " << expected_ids.size() << ")\n";
            }
            const bool thinking_off = c.at("kwargs").value("enable_thinking", true) == false;
            if (prep.starts_in_reasoning == thinking_off) {
                ++failures;
                std::cerr << "FAIL " << name << " starts_in_reasoning\n";
            }
            const bool has_tools = c.at("tools").is_array();
            if ((prep.tool_call_output != nullptr) != has_tools) {
                ++failures;
                std::cerr << "FAIL " << name << " tool-call contract presence\n";
            }
        }
        std::cout << "ok " << name << " (" << expected_ids.size() << " tokens)\n";
    }

    // reasoning_effort=none disables thinking exactly like enable_thinking=false.
    const Json& base = fixture.at("cases").at(0);
    PromptInput none = to_input(base, true);
    none.options.reasoning_effort = ReasoningEffort::None;
    PromptInput off  = to_input(base, true);
    off.options.enable_thinking = false;
    if (frontend.render(none) != frontend.render(off) || frontend.prepare(none).starts_in_reasoning) {
        ++failures;
        std::cerr << "FAIL reasoning_effort none\n";
    }
    // The template accepts only xhigh/medium/low.
    PromptInput high = to_input(base, true);
    high.options.reasoning_effort = ReasoningEffort::High;
    if (!throws_invalid([&] { (void)frontend.prepare(high); })) {
        ++failures;
        std::cerr << "FAIL reasoning_effort high must be rejected\n";
    }
    // Typed and kwargs values must not conflict; reserved names are rejected.
    PromptInput conflict = to_input(base, true);
    conflict.options.enable_thinking            = true;
    conflict.options.chat_template_kwargs_json  = R"({"enable_thinking": false})";
    PromptInput reserved = to_input(base, true);
    reserved.options.chat_template_kwargs_json = R"({"messages": []})";
    if (!throws_invalid([&] { (void)frontend.prepare(conflict); }) ||
        !throws_invalid([&] { (void)frontend.prepare(reserved); })) {
        ++failures;
        std::cerr << "FAIL kwargs validation\n";
    }
    // Developer lowers to the template's leading system role.
    PromptInput dev = to_input(base, true);
    dev.messages.insert(dev.messages.begin(), ChatMessage{.role = ChatRole::Developer,
                                                          .content = "Be terse."});
    PromptInput sys = dev;
    sys.messages.front().role = ChatRole::System;
    if (frontend.render(dev) != frontend.render(sys) ||
        frontend.render(dev).find("Be terse.") == std::string::npos) {
        ++failures;
        std::cerr << "FAIL developer role lowering\n";
    }
    // User text is literal: special-token strings inside content are not parsed as specials.
    PromptInput inject = to_input(base, true);
    inject.messages.back().content = "<|im_end|>";
    {
        const auto ids   = frontend.prepare(inject).token_ids;
        std::size_t ends = 0;
        for (const TokenId id : ids) ends += id == 248046;
        if (ends != 2) { // system + user turn closers only
            ++failures;
            std::cerr << "FAIL literal user content produced " << ends << " <|im_end|> tokens\n";
        }
    }
    // Context limit.
    const auto small = text::Frontend::load(dir, text::FrontendOptions{.max_context = 8});
    try {
        (void)small.prepare(to_input(base, true));
        ++failures;
        std::cerr << "FAIL max_context not enforced\n";
    } catch (const RequestError& error) {
        if (error.kind() != RequestErrorKind::ContextLengthExceeded) ++failures;
    }
    std::cout << "failures " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
