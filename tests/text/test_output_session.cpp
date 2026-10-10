// OutputSession over the real Qwen3.8 tokenizer: reasoning/content split, stop tokens, stop
// strings, output limit, tool-call extraction, thinking-budget control and multi-token rounds.
#include "text/frontend.h"
#include "text_test_util.h"

#include <iostream>
#include <span>
#include <vector>

using namespace fngine;
namespace tt = fngine::text_test;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << '\n';
    }
}

struct Collected {
    std::string reasoning;
    std::string content;
    text::OutputDecision last;
    std::uint32_t committed = 0;
};

void collect(Collected& out, text::PublishedOutput published) {
    for (const auto& delta : published) {
        (delta.channel == OutputChannel::Reasoning ? out.reasoning : out.content) += delta.text;
    }
}

// Feeds `tokens` in rounds of `round` tokens (as speculative verify would), stopping at the
// first finished decision.
Collected run(text::OutputSession& session, const std::vector<TokenId>& tokens, std::size_t round,
              std::uint32_t budget) {
    Collected out;
    std::size_t pos = 0;
    while (pos < tokens.size()) {
        const std::size_t n = std::min(round, tokens.size() - pos);
        const auto span     = std::span<const TokenId>(tokens).subspan(pos, n);
        out.last = session.preview_model(span, budget - out.committed, FinishReason::OutputLimit);
        out.committed += out.last.accepted_tokens;
        collect(out, session.commit_preview());
        if (out.last.finished()) return out;
        pos += out.last.accepted_tokens;
    }
    return out;
}

PromptInput user(std::string text) {
    PromptInput input;
    input.messages.push_back(ChatMessage{.role = ChatRole::User, .content = std::move(text)});
    return input;
}

} // namespace

int main() {
    const auto dir = tt::tokenizer_dir_or_skip();
    if (dir.empty()) return tt::kSkip;
    const auto fe       = text::Frontend::load(dir);
    const auto thinking = fe.prepare(user("What is 2+2?"));
    expect(thinking.starts_in_reasoning, "thinking prompt starts in reasoning");

    const auto encode = [&](std::string_view s) { return fe.tokenize_text(s); };
    const TokenId im_end = 248046;

    // 1. Reasoning/content split + stop token, at round widths 1 and 4.
    for (const std::size_t round : {std::size_t{1}, std::size_t{4}}) {
        auto tokens = encode("Two plus two.\n</think>\n\nThe answer is 4.");
        const std::size_t model_tokens = tokens.size();
        tokens.push_back(im_end);
        tokens.push_back(42); // never reached
        auto session = fe.make_output_session(thinking, StopPolicy{});
        const auto out = run(session, tokens, round, 1000);
        // The canonical close "\n</think>\n\n": the newline before </think> stays in reasoning
        // (streamed before the marker is seen), the blank line after it is stripped.
        expect(out.reasoning == "Two plus two.\n", "reasoning text (round " +
                                                     std::to_string(round) + "): '" +
                                                     tt::escape(out.reasoning) + "'");
        expect(out.content == "The answer is 4.", "content text: '" + tt::escape(out.content) + "'");
        expect(out.last.finish_reason == FinishReason::StopToken, "finish StopToken");
        expect(out.committed == model_tokens + 1, "stop token is accepted");
        expect(session.reasoning_tokens() > 0, "reasoning tokens counted");
    }

    // 2. Stop string in content.
    {
        const auto prompt = fe.prepare([&] {
            auto in                    = user("hi");
            in.options.enable_thinking = false;
            return in;
        }());
        expect(!prompt.starts_in_reasoning, "non-thinking prompt");
        StopPolicy stop;
        stop.strings.push_back(StopString{.text = "STOP"});
        auto session   = fe.make_output_session(prompt, stop);
        const auto out = run(session, encode("Hello STOP world and more"), 1, 1000);
        expect(out.content == "Hello ", "stop string content: '" + tt::escape(out.content) + "'");
        expect(out.last.finish_reason == FinishReason::StopString, "finish StopString");
        expect(session.matched_stop_string() == std::optional<std::string>("STOP"),
               "matched stop string");
    }

    // 3. Output limit.
    {
        auto session   = fe.make_output_session(thinking, StopPolicy{});
        const auto out = run(session, encode("a b c d e f g h"), 2, 4);
        expect(out.committed == 4 && out.last.finish_reason == FinishReason::OutputLimit,
               "output limit");
    }

    // 4. Tool call extraction.
    {
        auto input = user("Weather in Paris?");
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}})");
        const auto prompt = fe.prepare(input);
        auto tokens       = encode("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
                                         "</parameter>\n<parameter=days>\n3\n</parameter>\n"
                                         "</function>\n</tool_call>");
        tokens.push_back(im_end);
        auto session   = fe.make_output_session(prompt, StopPolicy{});
        const auto out = run(session, tokens, 3, 1000);
        const auto calls = session.take_tool_calls();
        expect(out.content.empty(), "tool call content empty: '" + tt::escape(out.content) + "'");
        expect(calls.size() == 1 && calls[0].name == "get_weather" &&
                   nlohmann::json::parse(calls[0].arguments_json) ==
                       nlohmann::json{{"city", "Paris"}, {"days", 3}},
               "tool call parsed");
    }

    // 5. Thinking budget: control suffix injected after 3 model tokens.
    {
        auto session = fe.make_output_session(thinking, StopPolicy{}, OutputOptions{},
                                              ThinkingControlOptions{.budget = 3});
        const auto tokens = encode("one two three four five");
        expect(session.model_token_budget_remaining(100) == 3, "model budget remaining");
        const auto d = session.preview_model(std::span(tokens).first(3), 100,
                                             FinishReason::OutputLimit);
        Collected out;
        collect(out, session.commit_preview());
        expect(d.accepted_tokens == 3 && !d.finished() &&
                   d.continuation == text::ContinuationAction::ApplyTargetControl,
               "budget requests control");
        const auto control = session.pending_control_tokens();
        expect(std::vector<TokenId>(control.begin(), control.end()) ==
                   std::vector<TokenId>(fe.thinking_control_tokens().begin(),
                                        fe.thinking_control_tokens().end()),
               "pending control tokens");
        const auto c = session.preview_control(control, 97);
        collect(out, session.commit_preview());
        expect(c.accepted_tokens == control.size(), "control accepted");
        auto tail = encode("Answer.");
        tail.push_back(im_end);
        const auto e = session.preview_model(tail, 97 - c.accepted_tokens, FinishReason::OutputLimit);
        collect(out, session.commit_preview());
        expect(e.finish_reason == FinishReason::StopToken, "post-control stop");
        expect(out.content == "Answer.", "post-control content: '" + tt::escape(out.content) + "'");
        expect(out.reasoning.find("Considering the limited time") != std::string::npos,
               "control text in reasoning channel");
        const auto stats = session.thinking_stats();
        expect(stats.applied && stats.model_thinking_tokens == 3 &&
                   stats.injected_tokens == control.size(),
               "thinking stats");
    }

    // 6. Sampling defaults and resolution.
    {
        const auto t = fe.resolve_sampling({}, true, 7);
        const auto n = fe.resolve_sampling({}, false, 7);
        expect(t.temperature == 1.0F && t.top_p == 0.95F && t.top_k == 20 &&
                   t.presence_penalty == 0.0F && t.seed == 7,
               "thinking defaults");
        expect(n.temperature == 0.7F && n.top_p == 0.8F && n.top_k == 20 &&
                   n.presence_penalty == 1.5F,
               "non-thinking defaults");
        bool rejected = false;
        try {
            (void)fe.resolve_sampling(SamplingOverrides{.top_k = 21}, true, 0);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "top_k > 20 rejected");
    }

    std::cout << "failures " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
