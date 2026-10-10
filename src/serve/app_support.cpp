#include "serve/app_support.h"

#include "serve/openai_chat.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fngine::serve {
namespace {

spdlog::level::level_enum to_spdlog(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    }
    return spdlog::level::info;
}

} // namespace

std::shared_ptr<spdlog::logger> make_logger(const std::string& name, LogLevel level) {
    auto sink   = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(name, std::move(sink));
    logger->set_level(to_spdlog(level));
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%e %^%l%$ [%n] %v");
    logger->flush_on(spdlog::level::warn);
    return logger;
}

PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking) {
    PromptInput input;
    input.messages.push_back(ChatMessage{.role = ChatRole::User, .content = std::move(text)});
    input.options.enable_thinking = enable_thinking;
    return input;
}

PromptInput prompt_from_messages_file(const std::filesystem::path& path,
                                      std::optional<bool> enable_thinking) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument("cannot read messages file: " + path.string());
    std::ostringstream bytes;
    bytes << file.rdbuf();
    RequestJson document;
    try {
        document = RequestJson::parse(bytes.str());
    } catch (const std::exception& error) {
        throw std::invalid_argument("malformed messages file " + path.string() + ": " +
                                    error.what());
    }
    RequestJson body = document.is_array() ? RequestJson{{"messages", std::move(document)}}
                                           : std::move(document);
    if (!body.is_object()) {
        throw std::invalid_argument("messages file must be a JSON array or object");
    }
    body["model"] = std::string(kDefaultModelId);
    try {
        const OpenAIChatRequest request = parse_chat_completion_request(body, RequestLimits{});
        ServeOptions server;
        server.enable_thinking = enable_thinking;
        return to_prompt_input(request.generation,
                               resolve_prompt_semantics(request.generation, server));
    } catch (const ApiException& error) {
        throw std::invalid_argument("messages file " + path.string() + ": " + error.what());
    }
}

} // namespace fngine::serve
