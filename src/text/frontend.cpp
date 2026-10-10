// Text frontend facade for Qwen3.8-Flash-Next.
// Parts adapted from cinference:src/models/qwen3_5/frontend/frontend.cpp and
// cinference:src/runtime/contract/sampling.cpp @ 383e5db (Apache-2.0). See NOTICE.
#include "text/frontend.h"

#include "text/chat_template.h"
#include "text/tokenizer.h"
#include "text/tool_call_parser.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace fngine::text {
namespace {

using Json  = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kThinkClose = "</think>";
static_assert(kThinkingControl.ends_with(kCanonicalReasoningCloseSerialization));
constexpr std::size_t kMaxResourceBytes = 64ULL << 20;

std::string read_file(const std::filesystem::path& path, std::size_t max_bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::invalid_argument("cannot read " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) > max_bytes) {
        throw std::invalid_argument("file is empty or too large: " + path.string());
    }
    std::string bytes(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::invalid_argument("cannot read " + path.string());
    }
    return bytes;
}

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::string token_content(const Json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_object() && value.contains("content") && value["content"].is_string())
        return value["content"].get<std::string>();
    return {};
}

// Qwen3.8-Flash-Next tokenizer_config.json semantics this frontend implements.
void validate_tokenizer_config(const Json& config) {
    const auto require = [](bool ok, const char* what) {
        if (!ok) {
            throw std::invalid_argument(std::string("tokenizer_config.json: ") + what +
                                        " does not match the Qwen3.8 tokenizer");
        }
    };
    require(!config.value("add_bos_token", true), "add_bos_token");
    require(!config.value("add_prefix_space", true), "add_prefix_space");
    require(config.contains("pad_token") && token_content(config["pad_token"]) == "<|endoftext|>",
            "pad_token");
    require(config.contains("eos_token") && token_content(config["eos_token"]) == "<|im_end|>",
            "eos_token");
    require(!config.contains("bos_token") || config["bos_token"].is_null(), "bos_token");
    require(!config.value("split_special_tokens", false), "split_special_tokens");
    require(!config.value("clean_up_tokenization_spaces", false), "clean_up_tokenization_spaces");
    if (config.contains("tokenizer_class")) {
        require(config["tokenizer_class"] == "Qwen2Tokenizer" ||
                    config["tokenizer_class"] == "Qwen2TokenizerFast",
                "tokenizer_class");
    }
}

nlohmann::ordered_json special_token_variables(const nlohmann::ordered_json& config) {
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                            "cls_token", "mask_token"}) {
        if (!config.contains(key) || config[key].is_null()) continue;
        const auto& value = config[key];
        if (value.is_string())
            tokens[key] = value;
        else if (value.is_object() && value.contains("content"))
            tokens[key] = value.at("content");
    }
    if (config.contains("additional_special_tokens"))
        tokens["additional_special_tokens"] = config["additional_special_tokens"];
    return tokens;
}

ChatRenderOptions render_options(const PromptOptions& options) {
    return ChatRenderOptions{
        .add_generation_prompt     = true,
        .enable_thinking           = options.enable_thinking,
        .reasoning_effort          = options.reasoning_effort,
        .preserve_thinking         = options.preserve_thinking,
        .chat_template_kwargs_json = options.chat_template_kwargs_json,
        .tool_jsons                = options.tool_jsons,
    };
}

[[noreturn]] void throw_context_length_exceeded(std::size_t tokens, std::uint32_t max_context) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prompt of " + std::to_string(tokens) +
                           " tokens exceeds max_context " + std::to_string(max_context));
}

StopPolicy merge_stop_policy(const Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

void validate(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F)
        throw std::invalid_argument("temperature must be in [0,2]");
    if (sampling.top_k < 1 || sampling.top_k > 20)
        throw std::invalid_argument("resolved top_k must be in [1,20]");
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F)
        throw std::invalid_argument("top_p must be in [0,1]");
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F)
        throw std::invalid_argument("min_p must be in [0,1]");
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F)
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F)
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
}

} // namespace

class Frontend::Impl {
public:
    Impl(const std::filesystem::path& dir, FrontendOptions options)
        : tokenizer_json(read_file(dir / "tokenizer.json", kMaxResourceBytes)),
          tokenizer_config_json(read_file(dir / "tokenizer_config.json", kMaxResourceBytes)),
          generation_config_json(read_file(dir / "generation_config.json", kMaxResourceBytes)),
          max_context(options.max_context) {
        if (max_context == 0) throw std::invalid_argument("frontend max_context must be nonzero");
        validate_tokenizer_config(
            parse_resource_json(tokenizer_config_json, "tokenizer_config.json"));
        (void)parse_resource_json(generation_config_json, "generation_config.json");
        tokenizer = std::make_shared<const Tokenizer>(TokenizerResources{
            tokenizer_json, tokenizer_config_json, generation_config_json});

        const auto template_path = options.chat_template_path.empty()
                                       ? dir / "chat_template.jinja"
                                       : options.chat_template_path;
        chat_template = std::make_unique<CompiledChatTemplate>(CompiledChatTemplate::resolve(
            read_file(template_path, 16ULL << 20), template_path.string(),
            special_token_variables(nlohmann::ordered_json::parse(tokenizer_config_json))));

        if (tokenizer->default_stop_token_ids().empty()) {
            throw std::invalid_argument("generation_config.json has no eos_token_id");
        }
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        std::vector<TokenId> encoded = tokenizer->encode(kThinkingControl);
        const std::string exact =
            tokenizer->decode(encoded, DecodeOptions{.skip_special_tokens = false});
        const std::string presented =
            tokenizer->decode(encoded, DecodeOptions{.skip_special_tokens = true});
        if (encoded.empty() || exact != kThinkingControl ||
            presented.find(kThinkClose) == std::string::npos) {
            throw std::invalid_argument(
                "tokenizer cannot present the canonical thinking control suffix");
        }
        for (const TokenId token : encoded) {
            if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
                defaults.token_ids.end()) {
                throw std::invalid_argument(
                    "canonical thinking control suffix contains a default terminal token");
            }
        }
        thinking_control = std::make_shared<const std::vector<TokenId>>(std::move(encoded));
    }

    RenderedChat render(const PromptInput& input, const PreparationControl& control) const {
        return chat_template->render(input.messages, render_options(input.options), control);
    }

    std::vector<TokenId> encode(const RenderedChat& rendered, std::size_t max_tokens) const {
        BoundaryEncodedText encoded = tokenizer->encode_with_boundaries(
            rendered.text, {}, EncodeOptions{.max_tokens = max_tokens}, rendered.literal_spans);
        return std::vector<TokenId>(encoded.input_ids.begin(), encoded.input_ids.end());
    }

    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
    std::shared_ptr<const Tokenizer> tokenizer;
    std::unique_ptr<CompiledChatTemplate> chat_template;
    StopPolicy defaults;
    ModelSamplingDefaults sampling = kQwen38SamplingDefaults;
    std::shared_ptr<const std::vector<TokenId>> thinking_control;
    std::uint32_t max_context = 0;
};

Frontend Frontend::load(const std::filesystem::path& tokenizer_dir, FrontendOptions options) {
    return Frontend(std::make_shared<const Impl>(tokenizer_dir, std::move(options)));
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

PreparedText Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    check_preparation_control(control, "prompt preparation");
    const auto start = Clock::now();
    PreparedText result;
    result.tool_call_output = build_tool_call_output_contract(input.options.tool_jsons,
                                                              !input.options.tool_jsons.empty());
    RenderedChat rendered      = impl_->render(input, control);
    const auto rendered_at     = Clock::now();
    result.starts_in_reasoning = rendered.starts_in_reasoning;
    result.token_ids = impl_->encode(rendered, static_cast<std::size_t>(impl_->max_context) + 1U);
    check_preparation_control(control, "tokenization");
    if (result.token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(result.token_ids.size(), impl_->max_context);
    }
    result.rendered = std::move(rendered.text);
    const auto end  = Clock::now();
    result.stats.render_seconds   = std::chrono::duration<double>(rendered_at - start).count();
    result.stats.tokenize_seconds = std::chrono::duration<double>(end - rendered_at).count();
    result.stats.seconds          = std::chrono::duration<double>(end - start).count();
    return result;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    check_preparation_control(control, "prompt preparation");
    const RenderedChat rendered = impl_->render(input, control);
    const auto count            = impl_->encode(rendered, SIZE_MAX).size();
    check_preparation_control(control, "tokenization");
    return static_cast<std::uint32_t>(
        std::min<std::size_t>(count, std::numeric_limits<std::uint32_t>::max()));
}

std::string Frontend::render(const PromptInput& input, const PreparationControl& control) const {
    return impl_->render(input, control).text;
}

PreparedText Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_reuse) const {
    const auto start = Clock::now();
    if (token_ids.empty()) throw std::invalid_argument("prompt must contain at least one token");
    if (token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(token_ids.size(), impl_->max_context);
    }
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the vocabulary: " +
                                    std::to_string(token));
        }
    }
    PreparedText result;
    result.token_ids          = std::move(token_ids);
    result.allow_prefix_reuse = allow_prefix_reuse;
    result.stats.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const auto ids = impl_->tokenizer->encode(text);
    return std::vector<TokenId>(ids.begin(), ids.end());
}

std::string Frontend::decode(std::span<const TokenId> ids, bool skip_special_tokens) const {
    return impl_->tokenizer->decode(ids, DecodeOptions{.skip_special_tokens = skip_special_tokens});
}

OutputSession Frontend::make_output_session(const PreparedText& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking) const {
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(impl_->tokenizer, std::move(policy), output, prompt.starts_in_reasoning,
                         thinking, impl_->thinking_control, prompt.tool_call_output);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

ResolvedSamplingParameters Frontend::resolve_sampling(const SamplingOverrides& overrides,
                                                      bool thinking,
                                                      std::uint64_t default_seed) const {
    const SamplingPreset& preset =
        impl_->sampling.for_mode(thinking ? SamplingMode::Thinking : SamplingMode::NonThinking);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .seed              = overrides.seed.value_or(default_seed),
    };
    // The sampling kernels have an exact top-20 candidate domain; zero means "the cap".
    if (resolved.top_k == 0) { resolved.top_k = 20; }
    validate(resolved);
    return resolved;
}

const Tokenizer& Frontend::tokenizer() const noexcept { return *impl_->tokenizer; }

std::shared_ptr<const Tokenizer> Frontend::shared_tokenizer() const noexcept {
    return impl_->tokenizer;
}

std::span<const TokenId> Frontend::thinking_control_tokens() const noexcept {
    return *impl_->thinking_control;
}

} // namespace fngine::text
