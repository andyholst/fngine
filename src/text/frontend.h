// Text frontend facade for Qwen3.8-Flash-Next: pack tokenizer + chat template + output sessions.
// Structure follows cinference:src/models/qwen3_5/frontend/frontend.h (Apache-2.0, see NOTICE).
#pragma once

#include "text/output_session.h"

#include <fngine/types.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <span>
#include <string_view>
#include <vector>

namespace fngine::text {

class Tokenizer;
struct ToolCallOutputContract;

// "\n\n Considering the limited time ...\n</think>\n\n": forced close for thinking budgets.
inline constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";

struct FrontendOptions {
    // Empty selects <tokenizer_dir>/chat_template.jinja.
    std::filesystem::path chat_template_path;
    std::uint32_t max_context = 131072;
};

// Fully prepared prompt text (engine-independent).
struct PreparedText {
    std::string rendered; // empty for prepare_tokens()
    std::vector<TokenId> token_ids;
    bool starts_in_reasoning = false;
    // Present iff the request declared tools; drives the streaming tool-call decoder.
    std::shared_ptr<const ToolCallOutputContract> tool_call_output;
    bool allow_prefix_reuse = true;
    PromptPreparationStats stats;
};

class Frontend {
public:
    // Loads tokenizer.json, tokenizer_config.json, generation_config.json and
    // chat_template.jinja from `tokenizer_dir` (the pack's tokenizer/ directory).
    static Frontend load(const std::filesystem::path& tokenizer_dir, FrontendOptions options = {});

    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    // Renders the chat template and tokenizes. Throws RequestError(ContextLengthExceeded) when
    // the prompt exceeds max_context, std::invalid_argument for invalid input.
    [[nodiscard]] PreparedText prepare(PromptInput input,
                                       const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    // Rendering only (no tokenization): exact template output.
    [[nodiscard]] std::string render(const PromptInput& input,
                                     const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedText prepare_tokens(std::vector<TokenId> token_ids,
                                              bool allow_prefix_reuse = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> ids,
                                     bool skip_special_tokens = false) const;

    [[nodiscard]] OutputSession
    make_output_session(const PreparedText& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {}) const;

    // Stop ids from generation_config.json eos_token_id ([248046, 248044]).
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    // Resolves request overrides against the mode preset (thinking iff starts_in_reasoning).
    [[nodiscard]] ResolvedSamplingParameters resolve_sampling(const SamplingOverrides& overrides,
                                                              bool thinking,
                                                              std::uint64_t default_seed) const;
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept;
    [[nodiscard]] std::shared_ptr<const Tokenizer> shared_tokenizer() const noexcept;
    [[nodiscard]] std::span<const TokenId> thinking_control_tokens() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;
};

} // namespace fngine::text
