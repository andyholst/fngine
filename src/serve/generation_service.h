// Modified for fngine from cinference:src/serve/generation_service.h @ 383e5db. See NOTICE.
// fngine changes: single-engine FIFO (1 active + max_pending queued), no media/context cache.
#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "fngine/engine.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fngine::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    double prepare_seconds         = 0.0;
    double ttft_seconds            = 0.0;
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    // Prompt tokens restored from a session snapshot (OpenAI usage cached_tokens).
    std::uint32_t reused_prompt_tokens = 0;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<fngine::GeneratedToolCall> tool_calls;
    fngine::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    fngine::ThinkingBudgetStats thinking;
    fngine::FinishReason finish_reason = fngine::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    GenerationMetrics metrics;
};

// Builds the protocol outcome from an Engine result (engine-independent).
GenerationOutcome make_generation_outcome(fngine::GenerationResult result, double prepare_seconds);

struct StreamSink {
    std::function<void(const fngine::GenerationStart& start)> on_start;
    std::function<void(const fngine::PromptProgress& progress)> on_progress;
    std::function<void(const fngine::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError request_error_to_api_error(const fngine::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its admission reservation until the HTTP response is released and is consumed
// exactly once by run().
struct PreparedRequest {
    fngine::GenerationHandle generation;
    fngine::ResolvedSamplingParameters sampling;
    double prepare_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<fngine::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options);

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }
    [[nodiscard]] const fngine::EngineOptions& engine_options() const { return engine_->options(); }
    [[nodiscard]] fngine::LoadSummary load_summary() const { return engine_->load_summary(); }
    [[nodiscard]] fngine::MemorySummary memory_summary() const { return engine_->memory_summary(); }
    [[nodiscard]] fngine::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }
    [[nodiscard]] bool is_available() const { return engine_->is_available(); }
    [[nodiscard]] fngine::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    // Admission: at most 1 active + engine max_pending_requests queued; beyond that 429.
    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          fngine::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

private:
    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest prepare_impl(const GenerationRequest& req,
                                               GenerationConsumerMode consumer_mode,
                                               fngine::GenerationObservationOptions observation,
                                               std::function<bool()> is_cancelled,
                                               DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;

    ServeOptions options_;
    std::unique_ptr<fngine::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
};

} // namespace fngine::serve
