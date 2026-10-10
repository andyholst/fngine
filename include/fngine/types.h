// Modified for fngine from cinference:include/ninfer/types.h @ 383e5db. See NOTICE.
// fngine changes: text-only single-request API for Qwen3.8-Flash-Next. Media, context-cache,
// DFlash/verify-tree/proposal-head, concurrency and materialization types removed; fngine
// engine options (pack, CPU expert pool, VRAM reserve, n-gram cache, snapshots) added.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fngine {

using TokenId = std::int32_t;

// QSA KV cache storage per token per layer: K8V8 (E4M3 K and V, 1032 B, default), K8V4 (E4M3 K,
// 4-bit V, 804 B), BF16 (2048 B).
enum class KvCacheStorage : std::uint8_t {
    K8V8,
    K8V4,
    BF16,
};

[[nodiscard]] constexpr std::string_view kv_cache_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::K8V8:
        return "k8v8";
    case KvCacheStorage::K8V4:
        return "k8v4";
    case KvCacheStorage::BF16:
        return "bf16";
    }
    return {};
}

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    Lookup,
    Auto,
};

[[nodiscard]] constexpr std::string_view speculative_backend_name(SpeculativeBackend b) noexcept {
    switch (b) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::Lookup:
        return "lookup";
    case SpeculativeBackend::Auto:
        return "auto";
    }
    return {};
}

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::Auto;
    // Upper bounds per source: MTP 1..3 drafts, prompt lookup 1..8 drafts. Zero = source maximum.
    std::uint32_t max_mtp_drafts    = 0;
    std::uint32_t max_lookup_drafts = 0;
};

struct EngineOptions {
    std::filesystem::path pack_path;
    // Empty selects <pack>/tokenizer/chat_template.jinja.
    std::filesystem::path chat_template_path;
    int device                         = 0;
    std::uint32_t max_context          = 131072;
    std::uint32_t max_pending_requests = 8;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 4096;
    KvCacheStorage kv_cache            = KvCacheStorage::K8V8;
    SpeculativeOptions speculative;
    bool use_cuda_graph           = true;
    std::uint32_t cpu_workers     = 0;  // 0: every worker core of the CPU placement policy
    std::uint32_t vram_reserve_mib = 512;
    std::uint32_t ngram_cache_mib = 1024;
    // Validation mode: BF16 copies of every FP8-row dense matrix.
    bool dense_bf16          = false;
    std::uint32_t snapshots  = 6;
    // In-session adaptive expert swaps between GPU slots and RAM, and elastic expert slots in the
    // unused context (--no-adaptive-experts).
    bool adaptive_experts = true;
    // Prefill: stream the warmest CPU-resident experts to the GPU over PCIe (--no-prefill-stream).
    bool prefill_stream = true;
    // Borrow low-ranked resident expert slots for prompt workspace (--no-borrow disables).
    bool prefill_borrow = true;
    // Apply the pack's ablation section (abliterated packs); --no-ablation runs the projection-free
    // control. Packs without the section are unaffected.
    bool ablation = true;
    // Default seed for requests that do not set SamplingOverrides::seed. Unset = random per
    // request (drawn once at submit and reported in ResolvedSamplingParameters).
    std::optional<std::uint64_t> seed;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Qwen3.8-Flash-Next recommended defaults.
inline constexpr ModelSamplingDefaults kQwen38SamplingDefaults{
    SamplingPreset{1.0F, 20, 0.95F, 0.0F, 0.0F, 0.0F},
    SamplingPreset{0.7F, 20, 0.8F, 0.0F, 1.5F, 0.0F},
};

// std::nullopt means "use the model/mode default"; explicit zero is a real override
// (temperature=0 selects exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the thinking phase remains open.
    // Omitted means unlimited. Injected control tokens consume the total output budget but not
    // this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Bounds only Qwen's emitted function-name grammar.
    std::uint32_t tool_name_max_length = 128;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                            = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation roles. The frontend lowers Developer to the template's system
// role.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

// Full OpenAI-style enum. The Qwen3.8 template accepts only xhigh (default), medium and low;
// None disables thinking; other values are rejected by the frontend.
enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

[[nodiscard]] constexpr std::optional<ReasoningEffort>
parse_reasoning_effort(std::string_view name) noexcept {
    for (auto e : {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Low,
                   ReasoningEffort::Medium, ReasoningEffort::High, ReasoningEffort::XHigh,
                   ReasoningEffort::Max}) {
        if (reasoning_effort_name(e) == name) {
            return e;
        }
    }
    return std::nullopt;
}

struct PromptOptions {
    // Unset typed fields leave template defaults intact (thinking on, xhigh, preserve on).
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of extra template parameters.
    std::string chat_template_kwargs_json;
    // One JSON object per tool (OpenAI {"type":"function","function":{...}} form).
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    InvalidRequest,
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
};

struct PromptPreparationStats {
    double seconds          = 0.0;
    double render_seconds   = 0.0;
    double tokenize_seconds = 0.0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Published once before any OutputDelta, after the snapshot-reuse choice.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested) : requested_(std::move(requested)) {}

    [[nodiscard]] bool requested() const { return requested_ && requested_(); }

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to host-side prompt preparation. Empty values = unbounded.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

struct GenerationTimings {
    double prepare_seconds         = 0.0;
    double first_token_seconds     = 0.0;
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend     = SpeculativeBackend::None;
    std::uint64_t rounds           = 0;
    std::uint64_t mtp_rounds       = 0;
    std::uint64_t lookup_rounds    = 0;
    std::uint64_t drafted_tokens   = 0;
    std::uint64_t accepted_tokens  = 0;
    std::uint64_t mtp_drafted      = 0;
    std::uint64_t mtp_accepted     = 0;
    std::uint64_t lookup_drafted   = 0;
    std::uint64_t lookup_accepted  = 0;
    std::uint64_t plain_steps      = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool applied                        = false;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    // Prompt tokens restored from a session snapshot / reused KV instead of prefilled.
    std::uint32_t reused_prompt_tokens = 0;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
};

struct MemorySummary {
    int device                         = 0;
    std::uint32_t max_context          = 0;
    KvCacheStorage kv_cache            = KvCacheStorage::K8V8;
    std::size_t device_total_bytes     = 0;
    std::size_t device_free_at_start   = 0;
    std::size_t dense_weight_bytes     = 0;
    std::size_t kv_cache_bytes         = 0;
    std::size_t state_bytes            = 0;
    std::size_t workspace_bytes        = 0;
    std::size_t expert_pool_bytes      = 0;
    std::uint32_t gpu_experts          = 0;
    std::uint32_t total_experts        = 0;
    double predicted_gpu_hit_rate      = 0.0;
    std::size_t host_pinned_bytes      = 0;
    std::size_t host_expert_bytes      = 0;
    std::size_t ngram_cache_bytes      = 0;
    std::uint32_t snapshot_slots       = 0;
    std::uint32_t snapshot_occupied    = 0;
};

// Monotonic counters plus derived rates since engine start (consumers diff two snapshots for
// interval values of the counters).
struct RuntimeStats {
    std::uint64_t requests                = 0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t reused_prompt_tokens    = 0; // snapshot/KV-reused prompt tokens
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_steps            = 0; // verify/decode graph launches
    double prefill_seconds                = 0.0;
    double decode_seconds                 = 0.0;
    // Routed (token, expert) pairs and their router-weight mass, total and GPU-resident.
    std::uint64_t routed_pairs     = 0;
    std::uint64_t gpu_routed_pairs = 0;
    // Σ_gpu w / Σ w over decode routing decisions (router weights after renormalization).
    double gpu_hit_rate_by_routing_mass = 0.0;
    // Cumulative decode routing mass (48 per decoded column) and its GPU-computed part; interval
    // hit rates are differences of these.
    double decode_routing_mass     = 0.0;
    double decode_gpu_routing_mass = 0.0;
    // Mean GPU wait for the CPU expert pool per layer per decode step (µs), and per layer.
    double cpu_wait_us_per_layer = 0.0;
    std::vector<double> cpu_wait_us_by_layer; // size 48 once the engine has run a step
    // Speculation: drafted/accepted tokens and rounds per source; accepted per round =
    // accepted / rounds (0 when no rounds).
    std::uint64_t mtp_rounds                 = 0;
    std::uint64_t mtp_drafted_tokens         = 0;
    std::uint64_t mtp_accepted_tokens        = 0;
    std::uint64_t lookup_rounds              = 0;
    std::uint64_t lookup_drafted_tokens      = 0;
    std::uint64_t lookup_accepted_tokens     = 0;
    double accepted_tokens_per_round_mtp     = 0.0;
    double accepted_tokens_per_round_lookup  = 0.0;
    // N-gram raw-row cache.
    std::uint64_t ngram_lookups   = 0;
    std::uint64_t ngram_cache_hits = 0;
    double ngram_cache_hit_rate   = 0.0;
    std::uint64_t snapshot_restores = 0;
    std::uint64_t expert_swaps      = 0; // completed adaptive GPU<->RAM expert swaps
    std::uint32_t waiting_requests  = 0;
    bool active_request             = false;
};

// Engine construction failure caused by machine resources (VRAM, pinned/host RAM, files).
// Applications exit with exit_code() (2 by default) after printing what().
class StartupError final : public std::runtime_error {
public:
    explicit StartupError(std::string message, int exit_code = 2)
        : std::runtime_error(std::move(message)), exit_code_(exit_code) {}

    [[nodiscard]] int exit_code() const noexcept { return exit_code_; }

private:
    int exit_code_;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::filesystem::path pack_path;
    double load_seconds               = 0.0;
    double upload_seconds             = 0.0;
    std::uint64_t pack_bytes_read     = 0;
    std::uint64_t host_to_device_bytes = 0;
};

} // namespace fngine
