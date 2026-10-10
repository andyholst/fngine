// Modified for fngine from cinference:src/serve/generation_service.cpp @ 383e5db. See NOTICE.
// fngine changes: no media acquisition or context-cache hints; FIFO of 1 active + max_pending.
#include "serve/generation_service.h"

#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace fngine::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_request_error(const fngine::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) {
        throw_request_error(fngine::RequestError(RequestErrorKind::Cancelled,
                                                 "client disconnected during preparation"));
    }
    if (Clock::now() >= deadline) {
        throw_request_error(fngine::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public fngine::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(fngine::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(fngine::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(fngine::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(fngine::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == fngine::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

GenerationService::GenerationService(ServeOptions options) : options_(std::move(options)) {
    engine_           = std::make_unique<fngine::Engine>(options_.engine);
    request_capacity_ = std::make_shared<RequestCapacity>(
        1U + static_cast<std::size_t>(options_.engine.max_pending_requests));
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(fngine::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.engine.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           fngine::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled) const {
    return prepare_impl(request, consumer_mode, observation, std::move(is_cancelled),
                        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                fngine::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                DeadlinePolicy deadline_policy) const {
    PreparedRequest prepared;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    fngine::RequestOptions request_options  = to_request_options(request, options_, semantics);
    prepared.thinking_budget   = request_options.execution.thinking.budget;
    prepared.reasoning_effort  = semantics.reasoning_effort;
    prepared.preserve_thinking = semantics.preserve_thinking;
    prepared.lifetime          = acquire_request_lifetime(deadline_policy);

    try {
        fngine::PromptInput input = to_prompt_input(request, semantics);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        fngine::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? fngine::OutputConsumerMode::Streaming
                                                  : fngine::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const fngine::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.engine.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        fngine::PromptInput input = to_prompt_input(request, semantics);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const fngine::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    fngine::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = fngine::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }
    fngine::GenerationResult result;
    try {
        result = prepared.generation.wait(output_sink.get(), cancellation);
    } catch (const fngine::RequestError& exception) { throw_request_error(exception); }
    return make_generation_outcome(std::move(result), prepared.prepare_seconds);
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    turn.content.push_back(ContentPart{.text = "hi", .type_raw = "text"});
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared = prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {},
                                            DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);
}

} // namespace fngine::serve
