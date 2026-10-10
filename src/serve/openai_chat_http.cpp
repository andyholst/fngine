// Modified for fngine from cinference:src/serve/openai_chat_http.cpp @ 383e5db. See NOTICE.
// fngine changes: spdlog request lifecycle instead of JSONL/operational request records.
#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fngine::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

ApiError internal_error(const std::string& message) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = message;
    return error;
}

std::string request_summary(const OpenAIChatRequest& request, int prompt_tokens) {
    return "chat stream=" + std::string(request.stream ? "true" : "false") +
           " messages=" + std::to_string(request.generation.messages.size()) +
           " tools=" + std::to_string(request.generation.tools.size()) +
           " prompt_tokens=" + std::to_string(prompt_tokens) +
           " max_tokens=" + std::to_string(request.generation.max_tokens);
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(parse_json_body(req), limits);
        validate_openai_model(request.model, options_.model_id);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        const fngine::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     observation, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        logger_->info("request {} rejected: {} ({})", req_id, exception.error().message,
                      exception.error().status);
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        logger_->error("request {} rejected: {}", req_id, exception.what());
        write_openai_error(res, internal_error(exception.what()));
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    auto lifecycle = std::make_shared<RequestLifecycle>(
        *this, req_id, request_summary(request, prepared.prompt_tokens));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(exception.error().message);
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(exception.what());
            write_openai_error(res, internal_error(exception.what()));
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(exception.what());
            write_openai_error(res, internal_error(exception.what()));
        }
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress,
             timings_per_token](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure("client disconnected");
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(exception.what());
                        return false;
                    }
                };
                try {
                    render_and_write(transport, [&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure("client disconnected");
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(exception.what());
                    return send_error(internal_error(exception.what()));
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const fngine::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const fngine::PromptProgress& progress) {
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const fngine::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure("client disconnected");
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(exception.what());
                    return send_error(internal_error(exception.what()));
                } catch (const ApiException& exception) {
                    lifecycle->failure(exception.error().message);
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(exception.what());
                    return send_error(internal_error(exception.what()));
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(exception.what());
                    return send_error(internal_error(exception.what()));
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure("client disconnected");
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure("client disconnected");
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(exception.what());
        write_openai_error(res, internal_error(exception.what()));
    }
}

} // namespace fngine::serve
