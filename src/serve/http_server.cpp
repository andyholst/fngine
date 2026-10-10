// Modified for fngine from cinference:src/serve/http_server.cpp @ 383e5db. See NOTICE.
// fngine changes: Chat Completions/models/health routes only; spdlog request and stats lines.
#include "serve/http_server.h"

#include "serve/openai_common.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace fngine::serve {
namespace {

void write_internal_error(httplib::Response& res, const std::string& message) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = message;
    write_openai_error(res, error);
}

} // namespace

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), logger_(std::move(logger)) {
    // One active request plus the FIFO; one extra worker answers health/models while busy.
    const std::size_t queued_requests = 1U + options_.engine.max_pending_requests;
    const std::size_t worker_count    = queued_requests + 1;
    server_.new_task_queue            = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, std::uint64_t id,
                                               std::string summary)
    : owner_(&owner), id_(id), started_(std::chrono::steady_clock::now()) {
    owner_->logger_->info("request {} start {}", id_, summary);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (finished_.exchange(true)) return;
    const auto& m = outcome.metrics;
    const double decode_rate =
        m.generation_wall_seconds > 0.0 && outcome.completion_tokens > 1
            ? (outcome.completion_tokens - 1) / m.generation_wall_seconds
            : 0.0;
    owner_->logger_->info(
        "request {} done prompt={} reused={} completion={} reasoning={} tool_calls={} "
        "ttft={:.3f}s prefill={:.3f}s decode={:.1f} tok/s spec={}/{} total={:.3f}s",
        id_, outcome.prompt_tokens, m.reused_prompt_tokens, outcome.completion_tokens,
        outcome.reasoning_tokens, outcome.tool_calls.size(), m.ttft_seconds, m.prefill_seconds,
        decode_rate, m.speculative_accepted_tokens, m.speculative_draft_tokens,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count());
}

void HttpServer::RequestLifecycle::failure(const std::string& reason) {
    if (finished_.exchange(true)) return;
    owner_->logger_->warn("request {} failed: {}", id_, reason);
}

void HttpServer::RequestLifecycle::response_failure(const std::string& reason) {
    owner_->logger_->warn("request {} response failed: {}", id_, reason);
}

void HttpServer::run_stats_reporter() {
    using Clock                       = std::chrono::steady_clock;
    fngine::RuntimeStats previous     = service_->runtime_stats();
    Clock::time_point previous_time   = Clock::now();
    const auto interval               = std::chrono::milliseconds(options_.log_stats_interval_ms);
    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) break;
        }
        const fngine::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const double seconds = std::chrono::duration<double>(now - previous_time).count();
        const auto prefill   = current.computed_prefill_tokens - previous.computed_prefill_tokens;
        const auto decode    = current.committed_decode_tokens - previous.committed_decode_tokens;
        if ((prefill != 0 || decode != 0) && seconds > 0.0) {
            logger_->info("stats prefill {:.1f} tok/s decode {:.1f} tok/s gpu-hit {:.1f}% "
                          "cpu-wait {:.1f} us/layer ngram-hit {:.1f}% waiting {}",
                          prefill / seconds, decode / seconds,
                          100.0 * current.gpu_hit_rate_by_routing_mass,
                          current.cpu_wait_us_per_layer, 100.0 * current.ngram_cache_hit_rate,
                          current.waiting_requests);
        }
        previous      = current;
        previous_time = now;
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_.max_request_bytes, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
        // CORS preflight carries no credentials; answer it without auth.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (!matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key)) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            write_openai_error(res, error);
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    logger_->error("{} {}: {}", req.method, req.path, e.error().message);
                }
                write_openai_error(res, e.error());
            } catch (const std::exception& e) {
                logger_->error("{} {}: {}", req.method, req.path, e.what());
                write_internal_error(res, e.what());
            } catch (...) {
                logger_->error("{} {}: unknown error", req.method, req.path);
                write_internal_error(res, "unknown error");
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(
        make_models_list(options_.model_id, unix_time_now(), options_.engine.max_context),
        "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != options_.model_id) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(
        make_model_object(options_.model_id, unix_time_now(), options_.engine.max_context),
        "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    service_ = &service;
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    logger_->info("serving {} on http://{}:{}/v1", options_.model_id, options_.host,
                  options_.port);
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace fngine::serve
