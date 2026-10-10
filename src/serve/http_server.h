// Modified for fngine from cinference:src/serve/http_server.h @ 383e5db. See NOTICE.
// fngine changes: OpenAI Chat Completions + models + health only; spdlog request lines replace
// the JSONL request log and operational log.
#pragma once

#include "serve/generation_service.h"
#include "serve/http_transport.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace spdlog {
class logger;
}

namespace fngine::serve {

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);

    // Reserves the configured address before model loading. The service is attached only after
    // its Engine is ready, then listen() enters the blocking accept loop on the bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept {
        return options_.model_id;
    }

private:
    // One request's terminal record: exactly one of done()/failure() is logged.
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, std::uint64_t id, std::string summary);

        void done(const GenerationOutcome& outcome);
        void failure(const std::string& reason);
        void response_failure(const std::string& reason);

    private:
        HttpServer* owner_ = nullptr;
        std::uint64_t id_  = 0;
        std::chrono::steady_clock::time_point started_;
        std::atomic<bool> finished_{false};
    };

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void run_stats_reporter();
    void stop_stats_reporter();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::shared_ptr<spdlog::logger> logger_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
};

} // namespace fngine::serve
