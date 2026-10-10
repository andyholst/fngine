// Modified for fngine from cinference:apps/serve/main.cpp @ 383e5db. See NOTICE.
// fngine changes: spdlog lines instead of the operational log; StartupError → exit code 2.
#include "serve/app_support.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

std::atomic<fngine::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    fngine::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    fngine::serve::ServeOptions options;
    try {
        options = fngine::serve::parse_serve_options(argc, argv);
    } catch (const std::exception& exception) {
        std::cerr << "fngine-serve: " << exception.what() << '\n';
        std::cerr << fngine::serve::serve_usage_text(argv[0]);
        return 1;
    }
    if (options.help_requested) {
        std::cout << fngine::serve::serve_usage_text(argv[0]);
        return 0;
    }
    const auto logger = fngine::serve::make_logger("fngine-serve", options.log_level);

    try {
        fngine::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            logger->error("cannot bind {}:{}", options.host, options.port);
            return 1;
        }

        fngine::serve::GenerationService service(options);
        const fngine::LoadSummary load       = service.load_summary();
        const fngine::MemorySummary memory   = service.memory_summary();
        logger->info("engine ready: {} ({:.1f} s), {} / {} experts on GPU, predicted hit {:.1f}%",
                     load.model_name, load.load_seconds, memory.gpu_experts,
                     memory.total_experts, 100.0 * memory.predicted_gpu_hit_rate);

        using Clock          = std::chrono::steady_clock;
        const auto warmup_t0 = Clock::now();
        service.warmup();
        logger->info("warmup {:.2f} s",
                     std::chrono::duration<double>(Clock::now() - warmup_t0).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        if (!options.api_key.empty()) logger->info("bearer API key required");

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            logger->error("listen failed on {}:{}", options.host, options.port);
            return 1;
        }
        logger->info("server stopped");
        return 0;
    } catch (const fngine::StartupError& error) {
        g_server.store(nullptr);
        logger->critical("startup failed: {}", error.what());
        return error.exit_code();
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        logger->critical("{}", exception.what());
        return 1;
    }
}
