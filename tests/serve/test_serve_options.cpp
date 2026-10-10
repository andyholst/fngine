// fngine-serve option parsing and server-default translation. Structure follows
// cinference:tests/test_serve_options.cpp @ 383e5db (Apache-2.0, see NOTICE).
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fngine::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "fngine-serve");
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(std::vector<std::string> arguments) {
    try {
        (void)parse(std::move(arguments));
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

GenerationRequest user_request() {
    GenerationRequest request;
    ChatTurn turn;
    turn.content.push_back(ContentPart{.text = "hi", .type_raw = "text"});
    request.messages.push_back(std::move(turn));
    request.max_tokens = 16;
    return request;
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"--pack", "/p"});
    failures += check(defaults.host == "127.0.0.1" && defaults.port == 8002 &&
                          defaults.model_id == "qwen3.8-flash-next",
                      "server defaults: 127.0.0.1:8002, model id qwen3.8-flash-next");
    const fngine::EngineOptions& e = defaults.engine;
    failures += check(e.pack_path == "/p" && e.max_context == 131072 && e.prefill_chunk == 4096 &&
                          e.speculative.backend == fngine::SpeculativeBackend::Auto &&
                          e.cpu_workers == 0 && e.vram_reserve_mib == 512 &&
                          e.ngram_cache_mib == 1024 && e.kv_cache == fngine::KvCacheStorage::K8V8 &&
                          !e.dense_bf16 && e.use_cuda_graph && e.snapshots == 6 &&
                          e.max_pending_requests == 8,
                      "engine defaults follow the plan");

    const ServeOptions all = parse(
        {"--pack", "/x", "--host", "0.0.0.0", "--port", "9000", "--model-id", "m", "--ctx",
         "262144", "--prefill-chunk", "8192", "--spec", "lookup", "--cpu-workers", "6",
         "--vram-reserve-mib", "1024", "--ngram-cache-mib", "256", "--kv", "bf16", "--dense",
         "bf16", "--no-cuda-graph", "--snapshots", "3", "--max-pending", "2", "--api-key", "k",
         "--no-thinking", "--no-preserve-thinking", "--default-thinking-budget", "100",
         "--temperature", "0.5", "--top-k", "7", "--seed", "11", "--log-level", "debug"});
    failures += check(all.host == "0.0.0.0" && all.port == 9000 && all.model_id == "m" &&
                          all.api_key == "k",
                      "server flags parsed");
    failures += check(all.engine.max_context == 262144 && all.engine.prefill_chunk == 8192 &&
                          all.engine.speculative.backend == fngine::SpeculativeBackend::Lookup &&
                          all.engine.cpu_workers == 6 && all.engine.vram_reserve_mib == 1024 &&
                          all.engine.ngram_cache_mib == 256 &&
                          all.engine.kv_cache == fngine::KvCacheStorage::BF16 &&
                          all.engine.dense_bf16 && !all.engine.use_cuda_graph &&
                          all.engine.snapshots == 3 && all.engine.max_pending_requests == 2,
                      "engine flags parsed");
    failures += check(all.enable_thinking == false && all.preserve_thinking == false &&
                          all.default_thinking_budget == 100u &&
                          all.sampling_overrides.temperature == 0.5F &&
                          all.sampling_overrides.top_k == 7 && all.sampling_overrides.seed == 11u &&
                          all.log_level == LogLevel::Debug,
                      "prompt/sampling defaults parsed");
    bool redacted = false;
    for (const std::string& arg : all.startup_argv) redacted = redacted || arg == "<redacted>";
    failures += check(redacted, "--api-key value is redacted from startup argv");

    failures += check(parse({"--help"}).help_requested, "--help requested");
    failures += check(rejects({}), "--pack is required");
    failures += check(rejects({"--pack", "/p", "--spec", "dflash"}), "unknown --spec rejected");
    failures += check(rejects({"--pack", "/p", "--kv", "nvfp4"}), "unknown --kv rejected");
    failures += check(parse({"--pack", "/p", "--kv", "k8v4"}).engine.kv_cache ==
                              fngine::KvCacheStorage::K8V4 &&
                          parse({"--pack", "/p", "--kv", "k8v8"}).engine.kv_cache ==
                              fngine::KvCacheStorage::K8V8,
                      "--kv k8v4 and k8v8 parse");
    failures += check(rejects({"--pack", "/p", "--dense", "fp4"}), "unknown --dense rejected");
    failures += check(rejects({"--pack", "/p", "--prefill-chunk", "100"}),
                      "prefill chunk must be a multiple of 16");
    failures += check(rejects({"--pack", "/p", "--port", "0"}), "port 0 rejected");
    failures += check(rejects({"--pack", "/p", "--top-k", "21"}), "top-k > 20 rejected");
    failures += check(rejects({"--pack", "/p", "--temperature", "2.5"}), "temperature > 2 rejected");
    failures += check(rejects({"--pack", "/p", "--ctx", "0"}), "ctx 0 rejected");
    failures += check(rejects({"--pack", "/p", "--cpu-workers", "65"}), "65 CPU workers rejected");
    failures += check(rejects({"--pack", "/p", "--max-pending", "0"}), "max-pending 0 rejected");
    failures += check(rejects({"--pack", "/p", "--port"}), "missing value rejected");
    failures += check(rejects({"--pack", "/p", "--bogus"}), "unknown flag rejected");

    // Server defaults layered under request fields.
    ServeOptions server = parse({"--pack", "/p", "--no-thinking", "--greedy", "--seed", "5"});
    GenerationRequest request = user_request();
    ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, server);
    failures += check(semantics.enable_thinking == false, "server --no-thinking is the default");
    request.enable_thinking = true;
    semantics               = resolve_prompt_semantics(request, server);
    failures += check(semantics.enable_thinking == true, "request enable_thinking overrides");
    request.sampling.temperature = 1.2;
    fngine::RequestOptions options = to_request_options(request, server, semantics);
    failures += check(options.execution.sampling.temperature == 0.0F,
                      "--greedy forces temperature 0 over the request");
    failures += check(options.execution.sampling.seed == 5u, "server seed used when omitted");
    request.sampling.seed = 9;
    options               = to_request_options(request, server, semantics);
    failures += check(options.execution.sampling.seed == 9u, "request seed wins");

    ServeOptions plain = parse({"--pack", "/p", "--default-thinking-budget", "64"});
    GenerationRequest r2 = user_request();
    const auto s2        = resolve_prompt_semantics(r2, plain);
    const auto o2        = to_request_options(r2, plain, s2);
    failures += check(o2.execution.thinking.budget == 64u && o2.execution.sampling.seed.has_value(),
                      "default thinking budget applies; omitted seed becomes concrete");

    if (failures == 0) std::cout << "serve options tests passed\n";
    return failures == 0 ? 0 : 1;
}
