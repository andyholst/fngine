// fngine CLI option parsing. Structure follows cinference:tests/test_cli_options.cpp @ 383e5db
// (Apache-2.0, see NOTICE).
#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

fngine::cli::Options parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "fngine");
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return fngine::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    using fngine::cli::Options;

    const Options defaults = parse({"--pack", "/p", "--prompt", "hi"});
    failures += check(defaults.max_new == 512 && defaults.engine.max_context == 131072 &&
                          defaults.engine.prefill_chunk == 4096 &&
                          defaults.engine.speculative.backend ==
                              fngine::SpeculativeBackend::Auto &&
                          defaults.engine.kv_cache == fngine::KvCacheStorage::K8V8 &&
                          !defaults.engine.dense_bf16 && defaults.engine.cpu_workers == 0 &&
                          !defaults.stats && !defaults.enable_thinking,
                      "CLI defaults");

    const Options all = parse(
        {"--pack", "/p", "--messages", "m.json", "--max-new", "64", "--ctx", "4096",
         "--prefill-chunk", "1024", "--spec", "none", "--greedy", "--top-p", "0.9", "--top-k",
         "5", "--min-p", "0.1", "--presence-penalty", "1.5", "--frequency-penalty", "-1", "--seed",
         "3", "--reasoning-effort", "low", "--cpu-workers", "4", "--vram-reserve-mib", "100",
         "--ngram-cache-mib", "64", "--kv", "bf16", "--dense", "bf16", "--no-cuda-graph",
         "--print-token-ids", "--stats", "--raw-output", "--stop", "END", "--reasoning-stop",
         "R", "--stop-token-id", "42", "--thinking-budget", "7", "--log-level", "warning"});
    failures += check(all.messages_path == "m.json" && all.max_new == 64 &&
                          all.engine.max_context == 4096 && all.engine.prefill_chunk == 1024 &&
                          all.engine.speculative.backend == fngine::SpeculativeBackend::None,
                      "request/engine flags parsed");
    failures += check(all.greedy && all.sampling.temperature == 0.0F && all.sampling.top_p == 0.9F &&
                          all.sampling.top_k == 5 && all.sampling.min_p == 0.1F &&
                          all.sampling.presence_penalty == 1.5F &&
                          all.sampling.frequency_penalty == -1.0F && all.sampling.seed == 3u,
                      "sampling flags parsed; --greedy forces temperature 0");
    failures += check(all.reasoning_effort == fngine::ReasoningEffort::Low &&
                          all.thinking_budget == 7u && all.engine.cpu_workers == 4 &&
                          all.engine.vram_reserve_mib == 100 && all.engine.ngram_cache_mib == 64 &&
                          all.engine.kv_cache == fngine::KvCacheStorage::BF16 &&
                          all.engine.dense_bf16 && !all.engine.use_cuda_graph &&
                          all.print_token_ids && all.stats && all.raw_output,
                      "engine/output flags parsed");
    failures += check(all.stop_strings.size() == 2 &&
                          all.stop_strings[0].channel == fngine::OutputChannel::Content &&
                          all.stop_strings[1].channel == fngine::OutputChannel::Reasoning &&
                          all.stop_token_ids == std::vector<fngine::TokenId>{42},
                      "stop flags parsed per channel");
    failures += check(all.log_level == fngine::serve::LogLevel::Warning, "log level parsed");

    const Options none = parse({"--pack", "/p", "--prompt", "x", "--reasoning-effort", "none"});
    failures += check(none.enable_thinking == false, "--reasoning-effort none disables thinking");

    failures += check(parse({"--help"}).help_requested, "--help");
    failures += check(rejects([] { (void)parse({"--prompt", "x"}); }), "--pack required");
    failures += check(rejects([] { (void)parse({"--pack", "/p"}); }), "prompt or messages required");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--messages",
                                                "m"}); }),
                      "--prompt and --messages are exclusive");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--no-thinking",
                                                "--reasoning-effort", "low"}); }),
                      "reasoning effort conflicts with --no-thinking");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--no-thinking",
                                                "--thinking-budget", "4"}); }),
                      "thinking budget conflicts with --no-thinking");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x",
                                                "--reasoning-effort", "high"}); }),
                      "efforts outside the template set rejected");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--top-k",
                                                "21"}); }),
                      "--top-k above 20 rejected");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--spec",
                                                "dflash"}); }),
                      "unknown spec rejected");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--max-new",
                                                "0"}); }),
                      "--max-new 0 rejected");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "--prompt", "x", "--stop",
                                                ""}); }),
                      "empty stop string rejected");
    const std::string help = fngine::cli::usage_text("fngine");
    failures += check(help.find("--pack") != std::string::npos &&
                          help.find("--spec auto|mtp|lookup|none") != std::string::npos &&
                          help.find("--ngram-cache-mib") != std::string::npos,
                      "usage lists fngine flags");
    if (failures == 0) std::cout << "CLI option tests passed\n";
    return failures == 0 ? 0 : 1;
}
