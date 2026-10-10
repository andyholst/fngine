// fngine-bench option parsing, test matrix, corpus and report formatting. Structure follows
// cinference:tests/test_ninfer_bench_support.cpp @ 383e5db (Apache-2.0, see NOTICE).
#include "fngine_bench_support.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace fngine::bench;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

BenchOptions parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "fngine-bench");
    std::vector<char*> argv;
    for (std::string& argument : arguments) argv.push_back(argument.data());
    return parse_args(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const BenchOptions options = parse({"--pack", "/p", "-p1024,4096,32768", "-n256",
                                        "-r3", "--spec", "none", "-ojson", "-f", "out.json"});
    failures += check(options.n_prompt == std::vector<int>{1024, 4096, 32768} &&
                          options.n_gen == 256 && options.repetitions == 3 &&
                          options.engine.speculative.backend == fngine::SpeculativeBackend::None &&
                          options.output == OutputFormat::Json && options.output_file == "out.json",
                      "plan command line parses");
    const auto tests = expand_tests(options);
    failures += check(tests.size() == 3 && tests[2].label == "pp32768+tg256" &&
                          tests[2].requested_output_tokens() == 257,
                      "one pp+tg test per prompt length");
    failures += check(options.engine.max_context == tests[2].required_context() &&
                          options.engine.max_context >= 32768 + 257,
                      "context auto-sized to the largest test");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "-p", "4096", "--ctx", "100"}); }),
                      "explicit --ctx below a test requirement rejected");
    failures += check(rejects([] { (void)parse({"-p", "10"}); }), "--pack required");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "-p", "0"}); }),
                      "non-positive prompt rejected");
    failures += check(rejects([] { (void)parse({"--pack", "/p", "-o", "csv"}); }),
                      "unsupported output format rejected");

    // Corpus building is deterministic and repeats to the requested size.
    const auto root = std::filesystem::temp_directory_path() / "fngine_bench_corpus_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "docs" / "b");
    std::ofstream(root / "README.md") << "R";
    std::ofstream(root / "docs" / "b" / "z.md") << "Z";
    std::ofstream(root / "docs" / "a.md") << "A";
    std::ofstream(root / "docs" / "skip.txt") << "S";
    std::string seen;
    const auto tokenize = [&](std::string_view text) {
        seen = std::string(text);
        return std::vector<fngine::TokenId>(text.begin(), text.end());
    };
    const auto corpus = build_text_corpus(root, tokenize, 20);
    failures += check(seen == "R\n\nA\n\nZ", "README then sorted docs/**/*.md, blank-line joined");
    failures += check(corpus.size() >= 20 && corpus.size() % seen.size() == 0,
                      "corpus repeats whole passes up to the minimum");
    failures += check(prompt_slice(corpus, 3) == std::vector<fngine::TokenId>{'R', '\n', '\n'},
                      "prompt slice takes the corpus prefix");
    failures += check(rejects([&] { (void)prompt_slice(corpus, 1000); }),
                      "prompt longer than corpus rejected");
    std::ofstream(root / "ids.txt") << "1 2\n3\t4\n";
    failures += check(load_corpus_ids(root / "ids.txt") == std::vector<fngine::TokenId>{1, 2, 3, 4},
                      "id corpus parses whitespace-separated ids");
    std::ofstream(root / "bad.txt") << "1 x";
    failures += check(rejects([&] { (void)load_corpus_ids(root / "bad.txt"); }),
                      "non-integer corpus rejected");
    std::filesystem::remove_all(root);

    // Statistics and reports.
    const Stats stats = compute_stats({1.0, 2.0, 3.0});
    failures += check(stats.mean == 2.0 && stats.stddev == 1.0 && stats.count == 3,
                      "sample mean/stddev");
    fngine::RuntimeStats before, after;
    before.decode_steps = 10;
    before.cpu_wait_us_per_layer = 100.0;
    after.decode_steps = 30;
    after.cpu_wait_us_per_layer = 40.0;
    before.decode_routing_mass = 100.0;
    before.decode_gpu_routing_mass = 50.0;
    after.decode_routing_mass = 300.0;
    after.decode_gpu_routing_mass = 230.0;
    const auto interval = stats_delta(before, after);
    failures += check(interval.cpu_wait_us_per_layer == 10.0 &&
                          interval.gpu_hit_rate_by_routing_mass == 0.9,
                      "interval metrics exclude prior requests and warmup");
    TestResult result{tests[0], {}};
    for (int i = 0; i < 2; ++i) {
        RepTiming rep;
        rep.prompt_tokens                       = 1024;
        rep.generated_output_tokens             = 257;
        rep.timings.prefill_seconds             = 0.5;
        rep.timings.decode_seconds              = 2.56;
        rep.engine.routed_pairs                 = 100;
        rep.engine.gpu_routed_pairs             = 90;
        rep.engine.gpu_hit_rate_by_routing_mass = 0.9;
        rep.engine.cpu_wait_us_per_layer        = 12.5;
        rep.engine.decode_steps                = 20;
        rep.engine.routing_mass                = 100.0;
        rep.engine.gpu_routing_mass            = 90.0;
        rep.engine.mtp_rounds                   = 10;
        rep.engine.mtp_accepted_tokens          = 15;
        rep.engine.ngram_lookups                = 10;
        rep.engine.ngram_cache_hits             = 8;
        result.reps.push_back(rep);
    }
    failures += check(prefill_tok_s_series(result) == std::vector<double>{2048.0, 2048.0} &&
                          decode_tok_s_series(result) == std::vector<double>{100.0, 100.0},
                      "prefill = prompt/prefill_s, decode = (generated-1)/decode_s");
    BenchEnvironment env;
    env.gpu_name = "RTX \"5090\"";
    env.engine   = options.engine;
    const std::string json_text = format_json(env, "fngine-bench -p 1024", {result});
    const auto json             = nlohmann::json::parse(json_text);
    const auto& t0              = json["tests"][0];
    failures += check(json["type"] == "fngine_bench_report" &&
                          json["environment"]["gpu"] == "RTX \"5090\"" &&
                          t0["prefill_tok_s_mean"] == 2048.0 && t0["decode_tok_s_mean"] == 100.0 &&
                          t0["gpu_hit_rate_by_routing_mass"] == 0.9 &&
                          t0["cpu_wait_us_per_layer"] == 12.5 &&
                          t0["accepted_tokens_per_round_mtp"] == 1.5 &&
                          t0["ngram_cache_hit_rate"] == 0.8 && t0["reps"].size() == 2,
                      "JSON report carries throughput and engine stats");
    const std::string table = format_table(env, {result});
    failures += check(table.find("pp1024+tg256") != std::string::npos &&
                          table.find("2048.00") != std::string::npos &&
                          table.find("90.0%") != std::string::npos,
                      "table report");
    if (failures == 0) std::cout << "bench support tests passed\n";
    return failures == 0 ? 0 : 1;
}
