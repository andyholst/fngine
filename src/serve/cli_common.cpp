// Number parsing adapted from cinference:src/serve/serve_options.cpp @ 383e5db. See NOTICE.
#include "serve/cli_common.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace fngine::serve {

LogLevel parse_log_level(std::string_view value) {
    if (value == "trace") return LogLevel::Trace;
    if (value == "debug") return LogLevel::Debug;
    if (value == "info") return LogLevel::Info;
    if (value == "warning" || value == "warn") return LogLevel::Warning;
    if (value == "error") return LogLevel::Error;
    if (value == "critical") return LogLevel::Critical;
    if (value == "off") return LogLevel::Off;
    throw std::invalid_argument("invalid log level: " + std::string(value));
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    errno            = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_storage(std::string_view value) {
    if (value == "k8v8") return KvCacheStorage::K8V8;
    if (value == "k8v4") return KvCacheStorage::K8V4;
    if (value == "bf16") return KvCacheStorage::BF16;
    throw std::invalid_argument("invalid --kv (expected k8v8|k8v4|bf16): " + std::string(value));
}

SpeculativeBackend parse_spec_backend(std::string_view value) {
    if (value == "auto") return SpeculativeBackend::Auto;
    if (value == "mtp") return SpeculativeBackend::Mtp;
    if (value == "lookup") return SpeculativeBackend::Lookup;
    if (value == "none") return SpeculativeBackend::None;
    throw std::invalid_argument("invalid --spec (expected auto|mtp|lookup|none): " +
                                std::string(value));
}

bool parse_dense_bf16(std::string_view value) {
    if (value == "fp8") return false;
    if (value == "bf16") return true;
    throw std::invalid_argument("invalid --dense (expected fp8|bf16): " + std::string(value));
}

namespace {

std::uint32_t parse_u32(const char* text, const char* label) {
    return static_cast<std::uint32_t>(parse_nonnegative_int(text, label));
}

} // namespace

bool parse_engine_flag(std::string_view arg, const RequireValue& value, EngineOptions& options) {
    if (arg == "--pack") {
        options.pack_path = value("--pack");
    } else if (arg == "--ctx") {
        options.max_context = parse_u32(value("--ctx"), "ctx");
    } else if (arg == "--prefill-chunk") {
        options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
    } else if (arg == "--spec") {
        options.speculative.backend = parse_spec_backend(value("--spec"));
    } else if (arg == "--cpu-workers") {
        options.cpu_workers = parse_u32(value("--cpu-workers"), "cpu-workers");
    } else if (arg == "--vram-reserve-mib") {
        options.vram_reserve_mib = parse_u32(value("--vram-reserve-mib"), "vram-reserve-mib");
    } else if (arg == "--ngram-cache-mib") {
        options.ngram_cache_mib = parse_u32(value("--ngram-cache-mib"), "ngram-cache-mib");
    } else if (arg == "--kv") {
        options.kv_cache = parse_kv_storage(value("--kv"));
    } else if (arg == "--dense") {
        options.dense_bf16 = parse_dense_bf16(value("--dense"));
    } else if (arg == "--no-cuda-graph") {
        options.use_cuda_graph = false;
    } else if (arg == "--no-adaptive-experts") {
        options.adaptive_experts = false;
    } else if (arg == "--no-prefill-stream") {
        options.prefill_stream = false;
    } else if (arg == "--no-borrow") {
        options.prefill_borrow = false;
    } else if (arg == "--no-ablation") {
        options.ablation = false;
    } else if (arg == "--snapshots") {
        options.snapshots = parse_u32(value("--snapshots"), "snapshots");
    } else if (arg == "--device") {
        options.device = parse_nonnegative_int(value("--device"), "device");
    } else if (arg == "--chat-template") {
        options.chat_template_path = value("--chat-template");
    } else {
        return false;
    }
    return true;
}

bool parse_sampling_flag(std::string_view arg, const RequireValue& value,
                         SamplingOverrides& overrides) {
    if (arg == "--temperature") {
        overrides.temperature = parse_float_in(value("--temperature"), "temperature", 0.0F, 2.0F);
    } else if (arg == "--top-p") {
        overrides.top_p = parse_float_in(value("--top-p"), "top-p", 0.0F, 1.0F);
    } else if (arg == "--top-k") {
        const int top_k = parse_nonnegative_int(value("--top-k"), "top-k");
        if (top_k > 20) throw std::invalid_argument("top-k must be in [0,20]");
        overrides.top_k = top_k;
    } else if (arg == "--min-p") {
        overrides.min_p = parse_float_in(value("--min-p"), "min-p", 0.0F, 1.0F);
    } else if (arg == "--presence-penalty") {
        overrides.presence_penalty =
            parse_float_in(value("--presence-penalty"), "presence-penalty", -2.0F, 2.0F);
    } else if (arg == "--frequency-penalty") {
        overrides.frequency_penalty =
            parse_float_in(value("--frequency-penalty"), "frequency-penalty", -2.0F, 2.0F);
    } else if (arg == "--seed") {
        overrides.seed = parse_u64(value("--seed"), "seed");
    } else {
        return false;
    }
    return true;
}

void validate_engine_options(const EngineOptions& options) {
    if (options.pack_path.empty()) throw std::invalid_argument("--pack is required");
    if (options.max_context == 0) throw std::invalid_argument("--ctx must be positive");
    if (options.prefill_chunk == 0 || options.prefill_chunk % 16 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 16");
    }
    if (options.cpu_workers > 64) throw std::invalid_argument("--cpu-workers must be in [0,64]");
    if (options.snapshots == 0) throw std::invalid_argument("--snapshots must be positive");
}

std::string engine_flags_usage() {
    return "  --pack PATH               fngine pack directory (required)\n"
           "  --ctx N                   max context tokens (default 131072)\n"
           "  --prefill-chunk N         prefill chunk tokens, multiple of 64 (default 4096)\n"
           "  --spec auto|mtp|lookup|none  speculative decoding source (default auto)\n"
           "  --cpu-workers N           CPU expert worker threads (default 0 = one per worker core)\n"
           "  --vram-reserve-mib N      VRAM left free after placement (default 512)\n"
           "  --ngram-cache-mib N       n-gram row cache in RAM (default 1024)\n"
           "  --kv k8v8|k8v4|bf16       KV cache storage (default k8v8)\n"
           "  --dense fp8|bf16          dense weights (bf16 = validation mode; default fp8)\n"
           "  --no-cuda-graph           eager decode steps\n"
           "  --no-adaptive-experts     no in-session GPU<->RAM expert swaps\n"
           "  --no-prefill-stream       prefill: no PCIe streaming of warm CPU experts\n"
           "  --no-borrow               reserve prompt workspace instead of borrowing expert slots\n"
           "  --no-ablation             ignore the pack's ablation section (un-projected control run)\n"
           "  --snapshots N             session snapshot slots (default 6)\n"
           "  --device N                CUDA device (default 0)\n"
           "  --chat-template FILE      override <pack>/tokenizer/chat_template.jinja\n";
}

std::string sampling_flags_usage() {
    return "  --temperature F [0,2]  --top-p F [0,1]  --top-k N [0,20]  --min-p F [0,1]\n"
           "  --presence-penalty F [-2,2]  --frequency-penalty F [-2,2]  --seed N\n";
}

} // namespace fngine::serve
