// Command-line parsing shared by fngine, fngine-serve and fngine-bench: numeric parsing, engine
// startup flags and sampling-override flags. Number parsing adapted from
// cinference:src/serve/serve_options.cpp @ 383e5db (Apache-2.0, see NOTICE).
#pragma once

#include <fngine/types.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace fngine::serve {

enum class LogLevel : std::uint8_t { Trace, Debug, Info, Warning, Error, Critical, Off };

[[nodiscard]] LogLevel parse_log_level(std::string_view value);
[[nodiscard]] int parse_nonnegative_int(const char* text, const char* label);
[[nodiscard]] float parse_float_in(const char* text, const char* label, float lo, float hi);
[[nodiscard]] std::uint64_t parse_u64(const char* text, const char* label);
[[nodiscard]] KvCacheStorage parse_kv_storage(std::string_view value); // k8v8 | k8v4 | bf16
[[nodiscard]] SpeculativeBackend parse_spec_backend(std::string_view value); // auto|mtp|lookup|none
[[nodiscard]] bool parse_dense_bf16(std::string_view value);                 // fp8 | bf16

// Supplies the value of the current flag (advancing argv) or throws "<flag> needs a value".
using RequireValue = std::function<const char*(const char* flag)>;

// Engine startup flags shared by all apps:
//   --pack PATH --ctx N --prefill-chunk N --spec auto|mtp|lookup|none --cpu-workers N
//   --vram-reserve-mib N --ngram-cache-mib N --kv k8v8|k8v4|bf16 --dense fp8|bf16 --no-cuda-graph
//   --no-adaptive-experts --no-prefill-stream --no-borrow --snapshots N --device N --chat-template FILE
// Returns false when `arg` is not one of them.
bool parse_engine_flag(std::string_view arg, const RequireValue& value, EngineOptions& options);

// Sampling flags: --temperature F [0,2] --top-p F [0,1] --top-k N [0,20] --min-p F [0,1]
// --presence-penalty F [-2,2] --frequency-penalty F [-2,2] --seed N. Returns false otherwise.
bool parse_sampling_flag(std::string_view arg, const RequireValue& value,
                         SamplingOverrides& overrides);

// Rejects inconsistent engine options (zero context, chunk not a multiple of 64, etc.).
void validate_engine_options(const EngineOptions& options);

// Usage lines for the shared flags.
[[nodiscard]] std::string engine_flags_usage();
[[nodiscard]] std::string sampling_flags_usage();

} // namespace fngine::serve
