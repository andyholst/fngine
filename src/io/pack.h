#pragma once

// fngine pack (see the pack spec): manifest.json parsing and structural validation.
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fngine::io {

class PackError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class TensorFormat { Fp8E4m3Row, Bf16, F32 };
enum class PackFile { Dense, DenseBf16, Experts, MtpExperts, Ngram };

const char* to_string(TensorFormat format) noexcept;

struct TensorInfo {
    std::string name;
    PackFile file        = PackFile::Dense;
    TensorFormat format  = TensorFormat::Bf16;
    std::vector<std::int64_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
    // fp8_e4m3_row only: BF16 row scales [rows].
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes  = 0;

    [[nodiscard]] std::int64_t numel() const noexcept;
    // Leading dimension (1 for vectors) and product of the remaining dimensions.
    [[nodiscard]] std::int64_t rows() const noexcept { return shape.size() > 1 ? shape[0] : 1; }
    [[nodiscard]] std::int64_t cols() const noexcept { return shape.empty() ? 0 : numel() / rows(); }
};

// Expected dense tensor (name, format, shape) as produced by tools/convert.py.
struct TensorSpec {
    std::string name;
    TensorFormat format;
    std::vector<std::int64_t> shape;
};
// The complete dense tensor set of the model (main model, PLE, MTP), in pack-spec order.
std::vector<TensorSpec> dense_tensor_specs();

// GGML type ids used by the expert blobs.
enum class GgmlType : std::int32_t {
    IQ3_XXS = 18,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    Q2_0    = 42,
};
const char* to_string(GgmlType type) noexcept;
// Bytes of one row of `elements` values (elements must be a multiple of the block size).
std::uint64_t ggml_row_bytes(GgmlType type, std::int64_t elements);

struct ExpertLayout {
    GgmlType gu_type   = GgmlType::IQ4_NL;
    GgmlType down_type = GgmlType::IQ4_NL;
    std::uint64_t gate_bytes  = 0;
    std::uint64_t up_bytes    = 0;
    std::uint64_t down_bytes  = 0;
    std::uint64_t blob_bytes  = 0;  // gate | up | down
    std::uint64_t blob_stride = 0;  // 4096-aligned
    std::uint64_t offset      = 0;  // file offset of expert 0
    [[nodiscard]] std::uint64_t blob_offset(int expert) const noexcept {
        return offset + std::uint64_t(expert) * blob_stride;
    }
};

struct NgramInfo {
    std::uint64_t rows          = 0;
    std::uint32_t row_bytes     = 0;
    std::uint32_t rows_per_page = 0;
    std::uint32_t header_bytes  = 0;
    std::uint32_t dim           = 0;
    std::int32_t layer          = 0;
    std::int32_t ngram_size     = 0;
    std::int32_t heads_per_ngram = 0;
    std::int32_t n_heads        = 0;
    std::array<std::uint64_t, 3> multipliers{};
    std::array<std::uint64_t, 16> head_sizes{};
    std::array<std::uint64_t, 16> head_offsets{};
    std::int32_t eos = 0;
    // File offset of row r in ngram.bin.
    [[nodiscard]] std::uint64_t row_offset(std::uint64_t r) const noexcept {
        return header_bytes + (r / rows_per_page) * 4096ull + (r % rows_per_page) * row_bytes;
    }
};

// Abliteration (optional pack section): per-layer unit refusal directions and their scales, in a
// standalone file next to the other pack files. The engine projects the summed routed-expert
// output of each main layer away from that layer's direction, which is exactly equivalent to
// orthogonalizing the down projection of every routed expert — without touching the packed
// (quantized) expert bytes.
struct AblationInfo {
    std::filesystem::path path;              // <pack>/ablation.bin
    std::int32_t layers = 0;
    std::int32_t dim = 0;
    std::uint64_t directions_offset = 0;     // f32 [layers][dim], row-major, unit vectors
    std::uint64_t scales_offset = 0;         // f32 [layers]; unused when scales_bytes == 0
    std::uint64_t scales_bytes = 0;
    // Scale applied to layers when no per-layer table is present.
    float scale = 1.0f;
};

class Pack {
public:
    // Parses and validates <dir>/manifest.json against the model dims and the file sizes.
    // Throws PackError with the offending field on any mismatch.
    static Pack open(const std::filesystem::path& dir);

    [[nodiscard]] const std::filesystem::path& dir() const noexcept { return dir_; }
    [[nodiscard]] std::filesystem::path file_path(PackFile file) const;
    [[nodiscard]] std::uint64_t file_bytes(PackFile file) const;
    [[nodiscard]] bool has_file(PackFile file) const;

    [[nodiscard]] const nlohmann::json& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const nlohmann::json& config() const { return manifest_.at("config"); }
    [[nodiscard]] const nlohmann::json& generation_config() const { return manifest_.at("generation_config"); }

    // Lookups throw PackError naming the tensor when it is missing.
    [[nodiscard]] const TensorInfo& tensor(std::string_view name) const;
    [[nodiscard]] const TensorInfo* find_tensor(std::string_view name) const noexcept;
    // BF16 copy (dense_bf16.bin) of an fp8_e4m3_row tensor.
    [[nodiscard]] const TensorInfo& tensor_bf16(std::string_view name) const;
    [[nodiscard]] const std::map<std::string, TensorInfo, std::less<>>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::map<std::string, TensorInfo, std::less<>>& tensors_bf16() const noexcept {
        return tensors_bf16_;
    }

    [[nodiscard]] const std::vector<ExpertLayout>& expert_layers() const noexcept { return expert_layers_; }
    [[nodiscard]] const ExpertLayout& mtp_experts() const noexcept { return mtp_experts_; }
    [[nodiscard]] const NgramInfo& ngram() const noexcept { return ngram_; }
    // Optional abliteration section; has_ablation() is false for an ordinary pack.
    [[nodiscard]] bool has_ablation() const noexcept { return !ablation_.path.empty(); }
    [[nodiscard]] const AblationInfo& ablation() const noexcept { return ablation_; }

    // Routing profile counts [48][512] (layer-major); empty when the pack has no profile.
    [[nodiscard]] bool has_profile() const noexcept { return !profile_.empty(); }
    [[nodiscard]] const std::vector<std::uint64_t>& profile() const noexcept { return profile_; }

    // Absolute paths of the tokenizer files listed in the manifest (all exist).
    [[nodiscard]] const std::vector<std::filesystem::path>& tokenizer_files() const noexcept { return tokenizer_; }
    // Absolute path of a tokenizer file by basename ("tokenizer.json"); empty if not listed.
    [[nodiscard]] std::filesystem::path tokenizer_file(std::string_view basename) const;

private:
    std::filesystem::path dir_;
    nlohmann::json manifest_;
    std::map<PackFile, std::string> files_;
    std::map<PackFile, std::uint64_t> file_sizes_;
    std::map<std::string, TensorInfo, std::less<>> tensors_;
    std::map<std::string, TensorInfo, std::less<>> tensors_bf16_;
    std::vector<ExpertLayout> expert_layers_;
    ExpertLayout mtp_experts_;
    NgramInfo ngram_;
    AblationInfo ablation_;
    std::vector<std::uint64_t> profile_;
    std::vector<std::filesystem::path> tokenizer_;
};

} // namespace fngine::io
