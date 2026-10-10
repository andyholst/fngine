#pragma once

// Device-resident dense weights of the model (everything except routed experts and the n-gram
// table), loaded from the pack with O_DIRECT staged uploads.
#include "core/arena.h"
#include "core/host_memory.h"
#include "io/direct_io.h"
#include "io/pack.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace fngine::model {

// FP8-E4M3 row-scaled matrix [n][k]: w[r][c] = e4m3(codes[r*k+c]) * bf16(scales[r]).
// `bf16` is the BF16 copy of the same matrix (pack dense_bf16), non-null only when loaded with
// `dense_bf16 = true` (validation mode).
struct Fp8Weight {
    const std::uint8_t* codes     = nullptr;
    const __nv_bfloat16* scales   = nullptr;
    const __nv_bfloat16* bf16     = nullptr;
    std::int32_t n = 0, k = 0;
};

// Row-major BF16 matrix [n][k].
struct Bf16Weight {
    const __nv_bfloat16* data = nullptr;
    std::int32_t n = 0, k = 0;
};

// FP32 tensor (effective norm multipliers, conv taps [channels][taps], A_log, dt_bias).
struct F32Weight {
    const float* data  = nullptr;
    std::int32_t count = 0;
};

// Hyper-connection gated residual: `inject` is unused by the mixers.
struct GrWeights {
    F32Weight hc_norm;   // [10240]
    Fp8Weight down;      // [320, 10240]
    Fp8Weight up;        // [10240, 384], row d*4+s, cols 320.. zero
    Bf16Weight inject;   // [4, 10240]
};

struct MixerWeights {
    F32Weight hc_norm;
    Fp8Weight down;
    Fp8Weight up;
};

struct GdnWeights {
    Fp8Weight qkvz;     // [16384, 2560]
    Bf16Weight ab;      // [96, 2560]
    F32Weight conv;     // [10240][4]
    F32Weight A_log;    // [48]
    F32Weight dt_bias;  // [48]
    F32Weight norm;     // [128], plain w
    Fp8Weight out;      // [2560, 6144]
};

struct QsaWeights {
    Fp8Weight qkv;          // [13312, 2560]
    F32Weight q_norm;       // [256]
    F32Weight k_norm;       // [256]
    Fp8Weight o;            // [2560, 6144]
    Bf16Weight idx_qk;      // [640, 2560]
    F32Weight idx_q_norm;   // [128]
    F32Weight idx_k_norm;   // [128]
};

struct MoeWeights {
    Bf16Weight router;          // [513, 2560] = router | shared gate
    Fp8Weight shared_gate_up;   // [1280, 2560]
    Fp8Weight shared_down;      // [2560, 640]
};

struct PleWeights {
    Fp8Weight key;         // [10240, 2560]
    Fp8Weight value;       // [2560, 2560]
    F32Weight norm_key;    // [10240]
    F32Weight norm_query;  // [10240]
    F32Weight norm_conv;   // [10240]
    F32Weight conv;        // [10240][4]
};

struct LayerWeights {
    bool qsa = false;
    GrWeights attn_gr;
    GrWeights mlp_gr;
    GdnWeights gdn;  // GDN layers only
    QsaWeights attn; // QSA layers only
    MoeWeights moe;
};

// A set of expert blobs (gate | up | down) at base + e * blob_stride.
struct ExpertBlobs {
    const std::uint8_t* base = nullptr;
    std::uint64_t blob_stride = 0;
    std::uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    io::GgmlType gu_type   = io::GgmlType::IQ4_NL;
    io::GgmlType down_type = io::GgmlType::IQ4_NL;
    std::int32_t count = 0;
};

struct MtpWeights {
    Fp8Weight fc_embedding;          // [2560, 2560]
    Fp8Weight fc_hidden;             // [2560, 2560]
    F32Weight pre_fc_norm_embedding; // [2560]
    F32Weight pre_fc_norm_hidden;    // [10240]
    GrWeights attn_gr;
    GrWeights mlp_gr;
    QsaWeights attn;
    MoeWeights moe;
    MixerWeights mixer;
    ExpertBlobs experts;  // 512 IQ4_NL experts, all in VRAM (null when not loaded)
};

// BF16 token embedding [248320][2560] in registered (device-mapped) host memory.
struct EmbeddingTable {
    const __nv_bfloat16* host   = nullptr;
    const __nv_bfloat16* device = nullptr;  // device-visible alias for the gather kernel
    std::int32_t rows = 0, cols = 0;
};

struct WeightsOptions {
    bool dense_bf16       = false;  // also upload BF16 copies of every FP8 matrix (validation)
    bool load_mtp_experts = true;
    int read_threads      = 2;  // parallel O_DIRECT reads per 64 MiB staging slot
};

struct WeightsLoadStats {
    io::TransferStats dense;
    io::TransferStats dense_bf16;
    io::TransferStats mtp_experts;
    io::TransferStats embedding;
    double embedding_register_seconds = 0;
};

class DenseWeights {
public:
    // Device bytes load() allocates for these options (for VRAM planning).
    static std::uint64_t device_bytes(const io::Pack& pack, const WeightsOptions& options);
    // Host bytes load() pins (the embedding table).
    static std::uint64_t host_bytes(const io::Pack& pack);

    // Allocates one device buffer for all dense tensors (+ MTP experts), an optional second one for
    // the BF16 copies, uploads them from the pack on `stream`, and fills + registers the host
    // embedding table. Throws on any I/O or CUDA failure.
    static DenseWeights load(const io::Pack& pack, cudaStream_t stream, const WeightsOptions& options = {});

    DenseWeights()                               = default;
    DenseWeights(DenseWeights&&) noexcept        = default;
    DenseWeights& operator=(DenseWeights&&) noexcept = default;

    [[nodiscard]] const LayerWeights& layer(int i) const { return layers_.at(std::size_t(i)); }
    [[nodiscard]] const Fp8Weight& lm_head() const noexcept { return lm_head_; }
    [[nodiscard]] const MixerWeights& mixer() const noexcept { return mixer_; }
    [[nodiscard]] const PleWeights& ple() const noexcept { return ple_; }
    [[nodiscard]] const MtpWeights& mtp() const noexcept { return mtp_; }
    [[nodiscard]] const EmbeddingTable& embedding() const noexcept { return embedding_; }
    [[nodiscard]] bool has_bf16() const noexcept { return bf16_.bytes != 0; }
    // MTP draft head: the lm_head rows of `vocab` (ascending token ids; vocab.size() rows), built
    // on the device from the loaded lm_head. Drafts are verified exactly, so the subset only
    // affects acceptance. Empty until built.
    void build_draft_head(const std::vector<std::int32_t>& vocab, cudaStream_t stream);
    [[nodiscard]] const Fp8Weight& draft_head() const noexcept { return draft_head_; }
    [[nodiscard]] const std::int32_t* draft_vocab() const noexcept { return draft_vocab_; }

    // Device address of a pack tensor's data (codes for FP8) / FP8 row scales / BF16 copy.
    [[nodiscard]] const void* device_data(std::string_view name) const;
    [[nodiscard]] const void* device_scales(std::string_view name) const;
    [[nodiscard]] const void* device_bf16(std::string_view name) const;

    [[nodiscard]] std::uint64_t vram_bytes() const noexcept { return dense_.bytes + bf16_.bytes + draft_.bytes; }
    [[nodiscard]] const WeightsLoadStats& stats() const noexcept { return stats_; }

private:
    struct Location {
        std::uint64_t data = 0, scales = 0, bf16 = 0;  // device offsets (bf16 into bf16_)
        bool has_scales = false, has_bf16 = false;
    };
    void bind(const io::Pack& pack);
    Fp8Weight fp8(const io::Pack& pack, const std::string& name) const;
    Bf16Weight bf16w(const io::Pack& pack, const std::string& name) const;
    F32Weight f32(const io::Pack& pack, const std::string& name) const;
    GrWeights gr(const io::Pack& pack, const std::string& prefix) const;
    MixerWeights mixer_w(const io::Pack& pack, const std::string& prefix) const;
    QsaWeights qsa(const io::Pack& pack, const std::string& prefix) const;
    MoeWeights moe(const io::Pack& pack, const std::string& prefix) const;

    DeviceBuffer dense_;  // dense tensors (except the embedding) followed by the MTP experts
    DeviceBuffer bf16_;
    HostRegion embed_;
    std::map<std::string, Location, std::less<>> locations_;
    std::uint64_t mtp_offset_ = 0;
    std::array<LayerWeights, 48> layers_{};
    Fp8Weight lm_head_;
    MixerWeights mixer_;
    PleWeights ple_;
    MtpWeights mtp_;
    EmbeddingTable embedding_;
    DeviceBuffer draft_;  // draft head codes | scales | vocab
    Fp8Weight draft_head_;
    const std::int32_t* draft_vocab_ = nullptr;
    WeightsLoadStats stats_;
};

} // namespace fngine::model
