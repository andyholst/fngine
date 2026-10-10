#include "model/weights.h"
#include "model/step_kernels.h"

#include "core/model_dims.h"
#include "core/device.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <vector>

namespace fngine::model {
namespace {

using io::TensorFormat;
using io::TensorInfo;
using namespace fngine::dims;

constexpr std::uint64_t kDeviceAlign = 256;

std::uint64_t align256(std::uint64_t v) { return (v + kDeviceAlign - 1) / kDeviceAlign * kDeviceAlign; }

// Dense tensors that live on the device: every pack tensor but the embedding, in file order.
std::vector<const TensorInfo*> device_tensors(const io::Pack& pack) {
    std::vector<const TensorInfo*> out;
    for (const auto& [name, info] : pack.tensors()) {
        if (name != "embed") { out.push_back(&info); }
    }
    std::sort(out.begin(), out.end(), [](const auto* a, const auto* b) { return a->offset < b->offset; });
    return out;
}

std::vector<const TensorInfo*> bf16_tensors(const io::Pack& pack) {
    std::vector<const TensorInfo*> out;
    for (const auto& [name, info] : pack.tensors_bf16()) { out.push_back(&info); }
    std::sort(out.begin(), out.end(), [](const auto* a, const auto* b) { return a->offset < b->offset; });
    return out;
}

std::uint64_t dense_layout_bytes(const io::Pack& pack) {
    std::uint64_t off = 0;
    for (const auto* t : device_tensors(pack)) {
        off = align256(off) + t->bytes;
        if (t->format == TensorFormat::Fp8E4m3Row) { off = align256(off) + t->scale_bytes; }
    }
    return align256(off);
}

std::uint64_t bf16_layout_bytes(const io::Pack& pack) {
    std::uint64_t off = 0;
    for (const auto* t : bf16_tensors(pack)) { off = align256(off) + t->bytes; }
    return align256(off);
}

std::uint64_t mtp_expert_bytes(const io::Pack& pack) {
    return std::uint64_t(kExperts) * pack.mtp_experts().blob_stride;
}

} // namespace

std::uint64_t DenseWeights::device_bytes(const io::Pack& pack, const WeightsOptions& options) {
    std::uint64_t bytes = dense_layout_bytes(pack);
    if (options.load_mtp_experts) { bytes += mtp_expert_bytes(pack); }
    if (options.dense_bf16) { bytes += bf16_layout_bytes(pack); }
    return bytes;
}

std::uint64_t DenseWeights::host_bytes(const io::Pack& pack) {
    return io::align_up(pack.tensor("embed").bytes, kHugePageBytes);
}

DenseWeights DenseWeights::load(const io::Pack& pack, cudaStream_t stream, const WeightsOptions& options) {
    DenseWeights w;
    if (options.dense_bf16 && !pack.has_file(io::PackFile::DenseBf16)) {
        throw io::PackError("pack: --dense bf16 requested but the pack has no dense_bf16.bin");
    }

    // Device layout: dense tensors packed in file order (256-aligned), then the MTP experts.
    std::vector<io::CopyRange> ranges;
    std::uint64_t off = 0;
    for (const auto* t : device_tensors(pack)) {
        Location loc;
        loc.data = align256(off);
        off      = loc.data + t->bytes;
        if (t->format == TensorFormat::Fp8E4m3Row) {
            loc.scales     = align256(off);
            loc.has_scales = true;
            off            = loc.scales + t->scale_bytes;
        }
        w.locations_.emplace(t->name, loc);
    }
    const std::uint64_t dense_bytes = align256(off);
    w.mtp_offset_                   = dense_bytes;
    const std::uint64_t total       = dense_bytes + (options.load_mtp_experts ? mtp_expert_bytes(pack) : 0);
    w.dense_                        = DeviceBuffer(static_cast<std::size_t>(total));
    auto* base                      = static_cast<std::byte*>(w.dense_.p);
    for (const auto* t : device_tensors(pack)) {
        const auto& loc = w.locations_.at(t->name);
        ranges.push_back({t->offset, t->bytes, base + loc.data});
        if (loc.has_scales) { ranges.push_back({t->scale_offset, t->scale_bytes, base + loc.scales}); }
    }
    {
        io::InputFile file(pack.file_path(io::PackFile::Dense));
        w.stats_.dense = io::upload_regions(file, std::move(ranges), stream, options.read_threads);
    }
    if (options.load_mtp_experts) {
        io::InputFile file(pack.file_path(io::PackFile::MtpExperts));
        const auto& mtp = pack.mtp_experts();
        w.stats_.mtp_experts =
            io::upload_regions(file, {{mtp.offset, mtp_expert_bytes(pack), base + w.mtp_offset_}}, stream, options.read_threads);
    }

    if (options.dense_bf16) {
        w.bf16_ = DeviceBuffer(static_cast<std::size_t>(bf16_layout_bytes(pack)));
        auto* bbase = static_cast<std::byte*>(w.bf16_.p);
        std::vector<io::CopyRange> bf_ranges;
        std::uint64_t boff = 0;
        for (const auto* t : bf16_tensors(pack)) {
            auto& loc    = w.locations_.at(t->name);
            loc.bf16     = align256(boff);
            loc.has_bf16 = true;
            boff         = loc.bf16 + t->bytes;
            bf_ranges.push_back({t->offset, t->bytes, bbase + loc.bf16});
        }
        io::InputFile file(pack.file_path(io::PackFile::DenseBf16));
        w.stats_.dense_bf16 = io::upload_regions(file, std::move(bf_ranges), stream, options.read_threads);
    }

    // Embedding table: pinned, device-mapped host memory (gathered zero-copy by the GPU).
    {
        const auto& embed = pack.tensor("embed");
        w.embed_          = HostRegion(static_cast<std::size_t>(embed.bytes));
        io::InputFile file(pack.file_path(io::PackFile::Dense));
        w.stats_.embedding = io::read_regions_to_host(file, {{embed.offset, embed.bytes, w.embed_.data()}}, 4);
        if (!w.embed_.register_with_cuda()) {
            throw std::runtime_error("cannot register the host embedding table with CUDA (cudaHostRegister failed)");
        }
        w.stats_.embedding_register_seconds = w.embed_.register_seconds();
        w.embedding_.host   = reinterpret_cast<const __nv_bfloat16*>(w.embed_.data());
        w.embedding_.device = static_cast<const __nv_bfloat16*>(w.embed_.device_ptr());
        w.embedding_.rows   = static_cast<std::int32_t>(embed.shape[0]);
        w.embedding_.cols   = static_cast<std::int32_t>(embed.shape[1]);
    }

    w.bind(pack);
    spdlog::info("dense weights: {:.2f} GiB VRAM (dense {:.2f} GB/s{}{}), embedding {:.2f} GiB host ({:.2f} GB/s, "
                 "register {:.2f} s)",
                 double(w.vram_bytes()) / double(1ull << 30), w.stats_.dense.gb_per_s(),
                 options.load_mtp_experts ? fmt::format(", mtp experts {:.2f} GB/s", w.stats_.mtp_experts.gb_per_s()) : "",
                 options.dense_bf16 ? fmt::format(", bf16 copies {:.2f} GB/s", w.stats_.dense_bf16.gb_per_s()) : "",
                 double(w.embed_.size()) / double(1ull << 30), w.stats_.embedding.gb_per_s(),
                 w.stats_.embedding_register_seconds);
    return w;
}

const void* DenseWeights::device_data(std::string_view name) const {
    const auto it = locations_.find(name);
    if (it == locations_.end()) { throw io::PackError("weights: no device tensor '" + std::string(name) + "'"); }
    return static_cast<const std::byte*>(dense_.p) + it->second.data;
}

const void* DenseWeights::device_scales(std::string_view name) const {
    const auto it = locations_.find(name);
    if (it == locations_.end() || !it->second.has_scales) {
        throw io::PackError("weights: no FP8 row scales for '" + std::string(name) + "'");
    }
    return static_cast<const std::byte*>(dense_.p) + it->second.scales;
}

const void* DenseWeights::device_bf16(std::string_view name) const {
    const auto it = locations_.find(name);
    if (it == locations_.end() || !it->second.has_bf16) {
        throw io::PackError("weights: no BF16 copy of '" + std::string(name) + "' (load with dense_bf16)");
    }
    return static_cast<const std::byte*>(bf16_.p) + it->second.bf16;
}

Fp8Weight DenseWeights::fp8(const io::Pack& pack, const std::string& name) const {
    const auto& t = pack.tensor(name);
    if (t.format != TensorFormat::Fp8E4m3Row) { throw io::PackError("weights: '" + name + "' is not fp8_e4m3_row"); }
    const auto& loc = locations_.at(name);
    Fp8Weight w;
    w.codes  = reinterpret_cast<const std::uint8_t*>(static_cast<const std::byte*>(dense_.p) + loc.data);
    w.scales = reinterpret_cast<const __nv_bfloat16*>(static_cast<const std::byte*>(dense_.p) + loc.scales);
    if (loc.has_bf16) {
        w.bf16 = reinterpret_cast<const __nv_bfloat16*>(static_cast<const std::byte*>(bf16_.p) + loc.bf16);
    }
    w.n = static_cast<std::int32_t>(t.shape[0]);
    w.k = static_cast<std::int32_t>(t.shape[1]);
    return w;
}

Bf16Weight DenseWeights::bf16w(const io::Pack& pack, const std::string& name) const {
    const auto& t = pack.tensor(name);
    if (t.format != TensorFormat::Bf16) { throw io::PackError("weights: '" + name + "' is not bf16"); }
    return {static_cast<const __nv_bfloat16*>(device_data(name)), static_cast<std::int32_t>(t.rows()),
            static_cast<std::int32_t>(t.cols())};
}

F32Weight DenseWeights::f32(const io::Pack& pack, const std::string& name) const {
    const auto& t = pack.tensor(name);
    if (t.format != TensorFormat::F32) { throw io::PackError("weights: '" + name + "' is not f32"); }
    return {static_cast<const float*>(device_data(name)), static_cast<std::int32_t>(t.numel())};
}

GrWeights DenseWeights::gr(const io::Pack& pack, const std::string& p) const {
    return {f32(pack, p + ".hc_norm"), fp8(pack, p + ".down"), fp8(pack, p + ".up"), bf16w(pack, p + ".inject")};
}

MixerWeights DenseWeights::mixer_w(const io::Pack& pack, const std::string& p) const {
    return {f32(pack, p + ".hc_norm"), fp8(pack, p + ".down"), fp8(pack, p + ".up")};
}

QsaWeights DenseWeights::qsa(const io::Pack& pack, const std::string& p) const {
    return {fp8(pack, p + ".qkv"),          f32(pack, p + ".q_norm"),     f32(pack, p + ".k_norm"),
            fp8(pack, p + ".o"),            bf16w(pack, p + ".idx_qk"),   f32(pack, p + ".idx_q_norm"),
            f32(pack, p + ".idx_k_norm")};
}

MoeWeights DenseWeights::moe(const io::Pack& pack, const std::string& p) const {
    return {bf16w(pack, p + ".router"), fp8(pack, p + ".shared_gate_up"), fp8(pack, p + ".shared_down")};
}

void DenseWeights::bind(const io::Pack& pack) {
    lm_head_ = fp8(pack, "lm_head");
    mixer_   = mixer_w(pack, "mixer");
    for (int i = 0; i < kLayers; ++i) {
        const std::string L = "L" + std::to_string(i);
        auto& layer         = layers_[std::size_t(i)];
        layer.qsa           = is_qsa_layer(i);
        layer.attn_gr       = gr(pack, L + ".attn_gr");
        layer.mlp_gr        = gr(pack, L + ".mlp_gr");
        if (layer.qsa) {
            layer.attn = qsa(pack, L + ".qsa");
        } else {
            layer.gdn = {fp8(pack, L + ".gdn.qkvz"),   bf16w(pack, L + ".gdn.ab"),   f32(pack, L + ".gdn.conv"),
                         f32(pack, L + ".gdn.A_log"),  f32(pack, L + ".gdn.dt_bias"), f32(pack, L + ".gdn.norm"),
                         fp8(pack, L + ".gdn.out")};
        }
        layer.moe = moe(pack, L + ".moe");
    }
    const std::string P = "L" + std::to_string(kPleLayer) + ".ple";
    ple_ = {fp8(pack, P + ".key"),       fp8(pack, P + ".value"),     f32(pack, P + ".norm_key"),
            f32(pack, P + ".norm_query"), f32(pack, P + ".norm_conv"), f32(pack, P + ".conv")};

    mtp_.fc_embedding          = fp8(pack, "mtp.fc_embedding");
    mtp_.fc_hidden             = fp8(pack, "mtp.fc_hidden");
    mtp_.pre_fc_norm_embedding = f32(pack, "mtp.pre_fc_norm_embedding");
    mtp_.pre_fc_norm_hidden    = f32(pack, "mtp.pre_fc_norm_hidden");
    mtp_.attn_gr               = gr(pack, "mtp.attn_gr");
    mtp_.mlp_gr                = gr(pack, "mtp.mlp_gr");
    mtp_.attn                  = qsa(pack, "mtp.qsa");
    mtp_.moe                   = moe(pack, "mtp.moe");
    mtp_.mixer                 = mixer_w(pack, "mtp.mixer");
    if (dense_.bytes > mtp_offset_) {
        const auto& m        = pack.mtp_experts();
        mtp_.experts.base    = reinterpret_cast<const std::uint8_t*>(static_cast<const std::byte*>(dense_.p) + mtp_offset_);
        mtp_.experts.blob_stride = m.blob_stride;
        mtp_.experts.gate_bytes  = m.gate_bytes;
        mtp_.experts.up_bytes    = m.up_bytes;
        mtp_.experts.down_bytes  = m.down_bytes;
        mtp_.experts.gu_type     = m.gu_type;
        mtp_.experts.down_type   = m.down_type;
        mtp_.experts.count       = kExperts;
    }
}

void DenseWeights::build_draft_head(const std::vector<std::int32_t>& vocab, cudaStream_t stream) {
    const auto rows = static_cast<std::int32_t>(vocab.size());
    const std::int32_t k = lm_head_.k;
    if (rows <= 0 || !std::is_sorted(vocab.begin(), vocab.end()) || vocab.front() < 0 ||
        vocab.back() >= lm_head_.n)
        throw io::PackError("draft vocabulary: ids must be ascending and inside the vocabulary");
    const std::uint64_t codes = (std::uint64_t(rows) * k + 255) & ~std::uint64_t{255};
    const std::uint64_t scales = (std::uint64_t(rows) * sizeof(__nv_bfloat16) + 255) & ~std::uint64_t{255};
    draft_ = DeviceBuffer(codes + scales + std::uint64_t(rows) * sizeof(std::int32_t));
    auto* base = static_cast<std::uint8_t*>(draft_.p);
    auto* ids = reinterpret_cast<std::int32_t*>(base + codes + scales);
    draft_.copy_from_host(vocab.data(), vocab.size() * sizeof(std::int32_t), codes + scales);
    gather_fp8_rows(lm_head_.codes, lm_head_.scales, ids, rows, k, base,
                    reinterpret_cast<__nv_bfloat16*>(base + codes), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    draft_head_ = Fp8Weight{base, reinterpret_cast<const __nv_bfloat16*>(base + codes), nullptr, rows, k};
    draft_vocab_ = ids;
}

} // namespace fngine::model
