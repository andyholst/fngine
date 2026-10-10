#include "io/pack.h"

#include "core/model_dims.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <utility>

namespace fngine::io {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace fngine::dims;

constexpr std::uint64_t kAlign = 4096;

[[noreturn]] void fail(const std::string& what) { throw PackError("pack: " + what); }

const json& field(const json& object, const char* key, const std::string& where) {
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { fail(where + ": missing field '" + key + "'"); }
    return *it;
}

template <class T>
T get(const json& object, const char* key, const std::string& where) {
    try {
        return field(object, key, where).get<T>();
    } catch (const json::exception& e) {
        fail(where + "." + key + ": " + e.what());
    }
}

void expect_eq(std::int64_t got, std::int64_t want, const std::string& what) {
    if (got != want) { fail(what + " is " + std::to_string(got) + ", expected " + std::to_string(want)); }
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) { s += (i ? "," : "") + std::to_string(shape[i]); }
    return s + "]";
}

TensorFormat parse_format(const std::string& text, const std::string& where) {
    if (text == "fp8_e4m3_row") { return TensorFormat::Fp8E4m3Row; }
    if (text == "bf16") { return TensorFormat::Bf16; }
    if (text == "f32") { return TensorFormat::F32; }
    fail(where + ": unknown tensor format '" + text + "'");
}

std::uint64_t element_bytes(TensorFormat f) {
    switch (f) {
    case TensorFormat::Fp8E4m3Row: return 1;
    case TensorFormat::Bf16: return 2;
    case TensorFormat::F32: return 4;
    }
    return 0;
}

GgmlType parse_ggml_type(std::int64_t id, const std::string& name, const std::string& where) {
    switch (id) {
    case 18: case 20: case 21: case 22: case 23: case 42: break;
    default: fail(where + ": unsupported ggml type id " + std::to_string(id));
    }
    const auto type = static_cast<GgmlType>(id);
    if (name != to_string(type)) {
        fail(where + ": type name '" + name + "' does not match type id " + std::to_string(id));
    }
    return type;
}

std::vector<std::int64_t> parse_shape(const json& j, const std::string& where) {
    std::vector<std::int64_t> shape;
    try {
        shape = j.get<std::vector<std::int64_t>>();
    } catch (const json::exception& e) { fail(where + ".shape: " + e.what()); }
    if (shape.empty()) { fail(where + ": empty shape"); }
    for (auto d : shape) {
        if (d <= 0) { fail(where + ": non-positive dimension in shape " + shape_string(shape)); }
    }
    return shape;
}

void add_gr(std::vector<TensorSpec>& s, const std::string& p) {
    s.push_back({p + ".hc_norm", TensorFormat::F32, {kStreamHidden}});
    s.push_back({p + ".down", TensorFormat::Fp8E4m3Row, {kGrLowRank, kStreamHidden}});
    s.push_back({p + ".up", TensorFormat::Fp8E4m3Row, {kStreamHidden, kGrLowRankPadded}});
    s.push_back({p + ".inject", TensorFormat::Bf16, {kStreams, kStreamHidden}});
}

void add_mixer(std::vector<TensorSpec>& s, const std::string& p) {
    s.push_back({p + ".hc_norm", TensorFormat::F32, {kStreamHidden}});
    s.push_back({p + ".down", TensorFormat::Fp8E4m3Row, {kGrLowRank, kStreamHidden}});
    s.push_back({p + ".up", TensorFormat::Fp8E4m3Row, {kStreamHidden, kGrLowRankPadded}});
}

void add_qsa(std::vector<TensorSpec>& s, const std::string& p) {
    s.push_back({p + ".qkv", TensorFormat::Fp8E4m3Row, {kQsaQkvRows, kHidden}});
    s.push_back({p + ".q_norm", TensorFormat::F32, {kQsaHeadDim}});
    s.push_back({p + ".k_norm", TensorFormat::F32, {kQsaHeadDim}});
    s.push_back({p + ".o", TensorFormat::Fp8E4m3Row, {kHidden, kQsaAttnOut}});
    s.push_back({p + ".idx_qk", TensorFormat::Bf16, {kIndexerRows, kHidden}});
    s.push_back({p + ".idx_q_norm", TensorFormat::F32, {kIndexerHeadDim}});
    s.push_back({p + ".idx_k_norm", TensorFormat::F32, {kIndexerHeadDim}});
}

void add_moe(std::vector<TensorSpec>& s, const std::string& p) {
    s.push_back({p + ".router", TensorFormat::Bf16, {kRouterRows, kHidden}});
    s.push_back({p + ".shared_gate_up", TensorFormat::Fp8E4m3Row, {2 * kSharedInter, kHidden}});
    s.push_back({p + ".shared_down", TensorFormat::Fp8E4m3Row, {kHidden, kSharedInter}});
}

std::uint64_t file_size_or_zero(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    return ec ? 0 : size;
}

struct Region {
    std::uint64_t begin, end;
    std::string what;
};

void check_disjoint(std::vector<Region>& regions, const std::string& file) {
    std::sort(regions.begin(), regions.end(), [](const Region& a, const Region& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < regions.size(); ++i) {
        if (regions[i].begin < regions[i - 1].end) {
            fail(file + ": regions of " + regions[i - 1].what + " and " + regions[i].what + " overlap");
        }
    }
}

} // namespace

const char* to_string(TensorFormat format) noexcept {
    switch (format) {
    case TensorFormat::Fp8E4m3Row: return "fp8_e4m3_row";
    case TensorFormat::Bf16: return "bf16";
    case TensorFormat::F32: return "f32";
    }
    return "?";
}

const char* to_string(GgmlType type) noexcept {
    switch (type) {
    case GgmlType::IQ3_XXS: return "IQ3_XXS";
    case GgmlType::IQ4_NL: return "IQ4_NL";
    case GgmlType::IQ3_S: return "IQ3_S";
    case GgmlType::IQ2_S: return "IQ2_S";
    case GgmlType::IQ4_XS: return "IQ4_XS";
    case GgmlType::Q2_0: return "Q2_0";
    }
    return "?";
}

std::uint64_t ggml_row_bytes(GgmlType type, std::int64_t elements) {
    std::int64_t block = 256, bytes = 0;
    switch (type) {
    case GgmlType::IQ2_S: bytes = 82; break;
    case GgmlType::IQ3_XXS: bytes = 98; break;
    case GgmlType::IQ3_S: bytes = 110; break;
    case GgmlType::IQ4_XS: bytes = 136; break;
    case GgmlType::IQ4_NL: block = 32; bytes = 18; break;
    case GgmlType::Q2_0: block = 64; bytes = 18; break;
    }
    if (elements % block) {
        fail(std::string("row of ") + std::to_string(elements) + " elements is not a multiple of the " +
             to_string(type) + " block");
    }
    return std::uint64_t(elements / block * bytes);
}

std::int64_t TensorInfo::numel() const noexcept {
    std::int64_t n = 1;
    for (auto d : shape) { n *= d; }
    return n;
}

std::vector<TensorSpec> dense_tensor_specs() {
    std::vector<TensorSpec> s;
    s.push_back({"embed", TensorFormat::Bf16, {kVocab, kHidden}});
    s.push_back({"lm_head", TensorFormat::Fp8E4m3Row, {kVocab, kHidden}});
    add_mixer(s, "mixer");
    for (int i = 0; i < kLayers; ++i) {
        const std::string L = "L" + std::to_string(i);
        add_gr(s, L + ".attn_gr");
        add_gr(s, L + ".mlp_gr");
        if (is_qsa_layer(i)) {
            add_qsa(s, L + ".qsa");
        } else {
            s.push_back({L + ".gdn.qkvz", TensorFormat::Fp8E4m3Row, {kGdnQkvzRows, kHidden}});
            s.push_back({L + ".gdn.ab", TensorFormat::Bf16, {2 * kGdnValueHeads, kHidden}});
            s.push_back({L + ".gdn.conv", TensorFormat::F32, {kGdnConvChannels, kGdnConvWidth}});
            s.push_back({L + ".gdn.A_log", TensorFormat::F32, {kGdnValueHeads}});
            s.push_back({L + ".gdn.dt_bias", TensorFormat::F32, {kGdnValueHeads}});
            s.push_back({L + ".gdn.norm", TensorFormat::F32, {kGdnHeadDim}});
            s.push_back({L + ".gdn.out", TensorFormat::Fp8E4m3Row, {kHidden, kGdnValueDim}});
        }
        add_moe(s, L + ".moe");
        if (i == kPleLayer) {
            s.push_back({L + ".ple.key", TensorFormat::Fp8E4m3Row, {kStreamHidden, kHidden}});
            s.push_back({L + ".ple.value", TensorFormat::Fp8E4m3Row, {kHidden, kHidden}});
            s.push_back({L + ".ple.norm_key", TensorFormat::F32, {kStreamHidden}});
            s.push_back({L + ".ple.norm_query", TensorFormat::F32, {kStreamHidden}});
            s.push_back({L + ".ple.norm_conv", TensorFormat::F32, {kStreamHidden}});
            s.push_back({L + ".ple.conv", TensorFormat::F32, {kStreamHidden, kPleConvTaps}});
        }
    }
    s.push_back({"mtp.fc_embedding", TensorFormat::Fp8E4m3Row, {kHidden, kHidden}});
    s.push_back({"mtp.fc_hidden", TensorFormat::Fp8E4m3Row, {kHidden, kHidden}});
    s.push_back({"mtp.pre_fc_norm_embedding", TensorFormat::F32, {kHidden}});
    s.push_back({"mtp.pre_fc_norm_hidden", TensorFormat::F32, {kStreamHidden}});
    add_gr(s, "mtp.attn_gr");
    add_gr(s, "mtp.mlp_gr");
    add_qsa(s, "mtp.qsa");
    add_moe(s, "mtp.moe");
    add_mixer(s, "mtp.mixer");
    return s;
}

Pack Pack::open(const fs::path& dir) {
    Pack pack;
    pack.dir_ = fs::absolute(dir);
    const auto manifest_path = pack.dir_ / "manifest.json";
    {
        std::ifstream in(manifest_path);
        if (!in) { fail("cannot open " + manifest_path.string()); }
        try {
            pack.manifest_ = json::parse(in);
        } catch (const json::exception& e) { fail(manifest_path.string() + ": " + e.what()); }
    }
    const json& m = pack.manifest_;
    if (get<std::string>(m, "format", "manifest") != "fngine-pack") { fail("manifest.format is not 'fngine-pack'"); }
    expect_eq(get<std::int64_t>(m, "version", "manifest"), 1, "manifest.version");

    // Config: the dimensions the structural checks below rely on.
    const json& cfg = field(m, "config", "manifest");
    expect_eq(get<std::int64_t>(cfg, "hidden_size", "config"), kHidden, "config.hidden_size");
    expect_eq(get<std::int64_t>(cfg, "num_hidden_layers", "config"), kLayers, "config.num_hidden_layers");
    expect_eq(get<std::int64_t>(cfg, "vocab_size", "config"), kVocab, "config.vocab_size");
    expect_eq(get<std::int64_t>(cfg, "num_experts", "config"), kExperts, "config.num_experts");
    expect_eq(get<std::int64_t>(cfg, "moe_intermediate_size", "config"), kExpertInter, "config.moe_intermediate_size");
    field(m, "generation_config", "manifest");

    // Files.
    const json& files = field(m, "files", "manifest");
    const std::pair<PackFile, const char*> keys[] = {{PackFile::Dense, "dense"},
                                                     {PackFile::DenseBf16, "dense_bf16"},
                                                     {PackFile::Experts, "experts"},
                                                     {PackFile::MtpExperts, "mtp_experts"},
                                                     {PackFile::Ngram, "ngram"}};
    for (const auto& [file, key] : keys) {
        const auto it = files.find(key);
        // dense_bf16.bin is validation-only (--dense bf16): serving installs may omit the file even
        // when the manifest lists it, and loading then proceeds without BF16 copies.
        if (it == files.end() || it->is_null()) {
            if (file == PackFile::DenseBf16) { continue; }
            fail(std::string("files.") + key + " is missing");
        }
        const auto name = it->get<std::string>();
        const auto path = pack.dir_ / name;
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) {
            if (file == PackFile::DenseBf16) { continue; }
            fail(std::string("files.") + key + ": " + path.string() + " does not exist");
        }
        pack.files_[file]      = name;
        pack.file_sizes_[file] = file_size_or_zero(path);
    }

    // Dense tensors.
    const auto specs = dense_tensor_specs();
    const json& tensors = field(m, "tensors", "manifest");
    std::vector<Region> dense_regions;
    for (const auto& [name, t] : tensors.items()) {
        const std::string where = "tensors." + name;
        TensorInfo info;
        info.name = name;
        if (get<std::string>(t, "file", where) != "dense") { fail(where + ".file is not 'dense'"); }
        info.file   = PackFile::Dense;
        info.format = parse_format(get<std::string>(t, "format", where), where);
        info.shape  = parse_shape(field(t, "shape", where), where);
        info.offset = get<std::uint64_t>(t, "offset", where);
        info.bytes  = get<std::uint64_t>(t, "bytes", where);
        if (info.bytes != std::uint64_t(info.numel()) * element_bytes(info.format)) {
            fail(where + ": bytes " + std::to_string(info.bytes) + " do not match shape " + shape_string(info.shape) +
                 " in " + to_string(info.format));
        }
        if (info.offset % kAlign) { fail(where + ": offset is not 4096-aligned"); }
        if (info.offset + info.bytes > pack.file_sizes_.at(PackFile::Dense)) { fail(where + ": exceeds dense file"); }
        dense_regions.push_back({info.offset, info.offset + info.bytes, name});
        if (info.format == TensorFormat::Fp8E4m3Row) {
            info.scale_offset = get<std::uint64_t>(t, "scale_offset", where);
            info.scale_bytes  = get<std::uint64_t>(t, "scale_bytes", where);
            if (info.shape.size() != 2) { fail(where + ": fp8_e4m3_row tensor must be 2-D"); }
            if (info.scale_bytes != 2 * std::uint64_t(info.shape[0])) { fail(where + ": scale_bytes != 2*rows"); }
            if (info.scale_offset % kAlign) { fail(where + ": scale_offset is not 4096-aligned"); }
            if (info.scale_offset + info.scale_bytes > pack.file_sizes_.at(PackFile::Dense)) {
                fail(where + ": scales exceed dense file");
            }
            dense_regions.push_back({info.scale_offset, info.scale_offset + info.scale_bytes, name + ".scales"});
        }
        pack.tensors_.emplace(name, std::move(info));
    }
    check_disjoint(dense_regions, "dense.bin");
    std::set<std::string, std::less<>> expected;
    for (const auto& spec : specs) {
        expected.insert(spec.name);
        const auto* info = pack.find_tensor(spec.name);
        if (!info) { fail("tensor '" + spec.name + "' is missing"); }
        if (info->format != spec.format || info->shape != spec.shape) {
            fail("tensor '" + spec.name + "' is " + to_string(info->format) + " " + shape_string(info->shape) +
                 ", expected " + to_string(spec.format) + " " + shape_string(spec.shape));
        }
    }
    for (const auto& [name, info] : pack.tensors_) {
        if (!expected.contains(name)) { fail("unexpected tensor '" + name + "'"); }
    }

    // BF16 copies of FP8 tensors (validation mode).
    if (pack.has_file(PackFile::DenseBf16)) {
        const json& bf = field(m, "tensors_bf16", "manifest");
        std::vector<Region> regions;
        for (const auto& [name, t] : bf.items()) {
            const std::string where = "tensors_bf16." + name;
            const auto* src         = pack.find_tensor(name);
            if (!src || src->format != TensorFormat::Fp8E4m3Row) { fail(where + ": not an fp8_e4m3_row tensor"); }
            TensorInfo info;
            info.name = name;
            if (get<std::string>(t, "file", where) != "dense_bf16") { fail(where + ".file is not 'dense_bf16'"); }
            info.file   = PackFile::DenseBf16;
            info.format = parse_format(get<std::string>(t, "format", where), where);
            info.shape  = parse_shape(field(t, "shape", where), where);
            info.offset = get<std::uint64_t>(t, "offset", where);
            info.bytes  = get<std::uint64_t>(t, "bytes", where);
            if (info.format != TensorFormat::Bf16 || info.shape != src->shape ||
                info.bytes != std::uint64_t(info.numel()) * 2) {
                fail(where + ": must be bf16 " + shape_string(src->shape));
            }
            if (info.offset % kAlign) { fail(where + ": offset is not 4096-aligned"); }
            if (info.offset + info.bytes > pack.file_sizes_.at(PackFile::DenseBf16)) { fail(where + ": exceeds dense_bf16 file"); }
            regions.push_back({info.offset, info.offset + info.bytes, name});
            pack.tensors_bf16_.emplace(name, std::move(info));
        }
        check_disjoint(regions, "dense_bf16.bin");
        for (const auto& [name, info] : pack.tensors_) {
            if (info.format == TensorFormat::Fp8E4m3Row && !pack.tensors_bf16_.contains(name)) {
                fail("tensors_bf16 lacks '" + name + "'");
            }
        }
    }

    // Routed experts.
    const json& experts = field(m, "experts", "manifest");
    expect_eq(get<std::int64_t>(experts, "n_layers", "experts"), kLayers, "experts.n_layers");
    expect_eq(get<std::int64_t>(experts, "n_experts", "experts"), kExperts, "experts.n_experts");
    expect_eq(get<std::int64_t>(experts, "hidden", "experts"), kHidden, "experts.hidden");
    expect_eq(get<std::int64_t>(experts, "intermediate", "experts"), kExpertInter, "experts.intermediate");
    const json& layers = field(experts, "layers", "experts");
    if (!layers.is_array() || layers.size() != std::size_t(kLayers)) { fail("experts.layers must have 48 entries"); }
    auto parse_layout = [](const json& j, const std::string& where, GgmlType gu, GgmlType down) {
        ExpertLayout e;
        e.gu_type     = gu;
        e.down_type   = down;
        e.gate_bytes  = get<std::uint64_t>(j, "gate_bytes", where);
        e.up_bytes    = get<std::uint64_t>(j, "up_bytes", where);
        e.down_bytes  = get<std::uint64_t>(j, "down_bytes", where);
        e.blob_bytes  = get<std::uint64_t>(j, "blob_bytes", where);
        e.blob_stride = get<std::uint64_t>(j, "blob_stride", where);
        e.offset      = get<std::uint64_t>(j, "offset", where);
        const auto gu_bytes = std::uint64_t(kExpertInter) * ggml_row_bytes(gu, kHidden);
        const auto dn_bytes = std::uint64_t(kHidden) * ggml_row_bytes(down, kExpertInter);
        if (e.gate_bytes != gu_bytes || e.up_bytes != gu_bytes) { fail(where + ": gate/up bytes do not match " + to_string(gu)); }
        if (e.down_bytes != dn_bytes) { fail(where + ": down bytes do not match " + to_string(down)); }
        if (e.blob_bytes != e.gate_bytes + e.up_bytes + e.down_bytes) { fail(where + ": blob_bytes != gate+up+down"); }
        if (e.blob_stride < e.blob_bytes || e.blob_stride % kAlign) { fail(where + ": invalid blob_stride"); }
        if (e.offset % kAlign) { fail(where + ": offset is not 4096-aligned"); }
        return e;
    };
    std::uint64_t next_offset = 0;
    for (int l = 0; l < kLayers; ++l) {
        const std::string where = "experts.layers[" + std::to_string(l) + "]";
        const json& j           = layers[std::size_t(l)];
        const auto gu   = parse_ggml_type(get<std::int64_t>(j, "gu_type_id", where), get<std::string>(j, "gu_type", where), where);
        const auto down = parse_ggml_type(get<std::int64_t>(j, "down_type_id", where), get<std::string>(j, "down_type", where), where);
        if (gu == GgmlType::IQ4_NL || gu == GgmlType::Q2_0) { fail(where + ": unsupported gate/up type " + to_string(gu)); }
        if (down != GgmlType::IQ4_NL && down != GgmlType::Q2_0) { fail(where + ": unsupported down type " + to_string(down)); }
        auto e = parse_layout(j, where, gu, down);
        if (e.offset != next_offset) { fail(where + ": offset is not contiguous with the previous layer"); }
        next_offset = e.offset + std::uint64_t(kExperts) * e.blob_stride;
        pack.expert_layers_.push_back(e);
    }
    if (next_offset > pack.file_sizes_.at(PackFile::Experts)) { fail("experts.bin is shorter than its layout"); }

    {
        const json& j = field(m, "mtp_experts", "manifest");
        expect_eq(get<std::int64_t>(j, "n_experts", "mtp_experts"), kExperts, "mtp_experts.n_experts");
        if (get<std::string>(j, "gu_type", "mtp_experts") != "IQ4_NL" ||
            get<std::string>(j, "down_type", "mtp_experts") != "IQ4_NL") {
            fail("mtp_experts must be IQ4_NL");
        }
        pack.mtp_experts_ = parse_layout(j, "mtp_experts", GgmlType::IQ4_NL, GgmlType::IQ4_NL);
        if (pack.mtp_experts_.blob_offset(kExperts) > pack.file_sizes_.at(PackFile::MtpExperts)) {
            fail("mtp_experts.bin is shorter than its layout");
        }
    }

    // N-gram table.
    {
        const json& j = field(m, "ngram", "manifest");
        auto& n       = pack.ngram_;
        n.rows            = get<std::uint64_t>(j, "rows", "ngram");
        n.row_bytes       = get<std::uint32_t>(j, "row_bytes", "ngram");
        n.rows_per_page   = get<std::uint32_t>(j, "rows_per_page", "ngram");
        n.header_bytes    = get<std::uint32_t>(j, "header_bytes", "ngram");
        n.dim             = get<std::uint32_t>(j, "dim", "ngram");
        n.layer           = get<std::int32_t>(j, "layer", "ngram");
        n.ngram_size      = get<std::int32_t>(j, "ngram_size", "ngram");
        n.heads_per_ngram = get<std::int32_t>(j, "heads_per_ngram", "ngram");
        n.n_heads         = get<std::int32_t>(j, "n_heads", "ngram");
        n.eos             = get<std::int32_t>(j, "eos", "ngram");
        if (get<std::string>(j, "type", "ngram") != "IQ4_NL") { fail("ngram.type must be IQ4_NL"); }
        expect_eq(n.row_bytes, kPleRowBytes, "ngram.row_bytes");
        expect_eq(n.dim, kPleHeadDim, "ngram.dim");
        expect_eq(n.rows_per_page, kAlign / kPleRowBytes, "ngram.rows_per_page");
        expect_eq(n.header_bytes, kAlign, "ngram.header_bytes");
        expect_eq(n.layer, kPleLayer, "ngram.layer");
        expect_eq(n.ngram_size, 3, "ngram.ngram_size");
        expect_eq(n.n_heads, kPleHeads, "ngram.n_heads");
        expect_eq(n.heads_per_ngram * (n.ngram_size - 1), kPleHeads, "ngram.heads_per_ngram*(ngram_size-1)");
        expect_eq(n.eos, kEosToken, "ngram.eos");
        const auto mult    = get<std::vector<std::uint64_t>>(j, "multipliers", "ngram");
        const auto sizes   = get<std::vector<std::uint64_t>>(j, "head_sizes", "ngram");
        const auto offsets = get<std::vector<std::uint64_t>>(j, "head_offsets", "ngram");
        if (mult.size() != 3 || sizes.size() != 16 || offsets.size() != 16) { fail("ngram: expected 3 multipliers and 16 heads"); }
        std::copy(mult.begin(), mult.end(), n.multipliers.begin());
        std::copy(sizes.begin(), sizes.end(), n.head_sizes.begin());
        std::copy(offsets.begin(), offsets.end(), n.head_offsets.begin());
        std::uint64_t acc = 0;
        for (int h = 0; h < 16; ++h) {
            if (n.head_offsets[h] != acc) { fail("ngram.head_offsets[" + std::to_string(h) + "] is not the prefix sum of head_sizes"); }
            acc += n.head_sizes[h];
        }
        if (acc > n.rows) { fail("ngram: heads exceed the table rows"); }
        for (auto mlt : n.multipliers) {
            // id * m must stay below 2^63 for every token id (the model's hash relies on it).
            if (mlt == 0 || mlt > (std::uint64_t(1) << 63) / std::uint64_t(kVocab)) { fail("ngram: multiplier out of range"); }
        }
        const auto pages = (n.rows + n.rows_per_page - 1) / n.rows_per_page;
        if (n.header_bytes + pages * kAlign > pack.file_sizes_.at(PackFile::Ngram)) { fail("ngram.bin is shorter than its rows"); }
        {
            std::ifstream in(pack.file_path(PackFile::Ngram), std::ios::binary);
            char header[32] = {};
            if (!in.read(header, sizeof(header))) { fail("ngram.bin: cannot read header"); }
            std::uint32_t version = 0, row_bytes = 0, rows_per_page = 0;
            std::uint64_t rows = 0;
            std::memcpy(&version, header + 4, 4);
            std::memcpy(&rows, header + 8, 8);
            std::memcpy(&row_bytes, header + 16, 4);
            std::memcpy(&rows_per_page, header + 20, 4);
            if (std::memcmp(header, "FNGR", 4) != 0 || version != 1 || rows != n.rows || row_bytes != n.row_bytes ||
                rows_per_page != n.rows_per_page) {
                fail("ngram.bin header does not match the manifest");
            }
        }
    }

    // Profile.
    {
        const auto it = m.find("profile");
        if (it != m.end() && !it->is_null()) {
            const auto name = get<std::string>(*it, "file", "profile");
            const auto path = pack.dir_ / name;
            const std::uint64_t want = std::uint64_t(kLayers) * kExperts * 8;
            if (file_size_or_zero(path) != want) { fail("profile " + path.string() + " must be " + std::to_string(want) + " bytes"); }
            std::ifstream in(path, std::ios::binary);
            pack.profile_.resize(std::size_t(kLayers) * kExperts);
            if (!in.read(reinterpret_cast<char*>(pack.profile_.data()), std::streamsize(want))) {
                fail("cannot read " + path.string());
            }
        }
    }

    // Abliteration (optional).
    {
        const auto it = m.find("ablation");
        if (it != m.end() && !it->is_null()) {
            auto& a = pack.ablation_;
            const std::string name = get<std::string>(*it, "file", "ablation");
            a.path = pack.dir_ / name;
            a.layers = static_cast<std::int32_t>(get<std::int64_t>(*it, "layers", "ablation"));
            a.dim    = static_cast<std::int32_t>(get<std::int64_t>(*it, "dim", "ablation"));
            expect_eq(a.layers, kLayers, "ablation.layers");
            expect_eq(a.dim, kHidden, "ablation.dim");
            a.directions_offset = static_cast<std::uint64_t>(get<std::int64_t>(*it, "directions_offset", "ablation"));
            const std::uint64_t dir_bytes = std::uint64_t(a.layers) * std::uint64_t(a.dim) * 4;
            if (a.directions_offset % kAlign) { fail("ablation.directions_offset is not 4096-aligned"); }
            const std::uint64_t size = file_size_or_zero(a.path);
            if (size < a.directions_offset + dir_bytes) {
                fail("ablation " + a.path.string() + " is shorter than its directions");
            }
            if (const auto s = it->find("scales_offset"); s != it->end() && !s->is_null()) {
                a.scales_offset = static_cast<std::uint64_t>(get<std::int64_t>(*it, "scales_offset", "ablation"));
                a.scales_bytes  = std::uint64_t(a.layers) * 4;
                if (size < a.scales_offset + a.scales_bytes) {
                    fail("ablation " + a.path.string() + " is shorter than its scales");
                }
            }
            a.scale = 1.0f;
            if (const auto sc = it->find("scale"); sc != it->end() && !sc->is_null()) {
                a.scale = sc->get<float>();
            }
        }
    }

    // Tokenizer.
    for (const auto& entry : field(m, "tokenizer", "manifest")) {
        const auto path = pack.dir_ / entry.get<std::string>();
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) { fail("tokenizer file " + path.string() + " does not exist"); }
        pack.tokenizer_.push_back(path);
    }
    if (pack.tokenizer_file("tokenizer.json").empty()) { fail("tokenizer list lacks tokenizer.json"); }
    return pack;
}

fs::path Pack::file_path(PackFile file) const {
    const auto it = files_.find(file);
    if (it == files_.end()) { fail("pack has no file of the requested kind"); }
    return dir_ / it->second;
}

std::uint64_t Pack::file_bytes(PackFile file) const {
    const auto it = file_sizes_.find(file);
    return it == file_sizes_.end() ? 0 : it->second;
}

bool Pack::has_file(PackFile file) const { return files_.contains(file); }

const TensorInfo* Pack::find_tensor(std::string_view name) const noexcept {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorInfo& Pack::tensor(std::string_view name) const {
    const auto* info = find_tensor(name);
    if (!info) { fail("tensor '" + std::string(name) + "' not found"); }
    return *info;
}

const TensorInfo& Pack::tensor_bf16(std::string_view name) const {
    const auto it = tensors_bf16_.find(name);
    if (it == tensors_bf16_.end()) { fail("bf16 copy of tensor '" + std::string(name) + "' not found"); }
    return it->second;
}

fs::path Pack::tokenizer_file(std::string_view basename) const {
    for (const auto& p : tokenizer_) {
        if (p.filename() == basename) { return p; }
    }
    return {};
}

} // namespace fngine::io
