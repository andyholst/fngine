// Loads the real pack (FNGINE_TEST_PACK): manifest validation, typed config, dense weights (+ MTP
// experts, host embedding) and expert placement with a capped VRAM pool, then checks uploaded bytes
// against the pack files. Env:
//   FNGINE_TEST_VRAM_MIB       expert pool cap (default 4096; another process may hold the GPU)
//   FNGINE_TEST_EXPERT_LAYERS  layers whose experts are loaded (default 12, bounds pinned host RAM)
//   FNGINE_TEST_DENSE_BF16=1   also upload the BF16 validation copies
#include "io/direct_io.h"
#include "io/pack.h"
#include "model/config.h"
#include "model/placement.h"
#include "model/weights.h"

#include "common/test_util.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace fngine;

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

long env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::strtol(v, nullptr, 10) : fallback;
}

std::vector<std::byte> file_bytes(const io::InputFile& file, std::uint64_t offset, std::uint64_t bytes) {
    std::vector<std::byte> out(bytes);
    file.read_exact(offset, out);
    return out;
}

std::vector<std::byte> device_bytes(const void* ptr, std::uint64_t bytes) {
    std::vector<std::byte> out(bytes);
    FNGINE_TEST_CUDA(cudaMemcpy(out.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    return out;
}

void expect_same(const std::vector<std::byte>& got, const std::vector<std::byte>& want, const char* what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: uploaded bytes differ from the pack\n", what);
        std::exit(1);
    }
    std::printf("ok %s (%zu bytes)\n", what, got.size());
}

// Device copy of [offset, offset+bytes) of a tensor region vs the dense file.
void check_region(const io::InputFile& file, const void* dev, std::uint64_t file_offset, std::uint64_t bytes,
                  const std::string& what) {
    expect_same(device_bytes(dev, bytes), file_bytes(file, file_offset, bytes), what.c_str());
}

} // namespace

int main() {
    const auto dir = test::env_or_skip("FNGINE_TEST_PACK");
    if (dir.empty()) { return test::kSkip; }
    if (test::skip_without_gpu()) { return test::kSkip; }
    const auto vram_mib   = static_cast<std::uint64_t>(env_long("FNGINE_TEST_VRAM_MIB", 4096));
    const int layers      = static_cast<int>(env_long("FNGINE_TEST_EXPERT_LAYERS", 12));
    const bool dense_bf16 = env_long("FNGINE_TEST_DENSE_BF16", 0) != 0;
    FNGINE_TEST_CUDA(cudaSetDevice(0));
    FNGINE_TEST_CUDA(cudaFree(nullptr));

    double t = now_s();
    const auto pack = io::Pack::open(dir);
    std::printf("manifest: %zu tensors, %zu bf16 copies, profile %s, %zu tokenizer files (%.3f s)\n",
                pack.tensors().size(), pack.tensors_bf16().size(), pack.has_profile() ? "yes" : "no",
                pack.tokenizer_files().size(), now_s() - t);
    FNGINE_CHECK(pack.tensors().size() == io::dense_tensor_specs().size());
    bool missing_throws = false;
    try {
        (void)pack.tensor("L0.does_not_exist");
    } catch (const io::PackError&) { missing_throws = true; }
    FNGINE_CHECK(missing_throws);
    // Abliterated packs carry an ablation section: unit directions for all 48 layers, and the
    // file must hold them exactly where the manifest says.
    if (pack.has_ablation()) {
        const auto& ab = pack.ablation();
        FNGINE_CHECK(ab.layers == 48 && ab.dim == 2560);
        std::ifstream in(ab.path, std::ios::binary);
        std::vector<float> dirs(std::size_t(ab.layers) * std::size_t(ab.dim));
        in.seekg(static_cast<std::streamoff>(ab.directions_offset));
        FNGINE_CHECK(static_cast<bool>(in.read(reinterpret_cast<char*>(dirs.data()),
                                               std::streamsize(dirs.size() * sizeof(float)))));
        double worst = 0;
        for (int l = 0; l < ab.layers; ++l) {
            double n = 0;
            for (int d = 0; d < ab.dim; ++d) {
                const double v = dirs[std::size_t(l) * std::size_t(ab.dim) + std::size_t(d)];
                n += v * v;
            }
            worst = std::max(worst, std::fabs(std::sqrt(n) - 1.0));
        }
        std::printf("ablation: %d layers x %d, scale %g, max |norm - 1| = %.2e\n", ab.layers, ab.dim,
                    double(ab.scale), worst);
        FNGINE_CHECK(worst < 1e-4);
    }
    // Serving installs omit the validation-only dense_bf16.bin although the manifest lists it: the
    // same pack (symlinked without that file) must load, with no BF16 copies.
    if (pack.has_file(io::PackFile::DenseBf16)) {
        namespace fs = std::filesystem;
        const fs::path bare = fs::temp_directory_path() / ("fngine-pack-no-bf16-" + std::to_string(::getpid()));
        fs::remove_all(bare);
        fs::create_directory(bare);
        const auto skipped = pack.file_path(io::PackFile::DenseBf16).filename();
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().filename() != skipped) {
                fs::create_symlink(fs::absolute(entry.path()), bare / entry.path().filename());
            }
        }
        const auto served = io::Pack::open(bare);
        fs::remove_all(bare);  // removes the links only
        std::printf("pack without %s: %zu tensors, %zu bf16 copies\n", skipped.c_str(), served.tensors().size(),
                    served.tensors_bf16().size());
        FNGINE_CHECK(!served.has_file(io::PackFile::DenseBf16));
        FNGINE_CHECK(served.tensors_bf16().empty());
        FNGINE_CHECK(served.tensors().size() == pack.tensors().size());
    }
    const auto cfg = model::ModelConfig::from_pack(pack);
    std::printf("config: hidden %d layers %d vocab %d experts %d top-%d eps %g theta %g rotary %d eos [",
                cfg.hidden, cfg.layers, cfg.vocab, cfg.experts, cfg.top_k, cfg.rms_eps, cfg.rope_theta, cfg.rotary_dim);
    for (auto id : cfg.eos_token_ids) { std::printf(" %d", id); }
    std::printf(" ]\n");
    FNGINE_CHECK(cfg.eos_token_ids.size() == 2 && cfg.ple_layer == 1);

    cudaStream_t stream;
    FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Dense weights.
    model::WeightsOptions wopt;
    wopt.dense_bf16 = dense_bf16;
    t               = now_s();
    const auto weights = model::DenseWeights::load(pack, stream, wopt);
    const auto& ws     = weights.stats();
    std::printf("dense load %.2f s: VRAM %.3f GiB (planned %.3f); dense %.2f GB in %.2f s = %.2f GB/s; "
                "mtp experts %.2f GB = %.2f GB/s; embedding %.2f GB host = %.2f GB/s, register %.2f s",
                now_s() - t, double(weights.vram_bytes()) / double(1ull << 30),
                double(model::DenseWeights::device_bytes(pack, wopt)) / double(1ull << 30), ws.dense.payload_bytes * 1e-9,
                ws.dense.seconds, ws.dense.gb_per_s(), ws.mtp_experts.payload_bytes * 1e-9, ws.mtp_experts.gb_per_s(),
                ws.embedding.payload_bytes * 1e-9, ws.embedding.gb_per_s(), ws.embedding_register_seconds);
    if (dense_bf16) { std::printf("; bf16 %.2f GB = %.2f GB/s", ws.dense_bf16.payload_bytes * 1e-9, ws.dense_bf16.gb_per_s()); }
    std::printf("\n");
    FNGINE_CHECK(weights.vram_bytes() == model::DenseWeights::device_bytes(pack, wopt));
    {
        io::InputFile dense(pack.file_path(io::PackFile::Dense));
        const auto& qkv = pack.tensor("L3.qsa.qkv");
        check_region(dense, weights.device_data("L3.qsa.qkv"), qkv.offset, 1 << 20, "L3.qsa.qkv codes head");
        check_region(dense, static_cast<const std::byte*>(weights.device_data("L3.qsa.qkv")) + qkv.bytes - 4096,
                     qkv.offset + qkv.bytes - 4096, 4096, "L3.qsa.qkv codes tail");
        check_region(dense, weights.device_scales("L3.qsa.qkv"), qkv.scale_offset, qkv.scale_bytes, "L3.qsa.qkv scales");
        const auto& lw = weights.layer(3);
        FNGINE_CHECK(lw.qsa && lw.attn.qkv.codes == weights.device_data("L3.qsa.qkv"));
        FNGINE_CHECK(lw.attn.qkv.n == 13312 && lw.attn.qkv.k == 2560);
        FNGINE_CHECK(!weights.layer(0).qsa && weights.layer(0).gdn.qkvz.n == 16384);
        FNGINE_CHECK((weights.layer(0).gdn.qkvz.bf16 != nullptr) == dense_bf16);
        for (const char* name : {"L0.gdn.conv", "L0.gdn.A_log", "L1.ple.norm_conv", "mtp.moe.router", "L47.mlp_gr.inject"}) {
            const auto& info = pack.tensor(name);
            check_region(dense, weights.device_data(name), info.offset, info.bytes, name);
        }
        const auto& lm = pack.tensor("lm_head");
        check_region(dense, static_cast<const std::byte*>(weights.device_data("lm_head")) + lm.bytes - (1 << 20),
                     lm.offset + lm.bytes - (1 << 20), 1 << 20, "lm_head codes tail");
        if (dense_bf16) {
            io::InputFile bf(pack.file_path(io::PackFile::DenseBf16));
            const auto& info = pack.tensor_bf16("L5.gdn.out");
            check_region(bf, weights.device_bf16("L5.gdn.out"), info.offset, info.bytes, "L5.gdn.out bf16 copy");
        }
        // Embedding: host rows and the device-mapped alias.
        const auto& embed = pack.tensor("embed");
        const auto& emb   = weights.embedding();
        FNGINE_CHECK(emb.rows == 248320 && emb.cols == 2560 && emb.device != nullptr);
        const std::uint64_t row_bytes = 2560 * 2;
        for (std::uint64_t row : {0ull, 12345ull, 248319ull}) {
            const auto want = file_bytes(dense, embed.offset + row * row_bytes, row_bytes);
            std::vector<std::byte> host(row_bytes);
            std::memcpy(host.data(), reinterpret_cast<const std::byte*>(emb.host) + row * row_bytes, row_bytes);
            expect_same(host, want, ("embed host row " + std::to_string(row)).c_str());
            expect_same(device_bytes(reinterpret_cast<const std::byte*>(emb.device) + row * row_bytes, row_bytes), want,
                        ("embed mapped row " + std::to_string(row)).c_str());
        }
        // MTP experts.
        io::InputFile mtpf(pack.file_path(io::PackFile::MtpExperts));
        const auto& mx = weights.mtp().experts;
        FNGINE_CHECK(mx.count == 512 && mx.blob_stride == pack.mtp_experts().blob_stride);
        for (int e : {0, 511}) {
            check_region(mtpf, mx.base + e * mx.blob_stride, pack.mtp_experts().blob_offset(e), pack.mtp_experts().blob_bytes,
                         "mtp expert " + std::to_string(e));
        }
    }

    // Placement as production would plan it right now (no cap, all layers; nothing allocated).
    {
        model::PlacementOptions full;
        full.preloaded_vram_bytes = weights.vram_bytes();
        try {
            const auto plan = model::plan_placement(pack, full);
            std::printf("full plan now: GPU %lld experts %.2f GiB, RAM %lld experts %.2f GiB, predicted hit %.1f%%\n",
                        (long long)plan.gpu_experts, double(plan.vram_bytes) / double(1ull << 30),
                        (long long)plan.cpu_experts, double(plan.host_bytes) / double(1ull << 30),
                        100 * plan.predicted_hit_rate);
        } catch (const model::HostMemoryError& e) { std::printf("full plan now: startup check refuses: %s\n", e.what()); }
        // The startup check refuses when the host needs exceed MemAvailable - 6 GiB.
        full.other_host_bytes = fngine::mem_available_bytes();
        bool refused          = false;
        try {
            (void)model::plan_placement(pack, full);
        } catch (const model::HostMemoryError& e) {
            refused = std::strstr(e.what(), "--ngram-cache-mib") != nullptr && std::strstr(e.what(), "--ctx") != nullptr;
        }
        FNGINE_CHECK(refused);
    }

    // Capped placement.
    model::PlacementOptions popt;
    popt.vram_cap_bytes       = vram_mib << 20;
    popt.expert_layers        = layers;
    popt.preloaded_vram_bytes = weights.vram_bytes();
    t                         = now_s();
    const auto placement      = model::ExpertPlacement::build(pack, stream, popt);
    const auto& plan          = placement.plan();
    const auto& ps            = placement.stats();
    std::printf("placement %.2f s: %s\n", now_s() - t, placement.summary().c_str());
    std::printf("expert reads: GPU %.2f GB at %.2f GB/s, RAM %.2f GB at %.2f GB/s (%.2f GB read)\n",
                ps.gpu_upload.payload_bytes * 1e-9, ps.gpu_upload.gb_per_s(), ps.host_fill.payload_bytes * 1e-9,
                ps.host_fill.gb_per_s(), ps.host_fill.read_bytes * 1e-9);
    FNGINE_CHECK(plan.gpu_experts + plan.cpu_experts == std::int64_t(layers) * 512);
    FNGINE_CHECK(plan.vram_bytes <= (vram_mib << 20));
    FNGINE_CHECK(plan.predicted_hit_rate >= 0 && plan.predicted_hit_rate <= 1);
    {
        std::vector<std::int32_t> dev(48 * 512);
        FNGINE_TEST_CUDA(cudaMemcpy(dev.data(), placement.slot_of_device(), dev.size() * 4, cudaMemcpyDeviceToHost));
        FNGINE_CHECK(dev == plan.slot_of);
    }
    io::InputFile experts(pack.file_path(io::PackFile::Experts));
    int checked_gpu = 0, checked_cpu = 0;
    for (int l = 0; l < layers; ++l) {
        const auto& layout = pack.expert_layers()[std::size_t(l)];
        for (int e = 0; e < 512; ++e) {
            const bool gpu = placement.slot_of(l, e) >= 0;
            if (gpu && placement.host_blob(l, e) != nullptr) { FNGINE_CHECK(false); }
            const bool first_gpu = gpu && checked_gpu == 0;
            const bool last_layer_gpu = gpu && l == layers - 1 && checked_gpu == 1;
            if (first_gpu || last_layer_gpu) {
                check_region(experts, placement.device_blob(l, e), layout.blob_offset(e), layout.blob_bytes,
                             "GPU expert L" + std::to_string(l) + " E" + std::to_string(e));
                ++checked_gpu;
            }
            if (!gpu && checked_cpu < 2 && (checked_cpu == 0 || l == layers - 1)) {
                const auto* host = placement.host_blob(l, e);
                FNGINE_CHECK(host != nullptr);
                std::vector<std::byte> got(host, host + layout.blob_bytes);
                expect_same(got, file_bytes(experts, layout.blob_offset(e), layout.blob_bytes),
                            ("RAM expert L" + std::to_string(l) + " E" + std::to_string(e)).c_str());
                ++checked_cpu;
            }
        }
    }
    FNGINE_CHECK(checked_gpu >= 1 && checked_cpu >= 1);
    FNGINE_TEST_CUDA(cudaStreamDestroy(stream));
    std::printf("PASS pack load\n");
    return 0;
}
