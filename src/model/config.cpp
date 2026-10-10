#include "model/config.h"

#include "core/model_dims.h"
#include "io/pack.h"

#include <cmath>
#include <string>

namespace fngine::model {
namespace {

using nlohmann::json;
using namespace fngine::dims;

[[noreturn]] void fail(const std::string& what) { throw io::PackError("config: " + what); }

template <class T>
T get(const json& j, const char* key) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) { fail(std::string("missing '") + key + "'"); }
    try {
        return it->get<T>();
    } catch (const json::exception& e) { fail(std::string(key) + ": " + e.what()); }
}

template <class T>
T get_or(const json& j, const char* key, T fallback) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) { return fallback; }
    try {
        return it->get<T>();
    } catch (const json::exception& e) { fail(std::string(key) + ": " + e.what()); }
}

void expect(std::int64_t got, std::int64_t want, const char* what) {
    if (got != want) { fail(std::string(what) + " is " + std::to_string(got) + ", fngine requires " + std::to_string(want)); }
}

void expect_f(double got, double want, const char* what) {
    if (std::fabs(got - want) > 1e-6 * std::fabs(want)) {
        fail(std::string(what) + " is " + std::to_string(got) + ", fngine requires " + std::to_string(want));
    }
}

} // namespace

ModelConfig ModelConfig::from_pack(const io::Pack& pack) {
    const json& c = pack.config();
    ModelConfig cfg;
    cfg.hidden        = get<std::int32_t>(c, "hidden_size");
    cfg.streams       = get<std::int32_t>(c, "hc_count");
    cfg.layers        = get<std::int32_t>(c, "num_hidden_layers");
    cfg.vocab         = get<std::int32_t>(c, "vocab_size");
    cfg.rms_eps       = get<float>(c, "rms_norm_eps");
    cfg.max_positions = get<std::int32_t>(c, "max_position_embeddings");
    cfg.gr_lowrank    = get<std::int32_t>(c, "hc_lowrank");
    expect(cfg.hidden, kHidden, "hidden_size");
    expect(cfg.streams, kStreams, "hc_count");
    expect(cfg.layers, kLayers, "num_hidden_layers");
    expect(cfg.vocab, kVocab, "vocab_size");
    expect_f(cfg.rms_eps, kRmsEps, "rms_norm_eps");
    expect(cfg.gr_lowrank, kGrLowRank, "hc_lowrank");
    if (cfg.max_positions <= 0) { fail("max_position_embeddings must be positive"); }

    const auto types = get<std::vector<std::string>>(c, "layer_types");
    if (types.size() != std::size_t(kLayers)) { fail("layer_types must list 48 layers"); }
    expect(get<std::int32_t>(c, "full_attention_interval"), kAttentionInterval, "full_attention_interval");
    for (int i = 0; i < kLayers; ++i) {
        const bool qsa = types[std::size_t(i)] == "full_attention";
        if (!qsa && types[std::size_t(i)] != "linear_attention") { fail("unknown layer type '" + types[std::size_t(i)] + "'"); }
        if (qsa != is_qsa_layer(i)) { fail("layer_types[" + std::to_string(i) + "] does not follow the 4-layer pattern"); }
        cfg.qsa_layer.push_back(qsa);
    }

    cfg.gdn_qk_heads   = get<std::int32_t>(c, "linear_num_key_heads");
    cfg.gdn_v_heads    = get<std::int32_t>(c, "linear_num_value_heads");
    cfg.gdn_head_dim   = get<std::int32_t>(c, "linear_key_head_dim");
    cfg.gdn_conv_width = get<std::int32_t>(c, "linear_conv_kernel_dim");
    expect(cfg.gdn_qk_heads, kGdnQkHeads, "linear_num_key_heads");
    expect(cfg.gdn_v_heads, kGdnValueHeads, "linear_num_value_heads");
    expect(cfg.gdn_head_dim, kGdnHeadDim, "linear_key_head_dim");
    expect(get<std::int32_t>(c, "linear_value_head_dim"), kGdnHeadDim, "linear_value_head_dim");
    expect(cfg.gdn_conv_width, kGdnConvWidth, "linear_conv_kernel_dim");
    const auto output_gate = get_or<std::string>(c, "output_gate_type", "sigmoid");
    if (output_gate != "sigmoid") { fail("output_gate_type must be 'sigmoid'"); }

    cfg.qsa_heads    = get<std::int32_t>(c, "num_attention_heads");
    cfg.qsa_kv_heads = get<std::int32_t>(c, "num_key_value_heads");
    cfg.qsa_head_dim = get<std::int32_t>(c, "head_dim");
    expect(cfg.qsa_heads, kQsaHeads, "num_attention_heads");
    expect(cfg.qsa_kv_heads, kQsaKvHeads, "num_key_value_heads");
    expect(cfg.qsa_head_dim, kQsaHeadDim, "head_dim");
    const json& rope = c.contains("rope_parameters") ? c.at("rope_parameters") : c;
    cfg.rope_theta   = get<float>(rope, "rope_theta");
    const double partial = rope.contains("partial_rotary_factor") ? get<double>(rope, "partial_rotary_factor")
                                                                  : get<double>(c, "partial_rotary_factor");
    cfg.rotary_dim   = static_cast<std::int32_t>(std::lround(partial * cfg.qsa_head_dim));
    expect_f(cfg.rope_theta, kRopeTheta, "rope_theta");
    expect(cfg.rotary_dim, kQsaRotaryDim, "rotary dims (head_dim * partial_rotary_factor)");
    cfg.indexer_heads    = get<std::int32_t>(c, "indexer_n_heads");
    cfg.indexer_head_dim = get<std::int32_t>(c, "indexer_head_dim");
    cfg.indexer_block    = get<std::int32_t>(c, "indexer_compress_ratio");
    expect(cfg.indexer_heads, kIndexerHeads, "indexer_n_heads");
    expect(cfg.indexer_head_dim, kIndexerHeadDim, "indexer_head_dim");
    expect(cfg.indexer_block, kIndexerBlock, "indexer_compress_ratio");
    expect(get<std::int32_t>(c, "indexer_kv_heads"), 1, "indexer_kv_heads");
    cfg.indexer_top_blocks = get<std::int32_t>(c, "indexer_budget") / cfg.indexer_block;
    expect(cfg.indexer_top_blocks, kIndexerTopBlocks, "indexer_budget / indexer_compress_ratio");

    cfg.experts        = get<std::int32_t>(c, "num_experts");
    cfg.top_k          = get<std::int32_t>(c, "num_experts_per_tok");
    cfg.norm_topk_prob = get_or<bool>(c, "norm_topk_prob", true);
    cfg.expert_inter   = get<std::int32_t>(c, "moe_intermediate_size");
    cfg.shared_inter   = get<std::int32_t>(c, "shared_expert_intermediate_size");
    expect(cfg.experts, kExperts, "num_experts");
    expect(cfg.top_k, kTopK, "num_experts_per_tok");
    if (!cfg.norm_topk_prob) { fail("norm_topk_prob must be true"); }
    expect(cfg.expert_inter, kExpertInter, "moe_intermediate_size");
    expect(cfg.shared_inter, kSharedInter, "shared_expert_intermediate_size");

    const auto ple_ids = get<std::vector<std::int32_t>>(c, "ple_layer_ids");  // one-indexed
    if (ple_ids.size() != 1) { fail("exactly one PLE layer is supported"); }
    cfg.ple_layer = ple_ids[0] - 1;
    expect(cfg.ple_layer, kPleLayer, "ple layer (ple_layer_ids - 1)");
    cfg.ple_conv_taps = get<std::int32_t>(c, "ple_conv_kernel_size");
    expect(cfg.ple_conv_taps, kPleConvTaps, "ple_conv_kernel_size");
    expect(get<std::int32_t>(c, "ple_embed_dim"), kHidden, "ple_embed_dim");
    expect(get<std::int32_t>(c, "ngram_size"), 3, "ngram_size");

    cfg.mtp_layers = get<std::int32_t>(c, "mtp_num_hidden_layers");
    expect(cfg.mtp_layers, 1, "mtp_num_hidden_layers");
    cfg.mtp_rope_theta = cfg.rope_theta;
    if (const auto it = c.find("mtp"); it != c.end() && it->is_object()) {
        cfg.mtp_rope_theta = get_or<float>(*it, "rope_theta", cfg.rope_theta);
    }

    const json& g = pack.generation_config();
    if (const auto it = g.find("eos_token_id"); it != g.end()) {
        if (it->is_array()) {
            cfg.eos_token_ids = it->get<std::vector<std::int32_t>>();
        } else if (it->is_number_integer()) {
            cfg.eos_token_ids = {it->get<std::int32_t>()};
        }
    }
    if (cfg.eos_token_ids.empty()) { cfg.eos_token_ids = {get<std::int32_t>(c, "eos_token_id")}; }
    cfg.pad_token_id         = get_or<std::int32_t>(g, "pad_token_id", -1);
    cfg.sampling.temperature = get_or<float>(g, "temperature", 1.0f);
    cfg.sampling.top_p       = get_or<float>(g, "top_p", 1.0f);
    cfg.sampling.top_k       = get_or<std::int32_t>(g, "top_k", 0);
    return cfg;
}

} // namespace fngine::model
