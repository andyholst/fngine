#pragma once

// Typed model configuration read from the pack manifest and checked against fngine::dims.
#include <cstdint>
#include <vector>

namespace fngine::io {
class Pack;
}

namespace fngine::model {

struct SamplingDefaults {
    float temperature = 1.0f;
    float top_p       = 1.0f;
    std::int32_t top_k = 0;
};

struct ModelConfig {
    std::int32_t hidden        = 0;
    std::int32_t streams       = 0;  // hc_count
    std::int32_t layers        = 0;
    std::int32_t vocab         = 0;
    float rms_eps              = 0;
    std::int32_t max_positions = 0;
    std::vector<bool> qsa_layer;  // per layer: full (QSA) attention vs GDN

    std::int32_t gr_lowrank = 0;

    std::int32_t gdn_qk_heads   = 0;
    std::int32_t gdn_v_heads    = 0;
    std::int32_t gdn_head_dim   = 0;
    std::int32_t gdn_conv_width = 0;

    std::int32_t qsa_heads      = 0;
    std::int32_t qsa_kv_heads   = 0;
    std::int32_t qsa_head_dim   = 0;
    std::int32_t rotary_dim     = 0;  // head_dim * partial_rotary_factor
    float rope_theta            = 0;
    std::int32_t indexer_heads    = 0;
    std::int32_t indexer_head_dim = 0;
    std::int32_t indexer_block    = 0;  // compress ratio
    std::int32_t indexer_top_blocks = 0;  // budget / compress ratio

    std::int32_t experts      = 0;
    std::int32_t top_k        = 0;
    bool norm_topk_prob       = true;
    std::int32_t expert_inter = 0;
    std::int32_t shared_inter = 0;

    std::int32_t ple_layer       = 0;  // zero-based decoder layer index
    std::int32_t ple_conv_taps   = 0;
    std::int32_t mtp_layers      = 0;
    float mtp_rope_theta         = 0;

    std::vector<std::int32_t> eos_token_ids;  // generation stop ids
    std::int32_t pad_token_id = -1;
    SamplingDefaults sampling;  // generation_config defaults (thinking mode)

    // Parses manifest config + generation_config; throws io::PackError on any value that
    // disagrees with fngine::dims.
    static ModelConfig from_pack(const io::Pack& pack);
};

} // namespace fngine::model
