#include "kernels/moe/moe.h"

namespace fngine::kernels {

int64_t moe_row_bytes(MoeFormat f, int32_t k) {
  switch (f) {
    case MoeFormat::IQ2_S: return int64_t(k) / 256 * 82;
    case MoeFormat::IQ3_XXS: return int64_t(k) / 256 * 98;
    case MoeFormat::IQ3_S: return int64_t(k) / 256 * 110;
    case MoeFormat::IQ4_XS: return int64_t(k) / 256 * 136;
    case MoeFormat::IQ4_NL: return int64_t(k) / 32 * 18;
    case MoeFormat::Q2_0: return int64_t(k) / 64 * 18;
  }
  return 0;
}

bool moe_gate_up_format_supported(MoeFormat f) {
  return f == MoeFormat::IQ2_S || f == MoeFormat::IQ3_XXS || f == MoeFormat::IQ3_S ||
         f == MoeFormat::IQ4_XS || f == MoeFormat::IQ4_NL;
}

bool moe_down_format_supported(MoeFormat f) {
  return f == MoeFormat::IQ4_NL || f == MoeFormat::Q2_0;
}

MoeExpertLayout make_moe_expert_layout(const uint8_t* slots, int64_t blob_stride,
                                       MoeFormat gate_up_format, MoeFormat down_format) {
  MoeExpertLayout l;
  l.slots = slots;
  l.blob_stride = blob_stride;
  const int64_t gate_bytes = dims::kExpertInter * moe_row_bytes(gate_up_format, dims::kHidden);
  l.gate_offset = 0;
  l.up_offset = gate_bytes;
  l.down_offset = 2 * gate_bytes;
  l.gate_up_format = gate_up_format;
  l.down_format = down_format;
  return l;
}

}  // namespace fngine::kernels
