// qsa_qk_norm_rope vs an FP64 oracle at model shapes (24 query + 2 key heads of 256, rotary 64).
#include "kernels/qsa/qsa.h"
#include "kernels/qsa/qsa_ref.h"

#include <cstdio>
#include <vector>

using namespace fngine::test;
using namespace fngine::test::qsa_ref;

static bool run_case(int tokens, int first_position, uint64_t seed) {
    Rng rng(seed);
    auto qkv = random_bf16(rng, size_t(tokens) * kQkvLd, 2.5);
    std::vector<float> q_norm(kD), k_norm(kD);
    for (auto& w : q_norm) w = static_cast<float>(rng.uniform(0.3, 1.8));
    for (auto& w : k_norm) w = static_cast<float>(rng.uniform(0.3, 1.8));
    std::vector<int32_t> positions(tokens);
    for (int t = 0; t < tokens; ++t) positions[t] = first_position + t;

    DeviceArray<__nv_bfloat16> d_qkv(qkv);
    DeviceArray<float> d_qn(q_norm), d_kn(k_norm);
    DeviceArray<int32_t> d_pos(positions);
    DeviceArray<__nv_bfloat16> d_q(size_t(tokens) * kHeads * kD), d_k(size_t(tokens) * kKvHeads * kD);
    fngine::kernels::qsa_qk_norm_rope(d_qkv.get(), d_qn.get(), d_kn.get(), d_pos.get(), tokens,
                                      d_q.get(), d_k.get(), nullptr);
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto q = to_float(d_q.download());
    const auto k = to_float(d_k.download());
    const auto x = to_float(qkv);

    std::vector<float> got;
    std::vector<double> ref, tol;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < kHeads + kKvHeads; ++h) {
            const bool query = h < kHeads;
            const float* row = x.data() + size_t(t) * kQkvLd +
                               (query ? h * 2 * kD : 12288 + (h - kHeads) * kD);
            const RowRef r = norm_rope_row(row, query ? q_norm.data() : k_norm.data(), kD,
                                           positions[t]);
            const float* g = query ? q.data() + (size_t(t) * kHeads + h) * kD
                                   : k.data() + (size_t(t) * kKvHeads + h - kHeads) * kD;
            got.insert(got.end(), g, g + kD);
            ref.insert(ref.end(), r.out.begin(), r.out.end());
            tol.insert(tol.end(), r.tol.begin(), r.tol.end());
        }
    }
    char name[96];
    std::snprintf(name, sizeof(name), "qk_norm_rope T=%d pos0=%d", tokens, first_position);
    return report_budget(name, got, ref, tol);
}

int main() {
    if (skip_without_gpu()) return kSkip;
    bool ok = true;
    ok &= run_case(1, 0, 1);
    ok &= run_case(9, 2045, 2);
    ok &= run_case(37, 98000, 3);
    ok &= run_case(300, 131000, 4);
    ok &= run_case(9, 262135, 5);
    return ok ? 0 : 1;
}
