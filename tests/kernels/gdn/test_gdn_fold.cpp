// Verify-record + fold bit-exactness (pattern of cinference:tests/ops/test_gdn_replay_fold.cpp).
//
// For each width T, all 36 layers run gdn_conv_record + gdn_gating + gdn_recurrent_record from a
// random state image in slot 0; then gdn_fold commits n = 0..T columns into slot 1. Each folded
// image (recurrent state + conv history, all layers) must equal, bit for bit, the image after n
// sequential plain T=1 steps (gdn_conv_decode + gdn_gating + gdn_recurrent_decode) on a separate
// pool, and every record output column must equal the corresponding T=1 step's output. Runs
// eagerly and with the record/fold/decode steps captured as CUDA graphs whose slot/commit values
// are written to the device control between replays (including a two-round ping-pong with
// gdn_swap_slots).
#include "kernels/gdn/gdn_test_common.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using namespace fngine;
using namespace fngine::kernels;
using namespace fngine::test;
using namespace fngine::test::gdn;

namespace {

constexpr int kW = kMaxRecordWidth;

struct LayerInputs {
    DeviceArray<__nv_bfloat16> qkvz;    // [36][16][16384]
    DeviceArray<float> conv_w;          // [36][10240][4]
    DeviceArray<float> ab;              // [36][16][96]
    DeviceArray<float> a_log, dt_bias;  // [36][48]
    explicit LayerInputs(Rng& rng)
        : qkvz(random_bf16(rng, size_t(kLayers) * kW * kQkvzLd, -1, 1)),
          conv_w(random_f32(rng, size_t(kLayers) * kConvChannels * 4, -0.5, 0.5)),
          ab(random_f32(rng, size_t(kLayers) * kW * kAbLd, -2.5, 1.5)),
          a_log(random_f32(rng, size_t(kLayers) * kValueHeads, -2, 1)),
          dt_bias(random_f32(rng, size_t(kLayers) * kValueHeads, -1, 1)) {}
};

struct StepBuffers {
    DeviceArray<__nv_bfloat16> conv_out;  // [36][16][10240]
    DeviceArray<float> g, beta;           // [36][16][48]
    DeviceArray<__nv_bfloat16> core;      // [36][16][6144]
    StepBuffers()
        : conv_out(size_t(kLayers) * kW * kConvChannels),
          g(size_t(kLayers) * kW * kValueHeads),
          beta(size_t(kLayers) * kW * kValueHeads),
          core(size_t(kLayers) * kW * kValueDim) {}
};

struct Pool {
    size_t bytes;
    DeviceArray<unsigned char> mem;
    GdnStatePool pool;
    explicit Pool(int slots)
        : bytes(gdn_state_pool_bytes(slots)), mem(bytes),
          pool(gdn_state_pool_bind(mem.get(), bytes, slots)) {}
};

// Copies slot `src` of pool a into slot `dst` of pool b (recurrent + conv images).
void copy_slot(const GdnStatePool& a, int src, const GdnStatePool& b, int dst) {
    FNGINE_TEST_CUDA(cudaMemcpy(gdn_recurrent_state(b, dst, 0), gdn_recurrent_state(a, src, 0),
                                gdn_state_slot_recurrent_bytes(), cudaMemcpyDeviceToDevice));
    FNGINE_TEST_CUDA(cudaMemcpy(gdn_conv_state(b, dst, 0), gdn_conv_state(a, src, 0),
                                gdn_state_slot_conv_bytes(), cudaMemcpyDeviceToDevice));
}

unsigned long long slot_mismatches(const GdnStatePool& a, int sa, const GdnStatePool& b, int sb) {
    FNGINE_TEST_CUDA(cudaDeviceSynchronize());
    return device_mismatches(gdn_recurrent_state(a, sa, 0), gdn_recurrent_state(b, sb, 0),
                             gdn_state_slot_recurrent_bytes()) +
           device_mismatches(gdn_conv_state(a, sa, 0), gdn_conv_state(b, sb, 0),
                             gdn_state_slot_conv_bytes());
}

void set_control(GdnStepControl* control, GdnStepControl value) {
    FNGINE_TEST_CUDA(cudaMemcpy(control, &value, sizeof value, cudaMemcpyHostToDevice));
}

struct Fixture {
    Rng rng{1234};
    LayerInputs in{rng};
    Pool pool{3};      // 0: committed source, 1: fold destination, 2: in-place checks
    Pool ref{1};       // sequential T=1 reference
    Pool initial{1};   // pristine initial image
    size_t records_bytes = gdn_records_bytes();
    DeviceArray<unsigned char> records_mem{records_bytes};
    GdnRecords records = gdn_records_bind(records_mem.get(), records_bytes);
    DeviceArray<GdnStepControl> control{1};
    DeviceArray<GdnStepControl> ref_control{1};
    StepBuffers rec, dec;
    cudaStream_t stream = nullptr;

    Fixture() {
        const auto rec_state = random_f32(rng, gdn_state_slot_recurrent_bytes() / 4, -0.1, 0.1);
        const auto conv_hist = random_bf16(rng, gdn_state_slot_conv_bytes() / 2, -1, 1);
        FNGINE_TEST_CUDA(cudaMemcpy(gdn_recurrent_state(initial.pool, 0, 0), rec_state.data(),
                                    gdn_state_slot_recurrent_bytes(), cudaMemcpyHostToDevice));
        FNGINE_TEST_CUDA(cudaMemcpy(gdn_conv_state(initial.pool, 0, 0), conv_hist.data(),
                                    gdn_state_slot_conv_bytes(), cudaMemcpyHostToDevice));
        set_control(ref_control.get(), {0, 0, 0, 0});
        FNGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Fixture() { cudaStreamDestroy(stream); }

    const __nv_bfloat16* qkvz(int l, int t) const {
        return in.qkvz.get() + (size_t(l) * kW + t) * kQkvzLd;
    }
    const float* ab(int l, int t) const { return in.ab.get() + (size_t(l) * kW + t) * kAbLd; }

    // Verify step of width T over all layers from control->source_slot.
    void record_step(int T, cudaStream_t s) {
        for (int l = 0; l < kLayers; ++l) {
            const size_t c = size_t(l) * kW;
            gdn_conv_record(qkvz(l, 0), in.conv_w.get() + size_t(l) * kConvChannels * 4, pool.pool, l,
                            control.get(), records, T, rec.conv_out.get() + c * kConvChannels, s);
            gdn_gating(ab(l, 0), in.a_log.get() + l * kValueHeads, in.dt_bias.get() + l * kValueHeads,
                       rec.g.get() + c * kValueHeads, rec.beta.get() + c * kValueHeads, T, s);
            gdn_recurrent_record(rec.conv_out.get() + c * kConvChannels, rec.g.get() + c * kValueHeads,
                                 rec.beta.get() + c * kValueHeads, pool.pool, l, control.get(),
                                 records, T, rec.core.get() + c * kValueDim, s);
        }
    }

    // Plain T=1 step on the reference pool using input row `row`; outputs land in column `col`.
    void decode_step(int row, int col, cudaStream_t s) {
        for (int l = 0; l < kLayers; ++l) {
            const size_t c = size_t(l) * kW + col;
            gdn_conv_decode(qkvz(l, row), in.conv_w.get() + size_t(l) * kConvChannels * 4, ref.pool, l,
                            ref_control.get(), dec.conv_out.get() + c * kConvChannels, s);
            gdn_gating(ab(l, row), in.a_log.get() + l * kValueHeads, in.dt_bias.get() + l * kValueHeads,
                       dec.g.get() + c * kValueHeads, dec.beta.get() + c * kValueHeads, 1, s);
            gdn_recurrent_decode(dec.conv_out.get() + c * kConvChannels, dec.g.get() + c * kValueHeads,
                                 dec.beta.get() + c * kValueHeads, ref.pool, l, ref_control.get(),
                                 dec.core.get() + c * kValueDim, s);
        }
    }

    // Record outputs of columns [0, T) vs the reference's columns [0, T) (all layers).
    unsigned long long output_mismatches(int T) {
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        unsigned long long bad = 0;
        for (int l = 0; l < kLayers; ++l) {
            const size_t c = size_t(l) * kW;
            bad += device_mismatches(rec.conv_out.get() + c * kConvChannels,
                                     dec.conv_out.get() + c * kConvChannels,
                                     size_t(T) * kConvChannels * 2);
            bad += device_mismatches(rec.core.get() + c * kValueDim, dec.core.get() + c * kValueDim,
                                     size_t(T) * kValueDim * 2);
        }
        return bad;
    }
};

struct Graph {
    cudaGraphExec_t exec = nullptr;
    void capture(cudaStream_t s, const std::function<void()>& body) {
        cudaGraph_t graph;
        FNGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        body();
        FNGINE_TEST_CUDA(cudaStreamEndCapture(s, &graph));
        FNGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        FNGINE_TEST_CUDA(cudaGraphDestroy(graph));
    }
    // Setup copies run on the legacy stream (D2D and pageable H2D cudaMemcpy may return before
    // the device sees them) while graphs replay on a non-blocking stream: order them explicitly.
    void launch(cudaStream_t s) const {
        FNGINE_TEST_CUDA(cudaDeviceSynchronize());
        FNGINE_TEST_CUDA(cudaGraphLaunch(exec, s));
        FNGINE_TEST_CUDA(cudaStreamSynchronize(s));
    }
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
    }
};

bool run_width(Fixture& f, int T, bool graphs) {
    const char* mode = graphs ? "graph" : "eager";
    copy_slot(f.initial.pool, 0, f.pool.pool, 0);
    FNGINE_TEST_CUDA(cudaMemset(gdn_recurrent_state(f.pool.pool, 1, 0), 0xAB,
                                gdn_state_slot_recurrent_bytes()));
    copy_slot(f.initial.pool, 0, f.ref.pool, 0);
    FNGINE_TEST_CUDA(cudaMemset(f.dec.conv_out.get(), 0, f.dec.conv_out.count * 2));
    FNGINE_TEST_CUDA(cudaMemset(f.rec.conv_out.get(), 0x55, f.rec.conv_out.count * 2));
    set_control(f.control.get(), {0, 1, 0, 0});

    Graph record, fold;
    std::vector<Graph> decode(graphs ? T : 0);
    if (graphs) {
        record.capture(f.stream, [&] { f.record_step(T, f.stream); });
        fold.capture(f.stream, [&] { gdn_fold(f.pool.pool, f.records, f.control.get(), T, f.stream); });
        for (int j = 0; j < T; ++j) decode[j].capture(f.stream, [&] { f.decode_step(j, j, f.stream); });
        record.launch(f.stream);
    } else {
        f.record_step(T, nullptr);
    }
    bool ok = true;
    const unsigned long long untouched = slot_mismatches(f.pool.pool, 0, f.initial.pool, 0);
    if (untouched != 0) {
        std::printf("FAIL %s T=%d: record modified the source slot (%llu words)\n", mode, T, untouched);
        ok = false;
    }
    for (int n = 0; n <= T; ++n) {
        set_control(f.control.get(), {0, 1, n, 0});
        if (graphs) fold.launch(f.stream);
        else gdn_fold(f.pool.pool, f.records, f.control.get(), T, nullptr);
        const unsigned long long bad = slot_mismatches(f.pool.pool, 1, f.ref.pool, 0);
        if (bad != 0) {
            std::printf("FAIL %s T=%d n=%d: folded image differs in %llu words\n", mode, T, n, bad);
            ok = false;
        }
        if (n < T) {
            if (graphs) decode[n].launch(f.stream);
            else f.decode_step(n, n, nullptr);
        }
    }
    const unsigned long long out_bad = f.output_mismatches(T);
    if (out_bad != 0) {
        std::printf("FAIL %s T=%d: record outputs differ from T=1 outputs in %llu words\n", mode, T,
                    out_bad);
        ok = false;
    }
    // In-place fold of all T columns (source == destination).
    copy_slot(f.initial.pool, 0, f.pool.pool, 2);
    set_control(f.control.get(), {2, 2, T, 0});
    if (graphs) fold.launch(f.stream);
    else gdn_fold(f.pool.pool, f.records, f.control.get(), T, nullptr);
    const unsigned long long in_place = slot_mismatches(f.pool.pool, 2, f.ref.pool, 0);
    if (in_place != 0) {
        std::printf("FAIL %s T=%d: in-place fold differs in %llu words\n", mode, T, in_place);
        ok = false;
    }
    if (ok) std::printf("PASS %s T=%d: n=0..%d folds, outputs and in-place fold bit-exact\n", mode, T, T);
    return ok;
}

// Two verify rounds with ping-pong slots: round 1 (T1, commit c1) 0 -> 1, swap, round 2 (T2,
// commit c2) 1 -> 0, all captured once and replayed; vs c1 + c2 plain steps.
bool run_ping_pong(Fixture& f, int T1, int c1, int T2, int c2) {
    copy_slot(f.initial.pool, 0, f.pool.pool, 0);
    copy_slot(f.initial.pool, 0, f.ref.pool, 0);
    set_control(f.control.get(), {0, 1, 0, 0});
    Graph r1, r2, d;
    r1.capture(f.stream, [&] {
        f.record_step(T1, f.stream);
        gdn_fold(f.pool.pool, f.records, f.control.get(), T1, f.stream);
        gdn_swap_slots(f.control.get(), f.stream);
    });
    r2.capture(f.stream, [&] {
        f.record_step(T2, f.stream);
        gdn_fold(f.pool.pool, f.records, f.control.get(), T2, f.stream);
        gdn_swap_slots(f.control.get(), f.stream);
    });
    set_control(f.control.get(), {0, 1, c1, 0});
    r1.launch(f.stream);
    GdnStepControl after{};
    FNGINE_TEST_CUDA(cudaMemcpy(&after, f.control.get(), sizeof after, cudaMemcpyDeviceToHost));
    bool ok = after.source_slot == 1 && after.destination_slot == 0;
    FNGINE_TEST_CUDA(cudaMemcpy(&f.control.get()->commit, &c2, sizeof c2, cudaMemcpyHostToDevice));
    r2.launch(f.stream);
    for (int j = 0; j < c1; ++j) f.decode_step(j, j, nullptr);
    for (int j = 0; j < c2; ++j) f.decode_step(j, j, nullptr);
    const unsigned long long bad = slot_mismatches(f.pool.pool, 0, f.ref.pool, 0);
    ok &= bad == 0;
    std::printf("%s ping-pong graphs T1=%d c1=%d T2=%d c2=%d: %llu differing words, swap %s\n",
                ok ? "PASS" : "FAIL", T1, c1, T2, c2, bad,
                after.source_slot == 1 ? "ok" : "wrong");
    return ok;
}

}  // namespace

int main() {
    if (skip_without_gpu()) return kSkip;
    Fixture f;
    bool ok = true;
    for (int T : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16}) ok &= run_width(f, T, false);
    for (int T : {2, 5, 9}) ok &= run_width(f, T, true);
    ok &= run_ping_pong(f, 5, 3, 4, 2);
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
