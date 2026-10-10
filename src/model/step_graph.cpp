#include "model/step_graph.h"

#include "core/device.h"
#include "core/exchange.h"
#include "core/model_dims.h"
#include "kernels/gdn/gdn.h"
#include "kernels/head/head.h"
#include "kernels/ple/ple.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace fngine::model {
namespace k = fngine::kernels;

namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct DevLayout {
  std::size_t in, out, scratch, config, counts, x_ar, total;
};

DevLayout dev_layout() {
  DevLayout l{};
  std::size_t off = 0;
  l.in = off;      off = align_up(off + sizeof(StepInDev));
  l.out = off;     off = align_up(off + sizeof(StepOutDev));
  l.scratch = off; off = align_up(off + sizeof(StepScratch));
  l.config = off;  off = align_up(off + sizeof(k::SamplingConfig));
  l.counts = off;  off = align_up(off + sizeof(int32_t) * dims::kVocab);
  l.x_ar = off;    off = align_up(off + sizeof(float) * dims::kStreamHidden);
  l.total = off;
  return l;
}

}  // namespace

StepRunner::StepRunner(ModelRuntime& rt, bool use_graphs, bool mtp, bool logit_diagnostics)
    : rt_(rt), use_graphs_(use_graphs), mtp_(mtp), logit_diagnostics_(logit_diagnostics) {
  if (mtp_ && !rt_.options().mtp) throw std::invalid_argument("StepRunner: MTP requires runtime MTP");
  const DevLayout l = dev_layout();
  dev_ = DeviceBuffer(l.total);
  dev_.fill(0);
  auto* b = static_cast<unsigned char*>(dev_.p);
  in_dev_ = reinterpret_cast<StepInDev*>(b + l.in);
  out_dev_ = reinterpret_cast<StepOutDev*>(b + l.out);
  scratch_ = reinterpret_cast<StepScratch*>(b + l.scratch);
  config_dev_ = reinterpret_cast<k::SamplingConfig*>(b + l.config);
  token_counts_ = reinterpret_cast<int32_t*>(b + l.counts);
  x_ar_ = reinterpret_cast<float*>(b + l.x_ar);
  host_ = MappedHostBuffer(align_up(sizeof(StepInDev)) + sizeof(StepOutDev));
  in_host_ = static_cast<StepInDev*>(host_.data());
  out_host_ = reinterpret_cast<StepOutDev*>(static_cast<unsigned char*>(host_.data()) + align_up(sizeof(StepInDev)));
  in_mapped_ = static_cast<const StepInDev*>(host_.device_ptr());
  out_mapped_ = reinterpret_cast<StepOutDev*>(static_cast<unsigned char*>(host_.device_ptr()) +
                                              align_up(sizeof(StepInDev)));
  set_sampling(k::SamplingConfig{});
}

void StepRunner::set_sampling(const k::SamplingConfig& config) {
  k::SamplingConfig c = config;
  c.token_counts = token_counts_;
  cudaStream_t s = rt_.stream();
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaMemsetAsync(token_counts_, 0, sizeof(int32_t) * dims::kVocab, s));
  CUDA_CHECK(cudaMemcpyAsync(config_dev_, &c, sizeof(c), cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

void StepRunner::reset_graphs() {
  for (auto& g : graphs_) g.reset();
}

// MTP bridge over T columns (hidden x_src rows, tokens = scratch alignment ids, positions =
// in.positions) and two autoregressive draft steps. Leaves d1..d3 in out.mtp_drafts.
void StepRunner::enqueue_mtp(int32_t T, const float* x_src, cudaStream_t s) {
  Forward& fwd = rt_.forward();
  Workspace& ws = rt_.ws();
  const MtpWeights& m = rt_.weights().mtp();
  step_mtp_transition(in_dev_, out_dev_, T, scratch_, s);
  StepIo bio;
  bio.T = T;
  bio.tokens = scratch_->alignment_ids;
  bio.positions = in_dev_->positions;
  bio.seq = &in_dev_->seq;
  bio.pass = T == 1 ? Pass::Decode : Pass::Verify;
  fwd.mtp_layer(x_src, bio, s);
  fwd.final_mixer(m.mixer, ws.X_mtp, T, ws.mix, s);
  draft_logits_argmax(T, s);
  step_select_accepted(ws.X_mtp, out_dev_, x_ar_, scratch_, s);
  for (int32_t i = 0; i + 1 < kMtpDrafts; ++i) {
    StepIo aio;
    aio.T = 1;
    aio.tokens = &scratch_->ar_token[i];
    aio.positions = &scratch_->ar_positions[i];
    aio.seq = &in_dev_->seq;
    aio.pass = Pass::Decode;
    fwd.mtp_layer(x_ar_, aio, s);
    fwd.final_mixer(m.mixer, ws.X_mtp, 1, ws.mix, s);
    draft_logits_argmax(1, s);
    step_ar_advance(ws.X_mtp, x_ar_, out_dev_, scratch_, i, s);
  }
}

// MTP drafts: argmax over the draft-vocabulary head when the pack provides one (then mapped back
// to token ids), over the full lm_head otherwise.
void StepRunner::draft_logits_argmax(int32_t rows, cudaStream_t s) {
  Forward& fwd = rt_.forward();
  Workspace& ws = rt_.ws();
  const DenseWeights& w = rt_.weights();
  if (w.draft_head().codes) {
    fwd.draft_head(ws.mix, rows, ws.logits, s);
    k::argmax(ws.logits, rows, w.draft_head().n, scratch_->mtp_argmax, s, w.draft_head().n);
    map_draft_ids(w.draft_vocab(), scratch_->mtp_argmax, rows, s);
  } else {
    fwd.lm_head(ws.mix, rows, ws.logits, s);
    k::argmax(ws.logits, rows, dims::kVocab, scratch_->mtp_argmax, s);
  }
}

void StepRunner::enqueue_step(int32_t T, cudaStream_t s, bool draft) {
  Forward& fwd = rt_.forward();
  Workspace& ws = rt_.ws();
  ModelState& st = rt_.state();
  step_copy(in_dev_, in_mapped_, sizeof(StepInDev), s);
  StepIo io;
  io.T = T;
  io.tokens = in_dev_->tokens;
  io.positions = in_dev_->positions;
  io.seq = &in_dev_->seq;
  io.pass = T == 1 ? Pass::Decode : Pass::Verify;
  fwd.embed(io, s);
  for (int32_t l = 0; l < dims::kLayers; ++l) fwd.layer(l, io, s);
  fwd.final_mixer(rt_.weights().mixer(), ws.X, T, ws.mix, s);
  fwd.lm_head(ws.mix, T, ws.logits, s);
  if (logit_diagnostics_) step_logit_gaps(ws.logits, T, out_dev_, s);
  if (T == 1) {
    k::sample(ws.logits, 1, dims::kVocab, config_dev_, in_dev_->positions, k::kSamplePurposeDecode,
              out_dev_->licensed, ws.sample_ws, ws.sample_ws_bytes, s);
    step_finish_single(in_dev_, out_dev_, s);
  } else {
    k::argmax(ws.logits, T, dims::kVocab, out_dev_->target, s);
    CUDA_CHECK(cudaMemcpyAsync(&out_dev_->length, &in_dev_->length, sizeof(int32_t),
                               cudaMemcpyDeviceToDevice, s));
    k::speculative_accept(out_dev_->target, ws.logits, in_dev_->drafts, &in_dev_->k, T - 1,
                          dims::kVocab, config_dev_, &out_dev_->length, &out_dev_->anchor,
                          out_dev_->licensed, &out_dev_->licensed_count, &out_dev_->accepted,
                          ws.accept_ws, ws.accept_ws_bytes, s);
    step_set_gdn_commit(out_dev_, st.gdn_control(), s);
    k::gdn_fold(st.gdn_pool(), st.gdn_records(), st.gdn_control(), T, s);
    k::ple_fold(st.ple_conv_state(), ws.c_record, &out_dev_->licensed_count, T, s);
  }
  if (draft) enqueue_mtp(T, ws.X, s);
  fwd.join_prefetch(s);
  step_copy(out_mapped_, out_dev_, sizeof(StepOutDev), s);
}

StepResult StepRunner::finish(double seconds, bool draft) const {
  StepResult r;
  r.count = out_host_->licensed_count;
  r.accepted = out_host_->accepted;
  if (r.count < 1 || r.count > kStepMaxTokens) throw std::runtime_error("step produced an invalid token count");
  std::memcpy(r.licensed, out_host_->licensed, sizeof(int32_t) * std::size_t(r.count));
  r.mtp_count = draft ? kMtpDrafts : 0;
  if (draft) std::memcpy(r.mtp_drafts, out_host_->mtp_drafts, sizeof(r.mtp_drafts));
  if (logit_diagnostics_)
    std::memcpy(r.target_gap, out_host_->target_gap, sizeof(float) * std::size_t(r.count));
  r.seconds = seconds;
  return r;
}

StepResult StepRunner::first_token(int32_t length) {
  const int32_t rows = rt_.last_rows();
  if (rows <= 0) throw std::logic_error("first_token: no prefill rows");
  cudaStream_t s = rt_.stream();
  Forward& fwd = rt_.forward();
  Workspace& ws = rt_.ws();
  const auto t0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::memset(in_host_, 0, sizeof(StepInDev));
  in_host_->positions[0] = rt_.position() - 1;  // the last prefilled position
  in_host_->length = length;
  in_host_->seq = rt_.next_seq();
  step_copy(in_dev_, in_mapped_, sizeof(StepInDev), s);
  const float* x_last = ws.X + std::size_t(rows - 1) * dims::kStreamHidden;
  fwd.final_mixer(rt_.weights().mixer(), x_last, 1, ws.mix, s);
  fwd.lm_head(ws.mix, 1, ws.logits, s);
  if (logit_diagnostics_) step_logit_gaps(ws.logits, 1, out_dev_, s);
  k::sample(ws.logits, 1, dims::kVocab, config_dev_, in_dev_->positions, k::kSamplePurposePrefill,
            out_dev_->licensed, ws.sample_ws, ws.sample_ws_bytes, s);
  step_finish_single(in_dev_, out_dev_, s);
  const bool draft = mtp_ && rt_.options().max_context - rt_.position() >= kMtpDrafts - 1;
  if (draft) {
    // The MTP layer reads its hidden from a buffer other than X_mtp; copy the last row first.
    CUDA_CHECK(cudaMemcpyAsync(x_ar_, x_last, sizeof(float) * dims::kStreamHidden,
                               cudaMemcpyDeviceToDevice, s));
    enqueue_mtp(1, x_ar_, s);
  }
  step_copy(out_mapped_, out_dev_, sizeof(StepOutDev), s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  rt_.check_exchange();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return finish(secs, draft);
}

StepResult StepRunner::step(int32_t anchor, std::span<const int32_t> drafts) {
  const int32_t k = int32_t(drafts.size());
  const int32_t T = k + 1;
  if (T > dims::kMaxGraphWidth) throw std::invalid_argument("step: too many drafts");
  const int32_t p0 = rt_.position();
  if (T > rt_.options().max_context - p0) throw std::length_error("step exceeds the context capacity");
  // MTP may accept all T-1 drafts, then write another kMtpDrafts-1 positions. At the
  // context tail keep the target step exact, but omit proposals that cannot be used safely.
  const bool draft = mtp_ && rt_.options().max_context - p0 - T >= kMtpDrafts - 1;
  const bool graph = use_graphs_ && (!mtp_ || draft);
  rt_.ensure_context(p0 + T);
  cudaStream_t s = rt_.stream();
  const uint32_t seq = rt_.next_seq();

  std::memset(in_host_, 0, sizeof(StepInDev));
  in_host_->tokens[0] = anchor;
  for (int32_t j = 0; j < k; ++j) {
    in_host_->tokens[j + 1] = drafts[std::size_t(j)];
    in_host_->drafts[j] = drafts[std::size_t(j)];
  }
  for (int32_t j = 0; j < T; ++j) in_host_->positions[j] = p0 + j;
  in_host_->k = k;
  in_host_->length = p0;
  in_host_->seq = seq;

  // Capture before timing: the controller's latency model must see steady-state step cost; a
  // first-use capture would make that width look many times slower than it is.
  if (graph && !graphs_[std::size_t(T)]) {
    auto& g = graphs_[std::size_t(T)];
    DecodeGraphDefinition def;
    def.capture(s, [&] { enqueue_step(T, s, draft); });
    g = std::make_unique<DecodeGraphExecutable>();
    g->instantiate(def);
    g->upload(s);
  }
  const auto t0 = std::chrono::steady_clock::now();
  rt_.pool().begin(seq, cpu::PoolMode::Decode);
  if (graph) {
    graphs_[std::size_t(T)]->launch(s);
  } else {
    enqueue_step(T, s, draft);
  }
  // n-gram rows for this step, staged while the first layers run.
  rt_.ngram().gather_tokens(in_host_->tokens, T, rt_.ngram_history(), rt_.ple_staging_host());
  host_store_release(rt_.exchange_host()->control.ple_ready.value, seq);
  // After the rows: swap copies through pageable staging can block this thread.
  rt_.overlap_swaps();
  CUDA_CHECK(cudaStreamSynchronize(s));
  rt_.pool().wait_idle();
  rt_.check_exchange();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  StepResult r = finish(secs, draft);
  // The step processed [anchor, d1..dA]; the licensed correction/bonus token is the next anchor.
  int32_t processed[kStepMaxTokens];
  processed[0] = anchor;
  for (int32_t j = 0; j < r.accepted; ++j) processed[j + 1] = r.licensed[j];
  rt_.commit(std::span<const int32_t>(processed, std::size_t(r.accepted + 1)));
  rt_.step_boundary();
  return r;
}

}  // namespace fngine::model
