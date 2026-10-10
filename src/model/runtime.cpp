#include "model/runtime.h"

#include "core/cpu_topology.h"
#include "core/device.h"
#include "core/model_dims.h"
#include "fngine/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace fngine::model {
namespace {

constexpr uint64_t kMiB = 1ull << 20;
// CUDA context extras not covered by the planners (graphs, module images, cuBLASLt).
constexpr uint64_t kRuntimeOverheadBytes = 384 * kMiB;
// Rows of the MTP draft head (the registered [32768, 2560] projection).
constexpr int32_t kDraftVocab = 32768;
// Positions beyond a step's tokens that its KV writes may reach (MTP drafting), rounded to a page.
constexpr int32_t kContextMargin = 64;

kernels::MoeFormat moe_format(io::GgmlType t) { return static_cast<kernels::MoeFormat>(int32_t(t)); }
cpu::WeightFormat cpu_format(io::GgmlType t) { return static_cast<cpu::WeightFormat>(int32_t(t)); }

}  // namespace

ModelRuntime::ModelRuntime(const RuntimeOptions& options) : options_(options) {
  StateOptions sopt;
  sopt.max_context = options_.max_context;
  sopt.kv_format = options_.kv;
  const uint64_t state_bytes = ModelState::device_bytes(sopt);
  if (options_.prefill_chunk < dims::kMaxStepTokens ||
      options_.prefill_chunk % dims::kMaxStepTokens != 0)
    throw std::invalid_argument("prefill_chunk must be a positive multiple of 16");
  if (options_.logit_rows < dims::kMaxStepTokens) options_.logit_rows = dims::kMaxStepTokens;

  const auto t0 = std::chrono::steady_clock::now();
  pack_ = std::make_unique<io::Pack>(io::Pack::open(options_.pack));
  config_ = ModelConfig::from_pack(*pack_);
  if (options_.max_context > config_.max_positions)
    throw std::invalid_argument("max_context exceeds pack max_position_embeddings (" +
                                std::to_string(config_.max_positions) + ")");
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, -5));

  WeightsOptions wopt;
  wopt.dense_bf16 = options_.dense_bf16;
  wopt.load_mtp_experts = options_.mtp;
  weights_ = DenseWeights::load(*pack_, stream_, wopt);
  if (options_.mtp) {
    // MTP draft head over the pack's draft vocabulary (tools/draft_vocab.py); the full lm_head
    // drafts otherwise. Built before placement so the expert budget accounts for it.
    const std::filesystem::path vocab_path = pack_->dir() / "draft-vocab.bin";
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(vocab_path, ec);
    if (!ec && bytes == kDraftVocab * sizeof(int32_t)) {
      std::vector<int32_t> vocab(kDraftVocab);
      std::ifstream in(vocab_path, std::ios::binary);
      if (!in.read(reinterpret_cast<char*>(vocab.data()), std::streamsize(bytes)))
        throw io::PackError("cannot read " + vocab_path.string());
      weights_.build_draft_head(vocab, stream_);
      spdlog::info("MTP draft head: {} draft-vocabulary rows ({:.1f} MiB)", kDraftVocab,
                   double(kDraftVocab) * dims::kHidden / double(1 << 20));
    } else if (!ec) {
      spdlog::warn("{} must hold {} int32 ids; MTP drafts with the full lm_head", vocab_path.string(), kDraftVocab);
    }
  }

  WorkspaceOptions wsopt;
  wsopt.max_tokens = options_.prefill_chunk;
  wsopt.logit_rows = options_.logit_rows;
  wsopt.max_blocks = (sopt.max_context + 63) / 64 * 64 / 4;
  wsopt.dense_bf16 = options_.dense_bf16;
  const uint64_t ws_bytes = WorkspaceBuffer::device_bytes(wsopt);
  WorkspaceOptions decode_wsopt = wsopt;
  decode_wsopt.max_tokens = dims::kMaxStepTokens;
  const uint64_t decode_ws_bytes = WorkspaceBuffer::device_bytes(decode_wsopt);
  const uint64_t ring_bytes = PrefillStream::ring_bytes(*pack_, options_.prefill_stream);
  fixed_vram_ = state_bytes + (options_.prefill_borrow ? decode_ws_bytes : ws_bytes + ring_bytes) +
                kRuntimeOverheadBytes;

  const uint64_t exchange_bytes = sizeof(DecodeExchange);
  const uint64_t prefill_bytes = prefill_exchange_bytes(options_.prefill_chunk);
  const uint64_t ple_bytes = uint64_t(options_.prefill_chunk) * dims::kPleHeads * dims::kPleRowBytes;
  const uint64_t ngram_bytes = options_.ngram_cache_mib * kMiB + options_.ngram_cache_mib * kMiB / 16;

  PlacementOptions popt;
  popt.fixed_vram_bytes = fixed_vram_;
  popt.vram_reserve_mib = options_.vram_reserve_mib;
  popt.vram_cap_bytes = options_.expert_vram_cap;
  popt.preloaded_vram_bytes = weights_.vram_bytes();
  popt.other_host_bytes = exchange_bytes + prefill_bytes + ple_bytes + ngram_bytes;
  popt.spare_slots = options_.adaptive_experts;
  popt.loan_bytes = options_.prefill_borrow ? ws_bytes + ring_bytes : 0;
  // Adaptive placement shares the state's allocation: experts fill its unused context tail.
  if (options_.adaptive_experts) {
    popt.state_bytes = state_bytes;
    popt.state_regions = ModelState::position_regions(sopt);
  }
  try {
    auto plan = plan_placement(*pack_, popt);
    if (options_.prefill_borrow && !plan.loan_bytes) {
      spdlog::warn("expert budget cannot back the prefill loan; using static workspace reservation");
      options_.prefill_borrow = false;
      fixed_vram_ = state_bytes + ws_bytes + ring_bytes + kRuntimeOverheadBytes;
      popt.fixed_vram_bytes = fixed_vram_;
      popt.loan_bytes = 0;
      plan = plan_placement(*pack_, popt);
    }
    placement_ = ExpertPlacement::build(*pack_, std::move(plan), stream_, popt);
    state_ = std::make_unique<ModelState>(sopt, placement_.state_memory());
    workspace_ = std::make_unique<WorkspaceBuffer>(options_.prefill_borrow ? decode_wsopt : wsopt);
    active_ws_ = &workspace_->get();
    if (options_.prefill_borrow) {
      prefill_wsopt_ = wsopt;
      spdlog::info("prefill borrowing {:.2f} GiB of expert slots for workspace/ring (decode scratch {:.2f} GiB)",
                   double(placement_.plan().loan_bytes) / double(1ull << 30),
                   double(decode_ws_bytes) / double(1ull << 30));
    }
    cublas_ = std::make_unique<kernels::CublasLtContext>();
  } catch (const HostMemoryError& e) {
    throw StartupError(e.what());
  } catch (const std::exception& e) {
    const std::string what = e.what();
    if (what.find("out of memory") != std::string::npos || what.find("cudaMalloc") != std::string::npos)
      throw StartupError("not enough GPU memory for the engine (" + what +
                         "); stop other GPU processes or lower --ctx / --prefill-chunk");
    throw;
  }

  exchange_buf_ = MappedHostBuffer(exchange_bytes);
  prefill_buf_ = MappedHostBuffer(prefill_bytes);
  ple_buf_ = MappedHostBuffer(ple_bytes);
  input_staging_ = std::make_unique<PinnedHostBuffer>(sizeof(int32_t) * (2 * std::size_t(options_.prefill_chunk) + 64));
  prefill_host_ = make_prefill_exchange_view(prefill_buf_.data(), options_.prefill_chunk);
  prefill_dev_ = make_prefill_exchange_view(prefill_buf_.device_ptr(), options_.prefill_chunk);

  // CPU expert pool.
  const CpuTopology topo = CpuTopology::detect();
  cpu::PoolConfig pcfg;
  pcfg.worker_cpus = topo.worker_cpus;
  if (options_.cpu_workers > 0 && std::size_t(options_.cpu_workers) < pcfg.worker_cpus.size())
    pcfg.worker_cpus.resize(std::size_t(options_.cpu_workers));
  for (int l = 0; l < dims::kLayers; ++l) {
    const io::ExpertLayout& el = pack_->expert_layers()[std::size_t(l)];
    cpu::ExpertLayerDesc d;
    d.gu_format = cpu_format(el.gu_type);
    d.down_format = cpu_format(el.down_type);
    d.gate_offset = 0;
    d.up_offset = el.gate_bytes;
    d.down_offset = el.gate_bytes + el.up_bytes;
    pcfg.layers[std::size_t(l)] = d;
  }
  pcfg.host_slot_of = placement_.host_slot_table();
  pcfg.decode = static_cast<DecodeExchange*>(exchange_buf_.data());
  pcfg.prefill = prefill_host_;
  // Prefill expert tables / streaming ring (its cost model runs before the pool workers exist).
  PrefillStreamOptions psopt;
  psopt.enabled = options_.prefill_stream;
  psopt.worker_cpus = pcfg.worker_cpus;
  if (options_.prefill_borrow && ring_bytes) {
    psopt.ring_memory = static_cast<uint8_t*>(placement_.loan_data()) + ws_bytes;
    psopt.ring_memory_bytes = ring_bytes;
  }
  prefill_stream_ = std::make_unique<PrefillStream>(*pack_, placement_, weights_.mtp().experts, psopt);
  pool_ = std::make_unique<cpu::ExpertPool>(pcfg);

  io::NgramStoreOptions nopt;
  nopt.cache_mib = options_.ngram_cache_mib;
  nopt.io_cpu = topo.io_cpu;
  ngram_ = std::make_unique<io::NgramStore>(options_.pack.string(),
                                            io::NgramParams::from_pack(options_.pack.string()), nopt);

  ForwardContext ctx;
  ctx.weights = &weights_;
  ctx.placement = &placement_;
  ctx.state = state_.get();
  ctx.ws = &workspace_->get();
  ctx.exchange = static_cast<DecodeExchange*>(exchange_buf_.device_ptr());
  ctx.prefill = prefill_dev_;
  ctx.ple_rows = static_cast<const uint8_t*>(ple_buf_.device_ptr());
  ctx.dense_bf16 = options_.dense_bf16;
  ctx.fp8_path = options_.fp8_path;
  ctx.cublas = cublas_.get();
  ctx.prefill_stream = prefill_stream_.get();
  routing_counts_ = DeviceBuffer(std::size_t(dims::kLayers) * dims::kExperts * sizeof(uint32_t));
  CUDA_CHECK(cudaMemsetAsync(routing_counts_.p, 0, routing_counts_.bytes, stream_));
  ctx.routing_counts = static_cast<uint32_t*>(routing_counts_.p);
  routing_stats_ = std::make_unique<RoutingStats>(
      options_.persist_routing_stats ? pack_->dir() / "routing-stats.bin" : std::filesystem::path{},
      static_cast<const uint32_t*>(routing_counts_.p));
  if (options_.adaptive_experts) {
    swapper_ = std::make_unique<ExpertSwapper>(placement_, stream_, static_cast<const uint32_t*>(routing_counts_.p),
                                               SwapOptions{});
    if (swapper_->usable())
      spdlog::info("adaptive experts: one spare GPU slot per layer, up to {} swaps planned every {} decode steps",
                   swapper_->options().max_per_plan, swapper_->options().every_steps);
  }
  for (int l = 0; l < dims::kLayers; ++l) {
    const io::ExpertLayout& el = pack_->expert_layers()[std::size_t(l)];
    ctx.gate_up_format[std::size_t(l)] = moe_format(el.gu_type);
    ctx.down_format[std::size_t(l)] = moe_format(el.down_type);
  }
  // Abliteration: upload the per-layer directions (and scales) once; the combine kernels read them
  // from every decode/verify step and prefill chunk.
  if (pack_->has_ablation() && !options_.ablation) {
    spdlog::info("abliterated pack: ablation disabled (--no-ablation); dense writers stay abliterated");
  }
  if (pack_->has_ablation() && options_.ablation) {
    const io::AblationInfo& ab = pack_->ablation();
    const std::size_t dir_bytes = std::size_t(dims::kLayers) * dims::kHidden * sizeof(float);
    const std::size_t scale_bytes =
        ab.scales_bytes != 0 ? std::size_t(dims::kLayers) * sizeof(float) : 0;
    std::ifstream in(ab.path, std::ios::binary);
    if (!in) throw StartupError("cannot read ablation file " + ab.path.string());
    std::vector<float> dirs(std::size_t(dims::kLayers) * dims::kHidden);
    in.seekg(static_cast<std::streamoff>(ab.directions_offset));
    if (!in.read(reinterpret_cast<char*>(dirs.data()), std::streamsize(dir_bytes))) {
      throw StartupError("ablation file " + ab.path.string() + " is shorter than its directions");
    }
    std::vector<float> scales(std::size_t(dims::kLayers), ab.scale);
    if (scale_bytes) {
      in.seekg(static_cast<std::streamoff>(ab.scales_offset));
      if (!in.read(reinterpret_cast<char*>(scales.data()), std::streamsize(scale_bytes))) {
        throw StartupError("ablation file " + ab.path.string() + " is shorter than its scales");
      }
    }
    ablation_ = DeviceBuffer(dir_bytes);
    CUDA_CHECK(cudaMemcpyAsync(ablation_.p, dirs.data(), dir_bytes, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    ctx.ablation_directions = static_cast<const float*>(ablation_.p);
    for (int l = 0; l < dims::kLayers; ++l) ctx.ablation_scales[std::size_t(l)] = scales[std::size_t(l)];
    spdlog::info("abliterated pack: projecting {} layers away from a refusal direction (scale {:.6g}{})",
                 ab.layers, double(ab.scale), scale_bytes ? ", per-layer table" : "");
  }
  uint64_t prefetch_mib = options_.l2_prefetch_mib;
  if (const char* env = std::getenv("FNGINE_L2_PREFETCH_MIB"); env && *env)
    prefetch_mib = std::strtoull(env, nullptr, 10);
  if (prefetch_mib > 0) {
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&prefetch_stream_, cudaStreamNonBlocking, least));
    ctx.prefetch_stream = prefetch_stream_;
    ctx.prefetch_budget = prefetch_mib << 20;
  }
  // One level below the step stream: when both have CTAs pending, the critical path goes first.
  CUDA_CHECK(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, -4));
  ctx.aux_stream = aux_stream_;
  forward_ = std::make_unique<Forward>(ctx);

  reset_sequence();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  spdlog::info("runtime ready in {:.1f} s: {}", secs, summary());
}

ModelRuntime::~ModelRuntime() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    pool_.reset();
    try {
      if (swapper_) swapper_->drain();
      if (routing_stats_) {
        routing_stats_->collect(stream_);
        routing_stats_->flush();
      }
    } catch (const std::exception& e) {
      spdlog::warn("routing stats flush at shutdown failed: {}", e.what());
    }
    swapper_.reset();
    forward_.reset();
    if (prefetch_stream_) {
      cudaStreamSynchronize(prefetch_stream_);
      cudaStreamDestroy(prefetch_stream_);
    }
    if (aux_stream_) {
      cudaStreamSynchronize(aux_stream_);
      cudaStreamDestroy(aux_stream_);
    }
    cudaStreamDestroy(stream_);
  }
}

DecodeExchange* ModelRuntime::exchange_host() const noexcept {
  return static_cast<DecodeExchange*>(exchange_buf_.data());
}
DecodeExchange* ModelRuntime::exchange_device() const noexcept {
  return static_cast<DecodeExchange*>(exchange_buf_.device_ptr());
}
uint8_t* ModelRuntime::ple_staging_host() const noexcept { return static_cast<uint8_t*>(ple_buf_.data()); }
const uint8_t* ModelRuntime::ple_staging_device() const noexcept {
  return static_cast<const uint8_t*>(ple_buf_.device_ptr());
}

void ModelRuntime::reset_sequence() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  state_->reset(stream_);
  history_ = io::NgramHistory::start(dims::kEosToken);
  position_ = 0;
  last_rows_ = 0;
  context_ = 0;
  placement_.revalidate(0);
}

void ModelRuntime::ensure_context(int32_t tokens) {
  tokens = int32_t(std::min<int64_t>(int64_t(tokens) + kContextMargin, state_->capacity()));
  if (tokens <= context_) return;
  context_ = tokens;
  if (swapper_ ? swapper_->vacate(tokens) : placement_.vacate(tokens, stream_)) prefill_stream_->invalidate();
}

uint32_t ModelRuntime::next_seq() {
  ++seq_;
  if (seq_ == 0) seq_ = 1;
  return seq_;
}

void ModelRuntime::check_exchange() {
  const uint32_t err = host_load_acquire(exchange_host()->control.error.value);
  if (err != 0) {
    const uint32_t layer = (err & 0xffffu);
    if ((err & 0xffff0000u) == kExchangeErrorMoeTimeout)
      throw std::runtime_error("CPU expert pool timed out at layer " + std::to_string(int(layer) - 1));
    if ((err & 0xffff0000u) == kExchangeErrorPleTimeout)
      throw std::runtime_error("n-gram rows were not staged in time (PLE wait timed out)");
    throw std::runtime_error("exchange error " + std::to_string(err));
  }
  if (pool_->failed()) throw std::runtime_error("CPU expert pool failed: " + pool_->failure());
}

void ModelRuntime::commit(std::span<const int32_t> tokens) {
  history_ = history_.advanced(tokens.data(), int(tokens.size()));
  position_ += int32_t(tokens.size());
}

void ModelRuntime::overlap_swaps() {
  if (!swapper_ || placement_.borrowed()) return;
  if (++steps_since_plan_ >= swapper_->options().every_steps) {
    swapper_->enqueue_plan(steps_since_plan_);
    steps_since_plan_ = 0;
  }
  swapper_->start();
}

void ModelRuntime::step_boundary() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  pool_->wait_idle();
  prefill_stream_->synchronize();
  if (!swapper_ || placement_.borrowed()) return;
  if (swapper_->boundary()) prefill_stream_->invalidate();
}

void ModelRuntime::collect_routing_stats() { routing_stats_->collect(stream_); }

std::vector<uint32_t> ModelRuntime::routing_counters() {
  std::vector<uint32_t> out(std::size_t(dims::kLayers) * dims::kExperts);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaMemcpy(out.data(), routing_counts_.p, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return out;
}

uint64_t ModelRuntime::force_random_swaps(int32_t n, uint64_t seed) {
  if (placement_.borrowed()) throw std::logic_error("cannot swap experts while prefill storage is borrowed");
  if (!swapper_) return 0;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  pool_->wait_idle();
  prefill_stream_->synchronize();
  swapper_->queue_random(n, seed);
  if (swapper_->drain()) prefill_stream_->invalidate();
  return swapper_->completed();
}

// The borrowed view is sized for this prompt's largest chunk, so a short prompt overwrites (and
// end_prefill_workspace restores) only a small prefix of the loan.
void ModelRuntime::begin_prefill_workspace(std::size_t prompt_tokens) {
  if (!options_.prefill_borrow) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  pool_->wait_idle();
  prefill_stream_->synchronize();
  if (swapper_) swapper_->drain();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  placement_.begin_loan(stream_);
  WorkspaceOptions o = prefill_wsopt_;
  const auto rows = (std::min<std::size_t>(prompt_tokens, std::size_t(o.max_tokens)) + 15) / 16 * 16;
  o.max_tokens = int32_t(std::max<std::size_t>(16, rows));
  // Prefix reuse can start above zero; score rows need the final absolute frontier, not
  // the configured capacity. The prefill grid already uses this same bound per chunk.
  o.max_blocks = std::max<int32_t>(1, (position_ + int32_t(prompt_tokens)) / 4);
  borrowed_bytes_ = WorkspaceBuffer::device_bytes(o);
  borrowed_workspace_ = WorkspaceBuffer::view(o, placement_.loan_data(), borrowed_bytes_);
  active_ws_ = &borrowed_workspace_;
  forward_->set_workspace(*active_ws_);
  WorkspaceBuffer::initialize(*active_ws_, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  prefill_stream_->invalidate();
  prefill_stream_->take_ring_written();  // count only this prompt's ring copies
}

void ModelRuntime::end_prefill_workspace(bool preserve_output) {
  if (!placement_.borrowed()) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  pool_->wait_idle();
  prefill_stream_->synchronize();
  Workspace& dst = workspace_->get();
  const Workspace& src = borrowed_workspace_;
  if (preserve_output && last_rows_ > 0) {
    // Generation only needs the final prompt row. Copy before expert restoration overwrites
    // the loan; persistent KV/GDN/PLE state and snapshots never reside in this allocation.
    const auto row = std::size_t(last_rows_ - 1);
    CUDA_CHECK(cudaMemcpyAsync(dst.X, src.X + row * dims::kStreamHidden,
                               dims::kStreamHidden * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst.tokens, src.tokens + row, 2 * sizeof(int32_t),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst.positions, src.positions + row, sizeof(int32_t),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst.seq, src.seq, sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst.logits, src.logits, std::size_t(src.options.logit_rows) *
                               dims::kVocab * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    if (options_.mtp && last_mtp_rows_ > 0)
      CUDA_CHECK(cudaMemcpyAsync(dst.X_mtp, src.X_mtp + std::size_t(last_mtp_rows_ - 1) * dims::kStreamHidden,
                                 dims::kStreamHidden * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    last_rows_ = 1;
  }
  std::vector<DeviceRange> written = prefill_stream_->take_ring_written();
  written.push_back({placement_.loan_data(), borrowed_bytes_});
  placement_.end_loan(stream_, written);
  active_ws_ = &dst;
  forward_->set_workspace(dst);
  prefill_stream_->invalidate();
}

void ModelRuntime::prefill(std::span<const int32_t> tokens, const PrefillHooks* hooks,
                           std::optional<int32_t> next_token) {
  if (tokens.empty()) return;
  if (position_ + int64_t(tokens.size()) > options_.max_context)
    throw std::length_error("prompt exceeds the context capacity");
  try {
    begin_prefill_workspace(tokens.size());
    std::size_t done = 0;
    while (done < tokens.size()) {
      int32_t T = int32_t(std::min<std::size_t>(std::size_t(options_.prefill_chunk), tokens.size() - done));
      // Snapshot hooks see exact 4096-token boundaries, also with nondivisor chunk sizes.
      if (hooks && hooks->after_chunk) T = std::min(T, 4096 - position_ % 4096);
      const bool last = done + std::size_t(T) == tokens.size();
      const bool has_next = !last || next_token.has_value();
      const int32_t next = last ? next_token.value_or(0) : tokens[done + std::size_t(T)];
      prefill_chunk(tokens.data() + done, T, has_next, next, hooks);
      done += std::size_t(T);
    }
    end_prefill_workspace(true);
  } catch (...) {
    // A callback can stop before later layers publish: release workers spinning on those
    // doorbells before waiting for idle or restoring any expert backing storage.
    cudaStreamSynchronize(stream_);
    host_store_release(exchange_host()->control.abort.value, 1);
    end_prefill_workspace(false);
    pool_->wait_idle();
    host_store_release(exchange_host()->control.abort.value, 0);
    throw;
  }
}

void ModelRuntime::prefill_chunk(const int32_t* tokens, int32_t T, bool has_next, int32_t next,
                                 const PrefillHooks* hooks) {
  Workspace& w = ws();
  const int32_t p0 = position_;
  const uint32_t seq = next_seq();

  // Inputs: tokens[0..T] (incl. the next token for the MTP), positions, seq.
  auto* stage = static_cast<int32_t*>(input_staging_->data());
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // staging reuse
  ensure_context(p0 + T);
  // Streamed-expert plan for T and the first ring copies (overlap the n-gram gather below).
  prefill_stream_->begin_chunk(T, stream_);
  std::memcpy(stage, tokens, sizeof(int32_t) * std::size_t(T));
  stage[T] = has_next ? next : 0;
  int32_t* pos = stage + T + 1;
  for (int32_t t = 0; t < T; ++t) pos[t] = p0 + t;
  auto* seq_slot = reinterpret_cast<uint32_t*>(pos + T);
  *seq_slot = seq;
  CUDA_CHECK(cudaMemcpyAsync(w.tokens, stage, sizeof(int32_t) * std::size_t(T + 1), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(w.positions, pos, sizeof(int32_t) * std::size_t(T), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(w.seq, seq_slot, sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));

  // n-gram rows for the chunk (host-mapped staging, read by the PLE kernel).
  ngram_->gather_tokens(tokens, T, history_, ple_staging_host());

  StepIo io;
  io.T = T;
  io.tokens = w.tokens;
  io.positions = w.positions;
  io.seq = w.seq;
  io.pass = Pass::Prefill;
  io.last_position = p0 + T - 1;

  pool_->begin(seq, cpu::PoolMode::Prefill);
  forward_->embed(io, stream_);
  if (hooks && hooks->after_layer) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    hooks->after_layer(-1, p0, T);
  }
  for (int32_t l = 0; l < dims::kLayers; ++l) {
    prefill_stream_->prefetch(l);
    forward_->layer(l, io, stream_);
    if (hooks && hooks->after_layer) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      check_exchange();
      hooks->after_layer(l, p0, T);
    }
  }
  if (hooks && hooks->on_logits) {
    forward_->final_mixer(weights_.mixer(), w.X, T, w.mix, stream_);
    for (int32_t r0 = 0; r0 < T; r0 += options_.logit_rows) {
      const int32_t rows = std::min(options_.logit_rows, T - r0);
      forward_->lm_head(w.mix + std::size_t(r0) * dims::kHidden, rows, w.logits, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      hooks->on_logits(p0 + r0, rows, w.logits);
    }
  }
  // MTP layer over the positions whose next token is known (keeps the MTP KV cache complete).
  last_mtp_rows_ = 0;
  if (options_.mtp) {
    const int32_t cols = has_next ? T : T - 1;
    last_mtp_rows_ = cols;
    if (cols > 0) {
      StepIo mio = io;
      mio.T = cols;
      mio.tokens = w.tokens + 1;
      mio.last_position = p0 + cols - 1;
      forward_->mtp_layer(w.X, mio, stream_);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  pool_->wait_idle();
  check_exchange();

  history_ = history_.advanced(tokens, T);
  position_ += T;
  last_rows_ = T;
  step_boundary();
  if (hooks && hooks->after_chunk) hooks->after_chunk(position_);
}

std::size_t ModelRuntime::snapshot_bytes() const {
  std::size_t total = 0;
  for (const auto& r : state_->snapshot_regions()) total += (r.bytes + 255) / 256 * 256;
  return total;
}

void ModelRuntime::snapshot_to(void* host) const {
  auto* dst = static_cast<unsigned char*>(host);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (const auto& r : state_->snapshot_regions()) {
    CUDA_CHECK(cudaMemcpyAsync(dst, r.data, r.bytes, cudaMemcpyDeviceToHost, stream_));
    dst += (r.bytes + 255) / 256 * 256;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void ModelRuntime::restore_from(const void* host, int32_t position, const io::NgramHistory& history) {
  if (position < 0 || position > options_.max_context) throw std::out_of_range("snapshot position");
  const auto* src = static_cast<const unsigned char*>(host);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (const auto& r : state_->snapshot_regions()) {
    CUDA_CHECK(cudaMemcpyAsync(r.data, src, r.bytes, cudaMemcpyHostToDevice, stream_));
    src += (r.bytes + 255) / 256 * 256;
  }
  const kernels::GdnStepControl control{0, 0, 0, 0};
  CUDA_CHECK(cudaMemcpyAsync(state_->gdn_control(), &control, sizeof(control), cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  position_ = position;
  history_ = history;
  last_rows_ = 0;
  // Positions from `position` on are dead: elastic slots there serve experts again.
  context_ = position;
  placement_.revalidate(position);
}

std::string ModelRuntime::summary() const {
  std::ostringstream o;
  const PlacementPlan& p = placement_.plan();
  o << "dense " << weights_.vram_bytes() / double(1ull << 30) << " GiB, experts GPU " << p.gpu_experts
    << " (" << p.vram_bytes / double(1ull << 30) << " GiB) / CPU " << p.cpu_experts << " ("
    << p.host_bytes / double(1ull << 30) << " GiB), predicted hit rate "
    << p.predicted_hit_rate << ", state+workspace " << fixed_vram_ / double(1ull << 30) << " GiB, ctx "
    << options_.max_context << ", chunk " << options_.prefill_chunk << ", kv "
    << (options_.kv == kernels::QsaKvFormat::K8V4   ? "k8v4"
        : options_.kv == kernels::QsaKvFormat::K8V8 ? "k8v8"
                                                    : "bf16")
    << ", dense "
    << (options_.dense_bf16 ? "bf16" : "fp8") << ", cpu workers " << pool_->workers() << ", "
    << prefill_stream_->summary() << ", prefill storage " << (options_.prefill_borrow ? "borrowed" : "static");
  return o.str();
}

}  // namespace fngine::model
