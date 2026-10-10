#include "io/ngram_store.h"

#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace fngine::io {
namespace {

constexpr int kMaxRetries = 3;
constexpr uint64_t kHashMul = 0x9E3779B97F4A7C15ull;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("ngram: " + msg); }

uint64_t now_us() {
  return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

inline uint64_t page_offset(uint64_t page) {
  return uint64_t(kNgramHeaderBytes) + page * uint64_t(kNgramPageBytes);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Params and hashing

NgramParams NgramParams::from_pack(const std::string& pack_dir) {
  const std::string path = pack_dir + "/manifest.json";
  std::ifstream in(path);
  if (!in) fail("cannot open " + path);
  nlohmann::json m;
  try {
    in >> m;
  } catch (const std::exception& e) {
    fail("cannot parse " + path + ": " + e.what());
  }
  if (!m.contains("ngram") || !m["ngram"].is_object()) fail(path + " has no \"ngram\" section");
  const auto& g = m["ngram"];
  NgramParams p;
  try {
    p.rows = g.at("rows").get<uint64_t>();
    if (g.at("row_bytes").get<int64_t>() != kNgramRowBytes ||
        g.at("rows_per_page").get<int64_t>() != kNgramRowsPerPage ||
        g.at("header_bytes").get<int64_t>() != kNgramHeaderBytes || g.at("dim").get<int64_t>() != 160 ||
        g.at("type").get<std::string>() != "IQ4_NL" || g.at("n_heads").get<int64_t>() != kNgramHeads ||
        g.at("heads_per_ngram").get<int64_t>() != kNgramHeadsPerOrder ||
        g.at("ngram_size").get<int64_t>() != 3)
      fail("unsupported ngram geometry in " + path);
    const auto& mul = g.at("multipliers");
    const auto& sizes = g.at("head_sizes");
    const auto& offs = g.at("head_offsets");
    if (mul.size() != 3 || sizes.size() != kNgramHeads || offs.size() != kNgramHeads)
      fail("bad multipliers/head_sizes/head_offsets lengths in " + path);
    for (int i = 0; i < 3; ++i) p.multipliers[i] = mul[i].get<int64_t>();
    for (int h = 0; h < kNgramHeads; ++h) {
      p.head_sizes[h] = sizes[h].get<int64_t>();
      p.head_offsets[h] = offs[h].get<int64_t>();
    }
    p.eos = g.at("eos").get<int32_t>();
  } catch (const nlohmann::json::exception& e) {
    fail("bad ngram section in " + path + ": " + e.what());
  }
  // Products id*m must fit int64 (HF computes in torch.long): m < 2^63 / vocab.
  int64_t vocab = 248320;
  if (m.contains("config") && m["config"].contains("vocab_size"))
    vocab = m["config"]["vocab_size"].get<int64_t>();
  for (int i = 0; i < 3; ++i)
    if (p.multipliers[i] <= 0 || p.multipliers[i] > std::numeric_limits<int64_t>::max() / vocab)
      fail("multiplier out of range in " + path);
  if (p.eos < 0 || p.eos >= vocab) fail("eos id out of range");
  for (int h = 0; h < kNgramHeads; ++h)
    if (p.head_sizes[h] <= 0 || p.head_offsets[h] < 0 ||
        uint64_t(p.head_offsets[h] + p.head_sizes[h]) > p.rows ||
        p.rows > std::numeric_limits<uint32_t>::max())
      fail("head " + std::to_string(h) + " exceeds the table");
  return p;
}

NgramHistory NgramHistory::advanced(const int32_t* tokens, int n) const {
  NgramHistory h = *this;
  if (n >= 2) {
    h.prev2 = tokens[n - 2];
    h.prev1 = tokens[n - 1];
  } else if (n == 1) {
    h.prev2 = h.prev1;
    h.prev1 = tokens[0];
  }
  return h;
}

void ngram_row_ids(const NgramParams& p, const int32_t* tokens, int T, const NgramHistory& history,
                   uint32_t* out) {
  const int64_t eos = p.eos;
  int64_t x1 = history.prev1, x2 = history.prev2;
  for (int t = 0; t < T; ++t) {
    const int64_t x0 = tokens[t];
    // A history token is visible only within the current EOS-delimited segment: an EOS at t-1
    // ends the previous segment, so it hides x_{t-2} (x_{t-1} itself reads as EOS anyway).
    const int64_t h2 = (x1 == eos) ? eos : x2;
    const uint64_t mixed2 = uint64_t(x0 * p.multipliers[0]) ^ uint64_t(x1 * p.multipliers[1]);
    const uint64_t mixed3 = mixed2 ^ uint64_t(h2 * p.multipliers[2]);
    uint32_t* o = out + size_t(t) * kNgramHeads;
    for (int h = 0; h < kNgramHeadsPerOrder; ++h)
      o[h] = uint32_t(mixed2 % uint64_t(p.head_sizes[h]) + uint64_t(p.head_offsets[h]));
    for (int h = kNgramHeadsPerOrder; h < kNgramHeads; ++h)
      o[h] = uint32_t(mixed3 % uint64_t(p.head_sizes[h]) + uint64_t(p.head_offsets[h]));
    x2 = x1;
    x1 = x0;
  }
}

// ---------------------------------------------------------------------------------------------
// io_uring context: one ring and queue_depth 4 KiB O_DIRECT page buffers.

struct NgramStore::Ring {
  io_uring ring{};
  uint8_t* buffers = nullptr;
  int depth = 0;
  std::vector<int> free_buffers;

  explicit Ring(int qd) : depth(qd) {
    const int rc = io_uring_queue_init(unsigned(qd), &ring, 0);
    if (rc < 0) fail(std::string("io_uring_queue_init: ") + std::strerror(-rc));
    void* p = nullptr;
    if (posix_memalign(&p, kNgramPageBytes, size_t(qd) * kNgramPageBytes) != 0) {
      io_uring_queue_exit(&ring);
      fail("posix_memalign for page buffers failed");
    }
    buffers = static_cast<uint8_t*>(p);
    free_buffers.reserve(size_t(qd));
    for (int i = qd - 1; i >= 0; --i) free_buffers.push_back(i);
  }
  ~Ring() {
    io_uring_queue_exit(&ring);
    std::free(buffers);
  }
  uint8_t* buffer(int i) const { return buffers + size_t(i) * kNgramPageBytes; }
};

// ---------------------------------------------------------------------------------------------
// Store

NgramStore::NgramStore(const std::string& pack_dir, const NgramParams& params,
                       const NgramStoreOptions& opt)
    : params_(params), opt_(opt) {
  if (opt_.queue_depth < 1 || opt_.queue_depth > 4096) fail("queue_depth must be in [1, 4096]");
  const std::string path = pack_dir + "/ngram.bin";
  fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd_ < 0) fail("cannot open " + path + " with O_DIRECT: " + std::strerror(errno));

  struct stat st{};
  if (fstat(fd_, &st) != 0) fail("fstat " + path + ": " + std::strerror(errno));
  const uint64_t pages = (params_.rows + kNgramRowsPerPage - 1) / kNgramRowsPerPage;
  if (uint64_t(st.st_size) < page_offset(pages)) {
    ::close(fd_);
    fail(path + " is truncated: " + std::to_string(st.st_size) + " bytes, need " +
         std::to_string(page_offset(pages)));
  }

  void* hdr = nullptr;
  if (posix_memalign(&hdr, kNgramPageBytes, kNgramHeaderBytes) != 0) fail("posix_memalign failed");
  const ssize_t got = ::pread(fd_, hdr, kNgramHeaderBytes, 0);
  const auto* h = static_cast<const uint8_t*>(hdr);
  uint32_t version = 0, row_bytes = 0, rows_per_page = 0, dim = 0, ggml_type = 0;
  uint64_t rows = 0;
  const bool magic_ok = got == kNgramHeaderBytes && std::memcmp(h, "FNGR", 4) == 0;
  if (magic_ok) {
    std::memcpy(&version, h + 4, 4);
    std::memcpy(&rows, h + 8, 8);
    std::memcpy(&row_bytes, h + 16, 4);
    std::memcpy(&rows_per_page, h + 20, 4);
    std::memcpy(&dim, h + 24, 4);
    std::memcpy(&ggml_type, h + 28, 4);
  }
  std::free(hdr);
  if (!magic_ok || version != 1 || rows != params_.rows || row_bytes != kNgramRowBytes ||
      rows_per_page != kNgramRowsPerPage || dim != 160 || ggml_type != 20) {
    ::close(fd_);
    fail(path + ": bad header (want FNGR v1, rows=" + std::to_string(params_.rows) +
         ", 90 B rows, 45 rows/page, dim 160, IQ4_NL)");
  }

  // Cache: 96 B slots; index = next power of two >= 1.5 * slots (load factor <= 2/3).
  n_slots_ = (opt_.cache_mib << 20) / sizeof(Slot);
  n_slots_ = std::min<size_t>(n_slots_, std::numeric_limits<uint32_t>::max() - 1);
  if (n_slots_ > 0) {
    slots_.reset(new Slot[n_slots_]);
    size_t index_size = 2;
    int bits = 1;
    while (index_size < n_slots_ + n_slots_ / 2) {
      index_size <<= 1;
      ++bits;
    }
    index_.assign(index_size, 0u);
    index_mask_ = index_size - 1;
    index_shift_ = 64 - bits;
  }

  gather_ring_ = std::make_unique<Ring>(opt_.queue_depth);
  prefetch_ring_ = std::make_unique<Ring>(opt_.queue_depth);
  io_thread_ = std::thread([this] { io_thread_main(); });
}

NgramStore::~NgramStore() {
  {
    std::lock_guard<std::mutex> lk(job_mu_);
    stop_ = true;
    jobs_.clear();
  }
  job_cv_.notify_all();
  if (io_thread_.joinable()) io_thread_.join();
  if (fd_ >= 0) ::close(fd_);
}

int64_t NgramStore::find_locked(uint32_t row) const {
  if (n_slots_ == 0) return -1;
  uint64_t i = (uint64_t(row) * kHashMul) >> index_shift_;
  for (;;) {
    const uint32_t e = index_[i];
    if (e == 0) return -1;
    if (slots_[e - 1].row == row) return int64_t(e - 1);
    i = (i + 1) & index_mask_;
  }
}

void NgramStore::erase_index_locked(uint32_t row) {
  uint64_t i = (uint64_t(row) * kHashMul) >> index_shift_;
  while (index_[i] != 0 && slots_[index_[i] - 1].row != row) i = (i + 1) & index_mask_;
  if (index_[i] == 0) return;
  // Backward-shift deletion keeps linear-probing chains intact without tombstones.
  uint64_t j = i;
  for (;;) {
    j = (j + 1) & index_mask_;
    const uint32_t e = index_[j];
    if (e == 0) break;
    const uint64_t home = (uint64_t(slots_[e - 1].row) * kHashMul) >> index_shift_;
    const bool stays = (i <= j) ? (i < home && home <= j) : (i < home || home <= j);
    if (stays) continue;
    index_[i] = e;
    i = j;
  }
  index_[i] = 0;
}

void NgramStore::insert_locked(uint32_t row, const uint8_t* data) {
  if (n_slots_ == 0 || find_locked(row) >= 0) return;
  size_t s;
  if (used_slots_ < n_slots_) {
    s = used_slots_++;
  } else {
    for (;;) {
      s = hand_;
      hand_ = (hand_ + 1 == n_slots_) ? 0 : hand_ + 1;
      if (slots_[s].ref) {
        slots_[s].ref = 0;
        continue;
      }
      erase_index_locked(slots_[s].row);
      break;
    }
  }
  Slot& slot = slots_[s];
  std::memcpy(slot.data, data, kNgramRowBytes);
  slot.ref = 0;
  slot.row = row;
  uint64_t i = (uint64_t(row) * kHashMul) >> index_shift_;
  while (index_[i] != 0) i = (i + 1) & index_mask_;
  index_[i] = uint32_t(s + 1);
}

void NgramStore::fetch(Ring& ring, const uint32_t* row_ids, size_t n, uint8_t* staging,
                       bool from_gather) {
  // 1. Cache lookups. misses: (row, destination index).
  std::vector<std::pair<uint32_t, uint32_t>> misses;
  uint64_t hits = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < n; ++i) {
      const uint32_t row = row_ids[i];
      if (row >= params_.rows) fail("row id " + std::to_string(row) + " out of range");
      const int64_t s = find_locked(row);
      if (s >= 0) {
        Slot& slot = slots_[size_t(s)];
        slot.ref = 1;
        if (staging) std::memcpy(staging + i * kNgramRowBytes, slot.data, kNgramRowBytes);
        ++hits;
      } else {
        misses.emplace_back(row, uint32_t(i));
      }
    }
  }
  c_.lookups += n;
  c_.hits += hits;
  c_.misses += misses.size();
  if (from_gather) {
    c_.gather_lookups += n;
    c_.gather_hits += hits;
  }
  if (misses.empty()) return;

  // 2. Group misses by page (sorted by row so duplicates and page-mates are adjacent).
  std::sort(misses.begin(), misses.end());
  struct PageGroup {
    uint64_t page;
    uint32_t begin, end;  // range in misses
    int retries;
  };
  std::vector<PageGroup> groups;
  for (uint32_t i = 0; i < misses.size(); ++i) {
    const uint64_t page = misses[i].first / kNgramRowsPerPage;
    if (groups.empty() || groups.back().page != page) groups.push_back({page, i, i, 0});
    groups.back().end = i + 1;
  }

  // 3. Reads: keep up to `depth` pages in flight; user_data = group << 16 | buffer.
  size_t next = 0, inflight = 0, done = 0;
  uint64_t rows_read = 0;
  std::string error;
  std::vector<size_t> retry_queue;
  auto submit = [&](size_t g) {
    const int b = ring.free_buffers.back();
    ring.free_buffers.pop_back();
    io_uring_sqe* sqe = io_uring_get_sqe(&ring.ring);
    io_uring_prep_read(sqe, fd_, ring.buffer(b), kNgramPageBytes, page_offset(groups[g].page));
    io_uring_sqe_set_data64(sqe, (uint64_t(g) << 16) | uint64_t(b));
    ++inflight;
  };
  while (done < groups.size()) {
    if (error.empty()) {
      while (!ring.free_buffers.empty() && (!retry_queue.empty() || next < groups.size())) {
        if (!retry_queue.empty()) {
          submit(retry_queue.back());
          retry_queue.pop_back();
        } else {
          submit(next++);
        }
      }
    }
    if (inflight == 0) break;  // only reachable after an error
    const int rc = io_uring_submit_and_wait(&ring.ring, 1);
    if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY)
      fail(std::string("io_uring_submit_and_wait: ") + std::strerror(-rc));
    io_uring_cqe* cqe = nullptr;
    unsigned head;
    unsigned seen = 0;
    io_uring_for_each_cqe(&ring.ring, head, cqe) {
      ++seen;
      const uint64_t ud = io_uring_cqe_get_data64(cqe);
      const size_t g = size_t(ud >> 16);
      const int b = int(ud & 0xffff);
      const int res = cqe->res;
      --inflight;
      PageGroup& grp = groups[g];
      if (res != kNgramPageBytes) {
        if (grp.retries < kMaxRetries && error.empty()) {
          ++grp.retries;
          retry_queue.push_back(g);
        } else if (error.empty()) {
          error = "read of page " + std::to_string(grp.page) + " failed after " +
                  std::to_string(kMaxRetries) + " retries: " +
                  (res < 0 ? std::string(std::strerror(-res)) : "short read " + std::to_string(res));
        }
        if (!error.empty()) ++done;  // abandoned
        ring.free_buffers.push_back(b);
        continue;
      }
      const uint8_t* page = ring.buffer(b);
      if (staging) {
        for (uint32_t k = grp.begin; k < grp.end; ++k) {
          const auto [row, dst] = misses[k];
          std::memcpy(staging + size_t(dst) * kNgramRowBytes,
                      page + size_t(row % kNgramRowsPerPage) * kNgramRowBytes, kNgramRowBytes);
        }
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        for (uint32_t k = grp.begin; k < grp.end; ++k) {
          if (k > grp.begin && misses[k].first == misses[k - 1].first) continue;
          insert_locked(misses[k].first,
                        page + size_t(misses[k].first % kNgramRowsPerPage) * kNgramRowBytes);
          ++rows_read;
        }
      }
      ring.free_buffers.push_back(b);
      ++done;
    }
    io_uring_cq_advance(&ring.ring, seen);
    if (!error.empty() && inflight == 0) break;
  }
  c_.rows_read += rows_read;
  const uint64_t pages = groups.size();
  c_.pages_read += pages;
  c_.bytes_read += pages * kNgramPageBytes;
  if (!error.empty()) {
    ring.free_buffers.clear();
    for (int i = ring.depth - 1; i >= 0; --i) ring.free_buffers.push_back(i);
    fail(error);
  }
}

void NgramStore::gather(const uint32_t* row_ids, size_t n, uint8_t* staging) {
  const uint64_t t0 = now_us();
  fetch(*gather_ring_, row_ids, n, staging, true);
  c_.gather_calls += 1;
  c_.gather_us += now_us() - t0;
}

void NgramStore::gather_tokens(const int32_t* tokens, int T, const NgramHistory& history,
                               uint8_t* staging) {
  std::vector<uint32_t> ids(size_t(T) * kNgramHeads);
  ngram_row_ids(params_, tokens, T, history, ids.data());
  gather(ids.data(), ids.size(), staging);
}

void NgramStore::prefetch_chunk(const int32_t* tokens, int T, const NgramHistory& history) {
  if (T <= 0) return;
  {
    std::lock_guard<std::mutex> lk(job_mu_);
    jobs_.push_back(Job{std::vector<int32_t>(tokens, tokens + T), history});
  }
  job_cv_.notify_one();
}

void NgramStore::wait_prefetch() {
  std::unique_lock<std::mutex> lk(job_mu_);
  idle_cv_.wait(lk, [this] { return jobs_.empty() && !busy_; });
  if (!prefetch_error_.empty()) {
    std::string e = std::move(prefetch_error_);
    prefetch_error_.clear();
    throw std::runtime_error(e);
  }
}

void NgramStore::io_thread_main() {
  if (opt_.io_cpu >= 0 && opt_.io_cpu < CPU_SETSIZE) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(opt_.io_cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);  // best effort
  }
  pthread_setname_np(pthread_self(), "fngine-ngram-io");
  std::vector<uint32_t> ids;
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lk(job_mu_);
      job_cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
      if (stop_) return;
      job = std::move(jobs_.front());
      jobs_.pop_front();
      busy_ = true;
    }
    const uint64_t t0 = now_us();
    std::string err;
    try {
      const int T = int(job.tokens.size());
      ids.resize(size_t(T) * kNgramHeads);
      ngram_row_ids(params_, job.tokens.data(), T, job.history, ids.data());
      fetch(*prefetch_ring_, ids.data(), ids.size(), nullptr, false);
    } catch (const std::exception& e) {
      err = e.what();
    }
    c_.prefetch_calls += 1;
    c_.prefetch_us += now_us() - t0;
    {
      std::lock_guard<std::mutex> lk(job_mu_);
      busy_ = false;
      if (!err.empty() && prefetch_error_.empty()) prefetch_error_ = err;
    }
    idle_cv_.notify_all();
  }
}

NgramStats NgramStore::stats() const {
  NgramStats s;
  s.lookups = c_.lookups;
  s.hits = c_.hits;
  s.misses = c_.misses;
  s.gather_lookups = c_.gather_lookups;
  s.gather_hits = c_.gather_hits;
  s.rows_read = c_.rows_read;
  s.pages_read = c_.pages_read;
  s.bytes_read = c_.bytes_read;
  s.gather_calls = c_.gather_calls;
  s.gather_us = c_.gather_us;
  s.prefetch_calls = c_.prefetch_calls;
  s.prefetch_us = c_.prefetch_us;
  {
    std::lock_guard<std::mutex> lk(mu_);
    s.cached_rows = used_slots_;
  }
  s.capacity_rows = n_slots_;
  return s;
}

void NgramStore::reset_stats() {
  for (auto* a : {&c_.lookups, &c_.hits, &c_.misses, &c_.gather_lookups, &c_.gather_hits,
                  &c_.rows_read, &c_.pages_read, &c_.bytes_read, &c_.gather_calls, &c_.gather_us,
                  &c_.prefetch_calls, &c_.prefetch_us})
    a->store(0);
}

}  // namespace fngine::io
