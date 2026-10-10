#pragma once

// N-gram (PLE) row store: model hashing of token n-grams into rows of the IQ4_NL table,
// a CLOCK cache of raw 90 B rows in host RAM, and io_uring O_DIRECT reads of the pack's
// page-aligned ngram.bin (see the pack spec, "ngram.bin").
//
// Hashing (modeling_qwen4_exp.py Qwen4ExpTextNGramEmbedding): for token x_t with history
//   h1 = x_{t-1},  h2 = (x_{t-1} == eos) ? eos : x_{t-2}
// (tokens before the sequence start read as eos; a segment starts right after an EOS and the
// EOS itself belongs to the previous segment, so only an EOS at t-1 hides x_{t-2}):
//   mixed2 = x_t*m0 ^ h1*m1,  mixed3 = mixed2 ^ h2*m2      (int64, never overflows, >= 0)
//   row[h] = mixed2 % size[h] + offset[h]  for heads 0..7,
//   row[h] = mixed3 % size[h] + offset[h]  for heads 8..15.
// The EOS rule is fully determined by the last two committed ids, so NgramHistory is the
// complete cross-step state (HF keeps the same two ids in conv_states[2]).
//
// Thread safety: gather() may run concurrently with an in-flight prefetch_chunk() (each uses
// its own io_uring; cache updates are serialized by an internal mutex). gather() itself must
// be called from one thread at a time.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fngine::io {

inline constexpr int kNgramHeads = 16;
inline constexpr int kNgramHeadsPerOrder = 8;
inline constexpr int kNgramRowBytes = 90;
inline constexpr int kNgramRowsPerPage = 45;
inline constexpr int kNgramPageBytes = 4096;
inline constexpr int kNgramHeaderBytes = 4096;

// Hash parameters and table geometry (manifest.json "ngram").
struct NgramParams {
  uint64_t rows = 0;
  int64_t multipliers[3] = {};
  int64_t head_sizes[kNgramHeads] = {};
  int64_t head_offsets[kNgramHeads] = {};
  int32_t eos = 248044;

  // Parses `<pack_dir>/manifest.json` and validates the section (row/page geometry, 3
  // multipliers in [1, 2^63/vocab), 16 heads, every row id < rows). Throws std::runtime_error.
  static NgramParams from_pack(const std::string& pack_dir);
};

// Last two committed token ids (eos-filled at sequence start).
struct NgramHistory {
  int32_t prev1 = 248044;  // x_{t-1}
  int32_t prev2 = 248044;  // x_{t-2}

  static NgramHistory start(int32_t eos) { return NgramHistory{eos, eos}; }
  // History after committing tokens[0..n).
  NgramHistory advanced(const int32_t* tokens, int n) const;
};

// Computes row ids out[T][16] for tokens[0..T) following `history`.
void ngram_row_ids(const NgramParams& p, const int32_t* tokens, int T, const NgramHistory& history,
                   uint32_t* out);

struct NgramStoreOptions {
  size_t cache_mib = 1024;  // raw-row cache (96 B slots); the hash index adds ~6 B per slot
  int queue_depth = 256;    // io_uring depth and number of 4 KiB page buffers per ring
  int io_cpu = 6;           // CPU for the prefetch thread; -1 = no pinning
};

struct NgramStats {
  uint64_t lookups = 0;         // row lookups (16 per token), gather + prefetch
  uint64_t hits = 0;            // lookups served from the cache
  uint64_t misses = 0;          // lookups not resident at lookup time (in-call repeats included)
  uint64_t gather_lookups = 0;  // lookups made by gather() only
  uint64_t gather_hits = 0;     // gather() lookups served from the cache
  uint64_t rows_read = 0;       // distinct rows fetched from disk
  uint64_t pages_read = 0;      // 4 KiB pages read
  uint64_t bytes_read = 0;
  uint64_t gather_calls = 0;
  uint64_t gather_us = 0;       // total wall time inside gather()
  uint64_t prefetch_calls = 0;
  uint64_t prefetch_us = 0;     // total wall time of prefetch jobs on the I/O thread
  uint64_t cached_rows = 0;     // currently resident rows
  uint64_t capacity_rows = 0;
  double hit_rate() const { return lookups ? double(hits) / double(lookups) : 0.0; }
  // Fraction of gather() lookups that did not wait for disk (what prefetch improves).
  double gather_hit_rate() const { return gather_lookups ? double(gather_hits) / double(gather_lookups) : 0.0; }
};

class NgramStore {
 public:
  // Opens `<pack_dir>/ngram.bin` with O_DIRECT and validates its header against `params`.
  NgramStore(const std::string& pack_dir, const NgramParams& params, const NgramStoreOptions& opt = {});
  ~NgramStore();
  NgramStore(const NgramStore&) = delete;
  NgramStore& operator=(const NgramStore&) = delete;

  const NgramParams& params() const { return params_; }

  // out[T][16] row ids (see ngram_row_ids).
  void row_ids(const int32_t* tokens, int T, const NgramHistory& history, uint32_t* out) const {
    ngram_row_ids(params_, tokens, T, history, out);
  }

  // Writes the raw 90 B row of each id to staging[i*90] for i in [0, n); n = T*16 for a step.
  // Cache hits are copied, misses read (one read per distinct page) and inserted.
  // `staging` may be host-mapped/pinned memory. Throws on I/O failure after 3 retries.
  void gather(const uint32_t* row_ids, size_t n, uint8_t* staging);

  // Convenience: row ids for tokens then gather into staging[T][16][90].
  void gather_tokens(const int32_t* tokens, int T, const NgramHistory& history, uint8_t* staging);

  // Queues an asynchronous cache fill for the rows of tokens[0..T) on the I/O thread (the next
  // prefill chunk). Returns immediately; data is copied.
  void prefetch_chunk(const int32_t* tokens, int T, const NgramHistory& history);
  // Blocks until all queued prefetch jobs finished; rethrows a prefetch I/O error.
  void wait_prefetch();

  NgramStats stats() const;
  void reset_stats();

 private:
  struct Ring;
  struct Slot {
    uint8_t data[kNgramRowBytes];
    uint8_t ref;  // CLOCK reference bit: set on hit, clear on insert
    uint8_t pad;
    uint32_t row;
  };
  static_assert(sizeof(Slot) == 96, "n-gram cache slot must be 96 bytes");

  // Cache primitives; caller holds mu_.
  int64_t find_locked(uint32_t row) const;
  void insert_locked(uint32_t row, const uint8_t* data);
  void erase_index_locked(uint32_t row);

  // Splits ids into hits (copied to staging when non-null) and misses, then reads the missing
  // pages through `ring` and inserts the rows. staging == nullptr: cache fill only.
  void fetch(Ring& ring, const uint32_t* row_ids, size_t n, uint8_t* staging, bool from_gather);
  void io_thread_main();

  NgramParams params_;
  NgramStoreOptions opt_;
  int fd_ = -1;

  // CLOCK cache.
  mutable std::mutex mu_;
  std::unique_ptr<Slot[]> slots_;  // uninitialized; [0, used_slots_) valid
  size_t n_slots_ = 0;
  std::vector<uint32_t> index_;    // open addressing (linear probing): slot+1, 0 = empty
  uint64_t index_mask_ = 0;
  int index_shift_ = 0;
  size_t used_slots_ = 0;
  size_t hand_ = 0;

  std::unique_ptr<Ring> gather_ring_;
  std::unique_ptr<Ring> prefetch_ring_;

  // Prefetch thread.
  struct Job {
    std::vector<int32_t> tokens;
    NgramHistory history;
  };
  std::thread io_thread_;
  std::mutex job_mu_;
  std::condition_variable job_cv_;
  std::condition_variable idle_cv_;
  std::deque<Job> jobs_;
  bool busy_ = false;
  bool stop_ = false;
  std::string prefetch_error_;

  struct Counters {
    std::atomic<uint64_t> lookups{0}, hits{0}, misses{0}, gather_lookups{0}, gather_hits{0},
        rows_read{0}, pages_read{0}, bytes_read{0}, gather_calls{0}, gather_us{0}, prefetch_calls{0},
        prefetch_us{0};
  };
  Counters c_;
};

}  // namespace fngine::io
