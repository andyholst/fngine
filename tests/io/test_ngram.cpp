// N-gram store test:
//  1. row ids == HF fixture (tests/fixtures/ngram_cases.json) for every case, fed as the
//     fixture's chunks, one token at a time, and as a single call;
//  2. (FNGINE_TEST_PACK) gathered bytes == pread of the pack rows for 10,000 random rows through a
//     small cache (hits, duplicates, evictions), prefetch + gather, and a cache-less store;
//  3. (FNGINE_TEST_REF with golden-manifest.json listing g2.safetensors) cache hit rate and
//     gather µs on the G2 token stream.
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/test_util.h"
#include "io/ngram_store.h"

using fngine::io::NgramHistory;
using fngine::io::NgramParams;
using fngine::io::NgramStore;
using fngine::io::NgramStoreOptions;
using fngine::io::kNgramHeads;
using fngine::io::kNgramRowBytes;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

bool exists(const std::string& p) { return ::access(p.c_str(), R_OK) == 0; }

nlohmann::json load_json(const std::string& path) {
  std::ifstream in(path);
  FNGINE_CHECK(in.good());
  nlohmann::json j;
  in >> j;
  return j;
}

NgramParams params_from_fixture(const nlohmann::json& p) {
  NgramParams r;
  for (int i = 0; i < 3; ++i) r.multipliers[i] = p["multipliers"][i].get<int64_t>();
  for (int h = 0; h < kNgramHeads; ++h) {
    r.head_sizes[h] = p["head_sizes"][h].get<int64_t>();
    r.head_offsets[h] = p["head_offsets"][h].get<int64_t>();
  }
  r.eos = p["eos"].get<int32_t>();
  r.rows = uint64_t(r.head_offsets[15] + r.head_sizes[15]);
  return r;
}

// Feeds tokens in the given chunk sizes, carrying NgramHistory across calls.
std::vector<uint32_t> ids_in_chunks(const NgramParams& p, const std::vector<int32_t>& tokens,
                                    const std::vector<int>& chunks) {
  std::vector<uint32_t> out(tokens.size() * kNgramHeads);
  NgramHistory h = NgramHistory::start(p.eos);
  size_t pos = 0;
  for (int n : chunks) {
    fngine::io::ngram_row_ids(p, tokens.data() + pos, n, h, out.data() + pos * kNgramHeads);
    h = h.advanced(tokens.data() + pos, n);
    pos += size_t(n);
  }
  FNGINE_CHECK(pos == tokens.size());
  return out;
}

void test_row_ids(const std::string& pack) {
  const auto fx = load_json(std::string(FNGINE_SOURCE_DIR) + "/tests/fixtures/ngram_cases.json");
  const NgramParams p = params_from_fixture(fx["params"]);
  if (!pack.empty()) {
    const NgramParams m = NgramParams::from_pack(pack);
    bool same = m.eos == p.eos;
    for (int i = 0; i < 3; ++i) same &= m.multipliers[i] == p.multipliers[i];
    for (int h = 0; h < kNgramHeads; ++h)
      same &= m.head_sizes[h] == p.head_sizes[h] && m.head_offsets[h] == p.head_offsets[h];
    expect(same, "pack manifest ngram params == fixture params");
  }
  int n_cases = 0;
  for (const auto& c : fx["cases"]) {
    const std::string name = c["name"].get<std::string>();
    const auto tokens = c["tokens"].get<std::vector<int32_t>>();
    const auto chunks = c["chunks"].get<std::vector<int>>();
    std::vector<uint32_t> want;
    for (const auto& r : c["row_ids"])
      for (const auto& v : r) want.push_back(v.get<uint32_t>());
    FNGINE_CHECK(want.size() == tokens.size() * kNgramHeads);
    const std::vector<int> whole{int(tokens.size())};
    const std::vector<int> singles(tokens.size(), 1);
    std::vector<int> pairs;
    for (size_t i = 0; i < tokens.size(); i += 2) pairs.push_back(int(std::min<size_t>(2, tokens.size() - i)));
    expect(ids_in_chunks(p, tokens, chunks) == want, name + ": fixture chunks");
    expect(ids_in_chunks(p, tokens, whole) == want, name + ": single call");
    expect(ids_in_chunks(p, tokens, singles) == want, name + ": one token at a time");
    expect(ids_in_chunks(p, tokens, pairs) == want, name + ": two tokens at a time");
    ++n_cases;
  }
  std::printf("row ids: %d fixture cases checked (fixture chunks, whole, 1-at-a-time, 2-at-a-time)\n",
              n_cases);
}

struct PackReader {
  int fd = -1;
  explicit PackReader(const std::string& pack) {
    fd = ::open((pack + "/ngram.bin").c_str(), O_RDONLY | O_CLOEXEC);
    FNGINE_CHECK(fd >= 0);
  }
  ~PackReader() { ::close(fd); }
  void row(uint32_t r, uint8_t* out) const {
    const off_t off = 4096 + off_t(r / 45) * 4096 + off_t(r % 45) * 90;
    FNGINE_CHECK(::pread(fd, out, kNgramRowBytes, off) == kNgramRowBytes);
  }
};

bool check_rows(const PackReader& pr, const uint32_t* ids, size_t n, const uint8_t* staging) {
  uint8_t want[kNgramRowBytes];
  for (size_t i = 0; i < n; ++i) {
    pr.row(ids[i], want);
    if (std::memcmp(want, staging + i * kNgramRowBytes, kNgramRowBytes) != 0) {
      std::printf("FAIL row %u (index %zu) bytes differ from pack\n", ids[i], i);
      return false;
    }
  }
  return true;
}

void test_gather(const std::string& pack) {
  const NgramParams p = NgramParams::from_pack(pack);
  const PackReader pr(pack);
  fngine::test::Rng rng(42);

  // Small cache (1 MiB = 10922 rows) so 10,000 random rows plus re-references see hits,
  // in-batch duplicates, same-page neighbours and CLOCK evictions.
  NgramStoreOptions opt;
  opt.cache_mib = 1;
  NgramStore store(pack, p, opt);
  std::vector<uint32_t> history;
  size_t checked = 0;
  bool ok = true;
  // At least 10,000 rows, and enough distinct reads to fill the cache and evict ~2,000 rows.
  while (checked < 10000 || store.stats().rows_read < store.stats().capacity_rows + 2000) {
    const size_t n = 144;  // T=9 verify step
    std::vector<uint32_t> ids(n);
    for (size_t i = 0; i < n; ++i) {
      const double u = rng.uniform();
      if (u < 0.3 && !history.empty()) {
        ids[i] = history[rng.next() % history.size()];  // re-reference (likely hit)
      } else if (u < 0.4 && i > 0) {
        ids[i] = ids[rng.next() % i];  // in-batch duplicate
      } else if (u < 0.5 && i > 0) {
        const uint32_t base = ids[i - 1] - ids[i - 1] % 45;  // page neighbour
        ids[i] = std::min<uint32_t>(base + uint32_t(rng.next() % 45), uint32_t(p.rows - 1));
      } else {
        ids[i] = uint32_t(rng.next() % p.rows);
      }
    }
    if (checked == 0) {  // edge rows: first, last, last full page boundary
      ids[0] = 0;
      ids[1] = uint32_t(p.rows - 1);
      ids[2] = uint32_t((p.rows - 1) / 45 * 45);
      ids[3] = 44;
      ids[4] = 45;
    }
    std::vector<uint8_t> staging(n * kNgramRowBytes, 0xAB);
    store.gather(ids.data(), n, staging.data());
    ok &= check_rows(pr, ids.data(), n, staging.data());
    history.insert(history.end(), ids.begin(), ids.end());
    checked += n;
  }
  const auto s = store.stats();
  expect(ok, "gather bytes == pread for random rows");
  expect(s.hits > 0 && s.misses > 0, "gather saw both hits and misses");
  expect(s.cached_rows == s.capacity_rows, "cache filled to capacity (evictions exercised)");
  std::printf("gather: %zu rows checked against pread; hit rate %.3f, rows read %llu, pages %llu, "
              "cache %llu/%llu rows, %.1f us/gather\n",
              checked, s.hit_rate(), (unsigned long long)s.rows_read, (unsigned long long)s.pages_read,
              (unsigned long long)s.cached_rows, (unsigned long long)s.capacity_rows,
              double(s.gather_us) / double(s.gather_calls));

  // Prefetch then gather: every lookup of the prefetched chunk must hit.
  {
    NgramStore st(pack, p, NgramStoreOptions{});
    std::vector<int32_t> toks(512);
    for (auto& t : toks) t = int32_t(rng.next() % 248320);
    toks[100] = p.eos;
    const NgramHistory h{17, p.eos};
    st.prefetch_chunk(toks.data(), int(toks.size()), h);
    st.wait_prefetch();
    const auto before = st.stats();
    std::vector<uint8_t> staging(toks.size() * kNgramHeads * kNgramRowBytes);
    st.gather_tokens(toks.data(), int(toks.size()), h, staging.data());
    const auto after = st.stats();
    std::vector<uint32_t> ids(toks.size() * kNgramHeads);
    st.row_ids(toks.data(), int(toks.size()), h, ids.data());
    expect(check_rows(pr, ids.data(), ids.size(), staging.data()), "prefetched gather bytes == pread");
    expect(after.misses == before.misses, "gather after prefetch is all hits");
    std::printf("prefetch: %llu rows in %.1f ms on the I/O thread; following gather %.1f us, all hits\n",
                (unsigned long long)before.rows_read, before.prefetch_us / 1e3,
                double(after.gather_us - before.gather_us));
  }

  // No cache: every gather reads from disk.
  {
    NgramStoreOptions o0;
    o0.cache_mib = 0;
    NgramStore st(pack, p, o0);
    std::vector<uint32_t> ids(64);
    for (auto& v : ids) v = uint32_t(rng.next() % p.rows);
    ids[5] = ids[3];
    std::vector<uint8_t> staging(ids.size() * kNgramRowBytes);
    st.gather(ids.data(), ids.size(), staging.data());
    expect(check_rows(pr, ids.data(), ids.size(), staging.data()), "cache-less gather bytes == pread");
  }
}

// Reads tensor "ids" (I32/I64) from a safetensors file.
std::vector<int32_t> read_safetensors_ids(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  FNGINE_CHECK(in.good());
  uint64_t hlen = 0;
  in.read(reinterpret_cast<char*>(&hlen), 8);
  std::string hdr(hlen, '\0');
  in.read(hdr.data(), std::streamsize(hlen));
  const auto j = nlohmann::json::parse(hdr);
  FNGINE_CHECK(j.contains("ids"));
  const auto& t = j["ids"];
  const std::string dt = t["dtype"].get<std::string>();
  const uint64_t a = t["data_offsets"][0].get<uint64_t>(), b = t["data_offsets"][1].get<uint64_t>();
  std::vector<int32_t> ids;
  in.seekg(std::streamoff(8 + hlen + a));
  if (dt == "I32") {
    ids.resize((b - a) / 4);
    in.read(reinterpret_cast<char*>(ids.data()), std::streamsize(b - a));
  } else {
    FNGINE_CHECK(dt == "I64");
    std::vector<int64_t> v((b - a) / 8);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(b - a));
    ids.assign(v.begin(), v.end());
  }
  FNGINE_CHECK(in.good());
  return ids;
}

void bench_g2(const std::string& pack, const std::string& ref) {
  const std::string gm = ref + "/golden-manifest.json";
  if (!exists(gm)) {
    std::printf("skip G2 stream: %s not present\n", gm.c_str());
    return;
  }
  const auto m = load_json(gm);
  if (!m.contains("files") || !m["files"].contains("g2.safetensors") || !exists(ref + "/g2.safetensors")) {
    std::printf("skip G2 stream: g2.safetensors not listed in %s\n", gm.c_str());
    return;
  }
  const auto toks = read_safetensors_ids(ref + "/g2.safetensors");
  const NgramParams p = NgramParams::from_pack(pack);
  const int T = int(toks.size());

  // Prefill: 4096-token chunks, next chunk prefetched while the current one "runs".
  {
    NgramStore st(pack, p, NgramStoreOptions{});
    const int chunk = 4096;
    NgramHistory h = NgramHistory::start(p.eos);
    std::vector<uint8_t> staging(size_t(chunk) * kNgramHeads * kNgramRowBytes);
    std::string per_chunk;
    const auto t0 = std::chrono::steady_clock::now();
    for (int pos = 0; pos < T; pos += chunk) {
      const int n = std::min(chunk, T - pos);
      st.wait_prefetch();
      const uint64_t us0 = st.stats().gather_us;
      st.gather_tokens(toks.data() + pos, n, h, staging.data());
      per_chunk += (per_chunk.empty() ? "" : ", ") + std::to_string(st.stats().gather_us - us0);
      const NgramHistory next = h.advanced(toks.data() + pos, n);
      if (pos + n < T) st.prefetch_chunk(toks.data() + pos + n, std::min(chunk, T - pos - n), next);
      h = next;
    }
    st.wait_prefetch();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const auto s = st.stats();
    std::printf("G2 prefill (%d tokens, chunk %d, next chunk prefetched): gather hit rate %.3f, "
                "all-lookup hit rate %.3f, gather us per chunk [%s], prefetch %.1f ms total, "
                "%llu pages read, wall %.1f ms\n",
                T, chunk, s.gather_hit_rate(), s.hit_rate(), per_chunk.c_str(), s.prefetch_us / 1e3,
                (unsigned long long)s.pages_read, ms);
  }
  // Decode: one token per gather, cold cache.
  {
    NgramStore st(pack, p, NgramStoreOptions{});
    NgramHistory h = NgramHistory::start(p.eos);
    uint8_t staging[kNgramHeads * kNgramRowBytes];
    for (int t = 0; t < T; ++t) {
      st.gather_tokens(&toks[size_t(t)], 1, h, staging);
      h = h.advanced(&toks[size_t(t)], 1);
    }
    const auto s = st.stats();
    std::printf("G2 decode (%d steps of T=1, cold cache): hit rate %.3f, gather %.1f us/step\n", T,
                s.gather_hit_rate(), double(s.gather_us) / double(s.gather_calls));
  }
}

}  // namespace

int main() {
  const char* pack_env = std::getenv("FNGINE_TEST_PACK");
  std::string pack = pack_env ? pack_env : "";
  if (!pack.empty() && !exists(pack + "/ngram.bin")) {
    std::printf("FNGINE_TEST_PACK=%s has no ngram.bin; pack checks skipped\n", pack.c_str());
    pack.clear();
  }
  try {
    test_row_ids(pack);
    if (pack.empty()) {
      std::printf("skip pack gather checks: FNGINE_TEST_PACK is not set\n");
    } else {
      test_gather(pack);
      const char* ref = std::getenv("FNGINE_TEST_REF");
      if (ref && *ref) bench_g2(pack, ref);
      else std::printf("skip G2 stream: FNGINE_TEST_REF is not set\n");
    }
  } catch (const std::exception& e) {
    std::printf("FAIL exception: %s\n", e.what());
    return 1;
  }
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS ngram\n");
  return 0;
}
