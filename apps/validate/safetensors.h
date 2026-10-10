#pragma once

// Minimal read-only safetensors view (mmap) for the golden reference files.
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fngine::validate {

struct StTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const unsigned char* data = nullptr;
  std::size_t bytes = 0;
  [[nodiscard]] int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
    struct stat st{};
    ::fstat(fd_, &st);
    size_ = std::size_t(st.st_size);
    map_ = static_cast<unsigned char*>(::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
    if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed for " + path.string());
    uint64_t hlen = 0;
    std::memcpy(&hlen, map_, 8);
    const auto header = nlohmann::json::parse(map_ + 8, map_ + 8 + hlen);
    const unsigned char* data = map_ + 8 + hlen;
    for (const auto& [name, v] : header.items()) {
      if (name == "__metadata__") continue;
      StTensor t;
      t.dtype = v.at("dtype").get<std::string>();
      t.shape = v.at("shape").get<std::vector<int64_t>>();
      const auto off = v.at("data_offsets").get<std::vector<uint64_t>>();
      t.data = data + off[0];
      t.bytes = off[1] - off[0];
      tensors_[name] = t;
    }
  }
  ~SafeTensors() {
    if (map_ && map_ != MAP_FAILED) ::munmap(map_, size_);
    if (fd_ >= 0) ::close(fd_);
  }
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  [[nodiscard]] bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  [[nodiscard]] const StTensor& at(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
  }

 private:
  int fd_ = -1;
  std::size_t size_ = 0;
  unsigned char* map_ = nullptr;
  std::map<std::string, StTensor> tensors_;
};

inline float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) { man <<= 1; --exp; }
      man &= 0x3ffu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline float bf16_bits_to_float(uint16_t h) {
  const uint32_t bits = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Element i of a float tensor (F32 / F16 / BF16) as float.
inline float st_float(const StTensor& t, int64_t i) {
  if (t.dtype == "F32") { float f; std::memcpy(&f, t.data + 4 * i, 4); return f; }
  uint16_t h;
  std::memcpy(&h, t.data + 2 * i, 2);
  if (t.dtype == "F16") return half_to_float(h);
  if (t.dtype == "BF16") return bf16_bits_to_float(h);
  throw std::runtime_error("unsupported float dtype " + t.dtype);
}

inline int32_t st_int(const StTensor& t, int64_t i) {
  if (t.dtype == "I32") { int32_t v; std::memcpy(&v, t.data + 4 * i, 4); return v; }
  if (t.dtype == "I64") { int64_t v; std::memcpy(&v, t.data + 8 * i, 8); return int32_t(v); }
  throw std::runtime_error("unsupported int dtype " + t.dtype);
}

}  // namespace fngine::validate
