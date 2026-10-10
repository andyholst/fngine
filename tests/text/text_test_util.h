// Helpers shared by the text frontend tests (host-only; no CUDA headers).
#pragma once

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fngine::text_test {

inline constexpr int kSkip = 77;

// <FNGINE_TEST_PACK>/tokenizer, or empty after printing a skip line.
inline std::filesystem::path tokenizer_dir_or_skip() {
    const char* pack = std::getenv("FNGINE_TEST_PACK");
    if (pack == nullptr || *pack == '\0') {
        std::printf("skip: FNGINE_TEST_PACK is not set\n");
        return {};
    }
    std::filesystem::path dir = std::filesystem::path(pack) / "tokenizer";
    if (!std::filesystem::exists(dir / "tokenizer.json")) {
        std::printf("skip: %s has no tokenizer.json\n", dir.c_str());
        return {};
    }
    return dir;
}

// Ordered so that re-serialized tool JSON keeps the fixture's key order (as Python does).
inline nlohmann::ordered_json load_fixture(const char* name) {
    const auto path = std::filesystem::path(FNGINE_SOURCE_DIR) / "tests" / "fixtures" / name;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open fixture " + path.string());
    return nlohmann::ordered_json::parse(file);
}

inline std::string escape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (c == '\n')
            out << "\\n";
        else if (c < 0x20)
            out << "\\x" << std::hex << int(c) << std::dec;
        else
            out << c;
    }
    return out.str();
}

} // namespace fngine::text_test
