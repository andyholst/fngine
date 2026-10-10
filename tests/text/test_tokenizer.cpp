// Tokenizer vs Hugging Face AutoTokenizer ids (tests/fixtures/tokenizer_cases.json), plus decode
// round trip, special-token handling and generation_config stop ids.
#include "text/frontend.h"
#include "text/tokenizer.h"
#include "text_test_util.h"

#include <iostream>
#include <vector>

using namespace fngine;
namespace tt = fngine::text_test;

int main() {
    const auto dir = tt::tokenizer_dir_or_skip();
    if (dir.empty()) return tt::kSkip;
    const auto frontend   = text::Frontend::load(dir);
    const auto& tokenizer = frontend.tokenizer();
    const auto fixture    = tt::load_fixture("tokenizer_cases.json");

    int failures = 0;
    std::size_t cases = 0;
    for (const auto& c : fixture.at("cases")) {
        ++cases;
        const std::string text       = c.at("text");
        const auto expected          = c.at("ids").get<std::vector<int>>();
        const std::vector<int> got   = tokenizer.encode(text);
        if (got != expected) {
            ++failures;
            std::cerr << "FAIL encode \"" << tt::escape(text) << "\": got [";
            for (int id : got) std::cerr << id << ' ';
            std::cerr << "] expected [";
            for (int id : expected) std::cerr << id << ' ';
            std::cerr << "]\n";
            continue;
        }
        const std::string decoded = tokenizer.decode(expected);
        if (decoded != text) {
            ++failures;
            std::cerr << "FAIL decode round trip \"" << tt::escape(text) << "\" -> \""
                      << tt::escape(decoded) << "\"\n";
        }
    }

    // Special tokens are atomic, and skip_special_tokens drops them on decode.
    const auto im = tokenizer.encode("<|im_start|>user\nhi<|im_end|>");
    if (im.empty() || im.front() != 248045 || im.back() != 248046 ||
        !tokenizer.is_special_token(248045) ||
        tokenizer.decode(im, text::DecodeOptions{.skip_special_tokens = true}) != "user\nhi") {
        ++failures;
        std::cerr << "FAIL special-token parse/skip\n";
    }
    // Literal (non-parsed) encoding splits marker text into ordinary BPE tokens.
    const auto literal =
        tokenizer.encode("<|im_end|>", text::EncodeOptions{.parse_added_tokens = false});
    if (literal.size() < 2 || tokenizer.decode(literal) != "<|im_end|>") {
        ++failures;
        std::cerr << "FAIL literal encoding of a special-token string\n";
    }
    const std::vector<int> stops = tokenizer.default_stop_token_ids();
    if (stops != std::vector<int>{248046, 248044}) {
        ++failures;
        std::cerr << "FAIL default stop ids\n";
    }
    if (frontend.default_stop_policy().token_ids != std::vector<TokenId>{248046, 248044}) {
        ++failures;
        std::cerr << "FAIL default stop policy\n";
    }
    std::cout << "tokenizer cases: " << cases << ", vocab " << tokenizer.vocab_size()
              << ", failures " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
