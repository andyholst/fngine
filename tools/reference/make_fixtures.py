#!/usr/bin/env python3
"""Write tokenizer / chat-template fixtures from the HF tokenizer.

    tools/.venv/bin/python tools/reference/make_fixtures.py [--hf /mnt/data/fngine/hf] [--out tests/fixtures]
"""
import argparse
import json
import os

from transformers import AutoTokenizer

TEXTS = [
    "Hello, world!",
    "int main(void) {\n    return 0;\n}\n",
    "def f(x):\n\treturn x ** 2  # square\n",
    "for (size_t i = 0; i < n; ++i) { a[i] = b[i] * c[i]; }",
    "#include <stdio.h>\n#include <stdlib.h>\n",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1;",
    "fn main() { let v: Vec<u8> = vec![1, 2, 3]; println!(\"{:?}\", v); }",
    "<div class=\"x\">&amp; &lt;tag&gt;</div>",
    "你好，世界！今天天气很好。",
    "日本語のテキストとカタカナ、ひらがな。",
    "한국어 문장입니다.",
    "Mixed 中文 and English 文本 123.",
    "Emoji: 😀🎉👍🏽 👨‍👩‍👧‍👦 🇺🇸",
    "🚀",
    "Ünïcödé àccents: café, naïve, Zürich, Ελληνικά, русский",
    "   leading spaces",
    "trailing spaces    ",
    "a\n\n\nb",
    "\t\t\ttabs\tand\nnewlines\r\n",
    "                                ",
    "\n\n\n\n\n\n\n\n",
    "1234567890 3.14159 -42 1e-6 0xDEADBEEF",
    "12345678901234567890123456789",
    "2026-10-01T12:34:56Z",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nreasoning here\n</think>\n\nanswer",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "<|endoftext|>",
    "text<|im_end|>more",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next?x=1&y=2",
    "user@example.com",
    "camelCaseIdentifier snake_case_identifier SCREAMING_CASE",
    "don't won't it's we'll they've I'd",
    "$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
    "λx. x + 1 → ∀ε>0 ∃δ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "!!!???...,,,;;;:::",
    "",
    " ",
    "The quick brown fox jumps over the lazy dog.",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def chat_cases():
    user = [{"role": "user", "content": "What is 2+2?"}]
    prior = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 = 4.", "reasoning_content": "Simple addition: 2 plus 2 is 4."},
        {"role": "user", "content": "And 3+3?"},
    ]
    return [
        dict(name="thinking_default", messages=user, tools=None, kwargs={}),
        dict(name="no_thinking", messages=user, tools=None, kwargs={"enable_thinking": False}),
        dict(name="tools", messages=[{"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": "What's the weather in Paris?"}],
             tools=[WEATHER_TOOL], kwargs={}),
        dict(name="reasoning_effort_low", messages=user, tools=None, kwargs={"reasoning_effort": "low"}),
        dict(name="preserve_thinking_false", messages=prior, tools=None, kwargs={"preserve_thinking": False}),
        dict(name="multi_turn", messages=prior, tools=None, kwargs={}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/mnt/data/fngine/hf")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/fixtures"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.hf)
    os.makedirs(a.out, exist_ok=True)
    assert len(TEXTS) == 40
    cases = [dict(text=t, ids=tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS]
    with open(os.path.join(a.out, "tokenizer_cases.json"), "w", encoding="utf-8") as f:
        json.dump(dict(tokenizer="Qwen/Qwen3.8-Flash-Next tokenizer.json", add_special_tokens=False, cases=cases),
                  f, ensure_ascii=False, indent=1)
    rendered = []
    for c in chat_cases():
        kw = dict(c["kwargs"])
        if c["tools"]:
            kw["tools"] = c["tools"]
        s = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(s, add_special_tokens=False)["input_ids"]
        rendered.append(dict(c, add_generation_prompt=True, rendered=s, ids=ids))
    with open(os.path.join(a.out, "chat_template_cases.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cases=rendered), f, ensure_ascii=False, indent=1)
    for r in rendered:
        print("==== %s\n%s" % (r["name"], r["rendered"]))


if __name__ == "__main__":
    main()
