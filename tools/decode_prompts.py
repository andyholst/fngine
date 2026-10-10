#!/usr/bin/env python3
"""Renders a fixed set of chat prompts (thinking on, the model's chat template) to token ids.

Each prompt becomes <out>/<name>.safetensors with an int32 `ids` tensor, the format the
fngine-validate --generate mode reads. Used to measure decode speed on the model's own output and
to collect decode routing counts for expert placement (decode traffic, not prompt text, decides
which experts must be GPU-resident).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from safetensors.numpy import save_file
from transformers import AutoTokenizer

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}]

PROMPTS = [
    ("p01", "Implement an LRU cache in Python with O(1) get and put, and explain the design.", False),
    ("p02", "Explain the difference between processes and threads on Linux, with concrete examples.", False),
    ("p03", "Write a Rust function that parses one CSV line, handling quoted fields and escaped quotes.", False),
    ("p04", "My React component re-renders forever when I call setState inside useEffect. Why, and how do I fix it?", False),
    ("p05", "Write a bash script that prints the ten largest files under a directory, human readable.", False),
    ("p06", "Explain how a B-tree works and when it beats a hash table.", False),
    ("p07", "Write a CUDA kernel that computes an inclusive prefix sum over a large array, and explain the algorithm.", False),
    ("p08", "Design a REST API for a todo application: endpoints, request bodies and status codes.", False),
    ("p09", "Find the bug: def mean(xs): return sum(xs) / len(xs) - it crashes sometimes. Make it robust and add tests.", False),
    ("p10", "Implement Dijkstra's algorithm in C++ with a priority queue and explain its complexity.", False),
    ("p11", "Read src/main.cpp and tell me what the program does.", True),
    ("p12", "Run the test suite with ctest and fix whatever fails.", True),
    ("p13", "Write pytest unit tests for a function that validates email addresses.", False),
    ("p14", "Convert this SQL to pandas: SELECT dept, AVG(salary) FROM emp GROUP BY dept HAVING COUNT(*) > 5", False),
    ("p15", "What is the average-case time complexity of quicksort? Give a short proof.", False),
    ("p16", "Write a Go HTTP server with graceful shutdown and a health endpoint.", False),
    ("p17", "Explain what a mutex deadlock is and show three ways to avoid it in C++.", False),
    ("p18", "Refactor this into idiomatic Python: for i in range(len(xs)):\n    if xs[i] > 0:\n        ys.append(xs[i]*2)", False),
    ("p19", "List the files in the current directory and show me the git status.", True),
    ("p20", "Write a TypeScript debounce function with proper generic types and explain how it works.", False),
    ("p21", "Review this C++ for bugs: std::vector<int> v; for (auto it = v.begin(); it != v.end(); ++it) if (*it == 0) v.erase(it);", False),
    ("p22", "Write a Python script that watches a directory and prints a line whenever a file changes.", False),
    ("p23", "Explain how git rebase differs from git merge, and when to use each.", False),
    ("p24", "Write a regular expression that matches ISO 8601 dates and explain each part.", False),
    ("p25", "Implement a thread-safe bounded queue in Java.", False),
    ("p26", "Explain virtual memory and page tables to a junior developer.", False),
    ("p27", "Write SQL to find the second highest salary per department.", False),
    ("p28", "Add error handling and retries with exponential backoff to a Python function that calls an HTTP API.", False),
    ("p29", "Search the codebase for TODO comments and summarize them.", True),
    ("p30", "Write a C program that reads a file line by line and counts word frequencies using a hash table.", False),
    ("p31", "Explain the CAP theorem with examples of real databases.", False),
    ("p32", "Write a Dockerfile for a Node.js app with a multi-stage build.", False),
    ("p33", "Implement binary search in Rust and prove it terminates.", False),
    ("p34", "Write a Makefile for a small C project with separate debug and release builds.", False),
    ("p35", "Explain what a closure is in JavaScript with three practical examples.", False),
    ("p36", "Create a new file hello.py that prints hello world, then run it.", True),
    ("p37", "Optimize this Python loop that sums squares of a large list; compare with numpy.", False),
    ("p38", "Write a Kotlin data class for a user profile and a function that validates it.", False),
    ("p39", "Explain how TLS 1.3 handshakes work, step by step.", False),
    ("p40", "Implement a trie in C++ with insert, search and prefix counting.", False),
    # Markdown documents and tables, Chinese and mathematics: answers whose tokens route differently
    # from English prose and code.
    ("p41", "Write a README for a command-line tool that resizes images: installation, usage examples and a table of options.", False),
    ("p42", "Create a markdown table comparing PostgreSQL, MySQL and SQLite on concurrency, replication and licensing, then summarize.", False),
    ("p43", "Write a changelog entry for version 2.3.0 of a web framework with sections for features, fixes and breaking changes.", False),
    ("p44", "Benchmark results: 1 thread 120 ms, 2 threads 64 ms, 4 threads 35 ms, 8 threads 21 ms, 16 threads 18 ms. Tabulate speedup and efficiency per thread count and explain the trend.", False),
    ("p45", "用 Python 实现一个 LRU 缓存，要求 get 和 put 都是 O(1)，并解释设计思路。", False),
    ("p46", "解释 Linux 中进程和线程的区别，并给出具体例子。", False),
    ("p47", "写一个 C++ 程序，逐行读取文件并用哈希表统计单词出现次数。", False),
    ("p48", "解释 TCP 三次握手和四次挥手的过程，以及为什么需要 TIME_WAIT 状态。", False),
    ("p49", "为一个推理引擎写一份架构设计文档的大纲，包括模块划分、请求生命周期和资源管理。", False),
    ("p50", "阅读 src/main.cpp 并告诉我这个程序做什么。", True),
    ("p51", "Prove that the square root of 2 is irrational.", False),
    ("p52", "Compute the integral of x^2 e^x dx step by step and check the result by differentiation.", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", default="/mnt/data/fngine/hf")
    ap.add_argument("--out", default="/mnt/data/fngine/prompts")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.hf)
    os.makedirs(args.out, exist_ok=True)
    for name, text, tools in PROMPTS:
        messages = [{"role": "user", "content": text}]
        enc = tok.apply_chat_template(messages, tools=TOOLS if tools else None, add_generation_prompt=True,
                                      tokenize=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        save_file({"ids": np.asarray(ids, dtype=np.int32)}, os.path.join(args.out, f"{name}.safetensors"))
        print(f"{name}: {len(ids)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
