"""Golden prompt definitions (G0/G1/G2). Token ids are saved with the outputs."""
import os

CINFERENCE = "/home/obelisk/Projects/cinference"
G0_USER = "Write a C function that reverses a singly linked list."
G1_MAX = 2048
G2_MAX = 8192
G2_DOCS = ["docs/maintainer/engine-architecture.md", "docs/maintainer/paged-kv-cache.md",
           "docs/maintainer/op-development.md"]


def _read(rel):
    with open(os.path.join(CINFERENCE, rel), encoding="utf-8") as f:
        return f.read()


def g0_ids(tok):
    msgs = [{"role": "user", "content": G0_USER}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)


def g1_ids(tok):
    msgs = [{"role": "user", "content": _read("README.md") + "\n\nSummarize the design in five bullet points."}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    return list(ids)[:G1_MAX]


def g2_ids(tok):
    text = "".join(_read(p) for p in G2_DOCS)
    return tok(text, add_special_tokens=False)["input_ids"][:G2_MAX]
