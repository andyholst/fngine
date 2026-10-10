"""Deterministic calibration corpus: cinference source/docs files, tokenized without a template."""
import os

CINFERENCE = "/home/obelisk/Projects/cinference"
ROOTS = ("src", "tools", "docs")
SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".py", ".md"}
MAX_TOKENS = 2048
MIN_TOKENS = 64
NUM_SEQS = 128


def files():
    out = []
    for root in ROOTS:
        for d, _, fs in os.walk(os.path.join(CINFERENCE, root)):
            for f in fs:
                if os.path.splitext(f)[1] in SUFFIXES:
                    out.append(os.path.relpath(os.path.join(d, f), CINFERENCE))
    return sorted(out)


def build(tok):
    """Returns [(relpath, ids)] for the first NUM_SEQS files (sorted path) with >= MIN_TOKENS tokens."""
    seqs = []
    for rel in files():
        with open(os.path.join(CINFERENCE, rel), encoding="utf-8", errors="replace") as f:
            ids = tok(f.read(), add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        if len(ids) >= MIN_TOKENS:
            seqs.append((rel, ids))
            if len(seqs) == NUM_SEQS:
                break
    return seqs
