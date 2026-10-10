#!/usr/bin/env python3
"""Builds the MTP draft vocabulary: the most frequent token ids of a local code + English corpus.

The MTP drafter scores only these rows of lm_head, which cuts its per-step lm_head traffic from
248320 rows to --size rows. Drafts are verified exactly by the target model, so the subset only
affects draft acceptance, never the output distribution.

Writes <pack>/draft-vocab.bin: little-endian int32 token ids, ascending. All added/special tokens
and the 256 single-byte tokens are always included.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

CODE_EXT = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".rs", ".go", ".js", ".ts",
            ".java", ".sh", ".cmake", ".md", ".txt", ".json", ".toml", ".yaml", ".yml")


def corpus_files(roots: list[str], max_bytes: int) -> list[str]:
    files: list[str] = []
    total = 0
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
            if not path.endswith(CODE_EXT) or not os.path.isfile(path):
                continue
            if any(part in path for part in ("/.git/", "/build", "/third_party/", "/.venv/", "__pycache__")):
                continue
            size = os.path.getsize(path)
            if size == 0 or size > 2 << 20:
                continue
            files.append(path)
            total += size
            if total >= max_bytes:
                return files
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", default="/mnt/data/fngine/hf", help="directory with tokenizer.json")
    ap.add_argument("--pack", default="/mnt/data/fngine/pack")
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--max-mib", type=int, default=96, help="corpus text cap")
    ap.add_argument("--root", action="append", default=None, help="corpus root (repeatable)")
    ap.add_argument("--ref", default="/mnt/data/fngine/ref", help="golden prompts for a coverage report")
    args = ap.parse_args()

    roots = args.root or ["/home/obelisk/Projects/cinference/src", "/home/obelisk/Projects/cinference/tools",
                          "/home/obelisk/Projects/cinference/tests", "/home/obelisk/Projects/cinference/docs",
                          "/home/obelisk/Projects/fngine/src", "/home/obelisk/Projects/fngine/apps",
                          "/home/obelisk/Projects/fngine/tools", *sorted(glob.glob("/usr/lib/python3*/"))]
    tok = Tokenizer.from_file(os.path.join(args.hf, "tokenizer.json"))
    files = corpus_files(roots, args.max_mib << 20)
    counts: collections.Counter[int] = collections.Counter()
    batch: list[str] = []
    n_tokens = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                batch.append(f.read())
        except (UnicodeDecodeError, OSError):
            continue
        if len(batch) >= 256:
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                counts.update(enc.ids)
                n_tokens += len(enc.ids)
            batch.clear()
    for enc in tok.encode_batch(batch, add_special_tokens=False):
        counts.update(enc.ids)
        n_tokens += len(enc.ids)

    meta = json.load(open(os.path.join(args.hf, "tokenizer.json")))
    forced = {a["id"] for a in meta.get("added_tokens", [])}
    vocab = meta["model"]["vocab"]
    # Byte-level BPE: the 256 single-byte symbols keep every byte representable.
    forced |= {i for s, i in vocab.items() if len(s) == 1}
    chosen = set(forced)
    for tid, _ in counts.most_common():
        if len(chosen) >= args.size:
            break
        chosen.add(tid)
    ids = np.array(sorted(chosen), dtype="<i4")
    out = os.path.join(args.pack, "draft-vocab.bin")
    ids.tofile(out)
    print(f"{len(files)} files, {n_tokens} tokens, {len(counts)} distinct; wrote {len(ids)} ids to {out}")

    chosen_arr = np.zeros(248320, dtype=bool)
    chosen_arr[ids] = True
    try:
        from safetensors.numpy import load_file
        for g in ("g0", "g1", "g2"):
            gi = load_file(os.path.join(args.ref, f"{g}.safetensors"))["ids"].astype(np.int64)
            print(f"  coverage {g}: {chosen_arr[gi].mean():.4f} of {len(gi)} tokens")
    except Exception as e:  # coverage report is optional
        print(f"  coverage report skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
