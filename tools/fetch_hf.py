#!/usr/bin/env python3
# Modified for fngine from Strata:tools/mtp_fetch.py @ 30ec18ec7094550fcc594fd948220d511d80464e. See NOTICE.
"""Range-fetch the dense (non-routed-expert) text weights of Qwen/Qwen3.8-Flash-Next.

    python tools/fetch_hf.py --repo Qwen/Qwen3.8-Flash-Next --out /mnt/data/fngine/hf \
        [--revision REV] [--seed-dir DIR] [--jobs 6]

Reads each shard's safetensors header by HTTP range request, selects the tensors fngine needs, fetches only
their byte ranges and writes one `dense-NNNNN.safetensors` per source shard (contiguous rewritten header).
Resumable: existing valid output shards are kept. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HF = "https://huggingface.co"
CHUNK = 64 << 20
FULL_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "chat_template.jinja", "model.safetensors.index.json"]
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, file=sys.stderr, flush=True)


def get(url, start=None, end=None, retries=5):
    """GET url; with start/end (inclusive) a range request whose length is verified."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fngine-fetch"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError("short range read: %d of %d" % (len(data), end - start + 1))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries - 1:
                raise
            log("retry %d %s: %s" % (attempt + 1, url, e))
        except Exception as e:
            if attempt == retries - 1:
                raise
            log("retry %d %s: %s" % (attempt + 1, url, e))
        time.sleep(2 ** attempt)


def selected(name):
    if name.startswith("model.visual."):
        return False
    if not (name.startswith("model.language_model.") or name.startswith("mtp.") or name == "lm_head.weight"):
        return False
    if re.match(r"model\.language_model\.layers\.\d+\.mlp\.experts\.", name):
        return False
    if re.search(r"\.ple_embedding\.ngram_embedding\.shard_", name):
        return False
    return True


def category(name):
    if name.startswith("mtp.layers.0.mlp.experts."):
        return "mtp_experts"
    if name.startswith("mtp."):
        return "mtp_other"
    if name == "lm_head.weight" or "embed_tokens" in name:
        return "embed/lm_head"
    if ".ple." in name:
        return "ple"
    if ".linear_attn." in name:
        return "gdn"
    if ".self_attn." in name:
        return "qsa"
    if ".mlp." in name:
        return "moe_dense"
    if "hyper_connection" in name:
        return "gr"
    return "other"


def shard_header(base, shard):
    url = base + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    return 8 + n, json.loads(get(url, 8, 8 + n - 1))


def load_seed(seed_dir):
    if not seed_dir:
        return {}
    m = json.load(open(os.path.join(seed_dir, "mtp-manifest.json")))
    return {r["name"]: dict(r, path=os.path.join(seed_dir, r["file"])) for r in m}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def seed_bytes(seed, t):
    s = seed.get(t["name"])
    if not s or s["shard"] != t["shard"] or s["start"] != t["start"] or s["end"] != t["start"] + t["bytes"] - 1 \
            or s["bytes"] != t["bytes"] or not os.path.exists(s["path"]) or os.path.getsize(s["path"]) != t["bytes"]:
        return None
    data = open(s["path"], "rb").read()
    return data if sha256_bytes(data) == s["sha256"] else None


def out_name(shard):
    m = re.match(r"model-(\d+)-of-\d+\.safetensors", shard)
    return "dense-%s.safetensors" % m.group(1)


def build_header(tensors, shard, revision):
    hdr = {"__metadata__": {"format": "pt", "source_shard": shard, "revision": revision}}
    off = 0
    for t in tensors:
        hdr[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + t["bytes"]]}
        off += t["bytes"]
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw


def validate_existing(path, tensors, shard, revision):
    """Existing output is valid if its header equals the one we would write and its size matches."""
    if not os.path.exists(path):
        return False
    head = build_header(tensors, shard, revision)
    if os.path.getsize(path) != len(head) + sum(t["bytes"] for t in tensors):
        return False
    with open(path, "rb") as f:
        return f.read(len(head)) == head


def fetch_shard(base, shard, tensors, out, revision, seed):
    """tensors sorted by source offset. Returns per-tensor manifest rows."""
    path = os.path.join(out, out_name(shard))
    rows = []
    if validate_existing(path, tensors, shard, revision):
        with open(path, "rb") as f:
            f.seek(len(build_header(tensors, shard, revision)))
            for t in tensors:
                h = hashlib.sha256()
                left = t["bytes"]
                while left:
                    b = f.read(min(left, 1 << 24))
                    h.update(b)
                    left -= len(b)
                rows.append(dict(t, out=os.path.basename(path), sha256=h.hexdigest()))
        log("%s: present, skipped" % shard)
        return rows
    part = path + ".part"
    done = 0
    with open(part, "wb") as f:
        f.write(build_header(tensors, shard, revision))
        # coalesce source-adjacent tensors not satisfied from seed into groups
        i = 0
        while i < len(tensors):
            t = tensors[i]
            local = seed_bytes(seed, t)
            if local is not None:
                f.write(local)
                rows.append(dict(t, out=os.path.basename(path), sha256=sha256_bytes(local), seeded=True))
                done += t["bytes"]
                i += 1
                continue
            j = i + 1
            while j < len(tensors) and tensors[j]["start"] == tensors[j - 1]["start"] + tensors[j - 1]["bytes"] \
                    and seed_bytes(seed, tensors[j]) is None:
                j += 1
            group = tensors[i:j]
            hashers = [hashlib.sha256() for _ in group]
            start, stop = group[0]["start"], group[-1]["start"] + group[-1]["bytes"]
            pos = start
            while pos < stop:
                end = min(pos + CHUNK, stop)
                data = get(base + shard, pos, end - 1)
                f.write(data)
                for k, g in enumerate(group):  # feed per-tensor hashers
                    a, b = max(pos, g["start"]), min(end, g["start"] + g["bytes"])
                    if a < b:
                        hashers[k].update(data[a - pos:b - pos])
                pos = end
                done += len(data)
            for k, g in enumerate(group):
                rows.append(dict(g, out=os.path.basename(path), sha256=hashers[k].hexdigest()))
            i = j
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    log("%s: wrote %s (%.2f GB)" % (shard, os.path.basename(path), done / 1e9))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--seed-dir")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rev = a.revision
    if not rev or not re.fullmatch(r"[0-9a-f]{40}", rev):
        info = json.loads(get("%s/api/models/%s/revision/%s" % (HF, a.repo, rev or "main")))
        rev = info["sha"]
    log("revision %s" % rev)
    base = "%s/%s/resolve/%s/" % (HF, a.repo, rev)

    missing = []
    for fn in FULL_FILES:
        dst = os.path.join(a.out, fn)
        if os.path.exists(dst):
            continue
        try:
            data = get(base + fn)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                missing.append(fn)
                continue
            raise
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
    if missing:
        log("not in repo (404): %s" % ", ".join(missing))

    weight_map = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({s for n, s in weight_map.items() if selected(n)})
    headers = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for shard, res in zip(shards, ex.map(lambda s: shard_header(base, s), shards)):
            headers[shard] = res
    plan, cats = {}, {}
    for shard in shards:
        hbase, hdr = headers[shard]
        ts = []
        for name, meta in hdr.items():
            if name == "__metadata__" or not selected(name) or weight_map.get(name) != shard:
                continue
            o0, o1 = meta["data_offsets"]
            ts.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                           start=hbase + o0, bytes=o1 - o0))
            c = cats.setdefault(category(name), [0, 0])
            c[0] += 1
            c[1] += o1 - o0
        ts.sort(key=lambda t: t["start"])
        if ts:
            plan[shard] = ts
    want = {n for n in weight_map if selected(n)}
    got = {t["name"] for ts in plan.values() for t in ts}
    if want != got:
        sys.exit("index/header mismatch: %s" % sorted(want ^ got)[:20])
    total = sum(c[1] for c in cats.values())
    print("inventory: %d tensors, %.3f GB, %d shards" % (len(got), total / 1e9, len(plan)))
    for k, (n, b) in sorted(cats.items()):
        print("  %-14s %5d tensors %9.3f GB" % (k, n, b / 1e9))
    sys.stdout.flush()

    seed = load_seed(a.seed_dir)
    rows = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(fetch_shard, base, s, ts, a.out, rev, seed) for s, ts in plan.items()]
        for fu in as_completed(futs):
            rows.extend(fu.result())
    rows.sort(key=lambda r: r["name"])
    manifest = dict(repo=a.repo, revision=rev, missing_files=missing, total_bytes=total,
                    tensors=[dict(name=r["name"], source_shard=r["shard"], out=r["out"], dtype=r["dtype"],
                                  shape=r["shape"], bytes=r["bytes"], sha256=r["sha256"]) for r in rows])
    tmp = os.path.join(a.out, "fetch-manifest.json.part")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(a.out, "fetch-manifest.json"))
    print("done: %d tensors, %.3f GB, seeded %d" % (len(rows), total / 1e9, sum(1 for r in rows if r.get("seeded"))))


if __name__ == "__main__":
    main()
