import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gguf import GGUFFile  # noqa: E402

DIR = "/home/obelisk/Projects/exploresrata/Strata-data/models/IQ3_S"
S1 = os.path.join(DIR, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
S2 = os.path.join(DIR, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")


def _open(p):
    if not os.path.exists(p):
        pytest.skip("missing %s" % p)
    return GGUFFile(p)


def _extents_ok(g):
    size = os.path.getsize(g.path)
    spans = sorted((t.offset, t.offset + t.nbytes) for t in g.tensors.values())
    assert spans[0][0] >= g.data_start
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        assert a1 <= b0
    assert spans[-1][1] <= size
    for t in g.tensors.values():
        assert (t.offset - g.data_start) % g.alignment == 0


def test_shard1_experts():
    g = _open(S1)
    assert g.version == 3
    assert g.metadata["general.architecture"] == "qwen4exp"
    assert g.metadata["qwen4exp.block_count"] == 48
    gu, dn = Counter(), Counter()
    for l in range(48):
        types = set()
        for p in ("gate", "up"):
            t = g.tensors["blk.%d.ffn_%s_exps.weight" % (l, p)]
            assert t.ne == [2560, 640, 512]
            types.add(t.type_name)
        assert len(types) == 1
        gu[types.pop()] += 1
        d = g.tensors["blk.%d.ffn_down_exps.weight" % l]
        assert d.ne == [640, 2560, 512]
        dn[d.type_name] += 1
        if d.type_name == "Q2_0":
            dn["layers"] = dn.get("layers", ()) + (l,)
    assert g.tensors["blk.47.ffn_gate_exps.weight"].type_name == "IQ4_XS"
    assert dict(gu) == {"IQ3_XXS": 17, "IQ2_S": 20, "IQ3_S": 10, "IQ4_XS": 1}
    assert dn["IQ4_NL"] == 39 and dn["Q2_0"] == 9
    assert dn["layers"] == (1, 8, 10, 12, 13, 20, 21, 30, 36)
    _extents_ok(g)


def test_shard1_ple_metadata():
    g = _open(S1)
    assert list(g.metadata["qwen4exp.ple.layer_multipliers"]) == [23703573157769, 20109073645365, 8052911324071]
    assert len(g.metadata["qwen4exp.ple.head_vocab_sizes"]) == 16
    sizes, offs = g.metadata["qwen4exp.ple.head_vocab_sizes"], g.metadata["qwen4exp.ple.head_offsets"]
    assert int(offs[0]) == 0
    assert all(int(offs[i + 1]) == int(offs[i]) + int(sizes[i]) for i in range(15))
    assert int(offs[15]) + int(sizes[15]) <= 320001536  # table has 90 trailing padding rows


def test_shard2_ngram():
    g = _open(S2)
    t = g.tensors["per_layer_token_embd.weight"]
    assert t.type_name == "IQ4_NL" and t.ne == [160, 320001536]
    assert t.nbytes == 90 * 320001536
    _extents_ok(g)
    assert len(g.tensor_bytes(t.name)[:90]) == 90
