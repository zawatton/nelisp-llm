#!/usr/bin/env python3
"""Can the model copy?  A content task with no world knowledge in it.

The remaining gap shows up as syntax working and content not: function words
rank in the tens, content words in the tens of thousands.  Copying separates
those.  A sequence of arbitrary tokens repeated twice is predictable only by
carrying token identity forward and reading it back out -- no facts, no
grammar.  Every language model does it, and the induction heads that do it are
already measurable in this one (p=0.59 from the second " is" to what followed
the first).  What is not yet known is whether that reaches the output.

Reports the rank of each token of the second repetition, for Bonsai and for
the donor through the same code.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))


def bonsai_logits(gguf_path, ids):
    R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
    R.FOLD = "linear"
    m = R.Model(gguf_path)
    kv = m.kv
    vocab = len(kv["tokenizer.ggml.tokens"])
    out = R.run(m, ids, False, True, "qkv", score=True, per_position=False)
    return out, vocab


def run_bonsai(gguf_path, ids, layers=None, skip=()):
    """Full logits, not just the mean, so ranks can be read per position."""
    R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
    R.FOLD = "linear"
    m = R.Model(gguf_path)
    saved = []
    orig = R.run

    # run() returns only the last position's logits unless scoring; recompute
    # the whole matrix from the dumped residual instead.
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    tmp.close()
    orig(m, ids, False, True, "qkv", dump=tmp.name, layers=layers, skip=skip)
    x = np.fromfile(tmp.name, dtype=np.float32).reshape(len(ids), -1)
    os.unlink(tmp.name)
    kv = m.kv
    eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
    rot = R.Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
                kv["prism.hadamard.block_size"], False)
    h = rot(R.rms(x, m.t("output_norm.weight"), eps, True))
    return h @ m.t("output.weight").T


def run_donor(table_path, ids):
    sys.path.insert(0, os.path.join(HERE, os.pardir, "build", "donor", "pylibs"))
    from nlwts import Table
    _ref = SourceFileLoader("qref", os.path.join(HERE, "qwen-forward-ref.py")).load_module()
    tb = Table(table_path)
    cfg = tb.config
    dim, heads = cfg["dim"], cfg["heads"]
    kvh, hd = cfg["kv-heads"], cfg["head-dim"]
    eps, base = cfg["rms-eps"], cfg["rope-base"]
    seq = len(ids)
    pos = np.arange(seq, dtype=np.float64)
    x = tb.rows(":wte", ids)
    for ly in range(cfg["layers"]):
        a = _ref.rmsnorm(x, tb.f32(":ln1g", ly), eps)
        q = (a @ tb.dequant(":wq", ly).T).reshape(seq, heads, hd)
        k = (a @ tb.dequant(":wk", ly).T).reshape(seq, kvh, hd)
        v = (a @ tb.dequant(":wv", ly).T).reshape(seq, kvh, hd)
        q = _ref.rope_half(_ref.rmsnorm(q, tb.f32(":q-norm", ly), eps), pos, base)
        k = _ref.rope_half(_ref.rmsnorm(k, tb.f32(":k-norm", ly), eps), pos, base)
        ctx = _ref.attend(q, k, v, heads, kvh, hd).reshape(seq, heads * hd)
        x1 = x + ctx @ tb.dequant(":wo", ly).T
        b = _ref.rmsnorm(x1, tb.f32(":ln2g", ly), eps)
        g = b @ tb.dequant(":wg", ly).T
        u = b @ tb.dequant(":wu", ly).T
        x = x1 + ((g / (1.0 + np.exp(-g))) * u) @ tb.dequant(":wd", ly).T
    return _ref.rmsnorm(x, tb.f32(":lnf"), eps) @ tb.dequant(":wte").T


def report(name, lg, ids, first_len):
    order = np.argsort(-lg, axis=-1)
    vocab = lg.shape[1]
    ranks = []
    print("  %s  (vocabulary %d)" % (name, vocab))
    for i in range(first_len - 1, len(ids) - 1):
        t = ids[i + 1]
        r = int(np.where(order[i] == t)[0][0]) + 1
        ranks.append(r)
        tag = "2nd pass" if i >= first_len else "boundary"
        print("     pos %2d -> token %6d   rank %8d   %s" % (i, t, r, tag))
    inside = ranks[1:]
    print("     second repetition: median rank %d, top-1 %d/%d, top-10 %d/%d"
          % (int(np.median(inside)), sum(r == 1 for r in inside), len(inside),
             sum(r <= 10 for r in inside), len(inside)))
    print()


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    if len(sys.argv) > 2 and sys.argv[2] == "--depths":
        pat = [int(x) for x in rng.integers(2000, 40000, 8)]
        ids = pat + pat
        print("copying, as a function of how much of the stack has run\n")
        for spec in ("0", "8", "16", "32", "48", "64",
                     "64 skip=deltanet", "64 skip=attn"):
            parts = spec.split()
            n = int(parts[0])
            sk = tuple(parts[1].split("=")[1].split(",")) if len(parts) > 1 else ()
            lg = run_bonsai("build/bonsai/f16.gguf", ids, layers=n, skip=sk)
            order = np.argsort(-lg, axis=-1)
            rs = [int(np.where(order[i] == ids[i + 1])[0][0]) + 1
                  for i in range(len(pat), len(ids) - 1)]
            print("  %-18s median rank %8d   best %8d   top-100 %d/%d"
                  % (spec, int(np.median(rs)), min(rs),
                     sum(r <= 100 for r in rs), len(rs)), flush=True)
        return
    # mid-frequency ids, away from the specials at the top of the vocabulary
    pat_b = [int(x) for x in rng.integers(2000, 40000, 8)]
    pat_d = [int(x) for x in rng.integers(2000, 40000, 8)]
    ids_b = pat_b + pat_b
    ids_d = pat_d + pat_d
    print("a sequence of 8 arbitrary tokens, repeated once.  Predicting the")
    print("second repetition needs only token identity carried and read back.\n")
    report("Ternary Bonsai 2 27B",
           run_bonsai("build/bonsai/f16.gguf", ids_b), ids_b, len(pat_b))
    report("Qwen3-0.6B (the donor, control)",
           run_donor("build/donor/qwen3-0.6b/weights.bin", ids_d), ids_d, len(pat_d))


if __name__ == "__main__":
    main()
