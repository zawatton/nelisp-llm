#!/usr/bin/env python3
"""Cross entropy of a passage under the imported Qwen3-0.6B, as a positive control.

Every configuration of the Bonsai forward lands near chance, and with no known-
good model through the same measurement there is no way to tell a broken
forward from a broken scorer.  This runs the donor -- a model this project
already imports and checks layer by layer -- through the same tokenise, forward
and cross-entropy path.  A working model on ordinary prose is a few nats; if
this comes out near chance too, the scorer is what is wrong.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlwts import Table                                        # noqa: E402
import importlib.machinery                                      # noqa: E402
_ref = importlib.machinery.SourceFileLoader(
    "qref", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "qwen-forward-ref.py")).load_module()


def main():
    table_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "The quick brown fox"
    tok_json = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(os.path.dirname(table_path), "tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    tokens = tok.encode(text, add_special_tokens=False).ids
    print("%d tokens" % len(tokens))

    tb = Table(table_path)
    cfg = tb.config
    dim, heads = cfg["dim"], cfg["heads"]
    kv_heads, hd = cfg["kv-heads"], cfg["head-dim"]
    eps, base = cfg["rms-eps"], cfg["rope-base"]
    nlayers = cfg["layers"]
    seq = len(tokens)
    positions = np.arange(seq, dtype=np.float64)

    x = tb.rows(":wte", tokens)
    for ly in range(nlayers):
        a = _ref.rmsnorm(x, tb.f32(":ln1g", ly), eps)
        q = (a @ tb.dequant(":wq", ly).T).reshape(seq, heads, hd)
        k = (a @ tb.dequant(":wk", ly).T).reshape(seq, kv_heads, hd)
        v = (a @ tb.dequant(":wv", ly).T).reshape(seq, kv_heads, hd)
        q = _ref.rmsnorm(q, tb.f32(":q-norm", ly), eps)
        k = _ref.rmsnorm(k, tb.f32(":k-norm", ly), eps)
        q = _ref.rope_half(q, positions, base)
        k = _ref.rope_half(k, positions, base)
        ctx = _ref.attend(q, k, v, heads, kv_heads, hd).reshape(seq, heads * hd)
        x1 = x + ctx @ tb.dequant(":wo", ly).T
        b = _ref.rmsnorm(x1, tb.f32(":ln2g", ly), eps)
        g = b @ tb.dequant(":wg", ly).T
        u = b @ tb.dequant(":wu", ly).T
        h = (g / (1.0 + np.exp(-g))) * u
        x = x1 + h @ tb.dequant(":wd", ly).T

    final = _ref.rmsnorm(x, tb.f32(":lnf"), eps)
    lg = final @ tb.dequant(":wte").T
    lg = lg - lg.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(lg).sum(axis=-1))
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    per = lse[:-1] - lg[np.arange(len(tgt)), tgt]
    order = np.argsort(-lg, axis=-1)
    ranks = np.array([int(np.where(order[i] == t)[0][0]) + 1
                      for i, t in enumerate(tgt)])
    vocab = lg.shape[1]
    print("cross entropy %.4f nats over %d targets  (chance %.2f)"
          % (per.mean(), len(tgt), np.log(vocab)))
    print("median rank of the true token: %d of %d;  top-1 %.1f%%, top-10 %.1f%%"
          % (int(np.median(ranks)), vocab,
             100.0 * (ranks == 1).mean(), 100.0 * (ranks <= 10).mean()))


if __name__ == "__main__":
    main()
