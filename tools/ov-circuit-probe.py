#!/usr/bin/env python3
"""Does any attention head's OV circuit map a token to its own unembedding?

Copying is the OV circuit: a head reads the residual at some earlier position
through W_V, writes it back through W_O, and the head reads the result out.
For a copy head the composition W_U . W_O . W_V . W_E has a large diagonal --
token t in, token t out.  That is a property of the weights alone, so it needs
no forward pass and cannot be blamed on the rest of the stack.

Run on Ternary Bonsai with its folded rotation in the right places, and on the
donor through the same measure.  The donor is tied, which makes copying easier
for it, so what matters is not that its number is larger but whether Bonsai's
is distinguishable from its own null at all.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
NTOK = 512


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def score(name, out_vecs, head_rows, rng):
    """Diagonal against off-diagonal cosine, in null sigmas."""
    o, h = unit(out_vecs), unit(head_rows)
    diag = (o * h).sum(axis=1)
    perm = rng.permutation(len(o))
    off = (o * h[perm]).sum(axis=1)
    z = (diag.mean() - off.mean()) / off.std()
    return diag.mean(), off.mean(), off.std(), z


def bonsai():
    R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
    R.FOLD = "linear"
    m = R.Model("build/bonsai/f16.gguf")
    kv = m.kv
    eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
    heads, kvh = kv["qwen35.attention.head_count"], kv["qwen35.attention.head_count_kv"]
    hd = kv["qwen35.attention.key_length"]
    iv = kv["qwen35.full_attention_interval"]
    rot = R.Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
                kv["prism.hadamard.block_size"], False)
    rng = np.random.default_rng(11)
    ids = np.unique(rng.integers(2000, 200000, NTOK * 2))[:NTOK]
    E = m.t("token_embd.weight")[ids].astype(np.float32)
    U = m.t("output.weight")[ids].astype(np.float32)
    # the residual at an early position is the embedding; normalise as block 1 does
    a = rot(R.rms(E, np.ones(E.shape[1], np.float32), eps))
    print("Ternary Bonsai 2 27B -- W_U . W_O . W_V . W_E, per attention head")
    print("  layer  best head   diagonal   off-diag       z")
    best = (None, -99)
    for ly in [3, 15, 31, 47, 63]:
        p = "blk.%d." % ly
        WV = m.t(p + "attn_v.weight")                 # (kvh*hd, dim)
        WO = m.t(p + "attn_output.weight")            # (dim, heads*hd)
        v = a @ WV.T                                  # (N, kvh*hd)
        rows = []
        for h in range(heads):
            kvi = h // (heads // kvh)
            g = np.zeros((len(ids), heads * hd), np.float32)
            g[:, h * hd:(h + 1) * hd] = v[:, kvi * hd:(kvi + 1) * hd]
            o = rot(g) @ WO.T                         # (N, dim)
            # the head reads A.h, and W_U already absorbed A'
            hh = rot(R.rms(o, np.ones(o.shape[1], np.float32), eps))
            rows.append(score("", U, hh, rng))
        z = max(r[3] for r in rows)
        hbest = int(np.argmax([r[3] for r in rows]))
        d, of, sd, _ = rows[hbest]
        print("  %5d  head %2d    %+9.4f  %+9.4f  %+6.2f" % (ly, hbest, d, of, z))
        if z > best[1]:
            best = ((ly, hbest), z)
    print("  strongest anywhere: layer %d head %d, z = %+.2f\n" % (*best[0], best[1]))


def donor():
    sys.path.insert(0, os.path.join(HERE, os.pardir, "build", "donor", "pylibs"))
    from nlwts import Table
    _ref = SourceFileLoader("qref", os.path.join(HERE, "qwen-forward-ref.py")).load_module()
    tb = Table("build/donor/qwen3-0.6b/weights.bin")
    cfg = tb.config
    heads, kvh, hd = cfg["heads"], cfg["kv-heads"], cfg["head-dim"]
    eps = cfg["rms-eps"]
    rng = np.random.default_rng(11)
    W = tb.dequant(":wte")
    ids = np.unique(rng.integers(2000, 140000, NTOK * 2))[:NTOK]
    E = W[ids]
    a = _ref.rmsnorm(E, np.ones(E.shape[1]), eps)
    print("Qwen3-0.6B (control) -- the same measure")
    print("  layer  best head   diagonal   off-diag       z")
    best = (None, -99)
    for ly in [0, 7, 13, 20, 27]:
        WV = tb.dequant(":wv", ly)
        WO = tb.dequant(":wo", ly)
        v = a @ WV.T
        rows = []
        for h in range(heads):
            kvi = h // (heads // kvh)
            g = np.zeros((len(ids), heads * hd))
            g[:, h * hd:(h + 1) * hd] = v[:, kvi * hd:(kvi + 1) * hd]
            o = g @ WO.T
            hh = _ref.rmsnorm(o, np.ones(o.shape[1]), eps)
            rows.append(score("", W[ids], hh, rng))
        z = max(r[3] for r in rows)
        hbest = int(np.argmax([r[3] for r in rows]))
        d, of, sd, _ = rows[hbest]
        print("  %5d  head %2d    %+9.4f  %+9.4f  %+6.2f" % (ly, hbest, d, of, z))
        if z > best[1]:
            best = ((ly, hbest), z)
    print("  strongest anywhere: layer %d head %d, z = %+.2f" % (*best[0], best[1]))


if __name__ == "__main__":
    bonsai()
    donor()
