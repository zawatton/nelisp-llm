#!/usr/bin/env python3
"""Where does the rotation belong in the copy circuit?  Sweep, on the weights.

The OV circuit W_U . W_O . W_V . W_E has a large diagonal in every model that
can copy -- the donor scores z = +8.6 -- and none at all under this reading of
the Bonsai weights (z = +0.17).  The circuit is four linear maps with a
transform possibly in front of each, and RMSNorm with a unit gain commutes
with an orthogonal transform, so the whole thing is linear and costs a matrix
product per arrangement instead of a forward pass.

Three choices at four places is eighty-one arrangements and about a minute.
An arrangement that makes a copy circuit appear is the one the weights were
folded for; if none does, the defect is not the rotation's placement.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
NTOK = 512
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 31


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def main():
    m = R.Model("build/bonsai/f16.gguf")
    kv = m.kv
    heads, kvh = kv["qwen35.attention.head_count"], kv["qwen35.attention.head_count_kv"]
    hd = kv["qwen35.attention.key_length"]
    W, V = kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"]
    B = kv["prism.hadamard.block_size"]
    fwd, inv = R.Rot(W, V, B, False), R.Rot(W, V, B, True)
    T = {"f": fwd, "i": inv, "-": (lambda x: x)}

    rng = np.random.default_rng(11)
    ids = np.unique(rng.integers(2000, 200000, NTOK * 3))[:NTOK]
    E = m.t("token_embd.weight")[ids].astype(np.float32)
    U = unit(m.t("output.weight")[ids].astype(np.float32))
    p = "blk.%d." % LAYER
    WV = m.t(p + "attn_v.weight")
    WO = m.t(p + "attn_output.weight")
    perm = rng.permutation(len(ids))

    # T(g) . W' equals g . Tinv(W)', so the transform goes onto the weight
    # once per choice instead of onto the activations once per head.
    WOx = {t: T["i" if t == "f" else ("f" if t == "i" else "-")](WO.copy())
           for t in "fi-"}
    AV = {}
    for te, tv in itertools.product("fi-", repeat=2):
        AV[(te, tv)] = (T[tv](T[te](E.copy())) @ WV.T).astype(np.float32)
    print("layer %d, %d tokens; z of the OV diagonal against its own null" % (LAYER, NTOK))
    print("places: E = the stored embedding, V = before attn_v,")
    print("        O = before attn_output, U = before the head\n")
    results = []
    for te, tv, to, tu in itertools.product("fi-", repeat=4):
        v = AV[(te, tv)]
        best = -99.0
        bh = -1
        for h in range(heads):
            kvi = h // (heads // kvh)
            o = v[:, kvi * hd:(kvi + 1) * hd] @ WOx[to][:, h * hd:(h + 1) * hd].T
            hh = unit(T[tu](o))
            d = (U * hh).sum(axis=1)
            off = (U * hh[perm]).sum(axis=1)
            z = (d.mean() - off.mean()) / (off.std() + 1e-12)
            if z > best:
                best, bh = z, h
        results.append((best, te, tv, to, tu, bh))
    results.sort(reverse=True)
    print("  %-4s %-4s %-4s %-4s  head    z" % ("E", "V", "O", "U"))
    for z, te, tv, to, tu, bh in results[:12]:
        print("  %-4s %-4s %-4s %-4s  %3d  %+7.2f" % (te, tv, to, tu, bh, z))
    print("  ...")
    for z, te, tv, to, tu, bh in results[-3:]:
        print("  %-4s %-4s %-4s %-4s  %3d  %+7.2f" % (te, tv, to, tu, bh, z))
    print("\n  for reference the donor reaches z = +8.61 with no transform at all")


if __name__ == "__main__":
    main()
