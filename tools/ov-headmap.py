#!/usr/bin/env python3
"""Is the head layout wrong?  Try every (output slot, kv head) pairing.

No arrangement of the rotation makes a copy circuit appear, so the next thing
the circuit depends on is which 256 columns of attn_output a given kv head's
value is written into.  This tries all of them -- every output slot against
every kv head, at several layers -- which exhausts the layout hypothesis in
one pass over the weights.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
NTOK = 512


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
    rng = np.random.default_rng(11)
    ids = np.unique(rng.integers(2000, 200000, NTOK * 3))[:NTOK]
    E = fwd(m.t("token_embd.weight")[ids].astype(np.float32))
    U = unit(m.t("output.weight")[ids].astype(np.float32))
    perm = rng.permutation(len(ids))
    # Which axis runs fastest is a reshape either way.  [head][dim] puts a
    # head's 256 values together; [dim][head] interleaves them with stride
    # `heads`.  The DeltaNet's tensors are head-major, but the attention
    # block's come through a different part of the conversion.
    def vslice(v, k, minor):
        return v[:, k::kvh] if minor else v[:, k * hd:(k + 1) * hd]

    def ocols(WOx, s, minor):
        return (WOx[:, s::heads] if minor else WOx[:, s * hd:(s + 1) * hd]).T

    print("every (attn_output slot, kv head) pairing and both axis orders")
    print("  layer  v-order   o-order   slot  kv       z   (donor +8.61)")
    for ly in (3, 31, 63):
        p = "blk.%d." % ly
        v = (E @ m.t(p + "attn_v.weight").T).astype(np.float32)
        WOx = inv(m.t(p + "attn_output.weight").copy())
        for vmin in (False, True):
            for omin in (False, True):
                best = (-99.0, -1, -1)
                for s in range(heads):
                    col = ocols(WOx, s, omin)
                    for k in range(kvh):
                        o = vslice(v, k, vmin) @ col
                        hh = unit(fwd(o))
                        d = (U * hh).sum(axis=1)
                        off = (U * hh[perm]).sum(axis=1)
                        z = (d.mean() - off.mean()) / (off.std() + 1e-12)
                        if z > best[0]:
                            best = (z, s, k)
                print("  %5d  %-9s %-9s %4d %3d  %+6.2f"
                      % (ly, "[dim][h]" if vmin else "[h][dim]",
                         "[dim][h]" if omin else "[h][dim]",
                         best[1], best[2], best[0]))


if __name__ == "__main__":
    main()
