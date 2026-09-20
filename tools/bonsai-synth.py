#!/usr/bin/env python3
"""A tiny model in Ternary Bonsai's shape, for checking gradients.

Finite differences need thousands of forward passes; the real model takes
fourteen seconds a block.  This writes the same nl-llm-wts-v1 table with the
same role names, the same 3:1 interleave and the same rotation machinery, at
dimensions where a full numerical check costs seconds -- including a Hadamard
block size of 8, which is what makes `:hadamard-block' worth reading from the
header rather than assuming 1024.

Deterministic: the same seed gives the same file, so a gradient check that
fails can be re-run on exactly the weights that failed it.
"""
import argparse, os, struct, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
_bx = SourceFileLoader("bx", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "bonsai-export.py")).load_module()

MAGIC = _bx.MAGIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=24)
    ap.add_argument("--block", type=int, default=8, help="Hadamard block size")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dim = args.dim
    heads, kvh, hdim = 2, 1, 8          # full-attention block
    rdims = 4                            # partial rotary: 4 of 8
    ff = 2 * dim
    nk, nv, sstate, kern = 1, 2, 4, 3    # DeltaNet: groups, heads, state, conv
    kd, vd = nk * sstate, nv * sstate
    cd = kd + kd + vd
    interval = 2                         # one DeltaNet then one attention
    widths = sorted({dim, vd, ff, heads * hdim})
    for w in widths:
        if w % args.block:
            raise SystemExit("width %d is not a multiple of block %d"
                             % (w, args.block))

    entries, chunks = [], []
    off = 0

    def emit(role, layer, arr, quant):
        nonlocal off
        if quant:
            w = arr if arr.ndim == 2 else arr.reshape(1, -1)
            q, sc = _bx.quantize_rows(w.astype(np.float32))
            blob, ng = _bx.pack_int8x4(q)
            payload = blob + sc.astype(np.float32).tobytes()
            entries.append(dict(name="%s.%d" % (role, layer), role=role,
                                layer=layer, kind="int8x4",
                                shape=[int(w.shape[0]), int(w.shape[1])],
                                words=int(ng), offset=off,
                                scale_offset=off + len(blob),
                                nbytes=len(payload)))
        else:
            a = np.asarray(arr, dtype=np.float32).ravel()
            payload = a.tobytes()
            shape = ([int(arr.shape[0]), int(arr.shape[1])] if arr.ndim == 2
                     else [1, int(a.size)])
            entries.append(dict(name="%s.%d" % (role, layer), role=role,
                                layer=layer, kind="f32", shape=shape,
                                words=0, offset=off, scale_offset=off,
                                nbytes=len(payload)))
        chunks.append(payload)
        off += len(payload)

    def w(rows, cols, s=0.15):
        return rng.normal(0.0, s, size=(rows, cols)).astype(np.float32)

    emit(":wte", -1, w(args.vocab, dim, 0.3), True)
    emit(":head", -1, w(args.vocab, dim, 0.2), True)
    emit(":lnf", -1, np.ones(dim, dtype=np.float32) + 0.05 * rng.normal(size=dim), False)

    for ly in range(args.layers):
        full = (ly % interval) == (interval - 1)
        emit(":ln1g", ly, np.ones(dim, dtype=np.float32) + 0.05 * rng.normal(size=dim), False)
        emit(":ln2g", ly, np.ones(dim, dtype=np.float32) + 0.05 * rng.normal(size=dim), False)
        if full:
            emit(":wq", ly, w(2 * heads * hdim, dim), True)
            emit(":wk", ly, w(kvh * hdim, dim), True)
            emit(":wv", ly, w(kvh * hdim, dim), True)
            emit(":wo", ly, w(dim, heads * hdim), True)
            emit(":q-norm", ly, np.ones(hdim, dtype=np.float32) + 0.05 * rng.normal(size=hdim), False)
            emit(":k-norm", ly, np.ones(hdim, dtype=np.float32) + 0.05 * rng.normal(size=hdim), False)
        else:
            emit(":wqkv", ly, w(cd, dim), True)
            emit(":wz", ly, w(vd, dim), True)
            emit(":walpha", ly, w(nv, dim), True)
            emit(":wbeta", ly, w(nv, dim), True)
            emit(":wout", ly, w(dim, vd), True)
            emit(":a-log", ly, rng.normal(0.0, 0.3, size=nv).astype(np.float32), False)
            emit(":dt-bias", ly, rng.normal(0.0, 0.3, size=nv).astype(np.float32), False)
            # channel-major, exactly as the exporter's reshape leaves the real one
            emit(":conv-w", ly, w(cd, kern, 0.3), False)
            emit(":ssm-norm", ly, np.ones(sstate, dtype=np.float32) + 0.05 * rng.normal(size=sstate), False)
        emit(":wg", ly, w(ff, dim), True)
        emit(":wu", ly, w(ff, dim), True)
        emit(":wd", ly, w(dim, ff), True)

    signs = []
    for width in widths:
        signs.extend(int(x) for x in rng.choice([-1, 1], size=width))

    header = {
        ":format": "nl-llm-wts-v1",
        ":donor": "synthetic",
        ":arch": "qwen35",
        ":dim": dim, ":heads": heads, ":kv-heads": kvh, ":head-dim": hdim,
        ":layers": args.layers, ":ff": ff, ":vocab": args.vocab,
        ":rope-base": 10000.0, ":rope-dims": rdims, ":rope-sections": [],
        ":rms-eps": 1e-6,
        ":full-attention-interval": interval,
        ":ssm-conv-kernel": kern, ":ssm-state": sstate,
        ":ssm-groups": nk, ":ssm-heads": nv, ":ssm-inner": vd,
        ":hadamard-block": args.block,
        ":hadamard-widths": list(widths),
        ":hadamard-signs": signs,
        ":hadamard-weights": [], ":hadamard-inverse": [],
        ":tied-head": False,
        ":tensors": entries,
    }
    blob = _bx.sexp(header).encode("utf-8")
    with open(args.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        for c in chunks:
            fh.write(c)
    print("wrote %s: %d tensors, %d bytes, widths %s, block %d"
          % (args.out, len(entries), os.path.getsize(args.out), widths, args.block))


if __name__ == "__main__":
    main()
