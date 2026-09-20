#!/usr/bin/env python3
"""Decide which way the incoherence rotation runs, from the weights themselves.

Both candidate transforms are orthogonal, so every norm-based instrument is
blind to the difference by construction.  What is not blind is coherence:
incoherence processing exists precisely to flatten a weight's heavy tails, so
undoing it must make the rows spikier again and doing it twice must not.  The
direction that raises kurtosis is the inverse of the one the runtime applies.

Reads F16 straight from the GGUF so no quantization noise enters the measure.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

BLOCK = 1024


def read_tensor(fh, base, t):
    fh.seek(base + t["offset"])
    raw = fh.read(gguf.tensor_bytes(t))
    tn = gguf.type_name(t["type"])
    if tn == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif tn == "F32":
        a = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        raise SystemExit("%s is %s" % (t["name"], tn))
    dims = list(t["dims"])
    if len(dims) == 1:
        return a
    return a.reshape(tuple(reversed(dims)))


def fwht(a):
    """In-place blockwise Walsh-Hadamard over the last axis, normalised."""
    n = a.shape[-1]
    assert n % BLOCK == 0, n
    a = a.reshape(a.shape[0], n // BLOCK, BLOCK).copy()
    h = 1
    while h < BLOCK:
        a = a.reshape(a.shape[0], a.shape[1], BLOCK // (2 * h), 2, h)
        x = a[:, :, :, 0, :].copy()
        y = a[:, :, :, 1, :].copy()
        a[:, :, :, 0, :] = x + y
        a[:, :, :, 1, :] = x - y
        a = a.reshape(a.shape[0], -1, BLOCK)
        h *= 2
    return (a / math.sqrt(BLOCK)).reshape(a.shape[0], n)


def rotate(rows, signs, invert):
    if invert:
        return fwht(rows) * signs
    return fwht(rows * signs)


def coherence(rows):
    """max-to-rms per row (averaged) and excess kurtosis over the matrix."""
    rms = np.sqrt(np.mean(rows * rows, axis=1))
    peak = np.max(np.abs(rows), axis=1)
    m2 = np.mean(rows * rows)
    m4 = np.mean(rows ** 4)
    return float(np.mean(peak / rms)), float(m4 / (m2 * m2) - 3.0)


def main(path):
    with open(path, "rb") as f:
        info = gguf.parse(f.read(512 << 20))
    base = gguf.data_start(info)
    byname = {t["name"]: t for t in info["tensors"]}
    kv = info["kv"]
    widths = list(kv["prism.hadamard.sign_widths"])
    values = np.asarray(kv["prism.hadamard.sign_values"], dtype=np.float32)
    names = set(kv.get("prism.hadamard.weight_names", []))
    runs = {}
    off = 0
    for w in widths:
        runs[w] = values[off:off + w].copy()
        off += w
    print(f"sign widths {widths}, {off} values, {len(names)} rotated tensors")

    probes = [
        ("blk.0.attn_qkv.weight", True),
        ("blk.0.ffn_down.weight", True),
        ("blk.0.ffn_up.weight", True),
        ("blk.0.ssm_alpha.weight", False),
        ("blk.0.ssm_beta.weight", False),
        ("token_embd.weight", False),
    ]
    rowcap = 512
    print()
    print(f"{'tensor':28s} {'rot?':5s} {'':>10s} {'stored':>9s} "
          f"{'forward':>9s} {'inverse':>9s}")
    for name, expect in probes:
        if name not in byname:
            print(f"{name:28s} MISSING")
            continue
        with open(path, "rb") as f:
            rows = read_tensor(f, base, byname[name])
        if rows.ndim != 2:
            print(f"{name:28s} ndim {rows.ndim}")
            continue
        cols = rows.shape[1]
        if cols not in runs:
            print(f"{name:28s} width {cols} has no sign run")
            continue
        rows = rows[:rowcap]
        s = runs[cols]
        stat = coherence(rows)
        fwd = coherence(rotate(rows, s, False))
        inv = coherence(rotate(rows, s, True))
        flag = "yes" if name in names else "no"
        agree = "" if (flag == "yes") == expect else "  <-- METADATA DISAGREES"
        print(f"{name:28s} {flag:5s} {'peak/rms':>10s} "
              f"{stat[0]:9.3f} {fwd[0]:9.3f} {inv[0]:9.3f}{agree}")
        print(f"{'':28s} {'':5s} {'kurtosis':>10s} "
              f"{stat[1]:9.3f} {fwd[1]:9.3f} {inv[1]:9.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/bonsai/f16.gguf")
