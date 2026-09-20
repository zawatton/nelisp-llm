#!/usr/bin/env python3
"""Is the ternary table bit-identical to the weights it came from?

The int8 export requantizes: every value moves a little, and "a little" was
measured at 0.0039 relative.  The ternary export does not -- the model's own
weights are ternary inside each 128-element block, so the block's absmax is
its scale and the values divide into it exactly.  That is a property worth
asserting rather than assuming, because an off-by-one in the packing produces
numbers that are still plausible.

Reads the exported table directly rather than through the Elisp, so a
disagreement between this and test/ternary-test.el separates "the exporter
wrote it wrong" from "the reader reads it wrong".
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_hd = SourceFileLoader("hd", os.path.join(HERE, "hadamard-direction.py")).load_module()


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(16)
        if not magic.startswith(b"nl-llm-wts-v1"):
            raise SystemExit("not an nl-llm-wts-v1 table")
        n = struct.unpack("<I", f.read(4))[0]
        blob = f.read(n).decode("utf-8")
    return blob, 16 + 4 + n


def read_sexp(text, i=0):
    """A minimal reader for the header's Lisp data: lists, keywords, numbers,
    strings, t and nil.  Written out rather than regexed because `:shape
    (10240 5120)' nests, and a scanner that does not nest reads the shape's
    closing paren as the entry's."""
    while i < len(text) and text[i] in " \t\n":
        i += 1
    if i >= len(text):
        raise ValueError("unexpected end of header")
    c = text[i]
    if c == "(":
        out, i = [], i + 1
        while True:
            while i < len(text) and text[i] in " \t\n":
                i += 1
            if text[i] == ")":
                return out, i + 1
            v, i = read_sexp(text, i)
            out.append(v)
    if c == '"':
        j = i + 1
        buf = []
        while text[j] != '"':
            if text[j] == "\\":
                j += 1
            buf.append(text[j])
            j += 1
        return "".join(buf), j + 1
    j = i
    while j < len(text) and text[j] not in " \t\n()":
        j += 1
    tok = text[i:j]
    if tok == "nil":
        return None, j
    if tok == "t":
        return True, j
    try:
        return int(tok), j
    except ValueError:
        pass
    try:
        return float(tok), j
    except ValueError:
        return tok, j


def plist(items):
    # the exporter writes :scale-offset for the key scale_offset
    return {k.lstrip(":").replace("-", "_"): v
            for k, v in zip(items[0::2], items[1::2])}


def sexp_entries(blob):
    header, _ = read_sexp(blob)
    return [plist(e) for e in plist(header)["tensors"]]


def main():
    table = sys.argv[1]
    ggufp = sys.argv[2] if len(sys.argv) > 2 else "build/bonsai/f16.gguf"
    blob, payload_at = read_header(table)
    ents = sexp_entries(blob)
    with open(ggufp, "rb") as f:
        info = gguf.parse(f.read(512 << 20))
    base = gguf.data_start(info)
    by = {t["name"]: t for t in info["tensors"]}
    role_to_g = {}
    bx = SourceFileLoader("bx", os.path.join(HERE, "bonsai-export.py")).load_module()
    for suffix, (role, _q) in list(bx.DELTANET_ROLES.items()) + \
            list(bx.ATTN_ROLES.items()):
        role_to_g.setdefault(role, suffix)
    for name, (role, _q) in bx.TOP_ROLES.items():
        role_to_g.setdefault(role, name)

    checked = worst = 0
    for e in ents:
        if e.get("kind") != "ternary2":
            continue
        role, layer = e["role"], e["layer"]
        suffix = role_to_g.get(role)
        gname = suffix if layer < 0 else "blk.%d.%s" % (layer, suffix)
        if gname not in by:
            continue
        rows, cols = e["shape"]
        words, block = e["words"], e["block"]
        nb = (cols + block - 1) // block
        off, soff = e["offset"], e["scale_offset"]
        nrow = min(rows, 8)
        with open(table, "rb") as f:
            f.seek(payload_at + off)
            raw = np.frombuffer(f.read(nrow * words * 4), dtype="<u4").reshape(nrow, words)
            f.seek(payload_at + soff)
            sc = np.frombuffer(f.read(nrow * nb * 4), dtype="<f4").reshape(nrow, nb)
        idx = np.arange(cols)
        v = (raw[:, idx // 16].astype(np.int64) >> (2 * (idx % 16))) & 3
        trits = np.where(v == 3, -1, v).astype(np.float32)
        deq = trits * sc[:, idx // block]
        with open(ggufp, "rb") as f:
            src = _hd.read_tensor(f, base, by[gname]).astype(np.float32)[:nrow]
        d = float(np.abs(deq - src).max())
        worst = max(worst, d)
        checked += 1
        if d != 0.0:
            print("  MISMATCH %-28s max |difference| %.6e" % (gname, d))
    if len(sys.argv) > 3:
        # a handful of rows, straight from the GGUF, for the Elisp reader to
        # be checked against -- so "the exporter is wrong" and "the reader is
        # wrong" stay separable
        rows_out = []
        for e in ents:
            if e.get("kind") != "ternary2":
                continue
            suffix = role_to_g.get(e["role"])
            gname = suffix if e["layer"] < 0 else "blk.%d.%s" % (e["layer"], suffix)
            if gname not in by or len(rows_out) >= 6:
                continue
            with open(ggufp, "rb") as f:
                src = _hd.read_tensor(f, base, by[gname]).astype(np.float32)
            r = 3 % src.shape[0]
            rows_out.append((e["role"], e["layer"], r,
                             [float(v) for v in src[r][:64]]))
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            f.write(";; -*- lisp-data -*-  AUTO-GENERATED by "
                    "tools/ternary-verify.py -- do not edit.\n")
            f.write(";; The first 64 values of one row of each of six tensors,\n"
                    ";; read straight from the donor GGUF in float32.\n")
            f.write("(")
            for role, layer, r, vals in rows_out:
                f.write("(:role %s :layer %d :row %d :values (%s))\n "
                        % (role, layer, r,
                           " ".join(repr(v) for v in vals)))
            f.write(")\n")
        print("  wrote %s (%d rows)" % (sys.argv[3], len(rows_out)))
    print("  %d ternary tensors checked against the GGUF, worst |difference| %.1e"
          % (checked, worst))
    if worst != 0.0:
        raise SystemExit(1)
    print("  bit-identical")


if __name__ == "__main__":
    main()
