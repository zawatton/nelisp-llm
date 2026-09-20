#!/usr/bin/env python3
"""A GGUF header reader: metadata and the tensor directory, nothing else.

The header sits at the front of the file, so a range request for a few
megabytes answers what a model is without downloading it.  That matters here:
the F16 build of Ternary-Bonsai-2-27B is 53.81 GB and whether it is worth
fetching depends on metadata in its first megabyte.
"""
import struct, sys

GGUF_MAGIC = b"GGUF"

# value type ids
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
          U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
          U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}

# ggml type -> (name, block elements, block bytes); only what we need to size
GGML_TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
              3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
              8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40), 10: ("Q2_K", 256, 84),
              11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
              13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
              15: ("Q8_K", 256, 292), 30: ("BF16", 1, 2)}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def need(self, n):
        if self.p + n > len(self.b):
            raise EOFError("header runs past the %d bytes read" % len(self.b))

    def raw(self, fmt, n):
        self.need(n)
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += n
        return v

    def string(self):
        n = self.raw("<Q", 8)
        self.need(n)
        s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def value(self, t):
        if t in _FIXED:
            fmt, n = _FIXED[t]
            return self.raw(fmt, n)
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.raw("<I", 4)
            n = self.raw("<Q", 8)
            return [self.value(et) for _ in range(n)]
        raise ValueError("unknown value type %d" % t)


def parse(buf):
    r = Reader(buf)
    if buf[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file (magic %r)" % buf[:4])
    r.p = 4
    version = r.raw("<I", 4)
    n_tensors = r.raw("<Q", 8)
    n_kv = r.raw("<Q", 8)
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.raw("<I", 4)
        kv[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.raw("<I", 4)
        dims = [r.raw("<Q", 8) for _ in range(nd)]
        tt = r.raw("<I", 4)
        off = r.raw("<Q", 8)
        tensors.append(dict(name=name, dims=dims, type=tt, offset=off))
    return dict(version=version, kv=kv, tensors=tensors, header_bytes=r.p)


def type_name(t):
    return GGML_TYPES.get(t, ("TYPE%d" % t, None, None))[0]


def align(x, a):
    return (x + a - 1) // a * a


def tensor_bytes(t):
    name, blk, nb = GGML_TYPES.get(t["type"], (None, None, None))
    if blk is None:
        raise ValueError("no size for ggml type %d" % t["type"])
    n = 1
    for d in t["dims"]:
        n *= d
    if n % blk:
        raise ValueError("%s: %d elements is not a whole number of %d-blocks"
                         % (t["name"], n, blk))
    return n // blk * nb


def data_start(info, alignment=32):
    """Byte offset of the tensor data region.  Tensor offsets are relative to it."""
    return align(info["header_bytes"], alignment)


def f16_to_f32(lo, hi):
    """Decode one IEEE-754 half from its two little-endian bytes."""
    h = lo | (hi << 8)
    s = -1.0 if h & 0x8000 else 1.0
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        return s * m * 2.0 ** -24
    if e == 31:
        return s * float("inf") if m == 0 else float("nan")
    return s * (1.0 + m / 1024.0) * 2.0 ** (e - 15)


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        buf = f.read(int(sys.argv[2]) if len(sys.argv) > 2 else 8 << 20)
    info = parse(buf)
    print("gguf v%d  %d tensors  %d metadata keys  header %d bytes"
          % (info["version"], len(info["tensors"]), len(info["kv"]),
             info["header_bytes"]))
    print("\n--- metadata ---")
    for k, v in info["kv"].items():
        s = repr(v)
        if len(s) > 110:
            s = s[:110] + " ... (%d items)" % (len(v) if isinstance(v, list) else 0)
        print("  %-44s %s" % (k, s))
    print("\n--- tensor types ---")
    counts = {}
    for t in info["tensors"]:
        counts[type_name(t["type"])] = counts.get(type_name(t["type"]), 0) + 1
    for k in sorted(counts):
        print("  %-10s %d" % (k, counts[k]))
    print("\n--- first tensors ---")
    for t in info["tensors"][:24]:
        print("  %-44s %-14s %s" % (t["name"], type_name(t["type"]), t["dims"]))


if __name__ == "__main__":
    sys.exit(main())
