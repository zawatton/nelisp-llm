"""Read an nl-llm-wts-v1 weight table.  Shared by the offline reference tools.

Doc 08 Phase 2c.  tools/qwen-weights-verify.py deliberately keeps its OWN,
cruder parse so that a malformed header fails in two independent places; this
module is for the tools whose job is numerics rather than format checking.

Dequantization happens in float64, not float32.  int8 * f32_scale can need more
than 24 mantissa bits, so a float32 reference would differ from the Elisp side's
arithmetic and force a tolerance -- and a tolerance is where an indexing bug
hides.  The GPU multiplies in f32; that is a separate comparison.
"""

import struct

import numpy as np

MAGIC = b"nl-llm-wts-v1" + b"\0" * 3


def _plist_int(body, key):
    return int(body.split(f":{key} ", 1)[1].split()[0].rstrip(")"))


class Table:
    """An opened weight table: header metadata plus a memory-mapped payload."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            if fh.read(16) != MAGIC:
                raise SystemExit(f"{path}: bad magic")
            (hlen,) = struct.unpack("<I", fh.read(4))
            text = fh.read(hlen).decode("utf-8")
        self.path = path
        self.payload_at = 16 + 4 + hlen
        self.blob = np.memmap(path, dtype=np.uint8, mode="r")

        head = text.split("(:name ", 1)[0]
        self.config = {}
        for key in ("dim", "heads", "kv-heads", "head-dim", "layers", "ff",
                    "vocab"):
            self.config[key] = _plist_int(head, key)
        self.config["rope-base"] = float(
            head.split(":rope-base ", 1)[1].split()[0])
        self.config["rms-eps"] = float(
            head.split(":rms-eps ", 1)[1].split()[0])
        self.config["tied-head"] = ":tied-head t" in head

        self.tensors = {}
        for chunk in text.split("(:name ")[1:]:
            body = chunk.rsplit(")", 1)[0] if chunk.endswith(")") else chunk
            e = {"name": body.split('"')[1],
                 "role": body.split(":role ", 1)[1].split()[0],
                 "kind": body.split(':kind "', 1)[1].split('"', 1)[0],
                 "shape": [int(v) for v in body.split(":shape (", 1)[1]
                           .split(")", 1)[0].split()]}
            for key in ("offset", "nbytes", "words", "scale-offset", "layer"):
                e[key] = _plist_int(body, key)
            self.tensors[(e["role"], e["layer"])] = e

    def entry(self, role, layer=-1):
        e = self.tensors.get((role, layer))
        if e is None:
            raise SystemExit(f"table has no {role} at layer {layer}")
        return e

    def f32(self, role, layer=-1):
        """Return an unquantized tensor as float64, shaped as declared."""
        e = self.entry(role, layer)
        if e["kind"] != "f32":
            raise SystemExit(f"{e['name']} is {e['kind']}, not f32")
        start = self.payload_at + e["offset"]
        raw = bytes(self.blob[start:start + e["nbytes"]])
        return (np.frombuffer(raw, dtype=np.float32)
                .astype(np.float64).reshape(e["shape"]))

    def dequant(self, role, layer=-1):
        """Return a quantized matrix dequantized to float64, shape (out, in)."""
        e = self.entry(role, layer)
        if e["kind"] != "int8x4":
            raise SystemExit(f"{e['name']} is {e['kind']}, not int8x4")
        out, cols = e["shape"]
        start = self.payload_at + e["offset"]
        lanes = np.frombuffer(bytes(self.blob[start:start + e["nbytes"]]),
                              dtype=np.int8).reshape(out, e["words"] * 4)
        sstart = self.payload_at + e["scale-offset"]
        scales = np.frombuffer(bytes(self.blob[sstart:sstart + 4 * out]),
                               dtype=np.float32).astype(np.float64)
        return lanes[:, :cols].astype(np.float64) * scales[:, None]

    def rows(self, role, indices, layer=-1):
        """Return selected rows of a quantized matrix, dequantized to float64."""
        e = self.entry(role, layer)
        out, cols = e["shape"]
        stride = 4 * e["words"]
        sstart = self.payload_at + e["scale-offset"]
        scales = np.frombuffer(bytes(self.blob[sstart:sstart + 4 * out]),
                               dtype=np.float32).astype(np.float64)
        got = np.zeros((len(indices), cols), dtype=np.float64)
        for n, i in enumerate(indices):
            base = self.payload_at + e["offset"] + i * stride
            lanes = np.frombuffer(bytes(self.blob[base:base + stride]),
                                  dtype=np.int8)[:cols]
            got[n] = lanes.astype(np.float64) * scales[i]
        return got
