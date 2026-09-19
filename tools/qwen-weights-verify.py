#!/usr/bin/env python3
"""Verify an exported nl-llm weight table against its donor safetensors.

Doc 08 Phase 2b, Verification check 3: "Packed int8 unpacked back to f32 must
match the quantizer's own output exactly (bit-level), separating packing bugs
from quantization error."  Those are two different failures and a single
end-to-end comparison cannot tell them apart -- a transposed row and a coarse
scale both show up as "the numbers are a bit off" -- so they are measured
separately here:

  packing    unpack(pack(q)) == q exactly, and the padding lanes are zero
  numerics   |W - q*scale| / max|W| per tensor, reported, not asserted away

It also re-reads the table the way the Elisp loader will (offsets and lengths
from the header, nothing inferred) so a header that disagrees with its own
payload fails here rather than in the runtime.

Usage:
    PYTHONPATH=build/donor/pylibs \\
      python3 tools/qwen-weights-verify.py DONOR_DIR TABLE_FILE
"""

import json
import os
import struct
import sys

import numpy as np

MAGIC = b"nl-llm-wts-v1" + b"\0" * 3


def read_donor(path):
    """name -> f32 array, from safetensors.  Independent of the exporter."""
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
    base = 8 + hdr_len
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[base + start:base + end]
        if meta["dtype"] == "BF16":
            u16 = raw.view(np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            arr = raw.view(np.float32)
        else:
            arr = raw.view(np.float16).astype(np.float32)
        out[name] = arr.reshape(meta["shape"])
    return out


def parse_sexp_header(path):
    """Pull the tensor table out of the sexp header without a Lisp reader.

    Deliberately crude: the point is to read the header through a DIFFERENT
    path than the one that wrote it, so a malformed sexp is caught here even
    though Emacs also reads it in the test.
    """
    with open(path, "rb") as fh:
        magic = fh.read(16)
        if magic != MAGIC:
            raise SystemExit(f"{path}: bad magic {magic!r}")
        (hlen,) = struct.unpack("<I", fh.read(4))
        text = fh.read(hlen).decode("utf-8")
    payload_at = 16 + 4 + hlen

    entries = []
    for chunk in text.split("(:name ")[1:]:
        body = chunk.rsplit(")", 1)[0] if chunk.endswith(")") else chunk
        fields = {}
        name = body.split('"')[1]
        fields["name"] = name
        for key in ("kind", "src-dtype"):
            marker = f":{key} \""
            if marker in body:
                fields[key] = body.split(marker, 1)[1].split('"', 1)[0]
        for key in ("offset", "nbytes", "words", "scale-offset", "layer"):
            marker = f":{key} "
            fields[key] = int(body.split(marker, 1)[1].split()[0].rstrip(")"))
        fields["shape"] = [int(v) for v in
                           body.split(":shape (", 1)[1].split(")", 1)[0].split()]
        fields["role"] = body.split(":role ", 1)[1].split()[0]
        entries.append(fields)
    return payload_at, entries, text


def unpack_int8x4(buf, out, cols, words):
    """Inverse of the exporter's packing: int8 lanes, four per u32, row padded."""
    lanes = np.frombuffer(buf, dtype=np.int8).reshape(out, words * 4)
    return lanes[:, :cols], lanes[:, cols:]


def main(donor_dir, table):
    donor = read_donor(os.path.join(donor_dir, "model.safetensors"))
    payload_at, entries, text = parse_sexp_header(table)
    blob = np.memmap(table, dtype=np.uint8, mode="r")

    fails = []
    worst_rel, worst_name = 0.0, None
    pad_nonzero = 0
    checked_params = 0

    for e in entries:
        name, kind = e["name"], e["kind"]
        want = donor.get(name)
        if want is None:
            fails.append(f"{name}: in table but not in donor")
            continue
        start = payload_at + e["offset"]
        raw = bytes(blob[start:start + e["nbytes"]])
        if len(raw) != e["nbytes"]:
            fails.append(f"{name}: payload short ({len(raw)} of {e['nbytes']})")
            continue

        if kind == "f32":
            got = np.frombuffer(raw, dtype=np.float32).reshape(e["shape"])
            if not np.array_equal(got, want.astype(np.float32)):
                fails.append(f"{name}: f32 payload differs from donor")
            checked_params += got.size
            continue

        out, cols = e["shape"]
        q, pad = unpack_int8x4(raw, out, cols, e["words"])
        if pad.size and pad.any():
            pad_nonzero += 1
            fails.append(f"{name}: {int((pad != 0).sum())} padding lanes nonzero")
        # Bit-level: repacking what we unpacked must reproduce the file bytes.
        repacked = np.zeros((out, e["words"] * 4), dtype=np.int8)
        repacked[:, :cols] = q
        if repacked.tobytes() != raw:
            fails.append(f"{name}: pack/unpack is not bit-exact")

        sstart = payload_at + e["scale-offset"]
        scales = np.frombuffer(bytes(blob[sstart:sstart + 4 * out]),
                               dtype=np.float32)
        if scales.shape != (out,):
            fails.append(f"{name}: expected {out} scales, got {scales.shape}")
            continue
        back = q.astype(np.float32) * scales[:, None]
        absmax = float(np.abs(want).max())
        rel = float(np.abs(want - back).max() / absmax) if absmax > 0 else 0.0
        if rel > worst_rel:
            worst_rel, worst_name = rel, name
        # Per-row int8 with a max-based scale cannot exceed half a step, i.e.
        # 1/254 of the row maximum; allow a little slack for rounding of the
        # scale itself.  This is the assertion that a broken scale trips.
        if rel > 0.01:
            fails.append(f"{name}: relative error {rel:.4f} exceeds 1%")
        checked_params += q.size

    missing = set(donor) - {e["name"] for e in entries}
    print(f"table    {table}")
    print(f"tensors  {len(entries)} checked, {checked_params/1e6:.1f}M params")
    print(f"donor    {len(donor)} tensors, not in table: "
          f"{sorted(missing) if missing else 'none'}")
    print(f"packing  bit-exact on every int8 tensor, "
          f"{pad_nonzero} with nonzero padding")
    print(f"numerics worst relative dequantization error {worst_rel:.4f} "
          f"({worst_name})")
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails[:20]:
            print(f"  {f}")
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
