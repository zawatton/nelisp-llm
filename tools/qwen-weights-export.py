#!/usr/bin/env python3
"""Convert a donor safetensors checkpoint to the nl-llm binary weight table.

Doc 08 (docs/design/08-weight-import.org) Phase 2b.  OFFLINE tool, not runtime.
The safetensors reader is written out here rather than imported so that Phase 2e
has something to port to Elisp; the format is small enough that a dependency
would buy nothing and hide what the Elisp side has to do.

Quantization is per-output-row int8: for a weight of shape (out, in),
scale[o] = max|W[o,:]| / 127 and q[o,i] = clip(round(W[o,i] / scale[o])).  Per
row rather than per tensor because a single scale is set by the largest row and
wastes most of the int8 range on every other one.  The DP4A kernel in
nelisp-gpu already indexes BIAS by the output row and has `o' in scope where it
reads BETA[0], so the kernel side of per-row scales is a one-token change in
Phase 2d, not a redesign.

Not quantized: RMSNorm gains and the Qwen3 q_norm/k_norm vectors.  They are a
few thousand floats in total and they are the tensors quantization error hurts
most, so they stay f32.

Binary format `nl-llm-wts-v1' (little-endian):

    magic     16 bytes  "nl-llm-wts-v1" + three NULs
    hdr_len   u32       byte length of the header that follows
    header    bytes     ONE Lisp sexp, utf-8: a plist describing config and
                        every tensor (:name :role :layer :shape :kind :offset
                        :nbytes :scale-offset).  A sexp, not JSON, because the
                        Phase 2e Elisp loader can `read' it with no parser and
                        NeLisp's JSON support differs from Emacs's.
    payload   bytes     per tensor, in header order: int8 lanes packed four per
                        little-endian u32 (kind int8x4), or raw f32 (kind f32).
                        int8 tensors are followed by `out' f32 scales at
                        :scale-offset.

Every donor tensor must be either mapped or explicitly ignored; an unmapped name
is a hard error, because a silently dropped tensor is exactly the kind of
conversion that loads, runs, and produces confident nonsense.

Usage:
    PYTHONPATH=build/donor/pylibs \\
      python3 tools/qwen-weights-export.py DONOR_DIR OUT_FILE
"""

import json
import os
import re
import struct
import sys

import numpy as np

MAGIC = b"nl-llm-wts-v1" + b"\0" * 3

# Donor tensor name -> (role, quantize?).  Roles are the plist keys the Elisp
# model uses, so the mapping is the whole translation layer.
PER_LAYER = {
    "self_attn.q_proj.weight": (":wq", True),
    "self_attn.k_proj.weight": (":wk", True),
    "self_attn.v_proj.weight": (":wv", True),
    "self_attn.o_proj.weight": (":wo", True),
    "self_attn.q_norm.weight": (":q-norm", False),
    "self_attn.k_norm.weight": (":k-norm", False),
    "mlp.gate_proj.weight": (":wg", True),
    "mlp.up_proj.weight": (":wu", True),
    "mlp.down_proj.weight": (":wd", True),
    "input_layernorm.weight": (":ln1g", False),
    "post_attention_layernorm.weight": (":ln2g", False),
}
TOP_LEVEL = {
    "model.embed_tokens.weight": (":wte", True),
    "model.norm.weight": (":lnf", False),
    "lm_head.weight": (":head", True),
}
# Handled by a check rather than by the table: Qwen3-0.6B declares
# tie_word_embeddings and ALSO ships lm_head.weight, bitwise identical to
# model.embed_tokens.weight.  Storing both would cost 155 MiB of int8 payload
# for a duplicate, so the duplicate is dropped -- but only after proving it is
# one.  A donor whose lm_head actually differs under that flag is a different
# model than the config claims, and says so loudly here.
DEDUP_AGAINST = {"lm_head.weight": "model.embed_tokens.weight"}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def read_safetensors(path):
    """Yield (name, numpy array) for every tensor, memory-mapping the payload.

    Format: u64 header length, that many bytes of JSON mapping name ->
    {dtype, shape, data_offsets}, then the tensor bytes.  bf16 is widened to
    f32 by placing its 16 bits in the high half of an f32, which is exact --
    bf16 and f32 share exponent width, so the conversion only pads mantissa.
    """
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
    base = 8 + hdr_len
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[base + start:base + end]
        dtype = meta["dtype"]
        if dtype == "BF16":
            u16 = raw.view(np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        elif dtype == "F32":
            arr = raw.view(np.float32)
        elif dtype == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise SystemExit(f"{name}: unsupported dtype {dtype}")
        yield name, arr.reshape(meta["shape"]), dtype


def quantize_rows(w):
    """Per-output-row int8 quantization of 2-D W.  Returns (int8 array, scales)."""
    absmax = np.abs(w).max(axis=1)
    # A row that is entirely zero has no scale; use 1.0 so dequantization gives
    # zeros back instead of NaN.
    scales = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales


def pack_int8x4(q):
    """Pack an int8 array into u32 words, four lanes per word, row-padded.

    Returns (bytes, words_per_row).  Each row is padded with zero lanes to a
    multiple of four so the kernel can read whole words; zero lanes contribute
    nothing to the dot product.
    """
    out, cols = q.shape
    ng = (cols + 3) // 4
    padded = np.zeros((out, ng * 4), dtype=np.int8)
    padded[:, :cols] = q
    return padded.tobytes(), ng


def dequant_error(w, q, scales):
    """Max absolute and relative error of the round trip, for the report."""
    back = q.astype(np.float32) * scales[:, None]
    absmax = np.abs(w).max()
    err = np.abs(w - back).max()
    return err, (err / absmax if absmax > 0 else 0.0)


def main(donor, out_path):
    cfg = json.load(open(os.path.join(donor, "config.json"), encoding="utf-8"))
    src = os.path.join(donor, "model.safetensors")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} (sharded donors are not handled yet)")

    entries, payload = [], bytearray()
    worst = (0.0, None)
    seen = set()
    tied = cfg.get("tie_word_embeddings", False)
    raw = {name: arr for name, arr, _ in read_safetensors(src)}
    dropped = []
    for dup, orig in DEDUP_AGAINST.items():
        if dup in raw and orig in raw:
            if not np.array_equal(raw[dup], raw[orig]):
                raise SystemExit(
                    f"{dup} differs from {orig} (max abs diff "
                    f"{float(np.abs(raw[dup] - raw[orig]).max())}), so this donor "
                    f"has a real separate head; tie_word_embeddings={tied} in "
                    "config.json is not the whole story -- decide explicitly")
            dropped.append(dup)
    del raw

    def emit(name, role, layer, arr, quantize, dtype):
        nonlocal worst
        if quantize:
            if arr.ndim != 2:
                raise SystemExit(f"{name}: expected a matrix, got {arr.shape}")
            q, scales = quantize_rows(arr)
            err, rel = dequant_error(arr, q, scales)
            if rel > worst[0]:
                worst = (rel, name)
            data, ng = pack_int8x4(q)
            offset = len(payload)
            payload.extend(data)
            scale_offset = len(payload)
            payload.extend(scales.tobytes())
            entries.append(dict(name=name, role=role, layer=layer,
                                shape=list(arr.shape), kind="int8x4",
                                offset=offset, nbytes=len(data), words=ng,
                                scale_offset=scale_offset, rel_err=float(rel),
                                src_dtype=dtype))
        else:
            data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            offset = len(payload)
            payload.extend(data)
            entries.append(dict(name=name, role=role, layer=layer,
                                shape=list(arr.shape), kind="f32",
                                offset=offset, nbytes=len(data), words=0,
                                scale_offset=-1, rel_err=0.0, src_dtype=dtype))

    for name, arr, dtype in read_safetensors(src):
        seen.add(name)
        if name in dropped:
            continue
        if name in TOP_LEVEL:
            role, quant = TOP_LEVEL[name]
            emit(name, role, -1, arr, quant, dtype)
            continue
        m = LAYER_RE.match(name)
        if m and m.group(2) in PER_LAYER:
            role, quant = PER_LAYER[m.group(2)]
            emit(name, role, int(m.group(1)), arr, quant, dtype)
            continue
        raise SystemExit(
            f"unmapped donor tensor {name!r} {list(arr.shape)}; add it to "
            "PER_LAYER/TOP_LEVEL or to IGNORED -- refusing to drop it silently")

    # After dedup, a tied donor must have no :head role and an untied one must.
    has_head = any(e["role"] == ":head" for e in entries)
    if tied and has_head:
        raise SystemExit("tied donor still emitted a :head tensor")
    if not tied and not has_head:
        raise SystemExit("config says untied but no lm_head tensor was found")

    n_layers = cfg["num_hidden_layers"]
    got = sorted({e["layer"] for e in entries if e["layer"] >= 0})
    if got != list(range(n_layers)):
        raise SystemExit(f"expected layers 0..{n_layers - 1}, got {got}")
    per_layer_roles = sorted({r for r, _ in PER_LAYER.values()})
    for ly in range(n_layers):
        have = sorted({e["role"] for e in entries if e["layer"] == ly})
        if have != per_layer_roles:
            raise SystemExit(f"layer {ly} has {have}, expected {per_layer_roles}")

    header = {
        ":format": "nl-llm-wts-v1",
        ":donor": os.path.basename(os.path.abspath(donor)),
        ":dim": cfg["hidden_size"],
        ":heads": cfg["num_attention_heads"],
        ":kv-heads": cfg["num_key_value_heads"],
        ":head-dim": cfg["head_dim"],
        ":layers": n_layers,
        ":ff": cfg["intermediate_size"],
        ":vocab": cfg["vocab_size"],
        ":rope-base": float(cfg["rope_theta"]),
        ":rms-eps": float(cfg["rms_norm_eps"]),
        ":tied-head": tied,
        ":tensors": entries,
    }

    with open(out_path, "wb") as fh:
        blob = sexp(header).encode("utf-8")
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        fh.write(payload)

    total = os.path.getsize(out_path)
    q_bytes = sum(e["nbytes"] for e in entries if e["kind"] == "int8x4")
    f_bytes = sum(e["nbytes"] for e in entries if e["kind"] == "f32")
    params = sum(int(np.prod(e["shape"])) for e in entries)
    print(f"wrote {out_path}: {len(entries)} tensors, {params/1e6:.1f}M params, "
          f"{total/2**20:.0f} MiB total")
    print(f"  int8 payload {q_bytes/2**20:.0f} MiB, f32 payload "
          f"{f_bytes/2**20:.2f} MiB, header {len(blob)} bytes")
    print(f"  worst per-tensor relative dequantization error: "
          f"{worst[0]:.4f} ({worst[1]})")
    print(f"  donor tensors seen {len(seen)}, all mapped"
          + (f", {len(dropped)} verified duplicate(s) dropped: "
             f"{', '.join(dropped)}" if dropped else ""))


def sexp(value):
    """Render VALUE as a Lisp datum: dicts become plists, bools t / nil."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            # Lisp keys use hyphens; the dicts above are built with Python
            # identifiers, so translate rather than leak scale_offset into a
            # plist the Elisp loader has to read.
            key = k if k.startswith(":") else ":" + k.replace("_", "-")
            parts.append(f"{key} {sexp(v)}")
        return "(" + " ".join(parts) + ")"
    if isinstance(value, bool):
        return "t" if value else "nil"
    if isinstance(value, (list, tuple)):
        return "(" + " ".join(sexp(v) for v in value) + ")"
    if isinstance(value, str):
        return value if value.startswith(":") else json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
