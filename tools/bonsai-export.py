#!/usr/bin/env python3
"""Turn Ternary-Bonsai-2-27B's F16 GGUF into the nl-llm-wts-v1 table.

The packed builds (PTQ1_0, PQ2_0) are PrismML's own quantization types and
stock tooling refuses them; the F16 build is ordinary GGUF and carries the same
weights, so this reads that and applies this project's own per-output-row int8
quantization.  The rotation the model folds into its weights is NOT undone --
it cannot be, the weights are stored rotated -- so the runtime has to apply the
matching transform to activations.  The signs and block size travel in the
header for exactly that.

Two architectures live in one file.  Three blocks in four are Gated DeltaNet
(ssm_* tensors, attn_qkv feeding a causal conv); the fourth is gated full
attention with its own q/k/v.  The role names below keep them apart.
"""
import argparse, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

MAGIC = b"nl-llm-wts-v1" + b"\0" * 3

# gguf tensor suffix -> (role, quantize?)
DELTANET_ROLES = {
    "attn_norm.weight": (":ln1g", False),
    "post_attention_norm.weight": (":ln2g", False),
    "attn_qkv.weight": (":wqkv", True),
    "attn_gate.weight": (":wz", True),
    "ssm_alpha.weight": (":walpha", True),
    "ssm_beta.weight": (":wbeta", True),
    "ssm_a": (":a-log", False),
    "ssm_dt.bias": (":dt-bias", False),
    "ssm_conv1d.weight": (":conv-w", False),
    "ssm_norm.weight": (":ssm-norm", False),
    "ssm_out.weight": (":wout", True),
    "ffn_gate.weight": (":wg", True),
    "ffn_up.weight": (":wu", True),
    "ffn_down.weight": (":wd", True),
}
ATTN_ROLES = {
    "attn_norm.weight": (":ln1g", False),
    "post_attention_norm.weight": (":ln2g", False),
    "attn_q.weight": (":wq", True),
    "attn_k.weight": (":wk", True),
    "attn_v.weight": (":wv", True),
    "attn_output.weight": (":wo", True),
    "attn_q_norm.weight": (":q-norm", False),
    "attn_k_norm.weight": (":k-norm", False),
    "ffn_gate.weight": (":wg", True),
    "ffn_up.weight": (":wu", True),
    "ffn_down.weight": (":wd", True),
}
TOP_ROLES = {"token_embd.weight": (":wte", True),
             "output.weight": (":head", True),
             "output_norm.weight": (":lnf", False)}


def quantize_rows(w):
    absmax = np.abs(w).max(axis=1)
    scales = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales


def pack_int8x4(q):
    out, cols = q.shape
    ng = (cols + 3) // 4
    padded = np.zeros((out, ng * 4), dtype=np.int8)
    padded[:, :cols] = q
    return padded.tobytes(), ng


def sexp(v):
    if isinstance(v, dict):
        return "(" + " ".join(
            "%s %s" % (k if k.startswith(":") else ":" + k.replace("_", "-"),
                       sexp(x)) for k, x in v.items()) + ")"
    if isinstance(v, bool):
        return "t" if v else "nil"
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(sexp(x) for x in v) + ")"
    if isinstance(v, str):
        return v if v.startswith(":") else '"%s"' % v.replace('"', '\\"')
    if isinstance(v, float):
        return repr(v)
    return str(v)


def read_tensor(fh, base, t):
    """Read one tensor as a float32 numpy array with its GGUF dims."""
    fh.seek(base + t["offset"])
    raw = fh.read(gguf.tensor_bytes(t))
    tn = gguf.type_name(t["type"])
    if tn == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif tn == "F32":
        a = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        raise SystemExit("tensor %s is %s; only F16/F32 are read here"
                         % (t["name"], tn))
    # GGUF dims are fastest-varying first, so [in, out] means a row is `in` long
    dims = list(t["dims"])
    if len(dims) == 1:
        return a
    return a.reshape(tuple(reversed(dims)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=None,
                    help="stop after this many blocks (for partial files)")
    args = ap.parse_args()

    with open(args.gguf, "rb") as fh:
        head = fh.read(16 << 20)
    info = gguf.parse(head)
    kv, base = info["kv"], gguf.data_start(info)
    by_name = {t["name"]: t for t in info["tensors"]}
    nblk = kv["qwen35.block_count"]
    if args.layers is not None:
        nblk = min(nblk, args.layers)
    interval = kv["qwen35.full_attention_interval"]

    entries, chunks, off = [], [], 0
    worst = (0.0, "")

    def emit(role, layer, arr, quant):
        nonlocal off, worst
        if quant:
            w = arr if arr.ndim == 2 else arr.reshape(1, -1)
            q, sc = quantize_rows(w)
            blob, ng = pack_int8x4(q)
            deq = q.astype(np.float32) * sc[:, None]
            denom = max(float(np.abs(w).max()), 1e-30)
            rel = float(np.abs(deq - w).max()) / denom
            if rel > worst[0]:
                worst = (rel, "%s@%d" % (role, layer))
            payload = blob + sc.tobytes()
            entries.append(dict(name="%s.%d" % (role, layer), role=role,
                                layer=layer, kind="int8x4",
                                shape=[int(w.shape[0]), int(w.shape[1])],
                                words=int(ng), offset=off,
                                scale_offset=off + len(blob),
                                nbytes=len(payload)))
        else:
            a = arr.astype(np.float32).ravel()
            payload = a.tobytes()
            shape = ([int(arr.shape[0]), int(arr.shape[1])] if arr.ndim == 2
                     else [1, int(a.size)])
            entries.append(dict(name="%s.%d" % (role, layer), role=role,
                                layer=layer, kind="f32", shape=shape,
                                words=0, offset=off, scale_offset=off,
                                nbytes=len(payload)))
        chunks.append(payload)
        off += len(payload)

    for name, (role, quant) in TOP_ROLES.items():
        if name in by_name:
            with open(args.gguf, "rb") as fh:
                emit(role, -1, read_tensor(fh, base, by_name[name]), quant)

    with open(args.gguf, "rb") as fh:
        for ly in range(nblk):
            full = (ly % interval) == (interval - 1)
            table = ATTN_ROLES if full else DELTANET_ROLES
            for suffix, (role, quant) in table.items():
                t = by_name.get("blk.%d.%s" % (ly, suffix))
                if t is None:
                    raise SystemExit("block %d has no %s" % (ly, suffix))
                emit(role, ly, read_tensor(fh, base, t), quant)
            print("  block %2d %-9s %d tensors" % (ly, "attention" if full
                                                   else "deltanet", len(table)),
                  flush=True)

    header = {
        ":format": "nl-llm-wts-v1",
        ":donor": os.path.basename(os.path.abspath(args.gguf)),
        ":arch": "qwen35",
        ":dim": kv["qwen35.embedding_length"],
        ":heads": kv["qwen35.attention.head_count"],
        ":kv-heads": kv["qwen35.attention.head_count_kv"],
        ":head-dim": kv["qwen35.attention.key_length"],
        ":layers": nblk,
        ":ff": kv["qwen35.feed_forward_length"],
        ":vocab": len(kv["tokenizer.ggml.tokens"]),
        ":rope-base": float(kv["qwen35.rope.freq_base"]),
        ":rope-dims": kv["qwen35.rope.dimension_count"],
        ":rope-sections": list(kv["qwen35.rope.dimension_sections"]),
        ":rms-eps": float(kv["qwen35.attention.layer_norm_rms_epsilon"]),
        ":full-attention-interval": interval,
        ":ssm-conv-kernel": kv["qwen35.ssm.conv_kernel"],
        ":ssm-state": kv["qwen35.ssm.state_size"],
        ":ssm-groups": kv["qwen35.ssm.group_count"],
        ":ssm-heads": kv["qwen35.ssm.time_step_rank"],
        ":ssm-inner": kv["qwen35.ssm.inner_size"],
        ":hadamard-block": kv["prism.hadamard.block_size"],
        ":hadamard-widths": list(kv["prism.hadamard.sign_widths"]),
        ":hadamard-signs": [int(x) for x in kv["prism.hadamard.sign_values"]],
        ":hadamard-weights": list(kv["prism.hadamard.weight_names"]),
        ":hadamard-inverse": list(kv["prism.hadamard.inverse_weight_names"]),
        ":tied-head": False,
        ":tensors": entries,
    }
    blob = sexp(header).encode("utf-8")
    with open(args.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        for c in chunks:
            fh.write(c)
    total = os.path.getsize(args.out)
    print("wrote %s: %d tensors, %.2f GB, header %d bytes"
          % (args.out, len(entries), total / 1e9, len(blob)))
    print("  worst per-tensor relative dequantization error: %.4f (%s)" % worst)


if __name__ == "__main__":
    sys.exit(main())
