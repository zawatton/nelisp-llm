#!/usr/bin/env python3
"""An independent forward over Ternary Bonsai 2 27B, in numpy, from the GGUF.

The Elisp path takes fifteen seconds a block, so sweeping a convention costs a
quarter of an hour an answer.  Several conventions in this architecture are not
written down anywhere in the weight file -- which way the folded rotation runs,
which side of the gated norm the gate falls on, the order inside `attn_qkv` --
and no cheap statistic distinguishes them, because every candidate is a
well-formed, norm-preserving, perfectly stable model that answers wrongly.
The only oracle is what the model says, so the thing to make cheap is asking it.

This reads the F16 weights directly (no int8 step, so a disagreement with the
Elisp path is not quantization) and runs all 64 blocks in a couple of minutes,
which makes a sweep affordable.

Usage:
  python3 tools/bonsai-ref-forward.py build/bonsai/f16.gguf \\
      --tokens "760 6511 314 9338 369" [--sweep]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf


def fwht(a, block):
    """Blockwise normalised Walsh-Hadamard over the last axis."""
    shape = a.shape
    n = shape[-1]
    a = a.reshape(-1, n // block, block).copy()
    h = 1
    while h < block:
        a = a.reshape(a.shape[0], a.shape[1], block // (2 * h), 2, h)
        x = a[:, :, :, 0, :].copy()
        y = a[:, :, :, 1, :].copy()
        a[:, :, :, 0, :] = x + y
        a[:, :, :, 1, :] = x - y
        a = a.reshape(a.shape[0], -1, block)
        h *= 2
    return (a / math.sqrt(block)).reshape(shape)


class Rot:
    def __init__(self, widths, values, block, invert):
        self.block, self.invert = block, invert
        self.runs, off = {}, 0
        for w in widths:
            self.runs[w] = np.asarray(values[off:off + w], dtype=np.float32)
            off += w

    def __call__(self, x):
        n = x.shape[-1]
        s = self.runs[n]
        if self.invert:
            return fwht(x, self.block) * s
        return fwht(x * s, self.block)


def rms(x, gain, eps):
    v = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(v + eps)) * gain


def silu(x):
    return x / (1.0 + np.exp(-x))


def l2n(x, eps=1e-6):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.info = gguf.parse(f.read(512 << 20))
        self.path = path
        self.kv = self.info["kv"]
        self.base = gguf.data_start(self.info)
        self.by = {t["name"]: t for t in self.info["tensors"]}
        self.fh = open(path, "rb")

    def t(self, name):
        t = self.by[name]
        self.fh.seek(self.base + t["offset"])
        raw = self.fh.read(gguf.tensor_bytes(t))
        tn = gguf.type_name(t["type"])
        a = (np.frombuffer(raw, dtype=np.float16).astype(np.float32) if tn == "F16"
             else np.frombuffer(raw, dtype=np.float32).copy())
        dims = list(t["dims"])
        return a if len(dims) == 1 else a.reshape(tuple(reversed(dims)))


def run(m, ids, invert, gate_first, qkv_order, verbose=False,
        layers=None, dump=None):
    kv = m.kv
    dim = kv["qwen35.embedding_length"]
    nblk = kv["qwen35.block_count"]
    iv = kv["qwen35.full_attention_interval"]
    eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
    heads = kv["qwen35.attention.head_count"]
    kvh = kv["qwen35.attention.head_count_kv"]
    hdim = kv["qwen35.attention.key_length"]
    rdims = kv["qwen35.rope.dimension_count"]
    rbase = float(kv["qwen35.rope.freq_base"])
    nk = kv["qwen35.ssm.group_count"]
    nv = kv["qwen35.ssm.time_step_rank"]
    sd = kv["qwen35.ssm.state_size"]
    kern = kv["qwen35.ssm.conv_kernel"]
    kd, vd = nk * sd, nv * sd
    grp = nv // nk
    rot = Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
              kv["prism.hadamard.block_size"], invert)

    emb = m.t("token_embd.weight")
    x = rot(emb[list(ids)].astype(np.float32))
    seq = len(ids)

    if layers:
        nblk = min(nblk, layers)
    for ly in range(nblk):
        p = "blk.%d." % ly
        ln1 = m.t(p + "attn_norm.weight")
        ln2 = m.t(p + "post_attention_norm.weight")
        if (ly % iv) == (iv - 1):
            a = rot(rms(x, ln1, eps))
            yq = a @ m.t(p + "attn_q.weight").T
            q, gate = yq[:, :heads * hdim], yq[:, heads * hdim:]
            k = a @ m.t(p + "attn_k.weight").T
            v = a @ m.t(p + "attn_v.weight").T
            qn, kn = m.t(p + "attn_q_norm.weight"), m.t(p + "attn_k_norm.weight")
            q = rms(q.reshape(seq, heads, hdim), qn, eps)
            k = rms(k.reshape(seq, kvh, hdim), kn, eps)
            pos = np.arange(seq, dtype=np.float32)[:, None, None]
            half = rdims // 2
            inv_f = 1.0 / (rbase ** (np.arange(half, dtype=np.float32) * 2.0 / rdims))
            th = pos * inv_f
            c, s = np.cos(th), np.sin(th)
            for t_ in (q, k):
                a1 = t_[:, :, :half].copy()
                a2 = t_[:, :, half:rdims].copy()
                t_[:, :, :half] = a1 * c - a2 * s
                t_[:, :, half:rdims] = a2 * c + a1 * s
            kk = np.repeat(k, heads // kvh, axis=1)
            vv = np.repeat(v.reshape(seq, kvh, hdim), heads // kvh, axis=1)
            sc = 1.0 / math.sqrt(hdim)
            att = np.einsum("qhd,khd->hqk", q, kk) * sc
            mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), 1)
            att = att + mask
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att = att / att.sum(axis=-1, keepdims=True)
            ctx = np.einsum("hqk,khd->qhd", att, vv).reshape(seq, heads * hdim)
            g = ctx * silu(gate)
            x = x + rot(g) @ m.t(p + "attn_output.weight").T
        else:
            plain = rms(x, ln1, eps)
            a = rot(plain)
            mixed = a @ m.t(p + "attn_qkv.weight").T
            z = a @ m.t(p + "attn_gate.weight").T
            al = plain @ m.t(p + "ssm_alpha.weight").T
            be = plain @ m.t(p + "ssm_beta.weight").T
            cw = m.t(p + "ssm_conv1d.weight")          # (cd, kern)
            cd = mixed.shape[1]
            pad = np.concatenate([np.zeros((kern - 1, cd), np.float32), mixed], 0)
            conv = np.zeros_like(mixed)
            for r in range(kern):
                conv += pad[r:r + seq] * cw[:, r]
            conv = silu(conv)
            parts = {"qkv": (conv[:, :kd], conv[:, kd:2 * kd], conv[:, 2 * kd:]),
                     "kqv": (conv[:, kd:2 * kd], conv[:, :kd], conv[:, 2 * kd:]),
                     "vqk": (conv[:, vd:vd + kd], conv[:, vd + kd:], conv[:, :vd])}
            qc, kc, vc = parts[qkv_order]
            # The 1/sqrt(dk) goes BEFORE the L2 norm, as transformers does it,
            # where its only effect is through the norm's epsilon.  After the
            # norm it divides a unit vector by 11.3, which collapses the
            # recurrence's output to 1e-5 and leaves the gated RMSNorm swamped
            # by its own eps -- stable, well-formed, and carrying nothing.
            qh = l2n(qc.reshape(seq, nk, sd) / math.sqrt(sd))
            kh = l2n(kc.reshape(seq, nk, sd) / math.sqrt(sd))
            vh = vc.reshape(seq, nv, sd)
            alog = m.t(p + "ssm_a")
            dtb = m.t(p + "ssm_dt.bias")
            gt = np.exp(-np.exp(alog) * np.log1p(np.exp(al + dtb)))
            bt = 1.0 / (1.0 + np.exp(-be))
            out = np.zeros((seq, nv, sd), dtype=np.float32)
            S = np.zeros((nv, sd, sd), dtype=np.float32)
            qg = np.repeat(np.arange(nk), grp)
            for t_ in range(seq):
                qt = qh[t_][qg]
                kt = kh[t_][qg]
                S = S * gt[t_][:, None, None]
                kv_ = np.einsum("hij,hi->hj", S, kt)
                d = (vh[t_] - kv_) * bt[t_][:, None]
                S = S + kt[:, :, None] * d[:, None, :]
                out[t_] = np.einsum("hij,hi->hj", S, qt)
            sn = m.t(p + "ssm_norm.weight")
            zz = z.reshape(seq, nv, sd)
            if gate_first:
                g = rms(out * silu(zz), sn, eps)
            else:
                g = rms(out, sn, eps) * silu(zz)
            x = x + rot(g.reshape(seq, vd)) @ m.t(p + "ssm_out.weight").T
        b = rot(rms(x, ln2, eps))
        gg = b @ m.t(p + "ffn_gate.weight").T
        uu = b @ m.t(p + "ffn_up.weight").T
        x = x + rot(silu(gg) * uu) @ m.t(p + "ffn_down.weight").T
        if verbose and (ly % 8 == 7 or ly == nblk - 1 or nblk <= 4):
            print("    block %2d  rms %.4f" % (ly, float(np.sqrt(np.mean(x * x)))),
                  flush=True)

    if dump:
        x.astype(np.float32).tofile(dump)
        print("    dumped %s (%s)" % (dump, x.shape), flush=True)
    h = rot(rms(x[-1], m.t("output_norm.weight"), eps))
    return h @ m.t("output.weight").T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tokens", default="760 6511 314 9338 369")
    ap.add_argument("--vocab", default="build/bonsai/vocab.tsv")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--gate-after", action="store_true")
    ap.add_argument("--order", default="qkv")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dump")
    args = ap.parse_args()

    vocab = None
    if os.path.exists(args.vocab):
        vocab = {}
        with open(args.vocab, encoding="utf-8") as f:
            for line in f:
                i, _, t = line.rstrip("\n").partition("\t")
                vocab[int(i)] = t.replace("Ġ", " ")

    m = Model(args.gguf)
    ids = [int(x) for x in args.tokens.split()]
    print("prompt: %s" % "|".join(vocab.get(i, str(i)) for i in ids)
          if vocab else "prompt: %s" % ids, flush=True)

    combos = ([(inv, gf, od) for inv in (False, True)
               for gf in (True, False) for od in ("qkv",)]
              if args.sweep else
              [(args.invert, not args.gate_after, args.order)])
    for inv, gf, od in combos:
        label = "rot=%-7s gate=%-6s order=%s" % (
            "inverse" if inv else "forward", "before" if gf else "after", od)
        lg = run(m, ids, inv, gf, od, verbose=not args.sweep,
                 layers=args.layers, dump=args.dump)
        top = np.argsort(-lg)[:8]
        print("  %s  ->  %s" % (label, "  ".join(
            "%s(%.2f)" % (repr(vocab.get(int(i), str(int(i)))) if vocab else int(i),
                          lg[i]) for i in top)), flush=True)


if __name__ == "__main__":
    main()
