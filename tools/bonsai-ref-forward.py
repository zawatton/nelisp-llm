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


_SEQ = {}


def sequency_perm(n):
    """Natural (Sylvester) order -> Walsh (sequency) order, for size N.

    The transform is named `normalized-sylvester-walsh-hadamard'.  The
    Sylvester construction and the Walsh ordering are the same matrix with its
    rows permuted -- which is still orthogonal, still norm-preserving, and a
    completely different transform.
    """
    if n in _SEQ:
        return _SEQ[n]
    m = n.bit_length() - 1
    p = np.empty(n, dtype=np.int64)
    for i in range(n):
        g = i ^ (i >> 1)
        r = 0
        for b in range(m):
            r = (r << 1) | ((g >> b) & 1)
        p[r] = i
    _SEQ[n] = p
    return p


def fwht(a, block, sequency=False):
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
    a = a / math.sqrt(block)
    if sequency:
        a = a[:, :, sequency_perm(block)]
    return a.reshape(shape)


class Rot:
    """The folded transform, with a per-width block size.

    `prism.hadamard.block_size` is 1024, but `prism.hadamard.gdn_v_grouped`
    is set, and the only widths the flag could be about are the ones the two
    mixing halves rotate before their output projection -- 6144, which is
    48 GDN heads of 128 and also 24 attention heads of 256.  A block-diagonal
    rotation is orthogonal whatever the block size, so getting it wrong costs
    nothing visible and everything real.
    """

    def __init__(self, widths, values, block, invert, blocks=None,
                 sequency=False):
        self.block, self.invert, self.sequency = block, invert, sequency
        self.blocks = blocks or {}
        self.runs, off = {}, 0
        for w in widths:
            self.runs[w] = np.asarray(values[off:off + w], dtype=np.float32)
            off += w

    def __call__(self, x):
        n = x.shape[-1]
        s = self.runs[n]
        b = self.blocks.get(n, self.block)
        if self.invert:
            return fwht(x, b, self.sequency) * s
        return fwht(x * s, b, self.sequency)


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
        layers=None, dump=None, mode="act", lens=None, skip=(), score=False, curve=0, blocks=None, no_head_rot=False,
        gate_first_half=False, kv_tile=False, swap_ab=False,
        sequency=False):
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
    _rot = Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
               kv["prism.hadamard.block_size"], invert, blocks=blocks,
               sequency=sequency)
    # Where the transform belongs at run time is not written down anywhere.
    #   act        the residual stream is unrotated and every rotated
    #              projection gets a rotated activation
    #   none       the whole network already lives in the rotated basis, so
    #              nothing is transformed at run time -- which is consistent
    #              because the RMSNorm gains would have been trained there too
    #   embed      the stream is rotated, entered once at the embedding
    ident = lambda v: v
    #   act        stream unrotated; every rotated projection and the stored
    #              embedding get the transform
    #   actplain   the same, but the stored embedding is already the true one
    #   none       the whole network lives in the rotated basis
    #   embed      the stream is rotated, entered once at the embedding
    rot = _rot if mode in ("act", "actplain") else ident
    erot = _rot if mode in ("act", "embed") else ident
    hrot = ident if no_head_rot else rot

    emb = m.t("token_embd.weight")
    x = erot(emb[list(ids)].astype(np.float32))
    lens_w = lens_n = None
    if curve:
        # The whole curve in one pass over the weights, because a pass is four
        # minutes of reading 53 GB and the question -- does depth help at all
        # -- needs every point on it, not one.
        cw_, cn_ = m.t("output.weight"), m.t("output_norm.weight")

        def ce(tag):
            hh = hrot(rms(x, cn_, eps))
            lg = hh @ cw_.T
            lg = lg - lg.max(axis=-1, keepdims=True)
            lse = np.log(np.exp(lg).sum(axis=-1))
            tgt = np.asarray(ids[1:], dtype=np.int64)
            v = float(np.mean(lse[:-1] - lg[np.arange(len(tgt)), tgt]))
            print("    %-10s rms %8.4f   cross entropy %7.4f nats" % (
                tag, float(np.sqrt(np.mean(x * x))), v), flush=True)
    if lens is not None:
        # The decisive question is not which convention is right but where the
        # answer stops being one.  Reading the residual through the head after
        # every block says whether the prompt survives block 0 or dissolves
        # over sixty-four of them, which no end-to-end sweep can tell apart.
        lens_w = m.t("output.weight")
        lens_n = m.t("output_norm.weight")

        def look(tag):
            h = hrot(rms(x[-1], lens_n, eps))
            lg = h @ lens_w.T
            top = np.argsort(-lg)[:4]
            print("    %-10s rms %8.4f  %s" % (
                tag, float(np.sqrt(np.mean(x * x))),
                "  ".join("%s(%.2f)" % (lens.get(int(i), int(i)), lg[i])
                          for i in top)), flush=True)
        look("embedding")
    if curve:
        ce("embedding")
    seq = len(ids)

    if layers is not None:
        nblk = min(nblk, layers)
    for ly in range(nblk):
        p = "blk.%d." % ly
        ln1 = m.t(p + "attn_norm.weight")
        ln2 = m.t(p + "post_attention_norm.weight")
        if (ly % iv) == (iv - 1) and "attn" in skip:
            pass
        elif (ly % iv) != (iv - 1) and "deltanet" in skip:
            pass
        elif (ly % iv) == (iv - 1):
            a = rot(rms(x, ln1, eps))
            yq = a @ m.t(p + "attn_q.weight").T
            # `attn_q` is twice as wide as the query: it carries the output
            # gate alongside.  Which half is which is not written down.
            if gate_first_half:
                gate, q = yq[:, :heads * hdim], yq[:, heads * hdim:]
            else:
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
            # `ssm_alpha` is read as the decay input (it meets dt_bias and
            # softplus) and `ssm_beta` as the write strength (it meets a
            # sigmoid).  The names suggest it; the file does not say it.
            al = plain @ m.t(p + "ssm_alpha.weight").T
            be = plain @ m.t(p + "ssm_beta.weight").T
            if swap_ab:
                al, be = be, al
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
            # 48 value heads served by 16 key heads: head h reads key group
            # h // 3 (repeat_interleave) or h % 16 (tile).  Both are shaped
            # correctly and only one is the model's.
            qg = (np.tile(np.arange(nk), grp) if kv_tile
                  else np.repeat(np.arange(nk), grp))
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
        if "ffn" not in skip:
            b = rot(rms(x, ln2, eps))
            gg = b @ m.t(p + "ffn_gate.weight").T
            uu = b @ m.t(p + "ffn_up.weight").T
            x = x + rot(silu(gg) * uu) @ m.t(p + "ffn_down.weight").T
        if lens is not None:
            look("block %d" % ly)
        if curve and ((ly + 1) % curve == 0 or ly == nblk - 1):
            ce("block %d" % ly)
        if verbose and (ly % 8 == 7 or ly == nblk - 1 or nblk <= 4):
            print("    block %2d  rms %.4f" % (ly, float(np.sqrt(np.mean(x * x)))),
                  flush=True)

    if dump:
        x.astype(np.float32).tofile(dump)
        print("    dumped %s (%s)" % (dump, x.shape), flush=True)
    hw = m.t("output.weight")
    hn = m.t("output_norm.weight")
    if score:
        # A graded oracle.  Whether a top token "looks right" is a judgement
        # call that eight wrong configurations all failed in different ways;
        # the cross entropy of real text is a number.  Chance is ln(vocab) =
        # 12.42 nats, a working model of this size is nearer 2, and a
        # configuration that is partly right lands in between instead of
        # looking exactly as wrong as one that is not.
        hh = hrot(rms(x, hn, eps))
        lg = hh @ hw.T
        lg = lg - lg.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(lg).sum(axis=-1))
        tgt = np.asarray(ids[1:], dtype=np.int64)
        return float(np.mean(lse[:-1] - lg[np.arange(len(tgt)), tgt]))
    h = hrot(rms(x[-1], hn, eps))
    return h @ hw.T


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
    ap.add_argument("--mode", default="act", choices=("act", "actplain", "none", "embed"))
    ap.add_argument("--lens", action="store_true")
    ap.add_argument("--skip", default="", help="deltanet and/or attn")
    ap.add_argument("--no-head-rot", action="store_true")
    ap.add_argument("--gate-first-half", action="store_true",
                    help="attn_q is [gate | query] rather than [query | gate]")
    ap.add_argument("--sequency", action="store_true",
                    help="Walsh (sequency) row order instead of Sylvester")
    ap.add_argument("--swap-ab", action="store_true",
                    help="ssm_beta is the decay and ssm_alpha the write strength")
    ap.add_argument("--kv-tile", action="store_true",
                    help="value head h reads key group h %% n_k, not h // grp")
    ap.add_argument("--block6144", type=int, default=0,
                    help="block size for the 6144-wide rotation (0 = default)")
    ap.add_argument("--curve", type=int, default=0,
                    help="report cross entropy every N blocks in one pass")
    ap.add_argument("--score", action="store_true",
                    help="report cross entropy of the prompt instead of a token")
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

    combos = ([(inv, gf, od, md)
               for md in ("none", "embed", "act")
               for inv in ((False,) if md == "none" else (False, True))
               for gf in (True, False) for od in ("qkv",)]
              if args.sweep else
              [(args.invert, not args.gate_after, args.order, args.mode)])
    for inv, gf, od, md in combos:
        label = "mode=%-5s rot=%-7s gate=%-6s order=%s" % (
            md, "-" if md == "none" else ("inverse" if inv else "forward"),
            "before" if gf else "after", od)
        lg = run(m, ids, inv, gf, od, verbose=not args.sweep,
                 layers=args.layers, dump=args.dump, mode=md,
                 lens=(vocab if args.lens else None),
                 skip=tuple(x for x in args.skip.split(",") if x),
                 score=args.score, curve=args.curve,
                 blocks=({6144: args.block6144} if args.block6144 else None),
                 no_head_rot=args.no_head_rot,
                 gate_first_half=args.gate_first_half, kv_tile=args.kv_tile,
                 swap_ab=args.swap_ab, sequency=args.sequency)
        if args.score:
            print("  %s  ->  cross entropy %.4f nats  (chance %.2f)"
                  % (label, lg, math.log(len(vocab) if vocab else 248320)),
                  flush=True)
            continue
        top = np.argsort(-lg)[:8]
        print("  %s  ->  %s" % (label, "  ".join(
            "%s(%.2f)" % (repr(vocab.get(int(i), str(int(i)))) if vocab else int(i),
                          lg[i]) for i in top)), flush=True)


if __name__ == "__main__":
    main()
