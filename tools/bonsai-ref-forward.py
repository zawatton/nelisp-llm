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

vocab_g = None


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
                 sequency=False, order=None):
        self.block, self.invert, self.sequency = block, invert, sequency
        self.blocks = blocks or {}
        # `sign_widths` lists the widths; whether the runs are concatenated in
        # that order is a separate claim.  The three lengths are distinct, so a
        # different file order means a different slice for every width -- and
        # every slice is random +/-1 either way, so nothing about the numbers
        # says which is which.
        order = order or list(range(len(widths)))
        self.runs, off = {}, 0
        for idx in order:
            w = widths[idx]
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


def rmsn(x, eps):
    """RMSNorm without the gain, for composing the gain on the other side."""
    v = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(v + eps))


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
        layers=None, dump=None, mode="act", lens=None, skip=(), score=False,
        curve=0, blocks=None, no_head_rot=False,
        gate_first_half=False, kv_tile=False, swap_ab=False,
        sequency=False, sign_order=None, decay_after_read=False,
        qscale="pre", gnorm_eps=None, gain_after_rot=False,
        no_hh_rot=False, no_out_rot=False, conv_reverse=False,
        head_minor=False, conv_act="all", no_l2norm=False,
        rot_ba=False, rope_interleaved=False, attn_kv_tile=False,
        swap_gate_up=False, per_position=False, embed_invert=False,
        no_attn_gate=False, attn_gate_silu=False,
        gate_per_head=False, embed_scale=0.0, ssm_a_is_A=True):
    kv = m.kv
    dim = kv["qwen35.embedding_length"]
    nblk = kv["qwen35.block_count"]
    iv = kv["qwen35.full_attention_interval"]
    eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
    gneps = gnorm_eps if gnorm_eps is not None else eps
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
               sequency=sequency, order=sign_order)
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
    _erot = _rot
    if embed_invert:
        # `inverse_weight_names` holds token_embd alone.  Reading that as "the
        # same transform the activations get" and reading it as "the other one"
        # are both defensible from the name, and the embedding is where content
        # enters the model -- so getting it wrong leaves position and syntax
        # intact and scrambles every token identity.
        _erot = Rot(kv["prism.hadamard.sign_widths"],
                    kv["prism.hadamard.sign_values"],
                    kv["prism.hadamard.block_size"], not invert,
                    blocks=blocks, sequency=sequency, order=sign_order)
    erot = _erot if mode in ("act", "embed") else ident
    hrot = ident if no_head_rot else rot

    emb = m.t("token_embd.weight")
    x = erot(emb[list(ids)].astype(np.float32))
    if embed_scale:
        # The stored embedding has rms 0.0129 while the first block's output
        # has rms 0.89 -- a factor of 69, against sqrt(5120) = 71.6.  Models
        # in the Gemma line multiply the embedding by sqrt(d_model) at the
        # input for exactly this reason.  Without it the token's identity is
        # 1.5% of the residual from the first block on, which is a model whose
        # syntax works and whose content does not.
        x = x * (embed_scale if embed_scale > 0 else math.sqrt(dim))
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
            # gate alongside.  Two things about the split are not written
            # down -- whether it is flat ([all q | all gate]) or per head
            # ([q gate] within each head), and which side is the gate.
            if gate_per_head:
                yqh = yq.reshape(seq, heads, 2 * hdim)
                lo, hi = yqh[:, :, :hdim], yqh[:, :, hdim:]
                q = (hi if gate_first_half else lo).reshape(seq, heads * hdim)
                gate = (lo if gate_first_half else hi).reshape(seq, heads * hdim)
            elif gate_first_half:
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
                if rope_interleaved:
                    # GPT-J order: the pair is (2i, 2i+1), adjacent.
                    a1 = t_[:, :, 0:rdims:2].copy()
                    a2 = t_[:, :, 1:rdims:2].copy()
                    t_[:, :, 0:rdims:2] = a1 * c - a2 * s
                    t_[:, :, 1:rdims:2] = a2 * c + a1 * s
                else:
                    # GPT-NeoX order: the pair is (i, i + rdims/2).
                    a1 = t_[:, :, :half].copy()
                    a2 = t_[:, :, half:rdims].copy()
                    t_[:, :, :half] = a1 * c - a2 * s
                    t_[:, :, half:rdims] = a2 * c + a1 * s
            rep = heads // kvh
            vr = v.reshape(seq, kvh, hdim)
            if attn_kv_tile:
                kk = np.tile(k, (1, rep, 1))
                vv = np.tile(vr, (1, rep, 1))
            else:
                kk = np.repeat(k, rep, axis=1)
                vv = np.repeat(vr, rep, axis=1)
            sc = 1.0 / math.sqrt(hdim)
            att = np.einsum("qhd,khd->hqk", q, kk) * sc
            mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), 1)
            att = att + mask
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att = att / att.sum(axis=-1, keepdims=True)
            ctx = np.einsum("hqk,khd->qhd", att, vv).reshape(seq, heads * hdim)
            # Qwen3-Next's attention gates its output with a sigmoid; the
            # GDN's RMSNormGated is the one that uses silu.  silu is negative
            # below zero and unbounded above it, so using it here does not
            # gate the context, it scrambles it.
            if no_attn_gate:
                g = ctx
            elif attn_gate_silu:
                g = ctx * silu(gate)
            else:
                g = ctx * (1.0 / (1.0 + np.exp(-gate)))
            # The 6144-wide rotation before the output projection is the one
            # thing both mixing halves do and the feed-forward never does --
            # and the feed-forward is the only half that improves with depth.
            x = x + (g if no_out_rot else rot(g)) @ m.t(p + "attn_output.weight").T
        else:
            plain = rms(x, ln1, eps)
            a = rot(plain)
            mixed = a @ m.t(p + "attn_qkv.weight").T
            z = a @ m.t(p + "attn_gate.weight").T
            # `ssm_alpha` is read as the decay input (it meets dt_bias and
            # softplus) and `ssm_beta` as the write strength (it meets a
            # sigmoid).  The names suggest it; the file does not say it.
            # `ssm_alpha` and `ssm_beta` are the one pair of projections the
            # rotation list leaves out -- and they are also the one pair that
            # was never quantized.  If the list names what was incoherence-
            # processed FOR QUANTIZATION, the transform may still have been
            # folded into these, and they would want the rotated activation
            # like everything else.  In the original model they are one matrix
            # (`in_proj_ba`) reading the same input as `in_proj_qkvz`.
            ba_in = a if rot_ba else plain
            al = ba_in @ m.t(p + "ssm_alpha.weight").T
            be = ba_in @ m.t(p + "ssm_beta.weight").T
            if swap_ab:
                al, be = be, al
            cw = m.t(p + "ssm_conv1d.weight")          # (cd, kern)
            cd = mixed.shape[1]
            pad = np.concatenate([np.zeros((kern - 1, cd), np.float32), mixed], 0)
            conv = np.zeros_like(mixed)
            for r in range(kern):
                # tap r covers position t + r - kern + 1, so r = kern-1 is the
                # current position.  A reversed file order puts it at r = 0.
                w_ = cw[:, kern - 1 - r] if conv_reverse else cw[:, r]
                conv += pad[r:r + seq] * w_
            # The activation after the causal conv: on everything, on the
            # value channels only, or not at all.
            if conv_act == "all":
                conv = silu(conv)
            elif conv_act == "v":
                conv = np.concatenate([conv[:, :2 * kd], silu(conv[:, 2 * kd:])], 1)
            parts = {"qkv": (conv[:, :kd], conv[:, kd:2 * kd], conv[:, 2 * kd:]),
                     "kqv": (conv[:, kd:2 * kd], conv[:, :kd], conv[:, 2 * kd:]),
                     "vqk": (conv[:, vd:vd + kd], conv[:, vd + kd:], conv[:, :vd])}
            qc, kc, vc = parts[qkv_order]
            # Where the 1/sqrt(dk) goes is not a rounding question here.  The
            # recurrence's output lands near 1e-3, which is the regime where
            # the gated RMSNorm's own 1e-6 dominates its denominator -- so the
            # norm stops normalising and the block's contribution becomes
            # proportional to this scale.  A factor applied in the wrong place
            # is then a systematic per-block error, which is exactly the shape
            # of the depth curve.
            sc = 1.0 / math.sqrt(sd)
            l2 = (lambda z: z) if no_l2norm else l2n

            def split0(a, nh):
                return (a.reshape(seq, sd, nh).transpose(0, 2, 1) if head_minor
                        else a.reshape(seq, nh, sd))
            if qscale == "pre":
                qh = l2(split0(qc, nk) * sc)
                kh = l2(split0(kc, nk) * sc)
            elif qscale == "post-q":
                qh = l2(split0(qc, nk)) * sc
                kh = l2(split0(kc, nk))
            elif qscale == "post-both":
                qh = l2(split0(qc, nk)) * sc
                kh = l2(split0(kc, nk)) * sc
            else:
                qh = l2(split0(qc, nk))
                kh = l2(split0(kc, nk))
            # Whether a projection's output runs [head][dim] or [dim][head]
            # is a reshape either way and a different model.
            def split(a, nh):
                return (a.reshape(seq, sd, nh).transpose(0, 2, 1) if head_minor
                        else a.reshape(seq, nh, sd))
            vh = split(vc, nv)
            alog = m.t(p + "ssm_a")
            dtb = m.t(p + "ssm_dt.bias")
            # The tensor is named `ssm_a`, not `ssm_a_log`.  llama.cpp's
            # Mamba conversion stores -exp(A_log) under that name, so the
            # exponential has already been taken and taking it again gives
            # exp(-60) = 1e-26, a decay of exactly 1.0 -- a gated delta rule
            # that never forgets, in every block, at every position.
            dt = np.log1p(np.exp(al + dtb))
            gt = np.exp(alog * dt) if ssm_a_is_A else np.exp(-np.exp(alog) * dt)
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
                # S_t = a_t S_{t-1} (I - b_t k k') + b_t v k' expands to a read
                # of the DECAYED state: a_t S_{t-1}' k.  Some implementations
                # read the undecayed one instead, which is a different
                # recurrence and differs at every position.
                if decay_after_read:
                    kv_ = np.einsum("hij,hi->hj", S, kt)
                    d = (vh[t_] - kv_) * bt[t_][:, None]
                    S = S * gt[t_][:, None, None] + kt[:, :, None] * d[:, None, :]
                else:
                    S = S * gt[t_][:, None, None]
                    kv_ = np.einsum("hij,hi->hj", S, kt)
                    d = (vh[t_] - kv_) * bt[t_][:, None]
                    S = S + kt[:, :, None] * d[:, None, :]
                out[t_] = np.einsum("hij,hi->hj", S, qt)
            sn = m.t(p + "ssm_norm.weight")
            zz = split(z, nv)
            if gate_first:
                g = rms(out * silu(zz), sn, gneps)
            else:
                g = rms(out, sn, gneps) * silu(zz)
            gv = g.reshape(seq, vd)
            x = x + (gv if no_out_rot else rot(gv)) @ m.t(p + "ssm_out.weight").T
        if "ffn" not in skip:
            # A rotation and a per-element gain do not commute, so whether the
            # gain belongs inside the rotation or outside it is a real choice
            # and it is made in every norm of every block.
            b = (rot(rmsn(x, eps)) * ln2 if gain_after_rot
                 else rot(rms(x, ln2, eps)))
            gg = b @ m.t(p + "ffn_gate.weight").T
            uu = b @ m.t(p + "ffn_up.weight").T
            # Which of the two feed-forward projections passes through the
            # activation.  Wrong, it is still a SwiGLU of the right shape.
            hh_ = silu(uu) * gg if swap_gate_up else silu(gg) * uu
            x = x + (hh_ if no_hh_rot else rot(hh_)) @ m.t(p + "ffn_down.weight").T
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
        per = lse[:-1] - lg[np.arange(len(tgt)), tgt]
        if per_position:
            # The mean hides everything.  A model that predicts the repeated
            # half of a repetitive passage and nothing else has the same mean
            # as one that predicts nothing, and only one of those is working.
            order = np.argsort(-lg, axis=-1)
            for i, t in enumerate(tgt):
                rank = int(np.where(order[i] == t)[0][0]) + 1
                print("    pos %2d -> %-14s ce %7.3f  rank %7d  top %s"
                      % (i, repr(vocab_g.get(int(t), int(t))) if vocab_g else t,
                         per[i], rank,
                         repr(vocab_g.get(int(order[i][0]), int(order[i][0])))
                         if vocab_g else int(order[i][0])), flush=True)
        return float(np.mean(per))
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
    ap.add_argument("--gain-after-rot", action="store_true",
                    help="rotate the normalised value, then apply the gain")
    ap.add_argument("--conv-reverse", action="store_true")
    ap.add_argument("--conv-act", default="all", choices=("all", "v", "none"))
    ap.add_argument("--swap-gate-up", action="store_true",
                    help="ffn_up passes through the activation, not ffn_gate")
    ap.add_argument("--rope-interleaved", action="store_true",
                    help="rotary pairs (2i, 2i+1) instead of (i, i + rdims/2)")
    ap.add_argument("--attn-kv-tile", action="store_true",
                    help="query head h reads kv head h mod kvh, not h div rep")
    ap.add_argument("--rot-ba", action="store_true",
                    help="feed ssm_alpha / ssm_beta the rotated activation")
    ap.add_argument("--no-l2norm", action="store_true",
                    help="feed the recurrence unnormalised q and k")
    ap.add_argument("--head-minor", action="store_true",
                    help="q/k/v/z run [dim][head] rather than [head][dim]")
    ap.add_argument("--no-out-rot", action="store_true",
                    help="feed ssm_out / attn_output the unrotated 6144 value")
    ap.add_argument("--no-hh-rot", action="store_true",
                    help="feed ffn_down the unrotated SwiGLU output")
    ap.add_argument("--decay-after-read", action="store_true",
                    help="read the undecayed state, decay when writing")
    ap.add_argument("--sign-order", default="",
                    help="file order of the sign runs, e.g. 2,1,0")
    ap.add_argument("--qscale", default="pre",
                    choices=("pre", "post-q", "post-both", "none"))
    ap.add_argument("--gnorm-eps", type=float, default=None)
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
    ap.add_argument("--embed-invert", action="store_true",
                    help="the stored embedding takes the other transform")
    ap.add_argument("--ssm-a-log", dest="ssm_a_is_A", action="store_false",
                    help="read ssm_a as A_log and exponentiate it")
    ap.add_argument("--embed-scale", type=float, nargs="?", const=-1.0,
                    default=0.0,
                    help="scale the embedding; bare flag means sqrt(d_model)")
    ap.add_argument("--attn-gate-silu", action="store_true",
                    help="gate the attention output with silu, not sigmoid")
    ap.add_argument("--gate-per-head", action="store_true",
                    help="attn_q splits [q gate] within each head")
    ap.add_argument("--no-attn-gate", action="store_true",
                    help="attn_output reads the context ungated")
    ap.add_argument("--per-position", action="store_true",
                    help="print the cross entropy and rank of every target")
    args = ap.parse_args()

    vocab = None
    if os.path.exists(args.vocab):
        vocab = {}
        with open(args.vocab, encoding="utf-8") as f:
            for line in f:
                i, _, t = line.rstrip("\n").partition("\t")
                vocab[int(i)] = t.replace("Ġ", " ")

    global vocab_g
    vocab_g = vocab
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
                 swap_ab=args.swap_ab, sequency=args.sequency,
                 qscale=args.qscale, gnorm_eps=args.gnorm_eps,
                 sign_order=([int(x) for x in args.sign_order.split(",")]
                             if args.sign_order else None),
                 decay_after_read=args.decay_after_read,
                 gain_after_rot=args.gain_after_rot, no_hh_rot=args.no_hh_rot,
                 no_out_rot=args.no_out_rot, conv_reverse=args.conv_reverse,
                 head_minor=args.head_minor, conv_act=args.conv_act,
                 no_l2norm=args.no_l2norm, rot_ba=args.rot_ba,
                 rope_interleaved=args.rope_interleaved,
                 attn_kv_tile=args.attn_kv_tile, swap_gate_up=args.swap_gate_up,
                 per_position=args.per_position, embed_invert=args.embed_invert,
                 no_attn_gate=args.no_attn_gate,
                 attn_gate_silu=args.attn_gate_silu,
                 gate_per_head=args.gate_per_head, embed_scale=args.embed_scale,
                 ssm_a_is_A=args.ssm_a_is_A)
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
