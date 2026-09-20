#!/usr/bin/env python3
"""Does any attention head do induction, at any depth?

The scoring passage is "The capital of France is Paris.  The capital of Japan
is Tokyo."  What makes its second half predictable is induction: from the
second " is" (position 11), attend to what followed the first " is" -- token
" Paris" at position 5 -- and copy the pattern.  Every language model that
works has heads doing this, usually several, usually sharply.

This looks for them directly.  It needs the blocks up to the head being
examined and nothing after, so it does not depend on the output head, the
final norm, or sixty blocks of accumulation, and it cannot be explained away
by a wrong readout.  If no head anywhere attends from 11 to 5, the queries and
keys are not carrying position or content, and the defect is upstream of
attention rather than inside it.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
R = SourceFileLoader("R", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "bonsai-ref-forward.py")).load_module()

gguf_path = sys.argv[1]
# The| capital| of| France| is| Paris|.| The| capital| of| Japan| is| Tokyo
IDS = [760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369, 25358]
QPOS, TARGET = 11, 5          # from the second " is" to what followed the first

m = R.Model(gguf_path)
kv = m.kv
eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
nblk = kv["qwen35.block_count"]
iv = kv["qwen35.full_attention_interval"]
heads, kvh = kv["qwen35.attention.head_count"], kv["qwen35.attention.head_count_kv"]
hdim = kv["qwen35.attention.key_length"]
rdims, rbase = kv["qwen35.rope.dimension_count"], float(kv["qwen35.rope.freq_base"])
rot = R.Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
            kv["prism.hadamard.block_size"], False)
seq = len(IDS)
uniform = math.log(QPOS + 1)


def probe(x, ly):
    p = "blk.%d." % ly
    a = rot(R.rms(x, m.t(p + "attn_norm.weight"), eps))
    yq = a @ m.t(p + "attn_q.weight").T
    q = yq[:, :heads * hdim]
    k = a @ m.t(p + "attn_k.weight").T
    q = R.rms(q.reshape(seq, heads, hdim), m.t(p + "attn_q_norm.weight"), eps)
    k = R.rms(k.reshape(seq, kvh, hdim), m.t(p + "attn_k_norm.weight"), eps)
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
    att = np.einsum("qhd,khd->hqk", q, kk) / math.sqrt(hdim)
    att = att + np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), 1)
    att = np.exp(att - att.max(axis=-1, keepdims=True))
    att = att / att.sum(axis=-1, keepdims=True)
    row = att[:, QPOS, :QPOS + 1]                  # (heads, QPOS+1)
    ent = -(row * np.log(row + 1e-30)).sum(axis=-1)
    return row, ent


x = rot(m.t("token_embd.weight")[IDS].astype(np.float32))
print("prompt positions 0..12; looking for attention from %d to %d\n" % (QPOS, TARGET))
print("  layer   entropy(mean/min)   p(->%d) max   best head   its peak" % TARGET)
best_overall = (0.0, -1, -1)
for ly in range(nblk):
    p = "blk.%d." % ly
    ln1 = m.t(p + "attn_norm.weight")
    ln2 = m.t(p + "post_attention_norm.weight")
    if (ly % iv) == (iv - 1):
        row, ent = probe(x, ly)
        h = int(row[:, TARGET].argmax())
        pmax = float(row[:, TARGET].max())
        if pmax > best_overall[0]:
            best_overall = (pmax, ly, h)
        print("  %5d   %.3f / %.3f        %.3f      head %2d     %2d (%.3f)"
              % (ly, ent.mean(), ent.min(), pmax, h,
                 int(row[h].argmax()), float(row[h].max())), flush=True)
    # advance the residual through the real block
    lg = None
    x = R.run(m, IDS, False, True, "qkv", layers=ly + 1,
              dump=None) if False else x
    # cheaper: run the block inline
    if (ly % iv) == (iv - 1):
        a = rot(R.rms(x, ln1, eps))
        yq = a @ m.t(p + "attn_q.weight").T
        q, gate = yq[:, :heads * hdim], yq[:, heads * hdim:]
        k = a @ m.t(p + "attn_k.weight").T
        v = a @ m.t(p + "attn_v.weight").T
        q = R.rms(q.reshape(seq, heads, hdim), m.t(p + "attn_q_norm.weight"), eps)
        k = R.rms(k.reshape(seq, kvh, hdim), m.t(p + "attn_k_norm.weight"), eps)
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
        att = np.einsum("qhd,khd->hqk", q, kk) / math.sqrt(hdim)
        att = att + np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), 1)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        ctx = np.einsum("hqk,khd->qhd", att, vv).reshape(seq, heads * hdim)
        x = x + rot(ctx * R.silu(gate)) @ m.t(p + "attn_output.weight").T
    else:
        x = R.run.__wrapped__(x) if False else R.deltanet_half(m, x, ly, rot, eps, kv) \
            if hasattr(R, "deltanet_half") else x
    b = rot(R.rms(x, ln2, eps))
    gg = b @ m.t(p + "ffn_gate.weight").T
    uu = b @ m.t(p + "ffn_up.weight").T
    x = x + rot(R.silu(gg) * uu) @ m.t(p + "ffn_down.weight").T

print("\n  strongest induction anywhere: p = %.3f at layer %d head %d"
      % best_overall)
print("  uniform over the %d visible positions is %.3f nats; chance p is %.3f"
      % (QPOS + 1, uniform, 1.0 / (QPOS + 1)))
