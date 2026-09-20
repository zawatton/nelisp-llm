#!/usr/bin/env python3
"""Block 0's intermediates from the F16 GGUF, for comparing against the Elisp.

The two implementations disagree at the first block.  Dumping the same named
stages from each localises that to one step instead of one block.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
R = SourceFileLoader("R", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "bonsai-ref-forward.py")).load_module()

m = R.Model(sys.argv[1])
out_dir = sys.argv[2]
ids = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3
                        else "760 6511 314 9338 369").split()]
kv = m.kv
eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
nk, nv = kv["qwen35.ssm.group_count"], kv["qwen35.ssm.time_step_rank"]
sd, kern = kv["qwen35.ssm.state_size"], kv["qwen35.ssm.conv_kernel"]
kd, vd, grp = nk * sd, nv * sd, nv // nk
rot = R.Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
            kv["prism.hadamard.block_size"], False)
seq = len(ids)


def put(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    a.tofile(os.path.join(out_dir, name + ".f32"))
    print("  %-8s %-14s rms %.6f" % (name, str(a.shape),
                                     float(np.sqrt(np.mean(a * a)))))


x = rot(m.t("token_embd.weight")[ids].astype(np.float32))
put("emb", x)
p = "blk.0."
ln1 = m.t(p + "attn_norm.weight")
plain = R.rms(x, ln1, eps)
put("plain", plain)
a = rot(plain)
put("arot", a)
mixed = a @ m.t(p + "attn_qkv.weight").T
put("mixed", mixed)
z = a @ m.t(p + "attn_gate.weight").T
put("z", z)
al = plain @ m.t(p + "ssm_alpha.weight").T
be = plain @ m.t(p + "ssm_beta.weight").T
put("alpha", al)
put("beta", be)
cw = m.t(p + "ssm_conv1d.weight")
cd = mixed.shape[1]
pad = np.concatenate([np.zeros((kern - 1, cd), np.float32), mixed], 0)
conv = np.zeros_like(mixed)
for r in range(kern):
    conv += pad[r:r + seq] * cw[:, r]
conv = R.silu(conv)
put("conv", conv)
qh = R.l2n(conv[:, :kd].reshape(seq, nk, sd) / math.sqrt(sd))
kh = R.l2n(conv[:, kd:2 * kd].reshape(seq, nk, sd) / math.sqrt(sd))
vh = conv[:, 2 * kd:].reshape(seq, nv, sd)
alog, dtb = m.t(p + "ssm_a"), m.t(p + "ssm_dt.bias")
gt = np.exp(-np.exp(alog) * np.log1p(np.exp(al + dtb)))
bt = 1.0 / (1.0 + np.exp(-be))
put("gt", gt)
put("bt", bt)
o = np.zeros((seq, nv, sd), np.float32)
S = np.zeros((nv, sd, sd), np.float32)
qg = np.repeat(np.arange(nk), grp)
for t_ in range(seq):
    qt = qh[t_][qg]
    kt = kh[t_][qg]
    S = S * gt[t_][:, None, None]
    d = (vh[t_] - np.einsum("hij,hi->hj", S, kt)) * bt[t_][:, None]
    S = S + kt[:, :, None] * d[:, None, :]
    o[t_] = np.einsum("hij,hi->hj", S, qt)
put("ctx", o.reshape(seq, vd))
sn = m.t(p + "ssm_norm.weight")
g = R.rms(o * R.silu(z.reshape(seq, nv, sd)), sn, eps)
put("gnorm", g.reshape(seq, vd))
xmid = x + rot(g.reshape(seq, vd)) @ m.t(p + "ssm_out.weight").T
put("xmid", xmid)
ln2 = m.t(p + "post_attention_norm.weight")
b = rot(R.rms(xmid, ln2, eps))
gg = b @ m.t(p + "ffn_gate.weight").T
uu = b @ m.t(p + "ffn_up.weight").T
hh = R.silu(gg) * uu
put("hh", hh)
xout = xmid + rot(hh) @ m.t(p + "ffn_down.weight").T
put("xout", xout)
