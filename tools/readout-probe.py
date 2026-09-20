#!/usr/bin/env python3
"""Does the residual ever point at the unembedding row of the token it means?

Copying fails at every depth while the induction heads that should drive it
are measurably present, so the question is whether token identity is in the
residual at all.  For each position this measures the cosine between the final
hidden state and three unembedding rows: the current token's, the next one's,
and a random one as the null.  A working model shows both the current and the
next clearly above the null; the donor is run through the same code to say
what that looks like.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def stats(name, h, head, ids, rng):
    hu = unit(h)
    cur = np.array([hu[i] @ unit(head[ids[i]]) for i in range(len(ids))])
    nxt = np.array([hu[i] @ unit(head[ids[i + 1]]) for i in range(len(ids) - 1)])
    rnd = np.array([hu[i] @ unit(head[int(rng.integers(0, head.shape[0]))])
                    for i in range(len(ids)) for _ in range(8)])
    print("  %s" % name)
    print("     cosine with the CURRENT token's row  mean %+.4f" % cur.mean())
    print("     cosine with the NEXT token's row     mean %+.4f" % nxt.mean())
    print("     cosine with a random row (null)      mean %+.4f  std %.4f"
          % (rnd.mean(), rnd.std()))
    print("     current is %+.1f null-sigmas away, next is %+.1f"
          % ((cur.mean() - rnd.mean()) / rnd.std(),
             (nxt.mean() - rnd.mean()) / rnd.std()))
    print()


def main():
    rng = np.random.default_rng(7)
    import tempfile
    R = SourceFileLoader("R", os.path.join(HERE, "bonsai-ref-forward.py")).load_module()
    R.FOLD = "linear"
    ids_b = [int(x) for x in rng.integers(2000, 40000, 24)]
    m = R.Model("build/bonsai/f16.gguf")
    tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    tmp.close()
    R.run(m, ids_b, False, True, "qkv", dump=tmp.name)
    x = np.fromfile(tmp.name, dtype=np.float32).reshape(len(ids_b), -1)
    os.unlink(tmp.name)
    kv = m.kv
    eps = float(kv["qwen35.attention.layer_norm_rms_epsilon"])
    rot = R.Rot(kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"],
                kv["prism.hadamard.block_size"], False)
    h = rot(R.rms(x, m.t("output_norm.weight"), eps, True))
    stats("Ternary Bonsai 2 27B", h, m.t("output.weight"), ids_b, rng)

    sys.path.insert(0, os.path.join(HERE, os.pardir, "build", "donor", "pylibs"))
    from nlwts import Table
    _ref = SourceFileLoader("qref", os.path.join(HERE, "qwen-forward-ref.py")).load_module()
    tb = Table("build/donor/qwen3-0.6b/weights.bin")
    cfg = tb.config
    heads, kvh, hd = cfg["heads"], cfg["kv-heads"], cfg["head-dim"]
    eps, base = cfg["rms-eps"], cfg["rope-base"]
    ids_d = [int(x) for x in rng.integers(2000, 40000, 24)]
    seq = len(ids_d)
    pos = np.arange(seq, dtype=np.float64)
    xd = tb.rows(":wte", ids_d)
    for ly in range(cfg["layers"]):
        a = _ref.rmsnorm(xd, tb.f32(":ln1g", ly), eps)
        q = (a @ tb.dequant(":wq", ly).T).reshape(seq, heads, hd)
        k = (a @ tb.dequant(":wk", ly).T).reshape(seq, kvh, hd)
        v = (a @ tb.dequant(":wv", ly).T).reshape(seq, kvh, hd)
        q = _ref.rope_half(_ref.rmsnorm(q, tb.f32(":q-norm", ly), eps), pos, base)
        k = _ref.rope_half(_ref.rmsnorm(k, tb.f32(":k-norm", ly), eps), pos, base)
        ctx = _ref.attend(q, k, v, heads, kvh, hd).reshape(seq, heads * hd)
        x1 = xd + ctx @ tb.dequant(":wo", ly).T
        b = _ref.rmsnorm(x1, tb.f32(":ln2g", ly), eps)
        g = b @ tb.dequant(":wg", ly).T
        u = b @ tb.dequant(":wu", ly).T
        xd = x1 + ((g / (1.0 + np.exp(-g))) * u) @ tb.dequant(":wd", ly).T
    hd_ = _ref.rmsnorm(xd, tb.f32(":lnf"), eps)
    stats("Qwen3-0.6B (control)", hd_, tb.dequant(":wte"), ids_d, rng)


if __name__ == "__main__":
    main()
