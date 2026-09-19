#!/usr/bin/env python3
"""Reference for Qwen3-Next's gated delta rule, and fixtures for the Elisp side.

Transcribed from transformers' torch_recurrent_gated_delta_rule.  Two details
are worth stating because they are invisible in the shapes:

  * the query is divided by sqrt(head_dim) *before* l2norm, and
    use_qk_l2norm_in_kernel is True -- so the division is undone and the query
    entering the recurrence is exactly l2norm(q).  Applying the division after
    the normalisation instead gives an answer that is wrong by sqrt(128) and
    looks entirely reasonable.

  * the state is (d_k, d_v); kv_mem is S^T k, and the write is the outer
    product k (x) delta.  The paper writes the transpose of this.

This file is the oracle for test/deltanet-test.el.  It is deliberately a
separate implementation rather than a port: comparing two transcriptions of
the same equations catches what re-reading does not.
"""
import argparse, json, math, os, sys

def l2norm(v, eps=1e-6):
    n = math.sqrt(sum(x * x for x in v))
    return [x / (n + eps) for x in v]

def softplus(x):
    # log1p(exp(x)), guarded the way torch does it
    return x if x > 20.0 else math.log1p(math.exp(x))

def gated_delta_rule(q, k, v, a, b, a_log, dt_bias, seq, dk, dv):
    """One head.  q,k: seq x dk (pre-norm).  v: seq x dv.  a,b: seq.
    a_log, dt_bias: scalars.  Returns (out seq x dv, states list)."""
    S = [[0.0] * dv for _ in range(dk)]
    out, states = [], []
    for t in range(seq):
        g = math.exp(-math.exp(a_log) * softplus(a[t] + dt_bias))
        beta = 1.0 / (1.0 + math.exp(-b[t]))
        qt = l2norm([x / math.sqrt(dk) for x in q[t]])   # the division is undone
        kt = l2norm([x / math.sqrt(dk) for x in k[t]])
        P = [[g * S[i][j] for j in range(dv)] for i in range(dk)]
        mem = [sum(P[i][j] * kt[i] for i in range(dk)) for j in range(dv)]
        delta = [(v[t][j] - mem[j]) * beta for j in range(dv)]
        S = [[P[i][j] + kt[i] * delta[j] for j in range(dv)] for i in range(dk)]
        out.append([sum(S[i][j] * qt[i] for i in range(dk)) for j in range(dv)])
        states.append([row[:] for row in S])
    return out, states

def det(n, mul, mod, scale):
    return [scale * ((i * mul) % mod - mod // 2) for i in range(1, n + 1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--dk", type=int, default=8)
    ap.add_argument("--dv", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seq, dk, dv = args.seq, args.dk, args.dv
    q = [det(dk, 7919, 101, 0.05) for _ in range(seq)]
    k = [det(dk, 5387, 101, 0.05) for _ in range(seq)]
    v = [det(dv, 3319, 101, 0.05) for _ in range(seq)]
    # vary per position so the scan is not doing the same thing every step
    for t in range(seq):
        q[t] = [x * (1.0 + 0.1 * t) for x in q[t]]
        k[t] = [x * (1.0 - 0.05 * t) for x in k[t]]
        v[t] = [x * (1.0 + 0.2 * t) for x in v[t]]
    a = [0.3 * ((t * 13) % 7 - 3) for t in range(seq)]
    b = [0.4 * ((t * 11) % 5 - 2) for t in range(seq)]
    a_log, dt_bias = 0.2, -0.1
    out, _ = gated_delta_rule(q, k, v, a, b, a_log, dt_bias, seq, dk, dv)
    blob = dict(seq=seq, dk=dk, dv=dv, q=q, k=k, v=v, a=a, b=b,
                a_log=a_log, dt_bias=dt_bias, out=out)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, "build", "deltanet-fixture.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f)
    print("wrote %s (seq %d, dk %d, dv %d)" % (os.path.relpath(path), seq, dk, dv))
    print("out[0][:4] =", ["%.9f" % x for x in out[0][:4]])
    print("out[-1][:4] =", ["%.9f" % x for x in out[-1][:4]])

if __name__ == "__main__":
    sys.exit(main())
