#!/usr/bin/env python3
"""Are token_embd and output.weight related by some orthogonal map?

cos(token_embd[t], output.weight[t]) is +0.0073 against an off-diagonal
spread of 0.014 -- the two are all but unrelated, which no language model's
embedding and unembedding are.  Applying the same transform to both preserves
that cosine and applying different ones destroys it, so none of the
file's declared transforms can account for it.  But some OTHER orthogonal map
might, and if one does it can be recovered rather than guessed.

Orthogonal Procrustes on half the sample, scored on the other half, so a fit
cannot be read off the tokens it was fitted to.  A control fits the same map
against a shuffled pairing, which must not fit.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

R = SourceFileLoader("R", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bonsai-ref-forward.py")).load_module()


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def fit(X, Y):
    """The orthogonal M minimising ||X M - Y||, by SVD of X'Y."""
    u, _, vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return u @ vt


def main():
    m = R.Model("build/bonsai/f16.gguf")
    E = m.t("token_embd.weight")
    H = m.t("output.weight")
    rng = np.random.default_rng(17)
    idx = np.unique(rng.integers(0, E.shape[0], 60000))
    rng.shuffle(idx)
    ntr = len(idx) // 2
    tr, te = idx[:ntr], idx[ntr:]
    print("%d training tokens, %d held out, dim %d" % (len(tr), len(te), E.shape[1]))

    Xtr = unit(E[tr].astype(np.float32))
    Ytr = unit(H[tr].astype(np.float32))
    Xte = unit(E[te].astype(np.float32))
    Yte = unit(H[te].astype(np.float32))

    print("\n  before any map: held-out diagonal cosine %+.4f"
          % float((Xte * Yte).sum(axis=1).mean()))

    M = fit(Xtr, Ytr)
    fitted = unit(Xte @ M)
    diag = (fitted * Yte).sum(axis=1)
    perm = rng.permutation(len(te))
    off = (fitted * Yte[perm]).sum(axis=1)
    print("  after the fitted orthogonal map:")
    print("     held-out diagonal %+.4f   off-diagonal %+.4f   std %.4f   z %+.2f"
          % (diag.mean(), off.mean(), off.std(),
             (diag.mean() - off.mean()) / off.std()))

    # control: fit against a shuffled pairing.  Any apparent fit here is what
    # the procedure produces from nothing.
    sh = rng.permutation(ntr)
    Mc = fit(Xtr, Ytr[sh])
    fc = unit(Xte @ Mc)
    dc = (fc * Yte).sum(axis=1)
    oc = (fc * Yte[perm]).sum(axis=1)
    print("  control, fitted to a shuffled pairing:")
    print("     held-out diagonal %+.4f   off-diagonal %+.4f   z %+.2f"
          % (dc.mean(), oc.mean(), (dc.mean() - oc.mean()) / (oc.std() + 1e-12)))

    # and the same on the donor, whose two are the same tensor, as a ceiling
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, "build", "donor", "pylibs"))
    from nlwts import Table
    tb = Table("build/donor/qwen3-0.6b/weights.bin")
    W = tb.dequant(":wte")
    print("\n  the donor is tied, so its diagonal is 1.0 by construction "
          "(ceiling, not a comparison)")


if __name__ == "__main__":
    main()
