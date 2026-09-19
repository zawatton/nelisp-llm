# COPY architecture comparison

`examples/compare-copy-architecture.el` is an opt-in comparison of two small
P5 COPY models. It changes depth and width together, so it cannot establish a
depth-only causal claim.

Both arms use initializer seed `439041101`, the same previous diversity
experiment's fresh 4,096-example exposure stream in the same order, and the
same 4,096-update budget: completion-only compact Adam, learning rate 0.003,
and sequence length 64. The baseline is dim 32/FF 64 with one block and one
head (27,264 parameters). The candidate is dim 24/FF 48 with two blocks and
one head (24,616 parameters). The original 128-example training split and
32-example held-out development split remain the reporting sets; candidate
exposure-stream accuracy is not substituted for the original train score.

Reporting includes both the unrestricted greedy exact-copy score and the
CPU/native gold-prefix teacher-forcing score. The latter measures per-byte
prediction under the expected prefix and is not free generation or a
competence benchmark. Before/after model hashes are checked, and evaluation is
performed outside GPU ownership without mutating either model.

Pure checks:

```sh
make test-copy-architecture
```

The experiment itself is an explicit GPU operation:

```sh
emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
  -l examples/compare-copy-architecture.el \
  --eval '(prin1 (nl-llm-compare-copy-architecture-run))'
```

## Verified paired run

The paired GPU run took 113.892 seconds wall time (the report records
113.8024513721466 seconds). It used the settings above and the same fresh
4,096-example exposure stream for both arms. On the original reporting sets,
the 27,264-parameter baseline scored 1/128 on train and 0/32 on development;
the 24,616-parameter candidate scored 0/128 and 0/32 respectively. These are
unrestricted greedy exact-copy scores, not accuracy on the exposure stream.

Gold-prefix teacher forcing on the held-out development set gave:

| arm | correct / tokens | mean cross-entropy | first byte | newline |
|---|---:|---:|---:|---:|
| baseline | 33 / 312 | 3.6456844469730307 | 0 / 32 | 31 / 32 |
| candidate | 35 / 312 | 3.6326901146242667 | 2 / 32 | 30 / 32 |

After excluding the terminal newline, the corresponding byte counts were only
2/280 and 5/280. The small cross-entropy difference and this newline-heavy
count are not evidence of a competence improvement. The comparison therefore
did not improve this seed, schedule, and model; neither arm was promoted.

The run used identical training metadata, including all 32 raw and encoded
batch hashes, and both inference checkpoints were read back from the same
artifact directory. The baseline final model hash was
`810d39bb6683d6ec9ad3146d1e2da64d22bc4d8173e1e4b857a8ce8050055939`; the
candidate hash was
`d2bbf49dd185a30346a89f3a51df4358af88c1607dddc9c2ecdf984c019d3359`.
The paired report SHA-256 is
`ce5df9cbbd54ea5046caa0fb3c02cd7b1755c4bd686a8625b971bb8c8a58b982`, with
source hash
`e6228a879c23003227172d92812198c10bd932f153493e962b7c7170fbbe3d41`.
The local, ignored, non-release artifacts are
[`paired.sexp`](../target/copy-architecture-AmJphI/paired.sexp),
[`baseline.checkpoint.sexp`](../target/copy-architecture-AmJphI/baseline.checkpoint.sexp),
and [`candidate.checkpoint.sexp`](../target/copy-architecture-AmJphI/candidate.checkpoint.sexp).
They are inference checkpoints, not Adam optimizer-resume checkpoints.

Before the paired run, the two-block candidate also passed independent
forward checks across GPU, CPU, and native execution (maximum difference
below `9e-7`) and the existing gradient/repeat/finite-difference checks. Those
checks establish numerical plumbing only; they do not establish competence.
Because this arm changes both depth and width, they also cannot isolate a
depth effect.

The next falsifiable diagnostic is to hold this budget fixed while separating
learning-order/length allocation and optimizer-condition effects; increasing
the budget alone is not the next conclusion from this run. That diagnostic is
not implemented here.
