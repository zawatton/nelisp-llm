# COPY optimization comparison

`examples/compare-copy-optimization.el` is an opt-in, paired ablation of
learning rate and example order for the small COPY candidate architecture. It
does not change service defaults or promote a model. All four arms use the
candidate geometry (dim 24, FF 48, two blocks, one head; 24,616 parameters),
the initializer `xorshift32` with seed `439041101`, completion-only compact
Adam, sequence length 64, and 4,096 updates.

The four fixed arms are:

| arm | learning rate | example order |
|---|---:|---|
| `sorted-high` | 0.003 | sorted |
| `sorted-low` | 0.0003 | sorted |
| `shuffled-high` | 0.003 | shuffled |
| `shuffled-low` | 0.0003 | shuffled |

The sorted schedule preserves the existing exposure identity. The shuffled
schedule uses the existing on-device epoch shuffle (seed `104729`) for each
128-example batch, with 32 consecutive batches. Within every batch the
multiset of examples is preserved; only order changes. Both schedules are
deterministic and are hashed in the returned report. The same fresh exposure
stream is used by every arm. The two sorted arms share the identity order, and
the two shuffled arms share the same deterministic permutations; within each
order condition the arms therefore have the same example multiset and order.
The comparison is about the specified learning-rate/order conditions rather
than a new dataset.

The original 128-example train split and 32-example held-out development split
remain the reporting sets. Reports contain before/after model hashes, greedy
scores, and gold-prefix teacher-forcing scores. Teacher forcing uses expected
completion prefixes and is not free generation or a competence benchmark.
The geometry comparison already changed depth and width together, so this
optimization comparison does not isolate either architecture factor. This
experiment covers dense P5 only; it does not test the validity of the
recurrent `nl-llm-recur.el` path or other model families. It also does not
establish that any one optimizer setting generalizes beyond this seed, data
schedule, and update budget. A bounded future architecture study could connect
recurrent depth through the same evaluation and data-distribution path; that
is not implemented here.

Pure contract checks:

```sh
make test-copy-optimization
```

The paired experiment is an explicit GPU operation:

```sh
emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
  -l examples/compare-copy-optimization.el \
  --eval '(prin1 (nl-llm-compare-copy-optimization-run))'
```

## Verified four-arm run

The independent GPU run took 247.024 seconds wall time (the report records
246.8441514968872 seconds). All four arms used the same initial model hash,
the same exposure multiset, and the same 4,096-update budget. The captured
context learning rates were `(0.003 0.0003 0.003 0.0003)`, confirming that the
per-arm learning-rate binding reached the trainer.

| arm | greedy dev | greedy original train | teacher-forced dev | mean CE | first byte | newline | non-newline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sorted-high` | 0/32 | 0/128 | 35/312 | 3.6326901146242667 | 2/32 | 30/32 | 5/280 |
| `sorted-low` | 2/32 | 17/128 | 67/312 | 3.1794641640160775 | 20/32 | 25/32 | 42/280 |
| `shuffled-high` | 2/32 | 16/128 | 65/312 | 3.0505265395876555 | 11/32 | 21/32 | 44/280 |
| `shuffled-low` | 4/32 | 18/128 | 69/312 | 3.1436292552601492 | 15/32 | 31/32 | 38/280 |

The best greedy development result was `shuffled-low` (4/32). Its four
correct development cases were three length-1 cases and one length-2 case;
all lengths 3 and above scored zero. The best teacher-forcing cross-entropy
was instead `shuffled-high`, while the best greedy score was `shuffled-low`,
so these indicators do not select the same arm. The small one-seed,
32-example development result is a limited diagnostic, not evidence for a
generally optimal setting or practical COPY/agent ability. No arm was
promoted.

The authoritative local, ignored, non-release report is
[`factorial.sexp`](../target/copy-optimization-ynZeRD/factorial.sexp), SHA-256
`a576ce6f3ee522d3459711b9cd9c065867b277367b2e3a5196f9f1fd9da4a198`.
All four inference checkpoints were read back and reproduced their reported
model hashes and teacher-forcing totals:

| arm | final model hash | checkpoint SHA-256 |
|---|---|---|
| `sorted-high` | `d2bbf49dd185a30346a89f3a51df4358af88c1607dddc9c2ecdf984c019d3359` | `ab03abb72eedddd35ace3dcd7b934994640d04fe4d49e821e28cba8c50f6292a` |
| `sorted-low` | `8eda970336a3f885932db83964323b6c518bb31638ebe06838a5fec462d86d8a` | `062538c0e93c0a8a71c343224c31085f2c1631a8556c8add433d49ee5f0e38a6` |
| `shuffled-high` | `339e7477858ec458ddf2d2cf8f927376d906296db54e72ada6711f533b61578d` | `3f2f13de49628a866f16c32eed4424d40c4faa1e7c380ae0dd5377e0acd6ed7d` |
| `shuffled-low` | `1e5c7129e25805ee627f3137bbc3cbe431741b873ed4c3a99ae5d974f64aee5d` | `86ae0c249fe906aaae5a29ff504008f0abfb58e7b660933288dc5d707e84806e` |

The shared initial model hash was
`b24f06554d3169e2b039cc031c09bbe8d6dcb6c1ed4a7f6c3feff3bad489d505`.
The run's architecture source identity was
`e6228a879c23003227172d92812198c10bd932f153493e962b7c7170fbbe3d41`.
These checkpoints contain inference weights only, not Adam optimizer-resume
state. Existing recurrent-depth integration remains a separate, unimplemented
candidate and is not a negative result of this dense-P5 experiment.

An independent process reproduced the `shuffled-low` winner's 4/32 greedy
development score, teacher-forcing totals, and weights. This run does not
justify more budget by itself. A next bounded candidate is to connect
recurrent depth through the same evaluation and data-distribution paths; that
work is unimplemented. Before adoption, multiple seeds and generalization on
lengths of at least three characters remain unverified.
