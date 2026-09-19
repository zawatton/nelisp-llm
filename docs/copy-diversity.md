# COPY diversity and teacher-forcing diagnostics

These are opt-in diagnostics for separating data exposure from the
free-running exact-copy score. They do not change service defaults, publish a
model, or claim competence on file tasks.

`examples/compare-copy-diversity.el` keeps the original 32-example development
set held out and compares the fixed 128-example training split with a fresh
exposure arm. The fresh arm supplies 32 batches of 128 examples (4,096 total
updates), using the same 27,264-parameter model: dim 32, FF 64, one block, one
head, byte vocabulary 256, sequence 64, Adam at 0.003, and the existing
initializer. Its purpose is to test whether repeating the same examples, or
seeing new literals at the same update budget, changes the result; it is not a
training recommendation.

The bounded CPU diagnostic
`examples/copy-teacher-forcing.el` reports per-case, per-length, and aggregate
next-byte accuracy and stable cross-entropy. It feeds the prompt and then the
preceding gold completion byte, so its values are teacher-forced and are not
free generation or exact-copy competence scores. The scorer uses fresh native
KV caches and never mutates the model.

Run the pure checks with:

```sh
make test-copy-diversity
make test-copy-teacher-forcing
```

The diversity experiment itself remains an explicit GPU operation:

```sh
emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
  -l examples/compare-copy-diversity.el \
  --eval '(prin1 (nl-llm-compare-copy-diversity-run))'
```

The paired run used the same 27,264-parameter initialization and 4,096-update
budget for both arms. The baseline repeated the original 128 literals; the
candidate saw 4,096 fresh exposures (not 4,096 unique literals). On the frozen
free-running score over the original train split, baseline was 57/128 and
candidate was 1/128; both were 0/32 on the held-out development split. The
teacher-forced reports give baseline 1,000/1,248 train tokens (mean CE
0.726341) and 34/312 dev tokens (CE 6.017117, first-byte 7/32, newline 7/32).
The candidate scored on that same original train split, not on its exposure
stream: 133/1,248 (CE 3.657517), and 33/312 dev tokens (CE 3.645684,
first-byte 0/32, newline 31/32). Thus fresh exposure alone did not improve this
seed, budget, and model; the low candidate teacher-forced content accuracy
does not support explaining the free-running result as exposure error alone.
This is not evidence that all small models or parameter budgets are
insufficient. A next falsifiable option is a depth-at-least-two comparison at
a similar parameter budget; it has not been implemented here.

The authoritative local artifacts are [`paired.sexp`](../target/copy-diversity-CCuCIj/paired.sexp),
[`baseline.teacher-forcing.sexp`](../target/copy-diversity-CCuCIj/baseline.teacher-forcing.sexp),
and [`candidate.teacher-forcing.sexp`](../target/copy-diversity-CCuCIj/candidate.teacher-forcing.sexp).
The files under `target/` are local, ignored experiment outputs rather than
release artifacts. The paired checkpoint files are inference checkpoints and
do not contain optimizer state for resume.

After a separately completed run, the same model object can be scored without
training by passing its COPY records to
`nl-llm-copy-teacher-forcing-score`. This second measurement must be described
as gold-prefix evaluation; it cannot by itself establish unseen-literal
generalization or native file-task competence.
