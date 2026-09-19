# Completion-only training

`nl-llm-agent-supervised` adds opt-in prompt/completion supervision to the
small P5 model. It does not reconstruct a hosted model, demonstrate general
agent competence, or automatically connect captured trajectories to training.
In addition to the direct CPU/GPU API, an opt-in supervised queue and its CPU
or GPU worker paths are available. They consume approved curation records via
the pending queue; capture alone never starts training. Existing legacy
full-sequence fine-tuning and worker paths retain their previous behavior.

## Canonical completion plans and checkpoint format

`nl-llm-agent-completion-plan-make` builds a detached, canonical
`nl-llm-completion-plan-v1` from token trajectories, completion boundaries,
and optional sparse loss masks. The plan binds tokenizer, vocabulary, padding,
sequence, optimizer settings, and a SHA-256 digest. Save the complete plan as
the `:completion-plan` field of a completion checkpoint whose format is
`nl-llm-agent-training-completion-v1`; loading requires the same plan, so an
omitted or changed plan is rejected. The older full-sequence checkpoint format
is not a completion-plan substitute and is intentionally incompatible with
this path. Run `make test-agent-completion-checkpoint` for the plan,
completion-checkpoint, and legacy checkpoint checks.

Completion plans are detached training-input records; callers can still mutate
Elisp objects, but validation detects changes against their digest. A plan is
also the binding for snapshot/resume of a low-level completion context. The
context must have been created with that canonical plan; unbound completion
contexts remain ephemeral and reject nonzero resume cursors, snapshots, and
optimizer-state restore. The public synchronous wrapper remains ephemeral,
while the background GPU configuration and runner connect the isolated worker
wire protocol that carries bound completion checkpoint/resume state. CPU
checkpoint/resume is unsupported. Sparse masks, compact transfer, and
epoch-shuffle options are not exposed on this worker wire path.

Restore weights and optimizer state in this order: load a completion
checkpoint with the expected plan, restore its model into a fresh candidate,
create a plan-bound context from that candidate, restore the optimizer state
and completed-step counter with the same expected plan, then call training from
the saved start step. Free the old context before rebuilding. A device I/O
failure can leave a candidate or context partially changed; discard and
rebuild them because rollback is not automatic. No claim of improved
capability follows from this data plumbing or from the numerical checks below.

After `nl-llm-agent-training-checkpoint-load` has validated the expected plan,
restore weights into a fresh candidate before creating the new context:

```elisp
(nl-llm-agent-training-checkpoint-restore-model candidate checkpoint)
(let ((ctx (nl-llm-agent-ondevice-from-model
            candidate sequence lr :loss-mode 'completion
            :optimizer optimizer
            :transfer-mode (plist-get plan :transfer-mode)
            :completion-plan plan)))
  (unwind-protect
      (progn
        (nl-llm-agent-ondevice-restore-training-state
         ctx completed optimizer-state plan)
        (nl-llm-agent-ondevice-train
         ctx trajectories epochs :start-step completed
         :loss-starts loss-starts :loss-masks loss-masks
         :shuffle-seed shuffle-seed)
        (nl-llm-agent-ondevice-sync ctx))
    (nl-llm-agent-ondevice-free ctx)))
```

The caller owns GPU enable/free and must discard both candidate and context
after a device I/O failure; restore-training-state restores optimizer state
and the counter, not model weights.

```elisp
(require 'nl-llm-agent-supervised)
(nl-llm-agent-supervised-train
 model [(:prompt "Question: " :completion "Answer")]
 :backend 'cpu :lr 0.05 :epochs 1)
```

The supplied model is updated **in place**. Defaults are CPU, SGD, learning
rate 0.05, and one epoch. Validation precedes training, but a runtime failure
after successful steps can leave partial updates; this API is not a transaction.
Use a separate candidate model when preserving the incumbent is required.

Examples are a vector of 1–128 exact `(:prompt STRING :completion STRING)`
records. Both strings must be nonempty and representable by the model's
tokenizer. Each combined example is bounded to 4096 characters and 4096 tokens;
the dataset is bounded to 65536 characters. Encoding returns detached token
trajectories, first-completion-token indices, the completion-token count, and
a SHA-256 digest binding text, tokenizer, and supervision boundaries.

For a token sequence of length `n` with its first completion token at index
`k`, target row `i` predicts token `i+1`. Only rows `k-1` through `n-2`
contribute to cross-entropy. CPU loss uses stable log-sum-exp and averages over
those `n-k` rows. Prompt logits have zero direct loss gradient, but prompt
embeddings can receive gradients through attention: the prompt is still context.
Reported dataset loss is weighted by completion-token counts. Training takes
one mean-completion-loss optimizer step per example, not one global batch step.

The GPU backend requires an already enabled GPU, supports SGD or Adam, and
accepts `:sequence` from 2 to 4096, at least the longest combined token length.
Its default is the longest length (minimum 2). CPU rejects `:sequence` and Adam.
GPU CE gradients, originally divided by the fixed sequence length `M`, are
multiplied by `M/(n-k)` on completion rows and by zero on prompt/padding rows.
The wrapper synchronizes weights back to the model and frees its owned context.

## Optional sparse completion targets

The low-level `nl-llm-agent-ondevice-train` accepts `:loss-masks` for
unbound completion-loss contexts and for plan-bound contexts. This is a vector of ordinary integer
0/1 vectors, one per trajectory and exactly as long as that trajectory.
Index `j` selects target token `j`, so its gradient is applied at row `j-1`.
Every position before the required `:loss-starts` boundary must be zero;
every trajectory must select at least one completion target. All tokens
remain in the input, including unselected completion tokens.

The mean loss is over selected targets in each example: GPU row scales are
`sequence-length / selected-count` for selected rows and zero elsewhere.
Omitting the option preserves the previous contiguous completion objective.
The entire request is validated and row plans detached before device updates.
Masks remain paired with trajectories during optional epoch shuffling.
Legacy/full-sequence contexts reject masks. Unbound completion contexts retain
their ephemeral snapshot/resume restrictions, while a plan-bound context
persists the masks as part of its canonical completion-plan binding. This
option is not yet exposed by the high-level supervised wrapper or durable
worker protocol.

`make test-agent-loss-masks` exercises planning and validation without a GPU.
`make test-agent-loss-masks-gpu` is a separate real-GPU numerical check.
Grammar-specific selection belongs to the agent experiment, not this engine.

The independent 2026-09-07 numerical run passed all four combinations of
GPU sequence length 8/16 and pad ID 0/255. An unpadded CPU SGD reference
used sparse target indices 2, 4, and 5; both dense and compact GPU updates
matched it to maximum absolute parameter error `4.82e-8`, and dense/compact
parameters were identical. All compared parameters were finite and all
three paths changed weights. The six planning/validation ERT tests also
passed. These checks establish the selected-target update under the tested
geometry, not an improvement in agent capability.

## Optional epoch shuffling

The low-level `nl-llm-agent-ondevice-train` accepts `:shuffle-seed` for
unbound completion-loss contexts and for plan-bound contexts. A nonzero uint32 seed selects a local
xorshift32/Fisher-Yates permutation for each epoch. The stream continues across
epochs within one call, and restarts from the supplied seed for a new call.
Every example is visited once per epoch with its own completion mask. This
uses modulo-index sampling for deterministic engineering experiments, not a
cryptographic or exact-uniform sampling guarantee.

Omitting the option keeps the existing fixed order. Adam's context step and
the flattened progress callback continue across epochs. Caller data is not
reordered in place. Invalid seeds and legacy loss contexts are rejected before
device updates. A plan-bound completion context resumes with the plan's
shuffle seed and saved step; an unbound context still rejects resumed
completion training. The checkpoint contract binds the shuffle algorithm,
seed, and plan position. Format v1 defines the xorshift32/Fisher--Yates
schedule; there is no separate algorithm field, so changing that schedule
requires a checkpoint-format/version change.

`make test-agent-epoch-shuffle` checks schedule determinism, token/mask pairing,
step/callback continuity, and fail-before-mutation behavior without a GPU.

The first short-copy comparison fixed the model initialization at xorshift32
seed `0x1A2B3C4D` and compared the existing order against epoch-shuffle seed
`104729`, declared before outcomes. Both arms used the same 31 training / 8
development examples, 27,264 parameters, and 992 Adam steps. The ordered arm
exactly reproduced its previous final model hash and loss.

| Training order | Training exact | Development exact | Training loss |
| --- | --- | --- | --- |
| Fixed | 24/31 | 5/8 | 0.4319283253323546 |
| Shuffled each epoch, seed `104729` | 11/31 | 2/8 | 0.588666715420851 |

Shuffling worsened this run and did not fix any of the three shorter
development cases. Only `bbb` and `cbc` were correct in the shuffled
development results. This is evidence against adopting this particular
configuration, not a universal claim about shuffling. No default changed.
The shuffled final model hash was
`f19e54716956276590b3b4514da8890b92dcd4d776776d5fa7945f189d814b4e`;
both initial hashes were
`bc4ae38649deda046482b1d8eb70b369766cd2a848b905ba44bede75285629c3`.

The comparison reused the frozen copy runner through a temporary, restored
override of its training call; its data encoding and decoder were untouched:

```sh
emacs -Q --batch \
  --eval '(setq load-prefer-newer t nl-llm-compare-copy-initialization-no-run t)' \
  -l examples/compare-copy-initialization.el \
  --eval '(let* ((print-length nil) (print-level nil) (original (symbol-function (quote nl-llm-agent-ondevice-train))) (ordered (nl-llm-compare-copy-initialization--run-candidate #x1A2B3C4D)) (shuffled (cl-letf (((symbol-function (quote nl-llm-agent-ondevice-train)) (lambda (&rest args) (apply original (append args (list :shuffle-seed 104729)))))) (nl-llm-compare-copy-initialization--run-candidate #x1A2B3C4D)))) (dolist (key (quote (:settings :dataset-sha256 :model-before-sha256 :steps))) (unless (equal (plist-get (plist-get ordered :report) key) (plist-get (plist-get shuffled :report) key)) (error "paired contract mismatch %s" key))) (prin1 (list :format "copy-epoch-order-comparison-v1" :shuffle-seed 104729 :ordered ordered :shuffled shuffled)) (terpri))'
```

Verification passed five new schedule ERT tests, the existing compact GPU
checks (nine manual checks and two ERT tests), and completion-supervision
checks (14 manual checks and eight ERT tests). Three real child-process GPU
resume tests also passed on the unchanged legacy path. Strict compilation
passed for the modified module and new test. The full service suite was not
rerun for this change; its prior result does not cover the new option.

The training module SHA256 for this comparison was
`49eda21ee4166ab966da2af24a390456622c82accb7c6bd2bd07cf5b7bc81908`.

On the same Vulkan environment, the small deterministic completion-resume
fixture passed all 12 combinations: SGD and Adam, dense and compact transfer,
and interruption at steps 2, 3, and 5. Interrupted-and-restored training had
zero reported maximum weight and optimizer-state difference from uninterrupted
training in every case; the final-step resume was also a no-op. This validates
the tested checkpoint/replay contract, not an improvement in model capability.
The public synchronous wrapper remains unconnected to this low-level resume
path. The background GPU queue configuration and service runner use the
isolated worker protocol with the bound plan and completion checkpoint format;
ordinary no-checkpoint supervised training remains unchanged.
An independent fault-injection run that suppressed Adam state restoration was
rejected as expected (dense split 2: weight difference `0.08081713318824768`,
optimizer-state difference about `0.0964`); the six SGD cases remained at zero.

## Compact GPU transfers

The supervised GPU wrapper selects the compact completion path. Direct
`nl-llm-agent-ondevice-from-model` callers opt in with both
`:loss-mode 'completion` and `:transfer-mode 'compact`. The low-level default remains
the previous dense transfer path, including legacy checkpoint behavior.
Compact transfer is rejected for legacy full-sequence loss.

For sequence length `M` and vocabulary size `V`, the compact graph reuses
embedding gather, index-target CE, and row scaling. It does not allocate the
explicit causal mask already implemented by the fused attention kernel, or
return unused full logits after each training step.

| Per-step float payload | Dense completion | Compact completion |
|---|---:|---:|
| Input tokens | `M × V` | `M` |
| Target tokens | `M × V` | `M` |
| Loss row scales | `M × V` | `M` |
| Returned training logits | `M × V` | `0` |

At `M=2048`, `V=256`, input/target/scale uploads fall from 6 MiB to 24 KiB
per step, and the unused 2 MiB logits response disappears. These are payload
counts, not a wall-clock speedup: protocol headers, Adam hyperparameter writes,
graph construction, and final parameter synchronization are separate. Model
geometry, supervision boundaries, optimizer, and training step counts do not
change. Compact contexts support the same plan-bound completion resume path;
unbound compact contexts remain ephemeral.

Run `make test-agent-compact` for GPU gradient, multi-step SGD/Adam parity,
transfer-shape, and validation checks. Run
`emacs -Q --batch -l examples/bench-completion-transfers.el` for a small paired
synthetic transfer benchmark. Its timings characterize that workload only;
they do not measure task quality or repository-scale training throughput.

One paired GTX 1060 run on 2026-09-06 (sequence 64, vocabulary 256, six SGD
steps) measured training at 1.0506 s dense versus 0.01817 s compact. Actual
float payload over those steps was 1,572,864 versus 4,608 bytes; final parameter
synchronization was separate. All trained weights matched exactly and both
models changed from initialization. This is a small synthetic run, not a
projection for long-context training; an unrelated CPU-heavy NeLisp process
was present, and the learning experiment and regression tests had finished
before this measurement.

Unbound completion-mode GPU contexts are intentionally ephemeral. Plan-bound
contexts may restore nonzero cursors, optimizer state, and training snapshots
only when the checkpoint carries the detached, digest-validated supervision
plan. Ordinary model inference artifacts are separate from resumable
optimizer checkpoints. The public synchronous wrapper does not expose this
low-level resume path. Background GPU configuration and the service runner
route bound completion checkpoint/resume through the isolated worker protocol;
CPU checkpointing remains unsupported.
No automatic publication, model switch, external inference, or training-data
collection is enabled by this module.

Run `make test-agent-supervised` for the CPU gradient, bounded wrapper, and
GPU masking tests. GPU checks require the local supported GPU backend; a skipped
GPU test is not evidence of GPU correctness.

`make test-agent-forward-parity` directly compares compact GPU forward logits
with the CPU batch model at every token position, and their final rows with
exported native sequential inference. It covers short repeated ASCII and
multibyte UTF-8 prefixes in a tiny two-head model, both at initialization and
with nonzero embedding, bias, normalization, and independent-head perturbations.
Inputs to the numerical comparison must be finite. After starting the resident
GPU server, the test restores pure-CPU tensor dispatch and verifies every
function cell against the saved CPU implementation before CPU/native calls.
The resident graph still dispatches directly to the GPU. An earlier version
left global tensor dispatch on GPU, so its nominal CPU reference was not an
independent pure-CPU check. The corrected local run passed all 18 comparisons:
maximum GPU/CPU difference 3.66e-7, CPU/native difference 4.44e-16. This checks these
short forward paths, not long-context numerical accuracy, training convergence,
or learned file-task competence.

`make test-agent-gradient-parity` checks the complementary backward path.
It compares full compact-GPU parameter gradients with pure-CPU autodiff for
short repeated-token and UTF-8 sequences, including completion boundaries and
padding. Initial and perturbed tiny models exercise the embedding, attention,
feed-forward, normalization, and independent output head. The GPU graph has no
optimizer; repeated execution checks that temporary gradients do not accumulate
across runs. Selected nonzero CPU gradient coordinates are also checked against
centered finite differences, restoring each weight afterwards. This is a
gradient check, not an optimizer-convergence or agent-capability benchmark.

The completed local run passed 160 full-tensor GPU/CPU comparisons (20
parameter tensors across four sequences and two model variants), 160 repeated
GPU-gradient checks, and 20 selected-coordinate CPU finite-difference checks.
Embedding, query, key, and output-head gradients were nonzero; weight arrays
remained exactly unchanged. GPU/CPU tolerances apply at every coordinate,
not just the coordinate with the largest absolute difference. CPU references
exclude padding; GPU inputs retain the fixed 16-row window. These results
support the tested backward path, not arbitrary models or optimizer behavior.

The opt-in [file-action grammar](file-action-grammar.md) now supports model-chosen
operations and variable strings. The agent's manual supervised file experiment
uses identical before/after budgets and checks actual held-out files. Neither
grammar validity nor lower training loss establishes improved agency, Japanese
competence, or safe self-modification.

## Manual long-context forward check

`make test-agent-long-forward-parity` extends numerical diagnosis to a
1,476-token synthetic UTF-8 prefix in a 2,048-row GPU window, with the
27,264-parameter file-task geometry (dimension 32, feed-forward 64, one
block/head, vocabulary 256). It checks all real rows, not just the final row.
The input contains paths, ASCII, Japanese, and supplementary Unicode; it is
not a held-out agent task or training dataset.

The two variants are fresh xorshift32 initialization and deterministic
nonzero weight/norm/bias perturbations, not trained artifacts. For each,
padding tokens 0 and 255 must not materially change earlier GPU rows.
Exported incremental inference is compared in both source and in-memory
byte-code modes, the latter matching the host service's normal numeric
execution mode. CPU dispatch and byte-code ownership are checked explicitly
to prevent an apparent CPU reference from silently invoking GPU operations.
All compared values must be finite, and model weights must remain unchanged.

The GPU/native tolerance is fixed at absolute `5e-4` plus relative `5e-4`
times the native magnitude, at every coordinate; padding-prefix differences
are bounded by `1e-6`. Source/byte-code differences are bounded by `1e-10`.
This is a manual numerical check without an optimizer, learning, publication,
or service-default changes. It is not part of the default test run.

The independent run completed on 2026-09-06 (98.48 seconds): one ERT test
passed without skips, including all 12 vector comparisons of 377,856
coordinates each. Source and byte-code results were identical in both
variants. GPU/native maximum absolute differences were `2.26e-6` for the
initial model and `2.24e-6` for the perturbed model; both padding choices
produced identical real-row logits. Every tolerance check passed and the
weights remained unchanged. Maximum relative errors near zero were 2.21
and 1.26 respectively, so this is an absolute-plus-relative tolerance result,
not a small-relative-error claim. The fixture SHA-256 was
`9cc42bb003742fd864d02915bd7c681543b08c672d3ffdca415229ef7fb6c860`.
This excludes a forward-path mismatch only under the tested conditions;
it does not establish trained-model quality or explain the failed file tasks.

## Short literal-copy learning probe

The manual `examples/learn-literal-copy.el` experiment isolates accurate
argument generation from the long service prompt and multi-step tool loop:

```sh
emacs -Q --batch -l examples/learn-literal-copy.el
```

Its fixed synthetic vocabulary consists of all 39 strings of length 1--3 over
`abc`, ordered by length and lexicographically within each length. Entries
whose zero-based index is divisible by five are development data (8); the
remaining 31 are training data. The exact prompt is `COPY: LITERAL\nOUTPUT:\n`,
and the completion is the literal followed by a newline. Development examples
never enter training. This is an engineering development split, not a new
held-out file-task benchmark.

The model keeps the previous 27,264-parameter UTF-8 P5 geometry (dimension 32,
feed-forward 64, one block/head). Training uses compact completion-only GPU
Adam at 0.003, sequence 32, and 32 epochs (992 example steps). Before and after
training, the probe measures training completion loss and exact native greedy
copy accuracy on both splits. Decoding considers all 256 byte tokens: no
grammar, answer list, expected length, or expected completion is supplied to
the decoder. A generated newline ends the response; otherwise generation stops
at eight tokens. Token IDs are reported to avoid displaying arbitrary control
bytes or invalid UTF-8. Model/dataset identities and numerical finiteness are
checked; errors invalidate the run rather than becoming quality scores.

`make test-literal-copy` runs cheap data-split, decoder, and exported-model
wiring tests without GPU training. Even a successful learning run would show
only short synthetic copying, not file editing, long-context copying, general
instruction following, or readiness for automatic self-modification. There is
no artifact publication, external inference, or real-user data collection.

The first completed run lowered training completion loss from 5.84828 to
1.29572, but greedy output became the same `[98 10]` (`b` plus newline) for
all 39 inputs. Exact training score changed from 0/31 to 1/31 only because
one training target is `b`; development stayed 0/8. This is not evidence that
copying was learned. The run completed all 992 steps with finite weights and
unchanged weight hashes during evaluation. It narrows the investigation to
learning input-dependent outputs, without establishing whether initialization,
optimization, gradients, data order, or capacity is responsible.

Reproduction identities for this fixed run:

- Example SHA256: `ac0742137067ad93eccdafd9e4182ffa56744480f4e2fe9e75edc4770994d459`
- Training dataset SHA256: `c5096df9d0a1b8d0efe91d871c5d1d91935dffaadda9d034b5ba165dd4678dbf`
- Initial model SHA256: `182e7670371862cb5144f8125d4f57a90d0eb83bd0252520b5cfa6ca845bb482`
- Trained model SHA256: `8b58f22c283d63b4511809b75ec02b2deec7e2de93f7d32d0a7d666254c5f45e`

These model hashes use this probe's canonical geometry/flattened-weight format,
not the different digest format used by the file-task experiment.

A subsequent engineering comparison changed only the epoch count from 32 to
256, starting from the same initial weights (not resuming the previous model):

```sh
emacs -Q --batch --eval '(setq nl-llm-learn-literal-copy-no-run t)' \
  -l examples/learn-literal-copy.el \
  --eval '(let ((nl-llm-learn-literal-copy-epochs 256) (print-length nil) (print-level nil)) (prin1 (nl-llm-learn-literal-copy-run)) (terpri))'
```

All 7,936 steps completed. Loss reached 1.05955, but every input produced
`[98 98 98 10]` (`bbb` plus newline). Training scored 0/31 and development
1/8 solely because the development split contains `bbb`. This does not show
input-dependent copying or generalization. The final model hash was
`9e4ed77b36e507307b6569158c4d8e8771a82f89c8b4cab769bd5928d86c3af8`;
the source, dataset, initial model, model geometry, and other training settings
were unchanged. Merely increasing epochs did not resolve the failure in these
two runs. Initialization and optimization remain hypotheses to test, not
established causes. Neither candidate was published or promoted to the service.

### Initialization-only comparison

The separate manual `examples/compare-copy-initialization.el` experiment runs
the unchanged 32-epoch baseline and a candidate with only matrix initialization
replaced. It retains the same data split, geometry, optimizer, learning rate,
and unrestricted native decoder. Candidate matrix values use a local xorshift32
stream, seed `0x1A2B3C4D`, mapped to the original amplitude interval
`[-1/sqrt(dim), 1/sqrt(dim))` in canonical parameter order. One-dimensional
normalization scales and biases remain unchanged. No global random state or
production constructor default is changed.

```sh
emacs -Q --batch -l examples/compare-copy-initialization.el
```

`make test-copy-initialization` checks the initializer and paired-run wiring
without GPU training. This is a single-seed engineering comparison: even an
improvement would not establish that this initializer is generally superior,
or that parameter-efficient general agency has been achieved.

The first completed paired GPU run executed 992 steps per arm. The baseline
exactly reproduced its previous loss and final model hash. Both arms started
at 0/31 training and 0/8 development accuracy. After training:

| Initialization | Training exact | Development exact | Training loss |
| --- | --- | --- | --- |
| Original modular sequence | 1/31 | 0/8 | 1.29571806198741 |
| Xorshift32, seed `0x1A2B3C4D` | 24/31 | 5/8 | 0.4319283253323546 |

The candidate correctly copied all five three-character development strings
(`aba`, `acc`, `bbb`, `caa`, `cbc`), but failed all three shorter development
strings (`a`, `ac`, `cb`). Its outputs were input-dependent, unlike the constant
baseline output. This supports initialization as a consequential factor in
this fixed experiment, not as the only cause of learning failures. The small,
previously inspected development split and single seed are not an independent
generalization benchmark. No model was promoted to the service.

Reproduction identities:

- Comparison source SHA256: `36638cd6a71c2384a0fc5c223a6f268bf1aa32ee8ad5ab279bd020d8e6dcac5b`
- Candidate initial model SHA256: `bc4ae38649deda046482b1d8eb70b369766cd2a848b905ba44bede75285629c3`
- Candidate trained model SHA256: `90920a38c5f38f62169bd5b22881c7a57d42e8c2d21793343aa327d236e53baf`

Settings and dataset hashes matched across arms; all 640 one-dimensional
values were preserved at initialization, and evaluation did not mutate model
weights. The focused nine ERT tests, the seven existing literal-copy tests,
and strict compilation passed. The full service/standalone suite was not
rerun for this experiment-only change.

Two further seeds were declared before running, with all other settings fixed.
Each candidate started fresh and completed 992 steps; the frozen baseline was
not rerun in this replication batch.

| Xorshift32 seed | Training exact | Development exact | Training loss |
| --- | --- | --- | --- |
| `0x243F6A88` | 22/31 | 4/8 | 0.2549587523154288 |
| `0x9E3779B9` | 24/31 | 5/8 | 0.31688231931389066 |

Both started at 0/31 and 0/8. Both still failed the three shorter development
strings; the first also failed `bbb`. Lower teacher-forced loss did not imply
higher greedy exact accuracy. Together with the original candidate, all three
seeds improved on the frozen legacy baseline, but these remain development
experiments rather than evidence of general instruction-following ability.

```sh
emacs -Q --batch \
  --eval '(setq load-prefer-newer t nl-llm-compare-copy-initialization-no-run t)' \
  -l examples/compare-copy-initialization.el \
  --eval '(let ((print-length nil) (print-level nil)) (dolist (seed (quote (#x243F6A88 #x9E3779B9))) (prin1 (list :seed seed :result (nl-llm-compare-copy-initialization--run-candidate seed))) (terpri)))'
```

Initial/final model SHA256 pairs for these seeds respectively:

- `a0ae631652840b0a33b1862bccc02ec8c525893d364153307042bc0077b59137` / `44699337be1a1694fe0a8f5316f45bd0bcba0f7bc85ee72ff18c1efbd02acd4f`
- `3c983939353d5534ffa392fc249f5e05e56bc2263470233ce03fc7d5e6594767` / `344b2d29ba632b7a6c49f21857e860f41ccb8e2393bc7bc9623019d8e824c535`

### Opt-in model creation API

`nl-llm-agent-initialization-create` makes the tested initialization available
without loading an experiment or temporarily overriding a constructor:

```elisp
(require 'nl-llm-agent-initialization)
(nl-llm-agent-initialization-create
 :initializer 'xorshift32 :seed #x1A2B3C4D
 :dim 32 :ff 64 :vocab 256 :nblocks 1 :heads 1
 :tokenizer "utf8-byte-v1")
```

It returns a fresh PAV model for the existing supervised-training and artifact
export APIs. Omitted geometry options keep the existing constructor defaults;
omitted `:initializer` selects `legacy`, preserving old weights. `xorshift32`
requires an explicit nonzero uint32 seed. This does not reinitialize an active
or trained model, change service defaults, or automatically publish a model.
The caller must record the initializer and seed in its experiment provenance;
the returned model/checkpoint schema is unchanged and does not store them.

Host verification covers eight API ERT tests, including exact legacy-model
equality, invalid initialization options before construction, preservation of
rank-1 values against a separate legacy model, and the original candidate's
full initial-weight hash. The existing comparison/copy tests (16) and artifact
tests (14 checks) also pass, as does strict compilation of the new module and
tests. An independent export/native-inference check on UTF-8 bytes for `A日A`
matched CPU last-position logits within `8.881784197001252e-16`, with unchanged
weight hashes. This is host-path verification; the full service and standalone
suite was not rerun for this opt-in addition.

## Length-ordered synthetic copy probe

`examples/learn-copy-curriculum.el` is an opt-in probe, not a service default
or a promoted model. It tests whether a small native model can copy unseen
ASCII literals before any attempt to transfer that skill to file actions.
The frozen literal-copy decoder/export helpers are guarded by their source
SHA-256; output selection remains unrestricted over all 256 byte IDs.

The local xorshift32 data seed is `608135816`, with the 42-character alphabet
`abcdefghijklmnopqrstuvwxyz0123456789/._-: ` (including its trailing space).
There are 20 unique literals at each length 1, 2, 3, 4, 8, 12, 16, and 24.
Accepted global indices divisible by five are development examples: 128
training and 32 development examples, with no duplicate literal across splits.
Whitespace is preserved. Independent JavaScript generation matched all 160
Elisp literals and the final PRNG state `2412661987`.

Each prompt is `COPY: <literal>\nOUTPUT:\n`; its completion is the literal
followed by a newline. Training uses only the training split: 1,248 completion
tokens per epoch, physical sequence length 64, completion-only compact Adam
at 0.003, and 32 fixed length-ordered epochs (4,096 updates). The model uses
27,264 parameters, dim 32, FF 64, one block/head, byte vocabulary 256, and
initializer seed `439041101`. There is no seed search, shuffle, or data-driven
stopping rule.

Evaluation uses fresh native caches, CPU byte-code inference outside GPU
ownership, and a fixed 32-generated-token bound. Exact success requires both
the expected bytes and newline termination; the decoder is not given the
expected literal or its length. Before/after evaluation weight hashes must
remain unchanged. This is a new dataset: its scores are not directly comparable
to the earlier 31/8-example three-letter probe, and there is no random-order
control establishing an advantage from length ordering.

Pure tests (no real GPU): `make test-copy-curriculum`.
The manual GPU run, from this repository, is:

```sh
emacs -Q --batch --eval '(setq load-prefer-newer t)' \
  -l examples/learn-copy-curriculum.el \
  --eval '(prin1 (nl-llm-learn-copy-curriculum-run))'
```

The fixed real-GPU run completed all 4,096 updates successfully. Exact train
copy rose from 0/128 to 57/128; development remained 0/32. At lengths
1, 2, 3, 4, 8, 12, 16, and 24, the after-training train successes were
14, 14, 13, 9, 7, 0, 0, and 0 (each out of 16). Every development length
group scored 0/4. For example, unseen `x` produced `j\n`, and `3j` produced
`j3\n`. These results do not establish unseen-literal copy generalization,
file-task transfer, or an advantage over other model architectures.
No file-task transfer run or model promotion was performed.

Provenance:

- Whole data hash: `495b41bc8b03c916158ab4613077c272dca17e02c1daa9505ea91b21496371ed`.
- Encoded training hash: `5dd393d95aba6fcaeeb0264013ecb6d8a4942c245f5a701f8d8c441e96e16c1f`.
- Initial model hash: `bc4ae38649deda046482b1d8eb70b369766cd2a848b905ba44bede75285629c3`.
- Trained model hash: `02c56aae4f493ffb518fce6c04d3783126d8ed7ba8b25a20c421dee3a217d8f5`.
- Probe source hash: `d2b862eea2ef254e6cb8642c7b4c010c53427cf8550d3709c7ce01533c8a133b`.

The original model object was retained and both evaluation passes preserved
weights. Root verification passed eight new pure ERT tests, 21 existing
copy/initialization/loss-mask ERT tests, strict compilation of both new files,
and `git diff --check`. The full service/standalone suite was not rerun for
this isolated example addition. The implementation was delegated to the
lightweight model; dataset reproduction, review, and real-GPU verification
were performed independently by the root agent.
