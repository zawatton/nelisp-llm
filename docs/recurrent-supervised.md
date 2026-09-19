# Recurrent supervised adapter

`lisp/nl-llm-agent-recur-supervised.el` is an opt-in CPU adapter that applies
the existing prompt/completion supervision boundary to the recurrent-depth
model in `lisp/nl-llm-recur.el`. It is an experimental adapter, not a provider,
artifact, GPU, or service integration.

The public entry points are:

```elisp
(require 'nl-llm-agent-recur-supervised)
(nl-llm-agent-recur-supervised-loss
 model examples
 :r 2 :k 2 :s0-seed 1 :tokenizer "utf8-byte-v1")
(nl-llm-agent-recur-supervised-train
 model examples
 :r 2 :k 2 :s0-seed 1 :tokenizer "utf8-byte-v1"
 :lr 0.01 :epochs 1)
```

The adapter validates the complete model, tokenizer, examples, and keyword
set before training. `:r` is the recurrent iteration count (default 2,
1..32), `:k` is the truncated-backpropagation suffix (default `:r`, 1..`:r`),
`:s0-seed` selects a deterministic prefix-stable initial latent state, and
`:tokenizer` defaults to `utf8-byte-v1`. Training uses CPU SGD with `:lr` in
(0, 1] (default 0.01) and `:epochs` in 1..32 (default 1). Prompt tokens remain
context; only completion target rows contribute to the weighted loss.

The recurrent model has separate prelude, shared recurrent core, and coda
stacks. The `:k` boundary controls recurrent tape retention, while the
completion boundary controls which target rows are scored; changing one does
not silently change the other. Caller models and examples are not rewritten
by loss evaluation, and training uses a local tape. This adapter does not
publish models, write checkpoints, alter provider configuration, or claim
practical COPY/agent capability.

Pure checks:

```sh
make test-agent-recur-supervised
```

The test fixture is deliberately tiny (byte vocabulary 256, dim 4, FF 8,
one head, one prelude/core/coda block) and exercises deterministic loss,
weighted completion scoring, CPU training, validation-before-mutation, and
the `:r`/`:k` forwarding boundary. It is a CPU experimental check only; no
GPU or service benchmark result is implied.
