# Recurrent artifact provider

`nl-llm-agent-recur-artifact.el` is a bounded, data-only interchange format
for recurrent-depth models.  It is separate from training checkpoints:
imports allocate fresh PAV parameters with zero gradients and do not carry
optimizer state or resume semantics.

```elisp
(let* ((saved (nl-llm-agent-recur-artifact-save
               "./models/recur-g1.sexp" model
               :tokenizer "utf8-byte-v1" :r 2 :s0-seed 104729 :step 16))
       (provider
        (nl-llm-agent-recur-provider
         "recur"
         (list (list :id "g1"
                     :path (plist-get saved :path)
                     :sha256 (plist-get saved :sha256)
                     :grammar grammar
                     :maxseq 128)))))
  (nl-llm-agent-provider-register registry provider))
```

The provider uses the existing `nl-llm-agent-provider` session protocol.
Catalog entries contain only public routing and capacity metadata; artifact
paths, digests, and trusted grammar functions remain private.  Opening a model
verifies the supplied SHA-256 and loads a fresh recurrent bundle.  Each
completion recomputes the complete token prefix with an explicit deterministic
`S0`; the recurrent decoder is therefore CPU-only and does not use a KV cache.
The provider refuses an active GPU dispatch and never changes backend function
bindings.  `:maxseq` counts tokenizer tokens, including the rendered message
prefix and generated completion.

On Emacs, the provider asks the shared `nl-llm-inference-runtime` to prepare its
in-memory compilation path before each forward.  The runtime owns one bounded
target group, including the three tensor kernels used by recurrent inference:
`photon-tensor-matmul`, `photon-tensor-softmax-rows`, and
`photon-tensor-scale`.  Preparation honors the configured source, byte-code, or
automatic mode and refreshes only definitions still owned by that runtime.
The provider checks GPU dispatch both before and after preparation, so a hook
that swaps a backend during preparation is rejected before forward computation.
This is in-memory compilation only: full-prefix recurrent computation remains
the same, and no separate compiler state or persistent artifact is created.

The artifact reader checks the format/family, tokenizer/vocabulary, model
geometry, tensor shapes, finite values, trailing data, file digest, and scalar
and byte bounds before allocating model parameters.  Saved artifacts are
immutable and are not automatically adopted or published.  Grammar callbacks
are trusted host configuration and are used only by constrained decoding;
artifact files contain no executable grammar or model code.

This is an experimental recurrent-depth adapter for the CPU provider boundary.
It is not a capability benchmark or a claim of practical generalization.  GPU
inference, the packaged standalone CLI, promotion, and background/service
configuration wiring are outside this adapter's scope.
