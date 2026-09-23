# Looped transformers after Astra: what this repo does not have yet

Survey, 2026-09-23. Not a design doc: nothing here is scheduled. It exists so
the next design doc starts from what is actually published and from what is
already implemented, instead of from the press.

## What is known about GPT-6 Astra

OpenAI released Astra on 2026-09-03 and framed it as the arrival of AGI. The
architecture is **not disclosed**. What the company itself states is narrow:
the model is trained to reason through reinforcement learning, it produces a
long internal chain of thought before answering, and it was pretrained on more
than 100,000 GPUs. The system card cites only safety and evaluation papers.

The "breakthrough" being discussed is **recurrent depth**, also called looped
transformers: reuse a block of layers several times so effective depth grows
without new parameters. That attribution comes from The Information
(2026-09-01) citing anonymous sources, not from OpenAI. Chief scientist Jakub
Pachocki's public response addressed depth rather than confirming the design —
"the depth of the computation graph for our present frontier models, including
Astra, is within a factor of two of GPT-4" — and the system card says the drop
in chain-of-thought monitorability is **not** attributed to an architecture
change.

So the honest framing for this repo: looping is an active research direction
with real published results, and Astra is at best weak evidence about it. What
follows is the published work, sorted by whether we have it.

## Already implemented here

| Technique | Paper | Where |
|---|---|---|
| Recurrent depth: prelude / shared core / coda, random R, truncated BPTT, zero-shot KL adaptive exit | Geiping et al., arXiv 2502.05171 (Huginn) | `lisp/nl-llm-recur.el`, design doc 07 |
| Continuous latent thoughts fed back as embeddings | Hao et al., arXiv 2412.06769 (Coconut) | `lisp/nl-llm-coconut.el`, design doc 06 |

The adaptive exit we have is *zero-shot*: iterate, run the coda, stop when the
output distribution stops moving (KL below a threshold), one decision for the
whole sequence. Everything below is missing.

## Missing, in the order worth doing

### 1. A learned exit gate, per token (Ouro)

*Zhu et al., "Scaling Latent Reasoning via Looped Language Models", arXiv
2510.25741.* Each step emits a halting probability from the final hidden
state, `lambda_t = sigmoid(Linear(h_t))`, which induces an exit distribution
over steps through the survival product. Stage I trains the model against the
step-weighted loss minus an entropy term (equivalently a KL to a uniform prior
over steps); Stage II trains the gate alone from a detached per-token loss
improvement `I = max(0, L_{t-1} - L_t)` turned into a soft label. Inference
exits at the first step whose cumulative exit probability passes a threshold q.

Why first: it replaces our threshold-and-hope exit with a trained one, it is
per token rather than per sequence, and it is the piece with fully published
formulas. It also gives depth allocation we can *measure* — does the gate spend
more steps on harder inputs?

Cost here: new autograd ops. Stage II alone needs a fused sigmoid + binary
cross-entropy-with-logits (one op, gradient `sigmoid(z) - w`); the full Stage I
additionally needs a per-row cross-entropy that returns a vector rather than a
scalar mean, and a log for the entropy term. Each needs a gradient check.

### 2. Per-token recursion depth by routing (Mixture-of-Recursions)

*Bae et al., arXiv 2507.10524.* A lightweight router assigns each token its own
number of recursions, with two caching strategies: recursion-wise caching,
where attention at depth d only runs over tokens still active at depth d, and
recursive KV sharing, where every depth reuses the first recursion's KV.

This is design doc 07's own follow-up ("per-token adaptive exit -- needs the
cache bookkeeping"). It subsumes item 1's inference-time behaviour but with a
router instead of a halting gate, and it is the one that actually saves compute
rather than just deciding when to stop.

### 3. Compute-matched looping on an MoE (SMELT)

*Wang et al., "SMELT: Scaling Laws for Compute-Matched MoE Looped
Transformers", arXiv 2609.01343.* Loop the middle half of the layers twice
while matching per-token FLOPs, non-embedding parameters **and** KV cache
against an unlooped baseline; they report 6.8–18.0% of training FLOPs saved on
the compute-optimal frontier, largest gains on code, and less attention-sink
behaviour in the looped layers.

We already have MoE (`lisp/nl-llm-moe.el`) and a stacked model, so the recipe
is close. The hard part is not the loop, it is the matching: a comparison that
does not hold all three budgets fixed measures nothing, which is exactly the
paper's complaint about earlier work.

### 4. KV cache sharing across iterations at decode

Huginn section 5, and Ouro's measurement that keeping only the last step's KV
gives a 4x cache reduction with little loss (averaging across steps also
works; prefill still needs the full caches). Design doc 07 lists this as a
follow-up on top of `nl-llm-decode`. Cheap relative to the others and it is
what makes deep recurrence affordable at inference.

### 5. On-device training of the recurrent stack

The resident-autograd builder (`nl-llm-gpu-ag.el`) trains the modern block on
the GPU, but the recurrent model still trains on the CPU path. Doc 07's first
follow-up: run the no-grad prefix as plain forward dispatches and unroll only
the last k iterations on the graph.

## Read this before investing

Three results argue against treating depth recurrence as a reasoning engine:

- **Recurrence is worth less than unique layers.** Schwethelm, Rueckert and
  Kaissis, arXiv 2604.21106, fit a recurrence-equivalence exponent of 0.46:
  between 0 (no capacity gain) and 1 (a shared recurrence equals a fresh
  block). A 410M model looped 4 times matched a 580M unlooped model while
  costing the training compute of a 1B one.
- **The latent steps do not look like hidden chain of thought.** Lu et al.,
  arXiv 2507.02199, probed Huginn-3.5B with logit lens and coda lens and found
  little interpretable latent CoT, probe results that depend heavily on layer
  index and decoding method, and only marginal gains from more recurrence
  compared with externalizing the steps.
- **A compressed loop is bounded by its state.** Zhang, arXiv 2605.30757: a
  loop whose recurrent state is compressed cannot decide problems that are
  P-complete under logspace reductions, while polynomial-length chain of
  thought can; full sequence-state loops sit in a different, memory-rich
  regime. The lever is the size of the carried state, not the number of loops.

Taken together: build these for **compute allocation** — spend fewer steps on
easy tokens, more on hard ones, and cache less at decode — and measure that.
Do not claim reasoning depth. Our own doc 07 already reports the honest version
at toy scale: held-out loss did not improve monotonically with R, and no R in
{1,2,4,8,16} was a clear winner.

## Out of scope for this repo

Astra's confirmed ingredients — reinforcement learning on long-horizon tasks,
computer use through screen and cursor, pretraining across 100,000+ GPUs — are
not things a single-machine Elisp lab reproduces. The architecture items above
are.

## Sources

- OpenAI, "GPT-6 Astra" (2026-09-03) and its system card at
  deploymentsafety.openai.com/gpt-6-astra
- The Information, 2026-09-01, via Fortune 2026-09-03 for the recurrent-depth
  report and the reaction to it; Sebastian Raschka, "GPT-6 Astra, Looped
  Transformers, and Hidden Reasoning" for Pachocki's statement and the paper
  trail
- arXiv: 1807.03819 (Universal Transformers, the origin of adaptive halting),
  2412.06769, 2502.05171, 2507.02199, 2507.10524, 2510.25741, 2604.21106,
  2605.30757, 2609.01343
