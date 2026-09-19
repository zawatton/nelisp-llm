# nelisp-llm -- experiment repo for small LMs on the nelisp-photon substrate.
EMACS ?= emacs
PHOTON ?= ../nelisp-photon/lisp
NELISP ?= ../nelisp/target/nelisp

.PHONY: test compile clean train train-modern train-modern-full gpu-test gpu-train-test gpu-ag-test gpu-block-test gpu-moe-test gpu-stack-test gpu-window-test gpu-gather-test gpu-adam-test gpu-tie-test gpu-sched-test agent-evolve-gpu-test bench-gpu bench-gpu-train bench-ondevice train-stacked-gpu train-corpus-gpu generate-gpu train-full-gpu checkpoint-gpu train-big-gpu stream-decode spec-decode bitnet-model bench-dp4a spec-chain integrated-decode bench-longctx agent-demo agent-model-demo agent-improve-demo agent-code-demo agent-sandbox-demo agent-tasks-demo agent-gpu-finetune-demo agent-ondevice-demo

test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/qwen-tokenizer-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/head-dim-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/rope-style-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-header-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-load-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-lora-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-backward-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-block-backward-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-train-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/distill-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/arch-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/attn-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/moe-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/block-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/autograd-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/lora-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/sample-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/decode-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/decode-capacity-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/inference-runtime-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-inference-runtime-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-recur-runtime-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/stream-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/spec-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/dropout-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/ckpt-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/lora-ckpt-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/coconut-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/recur-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-train-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-ag-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-lora-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-block-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-moe-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-stack-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-window-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-gather-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-adam-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-tie-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-sched-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-clip-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-decode-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-resume-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-batch-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-stream-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-bitnet-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-bitpack-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-dp4a-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-bitnet-dp4a-model-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-bitnet-wte-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-paged-spike-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-paged-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-paged-v-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-paged-cow-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-tree-attn-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-tree-verify-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-spec-chain-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/integrated-test.el
	$(EMACS) -Q --batch -L lisp -l test/agent-test.el
	$(EMACS) -Q --batch -L lisp -l test/agent-provider-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-native-provider-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-artifact-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -L ../nelisp-agent/lisp -l test/agent-recur-artifact-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-action-grammar-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-action-artifact-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-training-checkpoint-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-plan-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-checkpoint-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-resume-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-evolve-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-evolve-gpu-test.el
	$(EMACS) -Q --batch -L lisp -l test/agent-openai-provider-test.el
	$(EMACS) -Q --batch -L lisp -l test/agent-openai-http-test.el
	$(EMACS) -Q --batch -L lisp -l test/agent-sandbox-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-model-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-tokenizer-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-unicode-model-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-unicode-pipeline-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-improve-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-masked-loss-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-supervised-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-recur-supervised-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-supervised-evolve-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-masked-gpu-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-compact-gpu-seed-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-compact-ondevice-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-forward-parity-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-gradient-parity-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/literal-copy-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-curriculum-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-diversity-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-teacher-forcing-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-architecture-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-optimization-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/copy-initialization-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-initialization-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-epoch-shuffle-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-loss-masks-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/evolve-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/evolve-promotion-gate-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/evolve-queue-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/evolve-async-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-code-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-tasks-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-gpu-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-ondevice-test.el

.PHONY: test-agent-unicode test-agent-supervised test-agent-recur-supervised test-agent-recur-artifact test-agent-recur-runtime test-agent-actions test-agent-compact test-agent-forward-parity test-agent-gradient-parity test-literal-copy test-agent-completion-checkpoint test-agent-completion-resume test-agent-completion-resume-gpu test-evolve-promotion-gate

test-agent-recur-artifact:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -L ../nelisp-agent/lisp -l test/agent-recur-persistence-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -L ../nelisp-agent/lisp -l test/agent-recur-artifact-test.el

test-agent-recur-runtime:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-recur-runtime-test.el

test-evolve-promotion-gate:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/evolve-promotion-gate-test.el

test-agent-completion-checkpoint:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-plan-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-checkpoint-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-training-checkpoint-test.el

# Pure plan/checkpoint coverage for plan-bound completion resume.  The GPU
# target below is deliberately separate from both this target and `test`.
test-agent-completion-resume: test-agent-completion-checkpoint
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-resume-test.el

test-agent-completion-resume-gpu:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-completion-resume-gpu-test.el

.PHONY: test-copy-initialization
.PHONY: test-agent-supervised-evolve
test-agent-supervised-evolve:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-supervised-evolve-test.el

test-copy-initialization:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-initialization-test.el

.PHONY: test-agent-initialization
test-agent-initialization:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-initialization-test.el

.PHONY: test-agent-epoch-shuffle
test-agent-epoch-shuffle:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-epoch-shuffle-test.el

.PHONY: test-agent-loss-masks test-agent-loss-masks-gpu
test-agent-loss-masks:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-loss-masks-test.el

test-agent-loss-masks-gpu:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-loss-masks-gpu-test.el

test-agent-gradient-parity:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-gradient-parity-test.el

test-literal-copy:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/literal-copy-test.el

.PHONY: test-copy-curriculum
test-copy-curriculum:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-curriculum-test.el

.PHONY: test-copy-diversity test-copy-teacher-forcing test-copy-architecture test-copy-optimization
test-copy-diversity:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-diversity-test.el

test-copy-teacher-forcing:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-teacher-forcing-test.el

test-copy-architecture:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-architecture-test.el

test-copy-optimization:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/copy-optimization-test.el

test-agent-forward-parity:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-forward-parity-test.el

# Manual long-context numerical diagnostic; not part of the quick default run.
.PHONY: test-agent-long-forward-parity
test-agent-long-forward-parity:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-long-forward-parity-test.el

test-agent-compact:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-compact-gpu-seed-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-compact-ondevice-test.el

test-agent-actions:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-action-grammar-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-action-artifact-test.el

test-agent-supervised:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-masked-loss-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-supervised-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-masked-gpu-test.el

test-agent-recur-supervised:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-recur-supervised-test.el

test-agent-unicode:
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-tokenizer-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-unicode-model-test.el
	$(EMACS) -Q --batch --eval '(setq load-prefer-newer t)' -L lisp -L $(PHOTON) -l test/agent-unicode-pipeline-test.el
	$(NELISP) --load test/agent-tokenizer-test.el
	$(NELISP) --load test/agent-unicode-model-test.el

compile:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) \
	  --eval '(setq byte-compile-error-on-warn t)' \
	  -f batch-byte-compile lisp/nl-llm-arch.el

train:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-open.el

train-modern:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-modern.el

train-modern-full:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-modern-full.el

gpu-test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-test.el

gpu-train-test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-train-test.el

agent-evolve-gpu-test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-evolve-gpu-test.el

bench-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bench-gpu.el

bench-gpu-train:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bench-gpu-train.el

bench-ondevice:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bench-ondevice.el

train-stacked-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-stacked-gpu.el

train-corpus-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-corpus-gpu.el

generate-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/generate-gpu.el

train-full-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-full-gpu.el

checkpoint-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/checkpoint-gpu.el

train-big-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-big-gpu.el

stream-decode:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/stream-decode.el

spec-decode:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/spec-decode.el

bitnet-model:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bitnet-model.el

bench-dp4a:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bench-dp4a.el

spec-chain:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/spec-chain.el

integrated-decode:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/integrated-decode.el

bench-longctx:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/bench-longctx.el

agent-demo:
	$(EMACS) -Q --batch -L lisp -l examples/agent-demo.el

agent-model-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-model-demo.el

agent-improve-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-improve-demo.el

agent-code-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-code-demo.el

agent-sandbox-demo:
	$(EMACS) -Q --batch -L lisp -l examples/agent-sandbox-demo.el

agent-tasks-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-tasks-demo.el

agent-gpu-finetune-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-gpu-finetune-demo.el

agent-ondevice-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/agent-ondevice-demo.el

recur-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/recur-demo.el

coconut-demo:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/coconut-demo.el

# Suites that finish under the standalone reader (nelisp d145e3c02, one run,
# wall times measured 2026-09-05 with the exp shim and NaN-honest checks):
# arch 1s attn 9s moe 17s block 35s autograd 470s lora 39s sample 32s decode 55s
# stream 875s dropout 4s ckpt 44s lora-ckpt 88s agent 2s; provider, native
# provider, OpenAI provider, and evolve are sub-second.
# Not in this target because they did not finish within 40 minutes under NeLisp
# (every row they did print was a genuine PASS): spec coconut recur agent-model
# agent-improve agent-code agent-tasks.  agent-sandbox is Emacs-only: its
# contract launches an isolated emacs -Q --batch subprocess with a shell timeout.
test-nelisp:
	$(NELISP) --load test/arch-test.el
	$(NELISP) --load test/attn-test.el
	$(NELISP) --load test/moe-test.el
	$(NELISP) --load test/block-test.el
	$(NELISP) --load test/autograd-test.el
	$(NELISP) --load test/lora-test.el
	$(NELISP) --load test/sample-test.el
	$(NELISP) --load test/decode-test.el
	$(NELISP) --load test/decode-capacity-test.el
	$(NELISP) --load test/agent-inference-runtime-test.el
	$(NELISP) --load test/stream-test.el
	$(NELISP) --load test/dropout-test.el
	$(NELISP) --load test/ckpt-test.el
	$(NELISP) --load test/lora-ckpt-test.el
	$(NELISP) --load test/agent-test.el
	$(NELISP) --load test/agent-provider-test.el
	$(NELISP) --load test/agent-native-provider-test.el
	$(NELISP) --load test/agent-openai-provider-test.el
	$(NELISP) --load test/evolve-test.el
	$(NELISP) --load test/evolve-async-test.el

test-nelisp-fast:
	$(NELISP) --load test/arch-test.el
	$(NELISP) --load test/attn-test.el
	$(NELISP) --load test/moe-test.el
	$(NELISP) --load test/block-test.el
	$(NELISP) --load test/sample-test.el
	$(NELISP) --load test/decode-test.el
	$(NELISP) --load test/decode-capacity-test.el
	$(NELISP) --load test/agent-inference-runtime-test.el
	$(NELISP) --load test/dropout-test.el
	$(NELISP) --load test/ckpt-test.el
	$(NELISP) --load test/lora-ckpt-test.el
	$(NELISP) --load test/agent-provider-test.el
	$(NELISP) --load test/agent-native-provider-test.el
	$(NELISP) --load test/agent-openai-provider-test.el
	$(NELISP) --load test/evolve-test.el

.PHONY: coconut-demo recur-demo test-nelisp test-nelisp-fast

clean:
	rm -f lisp/*.elc

# --- Doc 08: weight import (docs/design/08-weight-import.org) -------------
.PHONY: qwen-tokenizer-table test-qwen-tokenizer ollama-provider test-head-dim
.PHONY: qwen-weights-table verify-qwen-weights test-weights-header
.PHONY: qwen-weights-rows test-weights-load test-rope-style
.PHONY: qwen-forward-ref test-weights-forward test-weights-gpu

# DONOR is a HuggingFace model directory holding config.json + tokenizer.json.
# Everything under build/donor/ is donor-derived and gitignored.
DONOR ?= build/donor/qwen3-0.6b
DONOR_REPO ?= Qwen/Qwen3-0.6B
PYLIBS ?= build/donor/pylibs
DONOR_URL = https://huggingface.co/$(DONOR_REPO)/resolve/main

# Fetch the donor's config and tokenizer, export the binary table, and
# regenerate the parity fixtures from the reference tokenizer.  The reference
# library goes into $(PYLIBS) via pip --target, so no virtualenv is needed and
# no system Python packages are touched.
qwen-tokenizer-table:
	mkdir -p $(DONOR) $(PYLIBS)
	cd $(DONOR) && curl -sfL -O $(DONOR_URL)/config.json
	cd $(DONOR) && curl -sfL -O $(DONOR_URL)/tokenizer_config.json
	cd $(DONOR) && curl -sfL -O $(DONOR_URL)/tokenizer.json
	python3 -m pip install -q --disable-pip-version-check --target $(PYLIBS) tokenizers regex
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-tokenizer-export.py $(DONOR) $(DONOR)
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-tokenizer-fixtures.py $(DONOR)/tokenizer.json test/fixtures/qwen-tokenizer.eld

# Donor token-id parity.  Skips cleanly when the table is absent, so a fresh
# clone is not red; run `make qwen-tokenizer-table' once to enable it.
test-qwen-tokenizer:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/qwen-tokenizer-test.el

# Phase 0: a locally served open-weight teacher, no conversion.
# Needs `ollama serve' running and `ollama pull qwen3:4b' done once.
ollama-provider:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/ollama-provider.el

# Attention with a head width decoupled from dim/heads, as Qwen3 has.
test-head-dim:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/head-dim-test.el

# Fetch the donor weights and convert them to the int8 table.  ~1.5 GiB of
# safetensors in, ~571 MiB of table out; both live under the gitignored
# build/donor/.  Needs `make qwen-tokenizer-table' first for config.json and
# the numpy/tokenizers install.
qwen-weights-table:
	mkdir -p $(DONOR) $(PYLIBS)
	cd $(DONOR) && curl -sfL -O $(DONOR_URL)/model.safetensors
	python3 -m pip install -q --disable-pip-version-check --target $(PYLIBS) numpy
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-weights-export.py $(DONOR) $(DONOR)/weights.bin

# Check the table against the donor: bit-exact packing, per-tensor
# dequantization error, and header/payload agreement.  Calibrated by corrupting
# a payload byte and a scale; both trip it.
verify-qwen-weights:
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-weights-verify.py $(DONOR) $(DONOR)/weights.bin

# The Elisp side: the header is one sexp and must `read' without a parser.
test-weights-header:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-header-test.el

# Sample rows straight out of the table with numpy, as the reference the Elisp
# reader is compared against: lanes as integers, scales, and float64 products.
qwen-weights-rows:
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-weights-rows.py $(DONOR)/weights.bin test/fixtures/qwen-weights-rows.eld

# The Elisp reader: unpacked lanes, per-row scales and dequantized values must
# match the exporter exactly.  Calibrated by flipping a payload byte and a
# fixture scale; both trip it.
test-weights-load:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-load-test.el

# The donor's rotation convention (half-split, not interleaved) and Qwen3's
# QK-norm.  Both are silent when ignored, so both are pinned against the
# suite's own reference.
test-rope-style:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/rope-style-test.el

# The numpy reference forward over the SAME int8 table, as a per-layer fixture.
# TOKENS and LAYERS override the defaults ("hello world", 2 layers).
TOKENS ?=
FWD_LAYERS ?= 2
qwen-forward-ref:
	PYTHONPATH=$(PYLIBS) python3 tools/qwen-forward-ref.py $(DONOR)/weights.bin test/fixtures/qwen-forward-ref.eld --layers $(FWD_LAYERS) $(TOKENS)

# Run the imported model forward in pure Elisp and compare layer by layer.
# ~6.5s per layer per token, the slowest suite here; it is the oracle the GPU
# path gets checked against.  Calibrated by using the wrong rotation convention
# (layer 0 red at rel 2.5e-1) and by perturbing one reference value.
#
# Deliberately NOT in the `test' aggregate: at two layers it is 26s, and a
# fixture covering all 28 would be minutes.  Run it when the import or the
# attention conventions change, which is when it has something to say.
test-weights-forward:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-forward-test.el

# Imported linears through the per-row DP4A kernel: uploaded as bytes, run on
# the GPU, and compared against the same W8A8 arithmetic on the CPU.  Needs a
# Vulkan device and the donor table; skips cleanly without either.  Calibrated
# by doubling one row's scale, which it detects.  About ten seconds.
#
# NL_LLM_GPU_E2E=1 adds the end-to-end check: all 28 layers on the GPU, whose
# greedy token must equal the CPU oracle's 12095.  That one takes ~4 minutes.
test-weights-gpu:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-gpu-test.el

# --- Doc 08 Phase 3: adaptation ------------------------------------------
.PHONY: test-weights-lora test-weights-backward test-block-backward test-train
.PHONY: test-distill distill

# A trainable LoRA over a frozen int8 base.  The transpose is pinned by the
# inner-product identity and every gradient against finite differences, since a
# transposed loop reads like the forward one and a wrong scale still descends.
# No donor table needed; the weight is quantized in the test.
test-weights-lora:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-lora-test.el

# The gates on teacher-generated data.  Teacher is injected, so no model runs.
test-distill:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/distill-test.el

# Build a dataset from the local open-weight teacher.  PROMPTS and OUT override
# the defaults; personal prompts belong in a file outside this repository.
# Needs `ollama serve' and a pulled model.
PROMPTS ?= examples/distill-prompts.txt
OUT ?= build/distilled.eld
distill:
	NL_LLM_DISTILL_PROMPTS=$(PROMPTS) NL_LLM_DISTILL_OUT=$(OUT) \
	  $(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/distill-from-teacher.el

# Every vjp between one linear and the next, each against finite differences on
# its own before anything is composed: RMSNorm, the half-split rotation,
# QK-norm, causal GQA, SwiGLU.  Two controls show what the dropped terms cost
# (RMSNorm's mean term, the softmax subtraction).
test-weights-backward:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-backward-test.el

# A gradient through a whole imported block: the taped forward must equal the
# oracle bit for bit, and dL/dA, dL/dB and dL/dx must match finite differences
# with a LoRA on each of the seven roles in turn.  seq is 2, so attention's
# cross-position terms are exercised.
test-block-backward:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-block-backward-test.el

# The training loop: completion-only cross-entropy over an imported model, with
# the stack gradient (head + blocks + loss) checked against finite differences
# and the completion boundary pinned.  A synthetic model, because one real step
# is 5.4 minutes at seq 1 -- see the design doc for the measurement.
test-train:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/weights-train-test.el
