# nelisp-llm -- experiment repo for small LMs on the nelisp-photon substrate.
EMACS ?= emacs
PHOTON ?= ../nelisp-photon/lisp

.PHONY: test compile clean train train-modern train-modern-full gpu-test gpu-train-test gpu-ag-test gpu-block-test gpu-moe-test gpu-stack-test gpu-window-test gpu-gather-test gpu-adam-test gpu-tie-test gpu-sched-test bench-gpu bench-gpu-train bench-ondevice train-stacked-gpu train-corpus-gpu generate-gpu train-full-gpu checkpoint-gpu train-big-gpu stream-decode spec-decode bitnet-model bench-dp4a spec-chain integrated-decode bench-longctx agent-demo agent-model-demo agent-improve-demo agent-code-demo agent-sandbox-demo

test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/arch-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/attn-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/moe-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/block-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/autograd-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/sample-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/decode-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/stream-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/spec-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/dropout-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/ckpt-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-train-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-ag-test.el
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
	$(EMACS) -Q --batch -L lisp -l test/agent-sandbox-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-model-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-improve-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/agent-code-test.el

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

clean:
	rm -f lisp/*.elc
