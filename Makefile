# nelisp-llm -- experiment repo for small LMs on the nelisp-photon substrate.
EMACS ?= emacs
PHOTON ?= ../nelisp-photon/lisp

.PHONY: test compile clean train train-modern train-modern-full gpu-test gpu-train-test bench-gpu bench-gpu-train

test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/arch-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/attn-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/moe-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/block-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/autograd-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/gpu-train-test.el

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

clean:
	rm -f lisp/*.elc
