# nelisp-llm -- experiment repo for small LMs on the nelisp-photon substrate.
EMACS ?= emacs
PHOTON ?= ../nelisp-photon/lisp

.PHONY: test compile clean train

test:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/arch-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/attn-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/moe-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/block-test.el
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l test/autograd-test.el

compile:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) \
	  --eval '(setq byte-compile-error-on-warn t)' \
	  -f batch-byte-compile lisp/nl-llm-arch.el

train:
	$(EMACS) -Q --batch -L lisp -L $(PHOTON) -l examples/train-open.el

clean:
	rm -f lisp/*.elc
