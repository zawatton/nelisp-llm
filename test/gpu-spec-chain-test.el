;;; gpu-spec-chain-test.el --- deep-tree (multi-head) chain speculative decode  -*- lexical-binding: t; -*-
;; Chain-draft speculative decode: from one hidden, multiple MTP heads draft a
;; depth-long token chain, verified in ONE GPU tree-verify forward, accepting the
;; longest greedy-matching prefix.  The emitted stream must be EXACTLY plain greedy
;; regardless of head quality -- checked here on an untrained (random) model with
;; head2 + head3 (depth 3).  Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-spec-chain-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-spec)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)

(defvar sc--fail 0)
(defun sc--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS" (progn (setq sc--fail (1+ sc--fail)) "FAIL")) (or extra ""))))
(defun sc--t (shape seed s) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* s 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun sc--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-SPEC-CHAIN SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (s 0.4)
       (maxseq 48) (nsteps 16) (maxdepth 6) (prompt '(0 3 1 4 2))
       (wte (sc--t (list vocab dim) 1 s)) (lnfg (sc--ones dim)) (bh (sc--t (list vocab) 19 0.1))
       (w2 (sc--t (list vocab dim) 70 s)) (b2 (sc--t (list vocab) 80 0.1))
       (w3 (sc--t (list vocab dim) 71 s)) (b3 (sc--t (list vocab) 81 0.1))
       (mkblk (lambda (s0) (list :ln1g (sc--ones dim) :wq (sc--t (list dim dim) (+ s0 1) s) :bq (sc--t (list dim) (+ s0 11) 0.1)
                                 :wk (sc--t (list kvdim dim) (+ s0 2) s) :bk (sc--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (sc--t (list kvdim dim) (+ s0 3) s) :bv (sc--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (sc--t (list dim dim) (+ s0 4) s) :bo (sc--t (list dim) (+ s0 14) 0.1) :ln2g (sc--ones dim)
                                 :wg (sc--t (list ff dim) (+ s0 5) s) :bg (sc--t (list ff) (+ s0 15) 0.1)
                                 :wu (sc--t (list ff dim) (+ s0 6) s) :bu (sc--t (list ff) (+ s0 16) 0.1)
                                 :wd (sc--t (list dim ff) (+ s0 7) s) :bd (sc--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (tables (nl-llm-gpu-rope-tables maxseq hd))
       (plain (nl-llm-greedy prompt nsteps blocks wte lnfg bh heads kvh dim vocab maxseq))
       (cr (nl-llm-gpu-spec-chain-decode prompt nsteps blocks wte lnfg bh
                                         (list (cons w2 b2) (cons w3 b3)) heads kvh dim vocab maxseq tables maxdepth))
       (chain (car cr)) (forwards (cdr cr)))
  (sc--ck "depth-3 chain spec == plain greedy" (equal chain plain)
          (format "%d toks / %d verifies = %.2f tok/fwd" nsteps forwards (/ (float nsteps) forwards)))
  ;; lossless sampling chain (rejection rule over the tree verify): deterministic
  ;; given a seed, in-vocab, and reproducible -- distributional losslessness is
  ;; pinned by spec-test (rejection-d ~ target).
  (random "nl-llm-chain-sample-seed")
  (let ((s1 (car (nl-llm-gpu-spec-chain-sample-decode prompt nsteps blocks wte lnfg bh
                  (list (cons w2 b2) (cons w3 b3)) heads kvh dim vocab maxseq tables maxdepth 0.8 6))))
    (random "nl-llm-chain-sample-seed")
    (let ((s2 (car (nl-llm-gpu-spec-chain-sample-decode prompt nsteps blocks wte lnfg bh
                    (list (cons w2 b2) (cons w3 b3)) heads kvh dim vocab maxseq tables maxdepth 0.8 6)))
          (inv t))
      (dolist (tk s1) (unless (and (>= tk 0) (< tk vocab)) (setq inv nil)))
      (sc--ck "chain-sample: in-vocab + deterministic per seed" (and inv (= (length s1) nsteps) (equal s1 s2)))))
  (nl-llm-gpu-disable)
  (princ (format "NL-LLM-GPU-SPEC-CHAIN %s (%d failures)\n" (if (= sc--fail 0) "ALL-PASS" "HAS-FAILURES") sc--fail))
  (kill-emacs (if (= sc--fail 0) 0 1)))
;;; gpu-spec-chain-test.el ends here
