;;; spec-test.el --- self-speculative decoding is lossless  -*- lexical-binding: t; -*-
;; The crucial property of speculative decoding: the emitted stream is EXACTLY
;; what plain greedy would produce, regardless of the draft head's quality.  We
;; check this on an UNTRAINED (random) model -- so it needs no GPU and isolates
;; the acceptance logic -- across several seeds, and report the accept rate.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/spec-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-spec)

(defvar sp--fail 0)
(defun sp--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS" (progn (setq sp--fail (1+ sp--fail)) "FAIL")) (or extra ""))))
(defun sp--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun sp--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4)
       (maxseq 48) (nsteps 24) (prompt '(0 3 1 4 2))
       (mkblk (lambda (s0) (list :ln1g (sp--ones dim) :wq (sp--t (list dim dim) (+ s0 1) sc) :bq (sp--t (list dim) (+ s0 11) 0.1)
                                 :wk (sp--t (list kvdim dim) (+ s0 2) sc) :bk (sp--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (sp--t (list kvdim dim) (+ s0 3) sc) :bv (sp--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (sp--t (list dim dim) (+ s0 4) sc) :bo (sp--t (list dim) (+ s0 14) 0.1) :ln2g (sp--ones dim)
                                 :wg (sp--t (list ff dim) (+ s0 5) sc) :bg (sp--t (list ff) (+ s0 15) 0.1)
                                 :wu (sp--t (list ff dim) (+ s0 6) sc) :bu (sp--t (list ff) (+ s0 16) 0.1)
                                 :wd (sp--t (list dim ff) (+ s0 7) sc) :bd (sp--t (list dim) (+ s0 17) 0.1)))))
  (dolist (seed '(1 2 3))
    (let* ((wte (sp--t (list vocab dim) (+ seed 50) sc)) (lnfg (sp--ones dim)) (bh (sp--t (list vocab) (+ seed 60) 0.1))
           (w2 (sp--t (list vocab dim) (+ seed 70) sc)) (b2 (sp--t (list vocab) (+ seed 80) 0.1))
           (blocks (list (funcall mkblk (* seed 100)) (funcall mkblk (* seed 200))))
           (plain (nl-llm-greedy prompt nsteps blocks wte lnfg bh heads kvh dim vocab maxseq))
           (sr (nl-llm-spec-greedy prompt nsteps blocks wte lnfg bh w2 b2 heads kvh dim vocab maxseq))
           (spec (car sr)) (rounds (cdr sr)))
      (sp--ck (format "seed %d: speculative == plain greedy" seed) (equal spec plain)
              (format "%d toks / %d rounds = %.2f tok/fwd" nsteps rounds (/ (float nsteps) rounds)))))
  (princ (format "NL-LLM-SPEC %s (%d failures)\n" (if (= sp--fail 0) "ALL-PASS" "HAS-FAILURES") sp--fail))
  (kill-emacs (if (= sp--fail 0) 0 1)))
;;; spec-test.el ends here
