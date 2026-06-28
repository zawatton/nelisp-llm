;;; agent-model-test.el --- P4: real model wired as the agent policy  -*- lexical-binding: t; -*-
;; Pins the constrained-decoding bridge that turns an nelisp-llm model into a
;; structurally-valid agent policy (CPU decode, no GPU):
;;   1. constrained generation under a grammar yields an exact, parseable action
;;      regardless of the model (mock step-fn).
;;   2. an UNTRAINED random model, driven through the grammar, still emits a valid
;;      Elisp action every step -- the grammar guarantees structure, the model
;;      only picks content.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-model-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)

(defvar am--fail 0)
(defun am--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq am--fail (1+ am--fail)) "FAIL")) (or extra ""))))
(defun am--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun am--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

;; --- (1) constrained generation is exact + parseable (mock model) -----------
(let* ((vocab nl-llm-agent-char-vocab)
       ;; mock logits: favor 'z' over all others; step-fn ignores the id
       (logits (let ((v (make-vector vocab -1.0))) (aset v (nl-llm-agent--char->id ?z) 9.0) v))
       (step (lambda (_id) logits))
       (out (nl-llm-agent-constrained-generate logits step (nl-llm-agent-grammar-message 5 "xyz")))
       (act (nl-llm-agent--parse out)))
  (am--ck "constrained generate is exact" (equal out "```elisp\n(message \"zzzzz\")\n```") (format "%S" out))
  (am--ck "constrained output parses to an Elisp action" (eq (car act) 'elisp))
  (am--ck "the Elisp action evaluates" (equal (nl-llm-agent--eval-elisp (nth 1 act)) "\"zzzzz\"")))

;; --- (2) an untrained random model still emits valid actions every step -----
(let* ((dim 32) (heads 4) (kvh 2) (ff 48) (vocab nl-llm-agent-char-vocab) (hd (/ dim heads)) (kvdim (* kvh hd))
       (mkblk (lambda (s0) (list :ln1g (am--ones dim) :wq (am--t (list dim dim) (+ s0 1) 0.4) :bq (am--t (list dim) (+ s0 11) 0.1)
                                 :wk (am--t (list kvdim dim) (+ s0 2) 0.4) :bk (am--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (am--t (list kvdim dim) (+ s0 3) 0.4) :bv (am--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (am--t (list dim dim) (+ s0 4) 0.4) :bo (am--t (list dim) (+ s0 14) 0.1) :ln2g (am--ones dim)
                                 :wg (am--t (list ff dim) (+ s0 5) 0.4) :bg (am--t (list ff) (+ s0 15) 0.1)
                                 :wu (am--t (list ff dim) (+ s0 6) 0.4) :bu (am--t (list ff) (+ s0 16) 0.1)
                                 :wd (am--t (list dim ff) (+ s0 7) 0.4) :bd (am--t (list dim) (+ s0 17) 0.1))))
       (model (list :blocks (list (funcall mkblk 100) (funcall mkblk 200))
                    :wte (am--t (list vocab dim) 1 0.4) :lnfg (am--ones dim) :bh (am--t (list vocab) 19 0.1)
                    :dim dim :heads heads :kvh kvh))
       (policy (nl-llm-agent-model-policy model (nl-llm-agent-grammar-message 6) 512))
       (res (nl-llm-agent-run "say something" policy
              :system "Emit one action." :max-steps 2 :workdir default-directory))
       (assistants (delq nil (mapcar (lambda (m) (and (eq (car m) 'assistant) (cdr m))) (plist-get res :messages))))
       (all-valid (and assistants (cl-every (lambda (s) (eq (car (nl-llm-agent--parse s)) 'elisp)) assistants))))
  (am--ck "random model drove >=2 steps" (>= (length assistants) 2) (format "%d assistant turns" (length assistants)))
  (am--ck "every step is a valid Elisp action (constrained)" all-valid
          (format "sample=%S" (and assistants (car (last (split-string (car assistants) "\n" t)))))))

(princ (format "NL-LLM-AGENT-MODEL %s (%d failures)\n" (if (= am--fail 0) "ALL-PASS" "HAS-FAILURES") am--fail))
(kill-emacs (if (= am--fail 0) 0 1))
;;; agent-model-test.el ends here
