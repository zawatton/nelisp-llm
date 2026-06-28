;;; nl-llm-spec.el --- self-speculative (MTP) greedy decoding  -*- lexical-binding: t; -*-

;; Speculative decoding (Leviathan et al. 2023) with Medusa/MTP-style heads
;; (Cai et al. 2024): the model carries an extra look-ahead head H2 that predicts
;; the token TWO ahead from the same hidden.  Each round drafts the next token t1
;; from the main head and a speculative token d from H2, then verifies in one
;; target forward; on a hit the round emits two tokens, otherwise one.  The
;; emitted stream is EXACTLY plain greedy regardless of H2's quality -- H2 only
;; changes how many tokens land per forward (the speedup).  This losslessness is
;; the property test/spec-test.el pins.
;;
;; The verify of [t1, d] is one batched target forward; here it is done as two
;; sequential KV-decode feeds but counted as one round, matching the batched cost.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-arch)
(require 'nl-llm-decode)   ; nl-llm-dcache-new, nl-llm-decode-h

(defun nl-llm-spec-argmax (data off n)
  "Index of the max of DATA[OFF..OFF+N)."
  (let ((bi 0) (bv (aref data off)) (i 1))
    (while (< i n) (when (> (aref data (+ off i)) bv) (setq bv (aref data (+ off i)) bi i)) (setq i (1+ i)))
    bi))

(defun nl-llm--head-argmax (h w bias vocab)
  "Argmax of the head W,BIAS applied to hidden H (1 x dim)."
  (nl-llm-spec-argmax (photon-tensor-data (photon-tensor-linear h w bias)) 0 vocab))

;;;###autoload
(defun nl-llm-greedy (prompt nsteps blocks wte lnfg bh heads kvh dim vocab maxseq)
  "Plain greedy decode: feed PROMPT (a list of ids), then generate NSTEPS tokens
with the main tied head.  Returns the list of generated ids."
  (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)) (h nil) (out nil))
    (dolist (tk prompt) (setq h (nl-llm-decode-h tk blocks caches wte lnfg dim)))
    (dotimes (_ nsteps)
      (let ((g (nl-llm--head-argmax h wte bh vocab)))
        (push g out)
        (setq h (nl-llm-decode-h g blocks caches wte lnfg dim))))
    (nreverse out)))

;;;###autoload
(defun nl-llm-spec-greedy (prompt nsteps blocks wte lnfg bh w2 b2 heads kvh dim vocab maxseq)
  "Self-speculative greedy decode with MTP look-ahead head W2,B2 (predicting the
token two ahead).  Returns (TOKENS . ROUNDS) where TOKENS is the generated id
list (identical to `nl-llm-greedy') and ROUNDS is the number of verify forwards
\(so NSTEPS/ROUNDS is the mean tokens accepted per forward = the speedup)."
  (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
        (h nil) (out nil) (n 0) (rounds 0))
    (dolist (tk prompt) (setq h (nl-llm-decode-h tk blocks caches wte lnfg dim)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (nl-llm--head-argmax h wte bh vocab))     ; guaranteed greedy next
            (d  (nl-llm--head-argmax h w2 b2 vocab)))      ; MTP draft for the token after t1
        (push t1 out) (setq n (1+ n))
        (let* ((h1 (nl-llm-decode-h t1 blocks caches wte lnfg dim))    ; verify: feed t1
               (true2 (nl-llm--head-argmax h1 wte bh vocab)))
          (if (and (= d true2) (< n nsteps))
              (progn (push d out) (setq n (1+ n))         ; draft correct: accept, keep d in cache
                     (setq h (nl-llm-decode-h d blocks caches wte lnfg dim)))
            (setq h h1)))))                               ; reject (or no room): d never entered the cache
    (cons (nreverse out) rounds)))

(provide 'nl-llm-spec)
;;; nl-llm-spec.el ends here
