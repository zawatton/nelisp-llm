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

;; --- tree draft (width-k): accept when the true token is in head2's top-k ----
(defun nl-llm-spec--topk-set (data off vocab k)
  "Set of the K largest indices of DATA[OFF..OFF+vocab)."
  (let ((idx nil) (j 0))
    (while (< j vocab) (push (cons (aref data (+ off j)) j) idx) (setq j (1+ j)))
    (setq idx (sort idx (lambda (a b) (> (car a) (car b)))))
    (let ((s nil) (i 0)) (while (and (< i k) idx) (push (cdr (pop idx)) s) (setq i (1+ i))) s)))

;;;###autoload
(defun nl-llm-spec-greedy-tree (prompt nsteps blocks wte lnfg bh w2 b2 heads kvh dim vocab maxseq k)
  "Greedy self-speculative decode with a width-K MTP draft: the speculative 2nd
token is accepted whenever the true greedy token lies in head2's top-K (a depth-1,
width-K draft tree).  Output is still exactly plain greedy; larger K raises the
mean tokens per forward.  Returns (TOKENS . ROUNDS)."
  (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
        (h nil) (out nil) (n 0) (rounds 0))
    (dolist (tk prompt) (setq h (nl-llm-decode-h tk blocks caches wte lnfg dim)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (nl-llm--head-argmax h wte bh vocab))
            (drafts (nl-llm-spec--topk-set (photon-tensor-data (photon-tensor-linear h w2 b2)) 0 vocab k)))
        (push t1 out) (setq n (1+ n))
        (let* ((h1 (nl-llm-decode-h t1 blocks caches wte lnfg dim))
               (true2 (nl-llm--head-argmax h1 wte bh vocab)))
          (if (and (memq true2 drafts) (< n nsteps))
              (progn (push true2 out) (setq n (1+ n))
                     (setq h (nl-llm-decode-h true2 blocks caches wte lnfg dim)))
            (setq h h1)))))
    (cons (nreverse out) rounds)))

;; --- lossless speculative sampling (rejection rule) -------------------------
(defun nl-llm-spec--randf () "Uniform random float in [0,1)." (/ (float (random 1000000)) 1000000.0))

(defun nl-llm-spec-probs (data off vocab temp topk)
  "Probability vector for logits DATA[OFF..) under temperature TEMP and optional
TOPK truncation (0 = no truncation), softmax-normalized."
  (let ((lg (make-vector vocab 0.0)) (j 0))
    (while (< j vocab) (aset lg j (/ (aref data (+ off j)) (max 1e-6 temp))) (setq j (1+ j)))
    (when (and topk (> topk 0) (< topk vocab))
      (let ((thr (car (last (sort (append lg nil) #'>) topk)))) ; kth largest
        (dotimes (i vocab) (when (< (aref lg i) thr) (aset lg i -1.0e30)))))
    (let ((mx -1.0e30)) (dotimes (i vocab) (when (> (aref lg i) mx) (setq mx (aref lg i))))
      (let ((s 0.0)) (dotimes (i vocab) (aset lg i (exp (- (aref lg i) mx))) (setq s (+ s (aref lg i))))
        (dotimes (i vocab) (aset lg i (/ (aref lg i) s)))))
    lg))

(defun nl-llm-spec--cat (p vocab)
  "Sample an index from probability vector P (length VOCAB)."
  (let ((u (nl-llm-spec--randf)) (c 0.0) (i 0) (r (1- vocab)))
    (while (< i vocab) (setq c (+ c (aref p i))) (when (>= c u) (setq r i i vocab)) (setq i (1+ i))) r))

;;;###autoload
(defun nl-llm-spec-rejection (p q vocab)
  "Lossless speculative-sampling step: draft d ~ Q, accept with prob
min(1, P[d]/Q[d]); on reject sample from the normalized residual max(0, P-Q).
The returned token is distributed EXACTLY as P, regardless of Q."
  (let* ((d (nl-llm-spec--cat q vocab))
         (acc (if (> (aref q d) 0.0) (min 1.0 (/ (aref p d) (aref q d))) 1.0)))
    (if (< (nl-llm-spec--randf) acc) d
      (let ((r (make-vector vocab 0.0)) (s 0.0))
        (dotimes (i vocab) (let ((v (- (aref p i) (aref q i)))) (when (> v 0.0) (aset r i v) (setq s (+ s v)))))
        (if (<= s 0.0) (nl-llm-spec--cat p vocab)
          (progn (dotimes (i vocab) (aset r i (/ (aref r i) s))) (nl-llm-spec--cat r vocab)))))))

(defun nl-llm-spec--residual (p q vocab)
  "Sample from the normalized residual max(0, P-Q); fall back to P if it vanishes."
  (let ((r (make-vector vocab 0.0)) (s 0.0))
    (dotimes (i vocab) (let ((v (- (aref p i) (aref q i)))) (when (> v 0.0) (aset r i v) (setq s (+ s v)))))
    (if (<= s 0.0) (nl-llm-spec--cat p vocab)
      (progn (dotimes (i vocab) (aset r i (/ (aref r i) s))) (nl-llm-spec--cat r vocab)))))

;;;###autoload
(defun nl-llm-spec-rejection-d (p q d vocab)
  "Speculative-sampling acceptance for an ALREADY-drafted token D ~ Q against
target P.  Returns (TOKEN . ACCEPTED): TOKEN is D if accepted (prob
min(1, P[d]/Q[d])) else a residual sample; TOKEN ~ P either way.  ACCEPTED (t/nil)
says whether D was kept -- used to stop verifying a draft chain at the first miss."
  (let ((acc (if (> (aref q d) 0.0) (min 1.0 (/ (aref p d) (aref q d))) 1.0)))
    (if (< (nl-llm-spec--randf) acc) (cons d t) (cons (nl-llm-spec--residual p q vocab) nil))))

;;;###autoload
(defun nl-llm-spec-sample-decode (prompt nsteps blocks wte lnfg bh w2 b2 heads kvh dim vocab maxseq temp topk)
  "Self-speculative decode with lossless sampling (temperature TEMP, TOPK): the
emitted stream is distributed exactly as plain temperature/top-k sampling from the
main head, while head2 drafts the look-ahead token (accepted via the rejection
rule).  Returns (TOKENS . ROUNDS)."
  (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
        (h nil) (out nil) (n 0) (rounds 0) (pending nil))
    (dolist (tk prompt) (setq h (nl-llm-decode-h tk blocks caches wte lnfg dim)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (or pending (nl-llm-spec--cat (nl-llm-spec-probs (photon-tensor-data (photon-tensor-linear h wte bh)) 0 vocab temp topk) vocab)))
            (q2 (nl-llm-spec-probs (photon-tensor-data (photon-tensor-linear h w2 b2)) 0 vocab temp topk)))
        (setq pending nil)
        (push t1 out) (setq n (1+ n))
        (let* ((d (nl-llm-spec--cat q2 vocab))
               (h1 (nl-llm-decode-h t1 blocks caches wte lnfg dim))
               (p2 (nl-llm-spec-probs (photon-tensor-data (photon-tensor-linear h1 wte bh)) 0 vocab temp topk))
               (acc (if (> (aref q2 d) 0.0) (min 1.0 (/ (aref p2 d) (aref q2 d))) 1.0)))
          (if (and (< (nl-llm-spec--randf) acc) (< n nsteps))
              (progn (push d out) (setq n (1+ n))                  ; accept draft (~ p2)
                     (setq h (nl-llm-decode-h d blocks caches wte lnfg dim)))
            ;; reject: residual sample (~ p2) becomes next round's first token
            (let ((r (make-vector vocab 0.0)) (s 0.0))
              (dotimes (i vocab) (let ((vv (- (aref p2 i) (aref q2 i)))) (when (> vv 0.0) (aset r i vv) (setq s (+ s vv)))))
              (setq pending (if (<= s 0.0) (nl-llm-spec--cat p2 vocab)
                              (progn (dotimes (i vocab) (aset r i (/ (aref r i) s))) (nl-llm-spec--cat r vocab))))
              (setq h h1))))))
    (cons (nreverse out) rounds)))

(provide 'nl-llm-spec)
;;; nl-llm-spec.el ends here
