;;; stream-test.el --- StreamingLLM bounded decode correctness  -*- lexical-binding: t; -*-
;; Two gates, pure CPU:
;;  (1) Equivalence: with cap >= sequence length, sink+window decode reduces
;;      EXACTLY to the plain KV-cache decode (nl-llm-decode-step) -- this pins the
;;      cache-relative RoPE math, the MVP correctness gate for docs/design/02.
;;  (2) Boundedness: the cache never holds more than nsink+win entries no matter
;;      how long we decode, and the kept slots hold the expected stream positions.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/stream-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-stream)

(defvar st--fail 0)
(defun st--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" (progn (setq st--fail (1+ st--fail)) "FAIL")) (or extra ""))))
(defun st--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun st--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4)
       (wte (st--t (list vocab dim) 1 sc)) (lnfg (st--ones dim)) (bh (st--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (st--ones dim) :wq (st--t (list dim dim) (+ s0 1) sc) :bq (st--t (list dim) (+ s0 11) 0.1)
                                 :wk (st--t (list kvdim dim) (+ s0 2) sc) :bk (st--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (st--t (list kvdim dim) (+ s0 3) sc) :bv (st--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (st--t (list dim dim) (+ s0 4) sc) :bo (st--t (list dim) (+ s0 14) 0.1) :ln2g (st--ones dim)
                                 :wg (st--t (list ff dim) (+ s0 5) sc) :bg (st--t (list ff) (+ s0 15) 0.1)
                                 :wu (st--t (list ff dim) (+ s0 6) sc) :bu (st--t (list ff) (+ s0 16) 0.1)
                                 :wd (st--t (list dim ff) (+ s0 7) sc) :bd (st--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (seq 8) (tokens (vector 0 3 1 4 2 5 3 6)))
  ;; (1) cap >= seq  ==>  streaming == plain decode, position for position
  (let* ((dcaches (mapcar (lambda (_) (nl-llm-dcache-new seq dim heads kvh)) blocks))
         ;; nsink+win = 4+8 = 12 >= seq 8, so nothing is ever evicted
         (scaches (mapcar (lambda (_) (nl-llm-scache-new 4 8 dim heads kvh)) blocks))
         (maxd 0.0) (p 0))
    (while (< p seq)
      (let ((dd (nl-llm-decode-step (aref tokens p) blocks dcaches wte lnfg bh dim))
            (ss (nl-llm-stream-step (aref tokens p) blocks scaches wte lnfg bh dim)) (j 0))
        (while (< j vocab) (setq maxd (max maxd (abs (- (aref dd j) (aref ss j))))) (setq j (1+ j))))
      (setq p (1+ p)))
    (st--ck "cap>=len: streaming == plain KV decode" (< maxd 1e-5) (format "maxdiff=%.2e" maxd)))
  ;; (2) boundedness + slot occupancy over a long decode (cap=4+8=12 << len 60)
  (let* ((nsink 4) (win 8) (cap (+ nsink win)) (len 60)
         (scaches (mapcar (lambda (_) (nl-llm-scache-new nsink win dim heads kvh)) blocks))
         (okfill t) (okslot t) (p 0))
    (while (< p len)
      (nl-llm-stream-step (aref tokens (mod p seq)) blocks scaches wte lnfg bh dim)
      (let ((c (car scaches)))
        (unless (= (nl-llm-scache-fill c) (min (1+ p) cap)) (setq okfill nil))
        ;; sink slots must still hold stream positions 0..nsink-1
        (when (>= p cap)
          (let ((sp (nl-llm-scache-spos c)) (i 0))
            (while (< i nsink) (unless (= (aref sp i) i) (setq okslot nil)) (setq i (1+ i))))))
      (setq p (1+ p)))
    (st--ck "boundedness: fill == min(seen, cap)" okfill (format "len=%d cap=%d" len cap))
    (st--ck "sink slots pinned to stream pos 0..nsink-1" okslot)
    ;; a no-sink window-only cache stays bounded too (boundedness is structural)
    (st--ck "window-only (nsink=0) also bounded"
            (let ((cs (mapcar (lambda (_) (nl-llm-scache-new 0 win dim heads kvh)) blocks)) (ok t) (q 0))
              (while (< q 40) (nl-llm-stream-step (aref tokens (mod q seq)) blocks cs wte lnfg bh dim)
                (unless (<= (nl-llm-scache-fill (car cs)) win) (setq ok nil)) (setq q (1+ q))) ok)))
  ;; (3) Cache-relative RoPE invariance under eviction (model-agnostic).
  ;; With period-WIN input, once the cache has cycled the kept set (sink + last
  ;; WIN tokens) and every cache-relative position repeats with period WIN, so the
  ;; logits must repeat with period WIN.  This pins the ring indexing + cache-rel
  ;; RoPE precisely, with no dependence on whether the model is trained.
  ;; (The *quality* benefit of the sink is a learned effect -- see
  ;; examples/stream-decode.el, which trains a model and shows it.)
  (let* ((nsink 2) (win 6) (cap (+ nsink win)) (len 40) (period win)
         (base (vector 0 7 2 9 4 11))   ; one period, length = win, all < vocab
         (caches (mapcar (lambda (_) (nl-llm-scache-new nsink win dim heads kvh)) blocks))
         (logs (make-vector len nil)) (maxd 0.0) (p 0))
    (while (< p len)
      (aset logs p (copy-sequence
                    (nl-llm-stream-step (aref base (mod p period)) blocks caches wte lnfg bh dim)))
      (setq p (1+ p)))
    (setq p (* 2 cap))                   ; compare only in the steady state
    (while (< (+ p period) len)
      (let ((a (aref logs p)) (b (aref logs (+ p period))) (j 0))
        (while (< j vocab) (setq maxd (max maxd (abs (- (aref a j) (aref b j))))) (setq j (1+ j))))
      (setq p (1+ p)))
    (st--ck "cache-relative RoPE: period-win logit invariance" (< maxd 1e-5) (format "maxdiff=%.2e" maxd)))
  (princ (format "NL-LLM-STREAM %s (%d failures)\n" (if (= st--fail 0) "ALL-PASS" "HAS-FAILURES") st--fail))
  (kill-emacs (if (= st--fail 0) 0 1)))
;;; stream-test.el ends here
