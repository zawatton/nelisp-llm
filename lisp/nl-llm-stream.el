;;; nl-llm-stream.el --- StreamingLLM attention-sink bounded-memory decode  -*- lexical-binding: t; -*-

;; Bounded-memory incremental decode (Xiao et al. 2023, "Efficient Streaming
;; Language Models with Attention Sinks").  A pure sliding window collapses once
;; the first tokens are evicted, because softmax dumps its surplus probability
;; mass onto the first few tokens -- the "attention sinks".  Keeping NSINK initial
;; tokens permanently plus a rolling window of the last WIN tokens bounds the KV
;; cache at NSINK+WIN entries while preserving quality.
;;
;; The one subtlety is RoPE: cached keys are rotated by their *cache-relative*
;; position (their rank in the kept set), not their absolute stream position, so
;; every relative distance the query sees stays within [0, NSINK+WIN).  We store
;; RAW (un-rotated) keys and apply RoPE at attention time; values are unrotated.
;; When the cap is never reached this reduces *exactly* to nl-llm-decode (the
;; equivalence checked in test/stream-test.el).
;;
;; CPU reference / oracle for the GPU streaming decode (docs/design/02-*).

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-arch)    ; nl-llm-rmsnorm
(require 'nl-llm-attn)    ; nl-llm--rope-block, nl-llm--rope-heads
(require 'nl-llm-decode)  ; nl-llm--swiglu-b

(cl-defstruct (nl-llm-scache (:constructor nl-llm-scache--make))
  kraw vraw spos (seen 0) nsink win cap kvdim dim heads kvh)

;;;###autoload
(defun nl-llm-scache-new (nsink win dim heads kvh)
  "Streaming KV cache: NSINK permanent sink slots + WIN rolling window slots.
Width DIM, HEADS query / KVH kv heads.  Memory is bounded at NSINK+WIN tokens
regardless of how many tokens are decoded."
  (let* ((hd (/ dim heads)) (kvdim (* kvh hd)) (cap (+ nsink win)))
    (nl-llm-scache--make
     :kraw (make-vector (* cap kvdim) 0.0) :vraw (make-vector (* cap kvdim) 0.0)
     :spos (make-vector cap -1) :seen 0 :nsink nsink :win win :cap cap
     :kvdim kvdim :dim dim :heads heads :kvh kvh)))

(defun nl-llm-scache-fill (cache)
  "Number of tokens currently resident in CACHE (<= cap)."
  (min (nl-llm-scache-seen cache) (nl-llm-scache-cap cache)))

(defun nl-llm--scache-slot (p nsink win)
  "Physical slot holding stream position P in an NSINK+WIN streaming cache."
  (if (< p nsink) p (+ nsink (mod (- p nsink) win))))

;;;###autoload
(defun nl-llm-stream-block (xrow blk cache &optional rope-base)
  "Decode one token XROW (1 x dim) through one pre-norm block with streaming CACHE.
BLK is the same weight plist as `nl-llm-decode-block'.  Appends this token's raw
key/value to CACHE (mutated, sink+window bounded) and returns the block output
\(1 x dim).  Keys are RoPE'd by cache-relative position at attention time."
  (let* ((dim (nl-llm-scache-dim cache)) (heads (nl-llm-scache-heads cache))
         (kvh (nl-llm-scache-kvh cache)) (hd (/ dim heads)) (kvdim (nl-llm-scache-kvdim cache))
         (grp (/ heads kvh)) (nsink (nl-llm-scache-nsink cache)) (win (nl-llm-scache-win cache))
         (p (nl-llm-scache-seen cache)) (base (or rope-base 10000.0))
         (scale (/ 1.0 (sqrt (float hd))))
         (a (nl-llm-rmsnorm xrow (plist-get blk :ln1g)))
         (qr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wq) (plist-get blk :bq))))
         (kr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wk) (plist-get blk :bk))))
         (vr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wv) (plist-get blk :bv))))
         (kc (nl-llm-scache-kraw cache)) (vc (nl-llm-scache-vraw cache)) (sp (nl-llm-scache-spos cache))
         (out (make-vector dim 0.0))
         (slot (nl-llm--scache-slot p nsink win))
         (start (max nsink (- p (1- win))))                ; oldest window stream pos kept
         (qcrel (if (< p nsink) p (+ nsink (- p start))))  ; query's cache-relative position
         (entries nil))
    ;; store this token's RAW key/value at its slot
    (let ((t0 0)) (while (< t0 kvdim)
      (aset kc (+ (* slot kvdim) t0) (aref kr t0))
      (aset vc (+ (* slot kvdim) t0) (aref vr t0)) (setq t0 (1+ t0))))
    (aset sp slot p)
    (setf (nl-llm-scache-seen cache) (1+ p))
    ;; kept entries as (slot . cache-relative-pos), sink first then window, in order
    (let ((s 0) (lim (min nsink (1+ p)))) (while (< s lim) (push (cons s s) entries) (setq s (1+ s))))
    (let ((s (max nsink start))) (while (<= s p)
      (push (cons (nl-llm--scache-slot s nsink win) (+ nsink (- s start))) entries) (setq s (1+ s))))
    (setq entries (nreverse entries))
    ;; query RoPE at its cache-relative position
    (nl-llm--rope-heads qr 0 heads hd qcrel base)
    (dotimes (h heads)
      (let* ((c0q (* h hd)) (c0k (* (/ h grp) hd)) (ne (length entries))
             (scores (make-vector ne 0.0)) (mx -1.0e30) (e entries) (j 0))
        (while e
          (let* ((slot (car (car e))) (crel (cdr (car e))) (kb (+ (* slot kvdim) c0k))
                 (kk (make-vector hd 0.0)) (t0 0) (acc 0.0))
            (while (< t0 hd) (aset kk t0 (aref kc (+ kb t0))) (setq t0 (1+ t0)))
            (nl-llm--rope-block kk 0 crel hd base)
            (setq t0 0) (while (< t0 hd) (setq acc (+ acc (* (aref qr (+ c0q t0)) (aref kk t0)))) (setq t0 (1+ t0)))
            (let ((scv (* acc scale))) (aset scores j scv) (when (> scv mx) (setq mx scv))))
          (setq e (cdr e) j (1+ j)))
        (let ((sm 0.0))
          (dotimes (jj ne) (let ((ex (exp (- (aref scores jj) mx)))) (aset scores jj ex) (setq sm (+ sm ex))))
          (let ((t0 0)) (while (< t0 hd)
            (let ((accv 0.0) (e2 entries) (jj 0))
              (while e2 (let ((slot (car (car e2))))
                (setq accv (+ accv (* (/ (aref scores jj) sm) (aref vc (+ (* slot kvdim) c0k t0))))))
                (setq e2 (cdr e2) jj (1+ jj)))
              (aset out (+ c0q t0) accv))
            (setq t0 (1+ t0)))))))
    (let* ((attn (photon-tensor-linear (photon-tensor (list 1 dim) out) (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (photon-tensor-add xrow attn))
           (bnorm (nl-llm-rmsnorm x1 (plist-get blk :ln2g))))
      (photon-tensor-add x1 (nl-llm--swiglu-b bnorm blk)))))

;;;###autoload
(defun nl-llm-stream-step (token blocks caches wte lnfg bh dim &optional rope-base)
  "Decode one TOKEN through BLOCKS with per-block streaming CACHES (mutated).
Same model as `nl-llm-decode-step' (gather embedding, blocks, final RMSNorm, tied
head) but with sink+window bounded KV.  Returns the (vocab) logit vector."
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim)
                           (let ((v (make-vector dim 0.0)))
                             (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl blocks) (cl caches))
    (while bl
      (setq x (nl-llm-stream-block x (car bl) (car cl) rope-base))
      (setq bl (cdr bl) cl (cdr cl)))
    (photon-tensor-data (photon-tensor-linear (nl-llm-rmsnorm x lnfg) wte bh))))

(provide 'nl-llm-stream)
;;; nl-llm-stream.el ends here
