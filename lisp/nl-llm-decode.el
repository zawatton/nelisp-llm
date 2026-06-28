;;; nl-llm-decode.el --- KV-cache incremental decode for the modern block  -*- lexical-binding: t; -*-

;; O(1)-projection / O(len)-attention per-token decoding for the full modern
;; block WITH biases and a tied head -- numerically the same model the on-device
;; path trains (RMSNorm + GQA/RoPE + SwiGLU).  Each block keeps a key/value cache
;; so generating token t costs ~O(dim^2 + t*hd) instead of re-running the whole
;; O(t^2) prefill every step.  Verified position-for-position against the prefill
;; forward (nl-llm-ag-block) in test/decode-test.el.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-arch)   ; nl-llm-rmsnorm, nl-llm-silu
(require 'nl-llm-attn)   ; nl-llm--rope-heads

(cl-defstruct (nl-llm-dcache (:constructor nl-llm-dcache--make))
  k v (len 0) kvdim dim heads kvh)

(defun nl-llm-dcache-new (max-seq dim heads kvh)
  "Empty KV cache for MAX-SEQ tokens, width DIM, HEADS query / KVH kv heads."
  (let* ((hd (/ dim heads)) (kvdim (* kvh hd)))
    (nl-llm-dcache--make :k (make-vector (* max-seq kvdim) 0.0)
                         :v (make-vector (* max-seq kvdim) 0.0)
                         :len 0 :kvdim kvdim :dim dim :heads heads :kvh kvh)))

(defun nl-llm--swiglu-b (x blk)
  "SwiGLU FFN with biases over X using BLK's :wg :bg :wu :bu :wd :bd."
  (photon-tensor-linear
   (photon-tensor-hadamard
    (nl-llm-silu (photon-tensor-linear x (plist-get blk :wg) (plist-get blk :bg)))
    (photon-tensor-linear x (plist-get blk :wu) (plist-get blk :bu)))
   (plist-get blk :wd) (plist-get blk :bd)))

;;;###autoload
(defun nl-llm-decode-block (xrow blk cache &optional rope-base)
  "Decode one token XROW (1 x dim) through one pre-norm block with KV CACHE.
BLK is a plist of tensor weights with biases: :ln1g :wq :bq :wk :bk :wv :bv
:wo :bo :ln2g :wg :bg :wu :bu :wd :bd.  Appends this token's RoPE'd key/value to
CACHE (mutated) and returns the block output (1 x dim)."
  (let* ((dim (nl-llm-dcache-dim cache)) (heads (nl-llm-dcache-heads cache))
         (kvh (nl-llm-dcache-kvh cache)) (hd (/ dim heads)) (kvdim (nl-llm-dcache-kvdim cache))
         (grp (/ heads kvh)) (pos (nl-llm-dcache-len cache)) (base (or rope-base 10000.0))
         (scale (/ 1.0 (sqrt (float hd))))
         (a (nl-llm-rmsnorm xrow (plist-get blk :ln1g)))
         (qr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wq) (plist-get blk :bq))))
         (kr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wk) (plist-get blk :bk))))
         (vr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wv) (plist-get blk :bv))))
         (kc (nl-llm-dcache-k cache)) (vc (nl-llm-dcache-v cache))
         (out (make-vector dim 0.0)))
    (nl-llm--rope-heads qr 0 heads hd pos base)
    (nl-llm--rope-heads kr 0 kvh hd pos base)
    (dotimes (t0 kvdim)
      (aset kc (+ (* pos kvdim) t0) (aref kr t0))
      (aset vc (+ (* pos kvdim) t0) (aref vr t0)))
    (setf (nl-llm-dcache-len cache) (1+ pos))
    (dotimes (h heads)
      (let ((c0q (* h hd)) (c0k (* (/ h grp) hd)) (scores (make-vector (1+ pos) 0.0)) (mx -1.0e30))
        (dotimes (j (1+ pos))
          (let ((kb (+ (* j kvdim) c0k)) (acc 0.0) (t0 0))
            (while (< t0 hd) (setq acc (+ acc (* (aref qr (+ c0q t0)) (aref kc (+ kb t0))))) (setq t0 (1+ t0)))
            (let ((sc (* acc scale))) (aset scores j sc) (when (> sc mx) (setq mx sc)))))
        (let ((sm 0.0))
          (dotimes (j (1+ pos)) (let ((e (exp (- (aref scores j) mx)))) (aset scores j e) (setq sm (+ sm e))))
          (let ((t0 0))
            (while (< t0 hd)
              (let ((acc 0.0) (j 0))
                (while (<= j pos)
                  (setq acc (+ acc (* (/ (aref scores j) sm) (aref vc (+ (* j kvdim) c0k t0)))))
                  (setq j (1+ j)))
                (aset out (+ c0q t0) acc))
              (setq t0 (1+ t0)))))))
    (let* ((attn (photon-tensor-linear (photon-tensor (list 1 dim) out) (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (photon-tensor-add xrow attn))
           (bnorm (nl-llm-rmsnorm x1 (plist-get blk :ln2g))))
      (photon-tensor-add x1 (nl-llm--swiglu-b bnorm blk)))))

;;;###autoload
(defun nl-llm-decode-step (token blocks caches wte lnfg bh dim &optional rope-base)
  "Decode one TOKEN: gather its embedding from WTE (vocab x dim), run it through
BLOCKS (each with its own entry in CACHES, mutated), final RMSNorm (LNFG), and a
tied head (logits = xf . WTE^T + BH).  Returns the (vocab) logit vector for the
next token.  Call once per position, in order, to generate."
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim)
                           (let ((v (make-vector dim 0.0)))
                             (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl blocks) (cl caches))
    (while bl
      (setq x (nl-llm-decode-block x (car bl) (car cl) rope-base))
      (setq bl (cdr bl) cl (cdr cl)))
    (photon-tensor-data
     (photon-tensor-linear (nl-llm-rmsnorm x lnfg) wte bh))))

;;;###autoload
(defun nl-llm-decode-h (token blocks caches wte lnfg dim &optional rope-base)
  "Like `nl-llm-decode-step' but return the post-final-RMSNorm hidden (1 x dim)
instead of logits, so that several heads (e.g. the main tied head and an MTP
look-ahead head) can be applied to the same hidden.  Feeds TOKEN and advances
the KV CACHES exactly as `nl-llm-decode-step'."
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim)
                           (let ((v (make-vector dim 0.0)))
                             (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl blocks) (cl caches))
    (while bl
      (setq x (nl-llm-decode-block x (car bl) (car cl) rope-base))
      (setq bl (cdr bl) cl (cdr cl)))
    (nl-llm-rmsnorm x lnfg)))

(provide 'nl-llm-decode)
;;; nl-llm-decode.el ends here
