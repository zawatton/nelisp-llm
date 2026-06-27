;;; nl-llm-attn.el --- RoPE causal attention (GQA) with a KV cache  -*- lexical-binding: t; -*-

;; Causal multi-head / grouped-query self-attention with rotary embeddings
;; (RoPE) and a reusable key/value cache for O(1)-per-token incremental
;; decoding.  GQA is the core: query has HEADS heads, key/value have
;; KV-HEADS heads (KV-HEADS <= HEADS, divides HEADS); each KV head is shared
;; by HEADS/KV-HEADS query heads, shrinking the KV cache.  MHA is the special
;; case KV-HEADS = HEADS.  The cached path is verified numerically identical
;; to full recomputation (see test/attn-test.el).  Built on photon-tensor.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)

(defun nl-llm--rope-block (vec base pos hd rbase)
  "Rotate one HD-long block at offset BASE of VEC by RoPE at position POS."
  (let ((half (/ hd 2)) (m 0))
    (while (< m half)
      (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 m) (float hd)))))
             (c (cos theta)) (s (sin theta))
             (i0 (+ base (* 2 m))) (i1 (+ base (* 2 m) 1))
             (a0 (aref vec i0)) (a1 (aref vec i1)))
        (aset vec i0 (- (* a0 c) (* a1 s)))
        (aset vec i1 (+ (* a0 s) (* a1 c))))
      (setq m (1+ m)))))

(defun nl-llm--rope-heads (vec rowbase nheads hd pos rbase)
  "Apply per-head RoPE to NHEADS blocks of HD starting at ROWBASE in VEC."
  (let ((h 0)) (while (< h nheads)
    (nl-llm--rope-block vec (+ rowbase (* h hd)) pos hd rbase) (setq h (1+ h)))))

;;;###autoload
(defun nl-llm-gqa (x layer heads kv-heads &optional rope-base)
  "Full causal grouped-query attention over X (seq x dim) with per-head RoPE.
LAYER holds :wq (dim x dim) and :wk :wv (kvdim x dim) where kvdim =
KV-HEADS*(dim/HEADS), and :wo (dim x dim).  Returns (seq x dim)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (hd (/ dim heads)) (kvdim (* kv-heads hd)) (grp (/ heads kv-heads))
         (base (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (q (photon-tensor-data (photon-tensor-linear x (plist-get layer :wq))))
         (k (photon-tensor-data (photon-tensor-linear x (plist-get layer :wk))))
         (v (photon-tensor-data (photon-tensor-linear x (plist-get layer :wv))))
         (out (make-vector (* seq dim) 0.0)))
    (dotimes (i seq)
      (nl-llm--rope-heads q (* i dim) heads hd i base)
      (nl-llm--rope-heads k (* i kvdim) kv-heads hd i base))
    (dotimes (h heads)
      (let ((c0q (* h hd)) (c0k (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((scores (make-vector (1+ i) 0.0)) (mx -1.0e30) (qb (+ (* i dim) c0q)))
            (dotimes (j (1+ i))
              (let ((kb (+ (* j kvdim) c0k)) (acc 0.0) (t0 0))
                (while (< t0 hd)
                  (setq acc (+ acc (* (aref q (+ qb t0)) (aref k (+ kb t0))))) (setq t0 (1+ t0)))
                (let ((sc (* acc scale))) (aset scores j sc) (when (> sc mx) (setq mx sc)))))
            (let ((sm 0.0))
              (dotimes (j (1+ i))
                (let ((e (exp (- (aref scores j) mx)))) (aset scores j e) (setq sm (+ sm e))))
              (let ((t0 0))
                (while (< t0 hd)
                  (let ((acc 0.0) (j 0))
                    (while (<= j i)
                      (setq acc (+ acc (* (/ (aref scores j) sm) (aref v (+ (* j kvdim) c0k t0)))))
                      (setq j (1+ j)))
                    (aset out (+ (* i dim) c0q t0) acc))
                  (setq t0 (1+ t0)))))))))
    (photon-tensor-linear (photon-tensor (list seq dim) out) (plist-get layer :wo))))

;;;###autoload
(defun nl-llm-mha (x layer heads &optional rope-base)
  "Full causal multi-head self-attention: GQA with KV-HEADS = HEADS."
  (nl-llm-gqa x layer heads heads rope-base))

(cl-defstruct (nl-llm-kv (:constructor nl-llm-kv--make))
  k v len dim heads kv-heads)

;;;###autoload
(defun nl-llm-kv-new (max-seq dim heads &optional kv-heads)
  "Return an empty KV cache for MAX-SEQ tokens, width DIM, HEADS query heads
and KV-HEADS key/value heads (default = HEADS, i.e. MHA)."
  (let* ((kvh (or kv-heads heads)) (hd (/ dim heads)) (kvdim (* kvh hd)))
    (nl-llm-kv--make :k (make-vector (* max-seq kvdim) 0.0)
                     :v (make-vector (* max-seq kvdim) 0.0)
                     :len 0 :dim dim :heads heads :kv-heads kvh)))

;;;###autoload
(defun nl-llm-attn-step (xi layer cache &optional rope-base)
  "Attend one new token XI (1 x dim) at the next position in CACHE.
Appends this token's RoPE'd key/value (KV-HEADS wide) to CACHE and returns
its attention output (1 x dim).  Incremental KV-cache decode path."
  (let* ((dim (nl-llm-kv-dim cache)) (heads (nl-llm-kv-heads cache))
         (kvh (nl-llm-kv-kv-heads cache)) (hd (/ dim heads))
         (kvdim (* kvh hd)) (grp (/ heads kvh)) (pos (nl-llm-kv-len cache))
         (base (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (qr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wq))))
         (kr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wk))))
         (vr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wv))))
         (kc (nl-llm-kv-k cache)) (vc (nl-llm-kv-v cache))
         (out (make-vector dim 0.0)))
    (nl-llm--rope-heads qr 0 heads hd pos base)
    (nl-llm--rope-heads kr 0 kvh hd pos base)
    (dotimes (t0 kvdim)
      (aset kc (+ (* pos kvdim) t0) (aref kr t0))
      (aset vc (+ (* pos kvdim) t0) (aref vr t0)))
    (setf (nl-llm-kv-len cache) (1+ pos))
    (dotimes (h heads)
      (let ((c0q (* h hd)) (c0k (* (/ h grp) hd)) (scores (make-vector (1+ pos) 0.0)) (mx -1.0e30))
        (dotimes (j (1+ pos))
          (let ((kb (+ (* j kvdim) c0k)) (acc 0.0) (t0 0))
            (while (< t0 hd)
              (setq acc (+ acc (* (aref qr (+ c0q t0)) (aref kc (+ kb t0))))) (setq t0 (1+ t0)))
            (let ((sc (* acc scale))) (aset scores j sc) (when (> sc mx) (setq mx sc)))))
        (let ((sm 0.0))
          (dotimes (j (1+ pos))
            (let ((e (exp (- (aref scores j) mx)))) (aset scores j e) (setq sm (+ sm e))))
          (let ((t0 0))
            (while (< t0 hd)
              (let ((acc 0.0) (j 0))
                (while (<= j pos)
                  (setq acc (+ acc (* (/ (aref scores j) sm) (aref vc (+ (* j kvdim) c0k t0)))))
                  (setq j (1+ j)))
                (aset out (+ c0q t0) acc))
              (setq t0 (1+ t0)))))))
    (photon-tensor-linear (photon-tensor (list 1 dim) out) (plist-get layer :wo))))

;;;###autoload
(defun nl-llm-gqa-cached (x layer heads kv-heads &optional rope-base)
  "GQA over X (seq x dim) computed incrementally via a KV cache.
Numerically identical to `nl-llm-gqa'; the decode-time path."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (xd (photon-tensor-data x))
         (cache (nl-llm-kv-new seq dim heads kv-heads))
         (out (make-vector (* seq dim) 0.0)))
    (dotimes (i seq)
      (let ((rowvec (make-vector dim 0.0)))
        (dotimes (t0 dim) (aset rowvec t0 (aref xd (+ (* i dim) t0))))
        (let ((oi (photon-tensor-data
                   (nl-llm-attn-step (photon-tensor (list 1 dim) rowvec)
                                     layer cache rope-base))))
          (dotimes (t0 dim) (aset out (+ (* i dim) t0) (aref oi t0))))))
    (photon-tensor (list seq dim) out)))

;;;###autoload
(defun nl-llm-mha-cached (x layer heads &optional rope-base)
  "Incremental MHA: `nl-llm-gqa-cached' with KV-HEADS = HEADS."
  (nl-llm-gqa-cached x layer heads heads rope-base))

(provide 'nl-llm-attn)
;;; nl-llm-attn.el ends here
