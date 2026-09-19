;;; nl-llm-attn.el --- RoPE causal attention (GQA) with a KV cache  -*- lexical-binding: t; -*-

;; Causal multi-head / grouped-query self-attention with rotary embeddings
;; (RoPE) and a reusable key/value cache for O(1)-per-token incremental
;; decoding.  GQA is the core: query has HEADS heads, key/value have
;; KV-HEADS heads (KV-HEADS <= HEADS, divides HEADS); each KV head is shared
;; by HEADS/KV-HEADS query heads, shrinking the KV cache.  MHA is the special
;; case KV-HEADS = HEADS.  The cached path is verified numerically identical
;; to full recomputation (see test/attn-test.el).  Built on photon-tensor.
;;
;; Head width is `:head-dim' on the layer when present, else (/ dim heads).
;; Qwen3 decouples the two -- Qwen3-0.6B is dim 1024 with 16 heads of width 128,
;; so the query side is 2048 wide and :wo folds it back to 1024.  Nothing in the
;; shapes objects when the width is wrong, so the loops just stride the wrong
;; distance; test/head-dim-test.el pins the numbers against its own reference.

;;; Code:

(require 'nl-llm-compat)
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

(defun nl-llm--rope-block-half (vec base pos hd rbase)
  "Rotate one HD-long block at BASE of VEC by half-split RoPE at POS.
Pairs element i with i + HD/2, the GPT-NeoX convention that Qwen3 and Llama
use (HuggingFace calls it rotate_half).  Same angles as
`nl-llm--rope-block', different pairing, and the two disagree substantially --
an 8-wide head at position 3 differs by 5.8.  Neither raises, because the
vector is the right length either way."
  (let* ((half (/ hd 2)) (orig (make-vector hd 0.0)) (i 0))
    (while (< i hd) (aset orig i (aref vec (+ base i))) (setq i (1+ i)))
    (setq i 0)
    (while (< i half)
      (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 i) (float hd)))))
             (c (cos theta)) (s (sin theta))
             (a (aref orig i)) (b (aref orig (+ i half))))
        (aset vec (+ base i) (- (* a c) (* b s)))
        (aset vec (+ base i half) (+ (* b c) (* a s))))
      (setq i (1+ i)))))

(defconst nl-llm-rope-styles '(interleaved half)
  "Rotation conventions `nl-llm--rope-heads' implements.")

(defun nl-llm-rope-style (layer &optional style)
  "Return the rotation convention for LAYER, defaulting to STYLE then
`interleaved'.  An unrecognised name signals rather than falling back, because
silently using the wrong pairing is the failure this option exists to fix."
  (let ((s (or (plist-get layer :rope-style) style 'interleaved)))
    (unless (memq s nl-llm-rope-styles)
      (error "nl-llm: unknown :rope-style %S (expected one of %S)"
             s nl-llm-rope-styles))
    s))

(defun nl-llm--rope-heads (vec rowbase nheads hd pos rbase &optional style)
  "Apply per-head RoPE to NHEADS blocks of HD starting at ROWBASE in VEC.
STYLE is `interleaved' (default, adjacent pairs) or `half' (i with i + HD/2)."
  (let ((h 0) (fn (if (eq style 'half)
                      #'nl-llm--rope-block-half
                    #'nl-llm--rope-block)))
    (while (< h nheads)
      (funcall fn vec (+ rowbase (* h hd)) pos hd rbase)
      (setq h (1+ h)))))

(defun nl-llm--rmsnorm-heads (vec rowbase nheads hd gain &optional eps)
  "RMSNorm each of NHEADS blocks of width HD at ROWBASE in VEC, scaled by GAIN.
Qwen3's QK-norm: applied to q and k per head BEFORE the rotation.  GAIN nil is
a no-op, so a donor without the tensors behaves as before."
  (when gain
    (let ((g (photon-tensor-data gain)) (e (or eps 1.0e-6)) (h 0))
      (unless (= (length g) hd)
        (error "nl-llm: QK-norm gain is %d wide, head is %d" (length g) hd))
      (while (< h nheads)
        (let ((base (+ rowbase (* h hd))) (ss 0.0) (i 0))
          (while (< i hd)
            (let ((v (aref vec (+ base i)))) (setq ss (+ ss (* v v))))
            (setq i (1+ i)))
          (let ((inv (/ 1.0 (sqrt (+ (/ ss (float hd)) e)))) (j 0))
            (while (< j hd)
              (aset vec (+ base j) (* (aref vec (+ base j)) inv (aref g j)))
              (setq j (1+ j)))))
        (setq h (1+ h))))))

;;;###autoload
(defun nl-llm-gqa (x layer heads kv-heads &optional rope-base head-dim)
  "Full causal grouped-query attention over X (seq x dim) with per-head RoPE.
Head width is LAYER's :head-dim, else HEAD-DIM, else (/ dim heads).  With HD as
that width, LAYER holds :wq (HEADS*HD x dim), :wk :wv (KV-HEADS*HD x dim) and
:wo (dim x HEADS*HD).  Returns (seq x dim)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (hd (or (plist-get layer :head-dim) head-dim (/ dim heads)))
         (qdim (* heads hd)) (kvdim (* kv-heads hd)) (grp (/ heads kv-heads))
         (base (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (style (nl-llm-rope-style layer))
         (qn (plist-get layer :q-norm)) (kn (plist-get layer :k-norm))
         (q (photon-tensor-data (photon-tensor-linear x (plist-get layer :wq))))
         (k (photon-tensor-data (photon-tensor-linear x (plist-get layer :wk))))
         (v (photon-tensor-data (photon-tensor-linear x (plist-get layer :wv))))
         (out (make-vector (* seq qdim) 0.0)))
    (dotimes (i seq)
      ;; QK-norm first: Qwen3 normalises each head before rotating it.
      (nl-llm--rmsnorm-heads q (* i qdim) heads hd qn)
      (nl-llm--rmsnorm-heads k (* i kvdim) kv-heads hd kn)
      (nl-llm--rope-heads q (* i qdim) heads hd i base style)
      (nl-llm--rope-heads k (* i kvdim) kv-heads hd i base style))
    (dotimes (h heads)
      (let ((c0q (* h hd)) (c0k (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((scores (make-vector (1+ i) 0.0)) (mx -1.0e30) (qb (+ (* i qdim) c0q)))
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
                    (aset out (+ (* i qdim) c0q t0) acc))
                  (setq t0 (1+ t0)))))))))
    (photon-tensor-linear (photon-tensor (list seq qdim) out) (plist-get layer :wo))))

;;;###autoload
(defun nl-llm-attn-reject-decoupled-head-dim (layer dim heads where)
  "Signal when LAYER asks for donor behaviour that WHERE does not implement.
Three keys describe how the donor's attention actually works and each is silent
when ignored, because the shapes stay valid either way:

  :head-dim    a head width other than (/ DIM HEADS)
  :rope-style  `half', the pairing Qwen3 and Llama use
  :q-norm / :k-norm   per-head normalisation before the rotation

Only `nl-llm-gqa', `nl-llm-block' / `nl-llm-model-forward' and the CPU KV
decode in `nl-llm-decode.el' implement them.  Every other attention path still
assumes the old conventions, so it calls this rather than ignoring the keys:
forgetting to migrate one is then an error instead of a confident wrong answer."
  (let ((hd (plist-get layer :head-dim)))
    (when (and hd (/= hd (/ dim heads)))
      (error "%s does not implement :head-dim %d yet (dim %d / heads %d = %d)"
             where hd dim heads (/ dim heads))))
  (let ((style (plist-get layer :rope-style)))
    (when (and style (not (eq style 'interleaved)))
      (error "%s does not implement :rope-style %S yet" where style)))
  (when (or (plist-get layer :q-norm) (plist-get layer :k-norm))
    (error "%s does not implement :q-norm / :k-norm yet" where)))

;;;###autoload
(defun nl-llm-mha (x layer heads &optional rope-base head-dim)
  "Full causal multi-head self-attention: GQA with KV-HEADS = HEADS."
  (nl-llm-gqa x layer heads heads rope-base head-dim))

(cl-defstruct (nl-llm-kv (:constructor nl-llm-kv--make))
  k v len dim heads kv-heads head-dim)

;;;###autoload
(defun nl-llm-kv-new (max-seq dim heads &optional kv-heads head-dim)
  "Return an empty KV cache for MAX-SEQ tokens, width DIM, HEADS query heads
and KV-HEADS key/value heads (default = HEADS, i.e. MHA).  HEAD-DIM is the
per-head width, defaulting to (/ DIM HEADS); the cache remembers it so the
decode step does not re-derive a width the weights disagree with."
  (let* ((kvh (or kv-heads heads)) (hd (or head-dim (/ dim heads)))
         (kvdim (* kvh hd)))
    (nl-llm-kv--make :k (make-vector (* max-seq kvdim) 0.0)
                     :v (make-vector (* max-seq kvdim) 0.0)
                     :len 0 :dim dim :heads heads :kv-heads kvh :head-dim hd)))

;;;###autoload
(defun nl-llm-attn-step (xi layer cache &optional rope-base)
  "Attend one new token XI (1 x dim) at the next position in CACHE.
Appends this token's RoPE'd key/value (KV-HEADS wide) to CACHE and returns
its attention output (1 x dim).  Incremental KV-cache decode path."
  (let* ((dim (nl-llm-kv-dim cache)) (heads (nl-llm-kv-heads cache))
         (kvh (nl-llm-kv-kv-heads cache))
         (hd (or (nl-llm-kv-head-dim cache) (/ dim heads)))
         (qdim (* heads hd))
         (kvdim (* kvh hd)) (grp (/ heads kvh)) (pos (nl-llm-kv-len cache))
         (base (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (style (nl-llm-rope-style layer))
         (qr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wq))))
         (kr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wk))))
         (vr (photon-tensor-data (photon-tensor-linear xi (plist-get layer :wv))))
         (kc (nl-llm-kv-k cache)) (vc (nl-llm-kv-v cache))
         (out (make-vector qdim 0.0)))
    (nl-llm--rmsnorm-heads qr 0 heads hd (plist-get layer :q-norm))
    (nl-llm--rmsnorm-heads kr 0 kvh hd (plist-get layer :k-norm))
    (nl-llm--rope-heads qr 0 heads hd pos base style)
    (nl-llm--rope-heads kr 0 kvh hd pos base style)
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
    (photon-tensor-linear (photon-tensor (list 1 qdim) out) (plist-get layer :wo))))

;;;###autoload
(defun nl-llm-gqa-cached (x layer heads kv-heads &optional rope-base head-dim)
  "GQA over X (seq x dim) computed incrementally via a KV cache.
Numerically identical to `nl-llm-gqa'; the decode-time path."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (xd (photon-tensor-data x))
         (cache (nl-llm-kv-new seq dim heads kv-heads
                               (or (plist-get layer :head-dim) head-dim
                                   (/ dim heads))))
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
(defun nl-llm-mha-cached (x layer heads &optional rope-base head-dim)
  "Incremental MHA: `nl-llm-gqa-cached' with KV-HEADS = HEADS."
  (nl-llm-gqa-cached x layer heads heads rope-base head-dim))

(provide 'nl-llm-attn)
;;; nl-llm-attn.el ends here
