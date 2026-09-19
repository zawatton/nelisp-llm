;;; nl-llm-decode.el --- KV-cache incremental decode for the modern block  -*- lexical-binding: t; -*-

;; O(1)-projection / O(len)-attention per-token decoding for the full modern
;; block WITH biases and a tied head -- numerically the same model the on-device
;; path trains (RMSNorm + GQA/RoPE + SwiGLU).  Each block keeps a key/value cache
;; so generating token t costs ~O(dim^2 + t*hd) instead of re-running the whole
;; O(t^2) prefill every step.  Verified position-for-position against the prefill
;; forward (nl-llm-ag-block) in test/decode-test.el.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-arch)   ; nl-llm-rmsnorm, nl-llm-silu
(require 'nl-llm-attn)   ; nl-llm--rope-heads

(cl-defstruct (nl-llm-dcache (:constructor nl-llm-dcache--make))
  k v (len 0) kvdim dim heads kvh head-dim)

(defun nl-llm--dcache-positive-integer-p (value)
  "Return non-nil when VALUE is a positive integer."
  (and (integerp value) (> value 0)))

(defun nl-llm--dcache-validate-layout (dim heads kvh &optional head-dim)
  "Validate cache layout dimensions DIM, HEADS, KVH and optional HEAD-DIM."
  (unless (nl-llm--dcache-positive-integer-p dim)
    (error "KV cache dim must be a positive integer, got %S" dim))
  (unless (nl-llm--dcache-positive-integer-p heads)
    (error "KV cache heads must be a positive integer, got %S" heads))
  (unless (nl-llm--dcache-positive-integer-p kvh)
    (error "KV cache kvh must be a positive integer, got %S" kvh))
  (when head-dim
    (unless (nl-llm--dcache-positive-integer-p head-dim)
      (error "KV cache head-dim must be a positive integer, got %S" head-dim)))
  ;; Kept unconditional.  Relaxing it for caches that state a head width looked
  ;; harmless -- a decoupled model does not need dim to divide heads -- but it
  ;; silently switched the check off for every cache, since one is now always
  ;; stored, and test/decode-capacity-test.el caught the loss.  Every Qwen3
  ;; dense size satisfies it anyway (1024/16, 2048/16, 2560/32), so there is no
  ;; case to relax it for yet.
  (unless (= (% dim heads) 0)
    (error "KV cache dim %d must be divisible by heads %d" dim heads))
  (unless (= (% heads kvh) 0)
    (error "KV cache heads %d must be divisible by kvh %d" heads kvh)))

(defun nl-llm-dcache-new (max-seq dim heads kvh &optional head-dim)
  "Empty KV cache for MAX-SEQ tokens, width DIM, HEADS query / KVH kv heads.
HEAD-DIM is the per-head width, defaulting to (/ DIM HEADS) and stored on the
cache so the decode step cannot re-derive a width the weights disagree with."
  (unless (nl-llm--dcache-positive-integer-p max-seq)
    (error "KV cache capacity must be a positive integer, got %S" max-seq))
  (nl-llm--dcache-validate-layout dim heads kvh head-dim)
  (let* ((hd (or head-dim (/ dim heads))) (kvdim (* kvh hd)))
    (nl-llm-dcache--make :k (make-vector (* max-seq kvdim) 0.0)
                         :v (make-vector (* max-seq kvdim) 0.0)
                         :len 0 :kvdim kvdim :dim dim :heads heads :kvh kvh
                         :head-dim hd)))

(defun nl-llm--dcache-preflight (cache &optional expected-dim)
  "Validate CACHE metadata and storage, and require room for one token.
When EXPECTED-DIM is non-nil, require CACHE to have that model dimension."
  (unless (nl-llm-dcache-p cache)
    (error "Invalid KV cache object: %S" cache))
  (let ((dim (nl-llm-dcache-dim cache))
        (heads (nl-llm-dcache-heads cache))
        (kvh (nl-llm-dcache-kvh cache))
        (kvdim (nl-llm-dcache-kvdim cache))
        (len (nl-llm-dcache-len cache))
        (kc (nl-llm-dcache-k cache))
        (vc (nl-llm-dcache-v cache))
        (hd (nl-llm-dcache-head-dim cache)))
    (nl-llm--dcache-validate-layout dim heads kvh hd)
    (when (and expected-dim (/= dim expected-dim))
      (error "KV cache dim %d does not match decoder dim %d" dim expected-dim))
    (let ((expected-kvdim (* kvh (or hd (/ dim heads)))))
      (unless (and (integerp kvdim) (= kvdim expected-kvdim))
        (error "KV cache kvdim %S does not match expected width %d"
               kvdim expected-kvdim)))
    (unless (and (vectorp kc) (vectorp vc))
      (error "KV cache key/value storage must be vectors"))
    (unless (= (length kc) (length vc))
      (error "KV cache key/value vector sizes differ: %d and %d"
             (length kc) (length vc)))
    (unless (= (% (length kc) kvdim) 0)
      (error "KV cache vector size %d is not divisible by kvdim %d"
             (length kc) kvdim))
    (unless (and (integerp len) (>= len 0))
      (error "KV cache length must be a non-negative integer, got %S" len))
    (let ((capacity (/ (length kc) kvdim)))
      (when (>= len capacity)
        (error "KV cache context capacity exceeded: length %d, capacity %d"
               len capacity))
      capacity)))

(defun nl-llm--decode-preflight (blocks caches dim)
  "Validate BLOCKS/CACHES cardinality and every cache before decoding."
  (unless (and (listp blocks) (listp caches))
    (error "Decoder blocks and caches must be lists"))
  (unless (= (length blocks) (length caches))
    (error "Decoder block/cache count mismatch: %d blocks, %d caches"
           (length blocks) (length caches)))
  (dolist (cache caches)
    (nl-llm--dcache-preflight cache dim))
  ;; Every block must decode the same token position.  Check this only after
  ;; validating every cache so a full cache still reports its capacity error.
  (when caches
    (let ((len (nl-llm-dcache-len (car caches))))
      (dolist (cache (cdr caches))
        (unless (= (nl-llm-dcache-len cache) len)
          (error "Decoder cache length mismatch: expected %d, got %d"
                 len (nl-llm-dcache-len cache)))))))

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
  (nl-llm--dcache-preflight cache)
  (let* ((dim (nl-llm-dcache-dim cache)) (heads (nl-llm-dcache-heads cache))
         (kvh (nl-llm-dcache-kvh cache))
         (hd (or (nl-llm-dcache-head-dim cache) (/ dim heads)))
         (qdim (* heads hd)) (kvdim (nl-llm-dcache-kvdim cache))
         (grp (/ heads kvh)) (pos (nl-llm-dcache-len cache)) (base (or rope-base 10000.0))
         (scale (/ 1.0 (sqrt (float hd))))
         (a (nl-llm-rmsnorm xrow (plist-get blk :ln1g)))
         (qr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wq) (plist-get blk :bq))))
         (kr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wk) (plist-get blk :bk))))
         (vr (photon-tensor-data (photon-tensor-linear a (plist-get blk :wv) (plist-get blk :bv))))
         (kc (nl-llm-dcache-k cache)) (vc (nl-llm-dcache-v cache))
         (out (make-vector qdim 0.0)))
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
    (let* ((attn (photon-tensor-linear (photon-tensor (list 1 qdim) out) (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (photon-tensor-add xrow attn))
           (bnorm (nl-llm-rmsnorm x1 (plist-get blk :ln2g))))
      (photon-tensor-add x1 (nl-llm--swiglu-b bnorm blk)))))

;;;###autoload
(defun nl-llm-decode-step
    (token blocks caches wte lnfg bh dim &optional rope-base head)
  "Decode one TOKEN: gather its embedding from WTE (vocab x dim), run it through
BLOCKS (each with its own entry in CACHES, mutated), final RMSNorm (LNFG), and a
tied head (logits = xf . WTE^T + BH).  Optional HEAD supplies an independent
vocab x dim output matrix instead.  Returns the (vocab) logit vector for the
next token.  Call once per position, in order, to generate."
  (nl-llm--decode-preflight blocks caches dim)
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim)
                           (let ((v (make-vector dim 0.0)))
                             (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl blocks) (cl caches))
    (while bl
      (setq x (nl-llm-decode-block x (car bl) (car cl) rope-base))
      (setq bl (cdr bl) cl (cdr cl)))
    (photon-tensor-data
     (photon-tensor-linear (nl-llm-rmsnorm x lnfg) (or head wte) bh))))

;;;###autoload
(defun nl-llm-decode-h (token blocks caches wte lnfg dim &optional rope-base)
  "Like `nl-llm-decode-step' but return the post-final-RMSNorm hidden (1 x dim)
instead of logits, so that several heads (e.g. the main tied head and an MTP
look-ahead head) can be applied to the same hidden.  Feeds TOKEN and advances
the KV CACHES exactly as `nl-llm-decode-step'."
  (nl-llm--decode-preflight blocks caches dim)
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
