;;; nl-llm-integrated.el --- end-to-end decode integrating all four techniques  -*- lexical-binding: t; -*-

;; One decode loop that composes the four GIGAZINE techniques at their four
;; natural layers, each doing its own job without conflicting with the others:
;;
;;   * BitNet b1.58 (weight layer)   -- every linear runs through a ternary LINFN
;;     (nl-llm-bitnet--run1 -> the GPU base-4 packed kernel), and the tied
;;     embedding/head is the packed WTE (unpacked row in, packed linear out), so
;;     no f32 weight matrix is touched during decode.
;;   * StreamingLLM (eviction layer) -- the KV cache keeps NSINK attention-sink
;;     slots plus a rolling WIN window, with cache-relative RoPE, so memory is
;;     bounded at NSINK+WIN regardless of how long we generate.
;;   * PagedAttention (storage layer) -- that bounded KV does not live in a flat
;;     array but in a block pool addressed by a per-cache block TABLE with a
;;     non-identity physical layout, blocks allocated on demand from a free-list.
;;   * Speculative MTP (generation layer) -- an MTP look-ahead head drafts the
;;     token after the greedy next; both verify in one round and a correct draft
;;     lands two tokens per forward.  Only accepted tokens ever enter the cache,
;;     so speculation never has to roll back the streaming ring.
;;
;; The composition is lossless by construction and pinned two ways in
;; test/integrated-test.el: paged storage reproduces the flat streaming cache to
;; f32 (the storage layer is transparent), and the speculative stream equals plain
;; greedy on the identical ternary/streaming/paged model (the generation layer is
;; transparent).  See examples/integrated-decode.el for a runnable demo.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-arch)     ; nl-llm-rmsnorm, nl-llm-silu
(require 'nl-llm-attn)     ; nl-llm--rope-block, nl-llm--rope-heads
(require 'nl-llm-decode)   ; nl-llm--swiglu-b (f32 reference path)
(require 'nl-llm-stream)   ; nl-llm--scache-slot, nl-llm-stream-block (oracle)
(require 'nl-llm-bitnet)   ; nl-llm-bitnet--run1, -pack-wte, -unpack-row
(require 'nl-llm-spec)     ; nl-llm-spec-argmax, nl-llm--head-argmax

;; ---- paged streaming KV cache: sink+window bounded, stored behind a block table

(cl-defstruct (nl-llm-spcache (:constructor nl-llm-spcache--make))
  kpool vpool table free (seen 0) nsink win cap bs nlblk kvdim dim heads kvh)

;;;###autoload
(defun nl-llm-spcache-new (nsink win dim heads kvh bs)
  "Paged streaming KV cache: NSINK sink + WIN window tokens (cap = NSINK+WIN),
stored in BS-token blocks addressed by a block table.  The free-list hands out
physical blocks in a reversed (non-identity) order, so the table indirection is
genuinely exercised, and blocks are allocated on demand as the prefix grows."
  (let* ((hd (/ dim heads)) (kvdim (* kvh hd)) (cap (+ nsink win))
         (nlblk (/ (+ cap bs -1) bs))
         (free nil))
    (dotimes (i nlblk) (push i free))           ; free = (0 1 ... nlblk-1) -> popped reversed
    (nl-llm-spcache--make
     :kpool (make-vector (* nlblk bs kvdim) 0.0) :vpool (make-vector (* nlblk bs kvdim) 0.0)
     :table (make-vector nlblk -1) :free free :seen 0
     :nsink nsink :win win :cap cap :bs bs :nlblk nlblk :kvdim kvdim :dim dim :heads heads :kvh kvh)))

(defun nl-llm-spcache-fill (cache)
  "Number of tokens currently resident in CACHE (<= cap)."
  (min (nl-llm-spcache-seen cache) (nl-llm-spcache-cap cache)))

(defun nl-llm-spcache-used-blocks (cache)
  "Physical blocks currently allocated (table entries that are mapped)."
  (let ((n 0)) (dotimes (i (nl-llm-spcache-nlblk cache)) (when (>= (aref (nl-llm-spcache-table cache) i) 0) (setq n (1+ n)))) n))

(defun nl-llm-spcache--base (cache slot)
  "Base index into the K/V pool for logical SLOT, allocating its block on demand."
  (let* ((bs (nl-llm-spcache-bs cache)) (kvdim (nl-llm-spcache-kvdim cache))
         (lb (/ slot bs)) (off (% slot bs)) (tbl (nl-llm-spcache-table cache))
         (phys (aref tbl lb)))
    (when (< phys 0)
      (setq phys (or (pop (nl-llm-spcache-free cache)) (error "nl-llm-spcache: out of blocks")))
      (aset tbl lb phys))
    (* (+ (* phys bs) off) kvdim)))

;; ---- one integrated pre-norm block: ternary linears + paged streaming attention

(defun nl-llm-integrated--attn (qr kr vr cache base)
  "Streaming sink+window paged attention for one token.  QR/KR/VR are the raw
\(un-RoPE'd) query/key/value vectors; this stores K/V at the token's paged slot,
attends the kept sink+window entries with cache-relative RoPE, and returns the
attention output (dim).  Shared by the LINFN and fused-resident blocks."
  (let* ((dim (nl-llm-spcache-dim cache)) (heads (nl-llm-spcache-heads cache))
         (kvh (nl-llm-spcache-kvh cache)) (hd (/ dim heads)) (kvdim (nl-llm-spcache-kvdim cache))
         (grp (/ heads kvh)) (nsink (nl-llm-spcache-nsink cache)) (win (nl-llm-spcache-win cache))
         (p (nl-llm-spcache-seen cache)) (scale (/ 1.0 (sqrt (float hd))))
         (kc (nl-llm-spcache-kpool cache)) (vc (nl-llm-spcache-vpool cache))
         (out (make-vector dim 0.0))
         (slot (nl-llm--scache-slot p nsink win))
         (start (max nsink (- p (1- win))))                ; oldest window stream pos kept
         (qcrel (if (< p nsink) p (+ nsink (- p start))))  ; query's cache-relative position
         (entries nil) (cbase (nl-llm-spcache--base cache slot)))
    (let ((t0 0)) (while (< t0 kvdim)
      (aset kc (+ cbase t0) (aref kr t0)) (aset vc (+ cbase t0) (aref vr t0)) (setq t0 (1+ t0))))
    (setf (nl-llm-spcache-seen cache) (1+ p))
    (let ((s 0) (lim (min nsink (1+ p)))) (while (< s lim)
      (push (cons (nl-llm-spcache--base cache s) s) entries) (setq s (1+ s))))
    (let ((s (max nsink start))) (while (<= s p)
      (push (cons (nl-llm-spcache--base cache (nl-llm--scache-slot s nsink win)) (+ nsink (- s start))) entries)
      (setq s (1+ s))))
    (setq entries (nreverse entries))
    (nl-llm--rope-heads qr 0 heads hd qcrel base)        ; query RoPE at its cache-relative position
    (dotimes (h heads)
      (let* ((c0q (* h hd)) (c0k (* (/ h grp) hd)) (ne (length entries))
             (scores (make-vector ne 0.0)) (mx -1.0e30) (e entries) (j 0))
        (while e
          (let* ((eb (car (car e))) (crel (cdr (car e))) (kb (+ eb c0k))
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
              (while e2 (let ((eb (car (car e2))))
                (setq accv (+ accv (* (/ (aref scores jj) sm) (aref vc (+ eb c0k t0))))))
                (setq e2 (cdr e2) jj (1+ jj)))
              (aset out (+ c0q t0) accv))
            (setq t0 (1+ t0)))))))
    out))

(defun nl-llm-integrated--blk (xrow blk cache linfn &optional rope-base)
  "Decode one token XROW (1 x dim) through one pre-norm block.  Every linear runs
through LINFN (XROW, weight-spec, bias-row) -> flat vector, so a ternary packed
LINFN makes the block BitNet; the KV is stored in the paged streaming CACHE and
attended sink+window with cache-relative RoPE (StreamingLLM + PagedAttention)."
  (let* ((dim (nl-llm-spcache-dim cache)) (base (or rope-base 10000.0))
         (a (nl-llm-rmsnorm xrow (plist-get blk :ln1g)))
         (qr (funcall linfn a (plist-get blk :wq) (plist-get blk :bq)))
         (kr (funcall linfn a (plist-get blk :wk) (plist-get blk :bk)))
         (vr (funcall linfn a (plist-get blk :wv) (plist-get blk :bv)))
         (out (nl-llm-integrated--attn qr kr vr cache base)))
    (let* ((attn (funcall linfn (photon-tensor (list 1 dim) out) (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (photon-tensor-add xrow (photon-tensor (list 1 dim) attn)))
           (bnorm (nl-llm-rmsnorm x1 (plist-get blk :ln2g)))
           (g (funcall linfn bnorm (plist-get blk :wg) (plist-get blk :bg)))
           (u (funcall linfn bnorm (plist-get blk :wu) (plist-get blk :bu)))
           (sd (photon-tensor-data (nl-llm-silu (photon-tensor (list 1 (length g)) g))))
           (hh (make-vector (length g) 0.0)))
      (dotimes (i (length g)) (aset hh i (* (aref sd i) (aref u i))))
      (photon-tensor-add x1 (photon-tensor (list 1 dim)
                                           (funcall linfn (photon-tensor (list 1 (length hh)) hh)
                                                    (plist-get blk :wd) (plist-get blk :bd)))))))

;; ---- fused resident block: 4 GPU dispatches/block (QKV, O, gate|up, D), weights
;; uploaded once.  Same math/cache as the LINFN block, far fewer dispatches/uploads.

(defun nl-llm-integrated-fused--blk (xrow rblk cache &optional rope-base)
  "Decode one token through a fused resident block RBLK
\(`nl-llm-bitnet-resident-block').  QKV is one dispatch, gate|up one dispatch, and
the resident weights are not re-uploaded -- otherwise identical to
`nl-llm-integrated--blk'."
  (let* ((dim (nl-llm-spcache-dim cache)) (base (or rope-base 10000.0))
         (a (nl-llm-rmsnorm xrow (plist-get rblk :ln1g)))
         (qkvspec (plist-get rblk :qkv)) (qkv (nl-llm-bitnet--run-fused-res a qkvspec))
         (sp (plist-get qkvspec :splits)) (dq (nth 0 sp)) (dk (nth 1 sp))
         (qr (cl-subseq qkv 0 dq)) (kr (cl-subseq qkv dq (+ dq dk))) (vr (cl-subseq qkv (+ dq dk)))
         (out (nl-llm-integrated--attn qr kr vr cache base)))
    (let* ((attn (nl-llm-bitnet--run-fused-res (photon-tensor (list 1 dim) out) (plist-get rblk :o)))
           (x1 (photon-tensor-add xrow (photon-tensor (list 1 dim) attn)))
           (bnorm (nl-llm-rmsnorm x1 (plist-get rblk :ln2g)))
           (guspec (plist-get rblk :gu)) (gu (nl-llm-bitnet--run-fused-res bnorm guspec))
           (ff (nth 0 (plist-get guspec :splits)))
           (sd (photon-tensor-data (nl-llm-silu (photon-tensor (list 1 ff) (cl-subseq gu 0 ff)))))
           (hh (make-vector ff 0.0)))
      (dotimes (i ff) (aset hh i (* (aref sd i) (aref gu (+ ff i)))))
      (photon-tensor-add x1 (photon-tensor (list 1 dim)
                                           (nl-llm-bitnet--run-fused-res (photon-tensor (list 1 ff) hh) (plist-get rblk :d)))))))

;;;###autoload
(defun nl-llm-bitnet-resident-block (blk)
  "Build a fused resident block from f32 weight plist BLK: Q|K|V and gate|up each
become one per-row-beta resident packed weight, O and down stay single, norms pass
through.  Returns a plist (:ln1g :ln2g :qkv :o :gu :d) for
`nl-llm-integrated-fused--blk'.  Free with `nl-llm-integrated-free-model'."
  (list :ln1g (plist-get blk :ln1g) :ln2g (plist-get blk :ln2g)
        :qkv (nl-llm-bitnet-pack-fused-res
              (list (plist-get blk :wq) (plist-get blk :wk) (plist-get blk :wv))
              (list (plist-get blk :bq) (plist-get blk :bk) (plist-get blk :bv)))
        :o (nl-llm-bitnet-pack-fused-res (list (plist-get blk :wo)) (list (plist-get blk :bo)))
        :gu (nl-llm-bitnet-pack-fused-res
             (list (plist-get blk :wg) (plist-get blk :wu)) (list (plist-get blk :bg) (plist-get blk :bu)))
        :d (nl-llm-bitnet-pack-fused-res (list (plist-get blk :wd)) (list (plist-get blk :bd)))))

;;;###autoload
(defun nl-llm-integrated-resident-model (blocks wte bh dim)
  "Upload a fully-resident fused model: each block via `nl-llm-bitnet-resident-block',
plus the packed tied embedding spec (for the embedding rows) and a resident head.
Returns a plist (:rblocks :wte-spec :head :dim :vocab)."
  (list :rblocks (mapcar #'nl-llm-bitnet-resident-block blocks)
        :wte-spec (nl-llm-bitnet-pack-wte wte)
        :head (nl-llm-bitnet-pack-fused-res (list wte) (list bh))
        :dim dim :vocab (car (photon-tensor-shape wte))))

(defun nl-llm-integrated-free-model (model)
  "Free every resident handle held by a `nl-llm-integrated-resident-model' MODEL."
  (dolist (rblk (plist-get model :rblocks))
    (dolist (k '(:qkv :o :gu :d)) (nl-llm-bitnet-free-fused-res (plist-get rblk k))))
  (nl-llm-bitnet-free-fused-res (plist-get model :head)))

(defun nl-llm-integrated-fused-h (token model caches lnfg &optional rope-base)
  "Embed TOKEN (packed-row), run the fused resident MODEL blocks with paged
streaming CACHES, return the post-final-RMSNorm hidden (1 x dim)."
  (let* ((ws (plist-get model :wte-spec)) (dim (plist-get model :dim))
         (packed (nth 0 ws)) (beta (nth 1 ws)) (fcount (nth 2 ws))
         (x (photon-tensor (list 1 dim) (nl-llm-bitnet-unpack-row packed beta fcount token dim)))
         (bl (plist-get model :rblocks)) (cl caches))
    (while bl (setq x (nl-llm-integrated-fused--blk x (car bl) (car cl) rope-base)) (setq bl (cdr bl) cl (cdr cl)))
    (nl-llm-rmsnorm x lnfg)))

;;;###autoload
(defun nl-llm-integrated-fused-spec-greedy (prompt nsteps model caches lnfg w2 b2 &optional rope-base)
  "Self-speculative greedy decode over the fused resident MODEL (MTP head W2,B2).
Output is exactly plain greedy; returns (TOKENS . ROUNDS).  Uses 4 GPU
dispatches/block + 1 head with no per-token weight upload."
  (let* ((vocab (plist-get model :vocab)) (head (plist-get model :head))
         (h nil) (out nil) (n 0) (rounds 0))
    (dolist (tk prompt) (setq h (nl-llm-integrated-fused-h tk model caches lnfg rope-base)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (nl-llm-spec-argmax (nl-llm-bitnet--run-fused-res h head) 0 vocab))
            (d  (nl-llm--head-argmax h w2 b2 vocab)))
        (push t1 out) (setq n (1+ n))
        (let* ((h1 (nl-llm-integrated-fused-h t1 model caches lnfg rope-base))
               (true2 (nl-llm-spec-argmax (nl-llm-bitnet--run-fused-res h1 head) 0 vocab)))
          (if (and (= d true2) (< n nsteps))
              (progn (push d out) (setq n (1+ n))
                     (setq h (nl-llm-integrated-fused-h d model caches lnfg rope-base)))
            (setq h h1)))))
    (cons (nreverse out) rounds)))

(defun nl-llm-integrated-h (token blocks caches embed-fn linfn lnfg dim &optional rope-base)
  "Embed TOKEN (via EMBED-FN -> dim vector), run BLOCKS with paged streaming
CACHES and ternary LINFN, return the post-final-RMSNorm hidden (1 x dim)."
  (let* ((x (photon-tensor (list 1 dim) (funcall embed-fn token)))
         (bl blocks) (cl caches))
    (while bl (setq x (nl-llm-integrated--blk x (car bl) (car cl) linfn rope-base)) (setq bl (cdr bl) cl (cdr cl)))
    (nl-llm-rmsnorm x lnfg)))

;; ---- f32 reference plumbing (no GPU): exercises the paged/streaming layers only

(defun nl-llm-integrated-linfn-f32 (a w b)
  "Plain f32 linear A.W^T + B as a flat vector (LINFN for the reference path)."
  (photon-tensor-data (photon-tensor-linear a w b)))

(defun nl-llm-integrated-embed-f32 (wte dim)
  "Return an EMBED-FN that gathers row TOKEN from the f32 WTE."
  (let ((wd (photon-tensor-data wte)))
    (lambda (token) (let ((v (make-vector dim 0.0)))
                      (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v))))

;; ---- full ternary model: greedy + MTP-speculative greedy (GPU packed kernel)

(defun nl-llm-integrated--ternary-fns (wte-spec dim)
  "Return (EMBED-FN . LINFN) for the fully-ternary model from packed WTE-SPEC."
  (let ((packed (nth 0 wte-spec)) (beta (nth 1 wte-spec)) (fcount (nth 2 wte-spec)))
    (cons (lambda (token) (nl-llm-bitnet-unpack-row packed beta fcount token dim))
          #'nl-llm-bitnet--run1)))

;;;###autoload
(defun nl-llm-integrated-greedy (prompt nsteps pblocks caches wte-spec lnfg bh dim vocab &optional rope-base)
  "Plain greedy decode over the fully-integrated model (ternary weights + paged
streaming KV).  Feeds PROMPT, generates NSTEPS tokens with the packed tied head.
Returns the generated id list.  CACHES are `nl-llm-spcache' (mutated)."
  (let* ((fns (nl-llm-integrated--ternary-fns wte-spec dim)) (embed-fn (car fns)) (linfn (cdr fns))
         (h nil) (out nil))
    (dolist (tk prompt) (setq h (nl-llm-integrated-h tk pblocks caches embed-fn linfn lnfg dim rope-base)))
    (dotimes (_ nsteps)
      (let ((g (nl-llm-spec-argmax (nl-llm-bitnet--run1 h wte-spec bh) 0 vocab)))
        (push g out)
        (setq h (nl-llm-integrated-h g pblocks caches embed-fn linfn lnfg dim rope-base))))
    (nreverse out)))

;;;###autoload
(defun nl-llm-integrated-spec-greedy (prompt nsteps pblocks caches wte-spec lnfg bh w2 b2 dim vocab &optional rope-base)
  "Self-speculative greedy decode over the fully-integrated model, with MTP head
W2,B2 (f32, predicting the token two ahead) drafting the look-ahead token.  Output
is EXACTLY `nl-llm-integrated-greedy'; returns (TOKENS . ROUNDS) so NSTEPS/ROUNDS
is the mean tokens accepted per target forward (the speedup).  Only accepted
tokens enter the cache, so the streaming ring is never rolled back."
  (let* ((fns (nl-llm-integrated--ternary-fns wte-spec dim)) (embed-fn (car fns)) (linfn (cdr fns))
         (h nil) (out nil) (n 0) (rounds 0))
    (dolist (tk prompt) (setq h (nl-llm-integrated-h tk pblocks caches embed-fn linfn lnfg dim rope-base)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (nl-llm-spec-argmax (nl-llm-bitnet--run1 h wte-spec bh) 0 vocab))   ; greedy next
            (d  (nl-llm--head-argmax h w2 b2 vocab)))                                ; MTP draft (token after t1)
        (push t1 out) (setq n (1+ n))
        (let* ((h1 (nl-llm-integrated-h t1 pblocks caches embed-fn linfn lnfg dim rope-base))
               (true2 (nl-llm-spec-argmax (nl-llm-bitnet--run1 h1 wte-spec bh) 0 vocab)))
          (if (and (= d true2) (< n nsteps))
              (progn (push d out) (setq n (1+ n))                                    ; draft correct: keep d
                     (setq h (nl-llm-integrated-h d pblocks caches embed-fn linfn lnfg dim rope-base)))
            (setq h h1)))))                                                          ; reject: d never entered cache
    (cons (nreverse out) rounds)))

;; ---- attention ON the GPU: KV pools resident, cache-relative RoPE + softmax in
;; a kernel, no q/k/v read-back and no CPU attention loop.  The streaming/paged
;; bookkeeping (which slots are kept, their cache-relative ranks) stays on the CPU
;; -- it is cheap integer work -- and is shipped to the kernel as a small entry
;; list.  Only the heavy per-head RoPE/score/softmax/weighted-sum runs on the GPU.

(defun nl-llm-integrated--rope-tables (cap hd rbase)
  "cos/sin tables (CAP x hd/2) for cache-relative RoPE, indexed [pos*half + m],
matching `nl-llm--rope-block' (theta = pos / RBASE^(2m/hd))."
  (let* ((half (/ hd 2)) (co (make-vector (* cap half) 0.0)) (si (make-vector (* cap half) 0.0)))
    (dotimes (p cap) (dotimes (m half)
      (let ((th (/ (float p) (expt rbase (/ (* 2.0 m) (float hd))))))
        (aset co (+ (* p half) m) (cos th)) (aset si (+ (* p half) m) (sin th)))))
    (cons co si)))

(defun nl-llm-integrated--stream-plan (cache)
  "Advance streaming CACHE by one token; return (WBASE ENTV QCREL NE): physical
base to append this token's K/V, the kept-entry list as a flat float vector
\[base crel base crel ...], the query's cache-relative position, and the count."
  (let* ((nsink (nl-llm-spcache-nsink cache)) (win (nl-llm-spcache-win cache))
         (p (nl-llm-spcache-seen cache)) (slot (nl-llm--scache-slot p nsink win))
         (start (max nsink (- p (1- win)))) (qcrel (if (< p nsink) p (+ nsink (- p start))))
         (wbase (nl-llm-spcache--base cache slot)) (ents nil))
    (setf (nl-llm-spcache-seen cache) (1+ p))
    (let ((s 0) (lim (min nsink (1+ p)))) (while (< s lim) (push (cons (nl-llm-spcache--base cache s) s) ents) (setq s (1+ s))))
    (let ((s (max nsink start))) (while (<= s p)
      (push (cons (nl-llm-spcache--base cache (nl-llm--scache-slot s nsink win)) (+ nsink (- s start))) ents) (setq s (1+ s))))
    (setq ents (nreverse ents))
    (let* ((ne (length ents)) (ev (make-vector (* 2 ne) 0.0)) (i 0))
      (dolist (e ents) (aset ev (* 2 i) (float (car e))) (aset ev (+ (* 2 i) 1) (float (cdr e))) (setq i (1+ i)))
      (list wbase ev qcrel ne))))

(defun nl-llm-integrated-gpattn--blk (xrow rblk pool model wbase entv ne qcrel)
  "One block with attention ON the GPU.  RBLK = resident fused weights, POOL =
this block's (CK . CV) resident KV handles, MODEL holds shared scratch + RoPE
tables + dims.  QKV stays resident, is appended + attended on the GPU, and only
the post-O attention vector returns to the CPU for the residual + FFN."
  (let* ((dim (plist-get model :dim)) (heads (plist-get model :heads)) (kvh (plist-get model :kvh))
         (hd (/ dim heads)) (kvdim (* kvh hd)) (qkvn (+ dim (* 2 kvdim))) (psz (plist-get model :poolsz))
         (co (plist-get model :co)) (si (plist-get model :si)) (cosz (plist-get model :cosz))
         (a-h (plist-get model :a-h)) (qkv-h (plist-get model :qkv-h)) (attn-h (plist-get model :attn-h))
         (ck (car pool)) (cv (cdr pool))
         (a (nl-llm-rmsnorm xrow (plist-get rblk :ln1g))))
    (nelisp-gpu-server-write-resident a-h (copy-sequence (photon-tensor-data a)))
    (nl-llm-bitnet--run-fused-res-io a-h qkv-h (plist-get rblk :qkv))                  ; QKV (resident in/out)
    (nelisp-gpu-server-run2 'qkv-append                                                ; append raw K/V to pools
      (list (list 'res qkv-h qkvn) (list 'res ck psz) (list 'res cv psz))
      (vector dim kvdim wbase) (/ (+ kvdim 63) 64))
    (nelisp-gpu-server-run2 'attn-stream-entries                                       ; GPU streaming attention
      (list (list 'res qkv-h qkvn) (list 'res ck psz) (list 'res cv psz)
            (cons 'in entv) (list 'res co cosz) (list 'res si cosz) (list 'res attn-h dim))
      (vector dim heads kvh ne qcrel) (/ (+ dim 63) 64))
    (let* ((attn (nl-llm-bitnet--run-fused-res-rin attn-h (plist-get rblk :o)))        ; O (resident in -> CPU)
           (x1 (photon-tensor-add xrow (photon-tensor (list 1 dim) attn)))
           (bnorm (nl-llm-rmsnorm x1 (plist-get rblk :ln2g)))
           (guspec (plist-get rblk :gu)) (gu (nl-llm-bitnet--run-fused-res bnorm guspec))
           (ff (nth 0 (plist-get guspec :splits)))
           (sd (photon-tensor-data (nl-llm-silu (photon-tensor (list 1 ff) (cl-subseq gu 0 ff)))))
           (hh (make-vector ff 0.0)))
      (dotimes (i ff) (aset hh i (* (aref sd i) (aref gu (+ ff i)))))
      (photon-tensor-add x1 (photon-tensor (list 1 dim)
                                           (nl-llm-bitnet--run-fused-res (photon-tensor (list 1 ff) hh) (plist-get rblk :d)))))))

;;;###autoload
(defun nl-llm-integrated-gpattn-model (blocks wte bh dim heads kvh nsink win bs &optional rope-base)
  "Build a fully-resident model whose attention runs on the GPU: resident fused
weights per block (`nl-llm-bitnet-resident-block'), a resident (CK . CV) KV pool
per block, shared scratch + cache-relative RoPE tables + a resident head.  Free
with `nl-llm-integrated-gpattn-free'."
  (let* ((hd (/ dim heads)) (kvdim (* kvh hd)) (cap (+ nsink win)) (nlblk (/ (+ cap bs -1) bs))
         (psz (* nlblk bs kvdim)) (cosz (* cap (/ hd 2)))
         (tabs (nl-llm-integrated--rope-tables cap hd (or rope-base 10000.0))))
    (list :rblocks (mapcar #'nl-llm-bitnet-resident-block blocks)
          :pools (mapcar (lambda (_) (cons (nelisp-gpu-server-upload (make-vector psz 0.0))
                                           (nelisp-gpu-server-upload (make-vector psz 0.0)))) blocks)
          :wte-spec (nl-llm-bitnet-pack-wte wte)
          :head (nl-llm-bitnet-pack-fused-res (list wte) (list bh))
          :co (nelisp-gpu-server-upload (car tabs)) :si (nelisp-gpu-server-upload (cdr tabs))
          :a-h (nelisp-gpu-server-upload (make-vector dim 0.0))
          :qkv-h (nelisp-gpu-server-upload (make-vector (+ dim (* 2 kvdim)) 0.0))
          :attn-h (nelisp-gpu-server-upload (make-vector dim 0.0))
          :dim dim :heads heads :kvh kvh :vocab (car (photon-tensor-shape wte))
          :nsink nsink :win win :bs bs :poolsz psz :cosz cosz)))

(defun nl-llm-integrated-gpattn-free (model)
  "Free every resident handle of a GPU-attention MODEL."
  (dolist (rblk (plist-get model :rblocks))
    (dolist (k '(:qkv :o :gu :d)) (nl-llm-bitnet-free-fused-res (plist-get rblk k))))
  (nl-llm-bitnet-free-fused-res (plist-get model :head))
  (dolist (p (plist-get model :pools)) (ignore-errors (nelisp-gpu-server-free (car p))) (ignore-errors (nelisp-gpu-server-free (cdr p))))
  (dolist (k '(:co :si :a-h :qkv-h :attn-h)) (ignore-errors (nelisp-gpu-server-free (plist-get model k)))))

(defun nl-llm-integrated-gpattn-h (token model book lnfg)
  "Embed TOKEN, run MODEL's GPU-attention blocks with streaming bookkeeping BOOK
\(an `nl-llm-spcache' used only for slot/entry computation), return the hidden."
  (let* ((plan (nl-llm-integrated--stream-plan book))
         (wbase (nth 0 plan)) (entv (nth 1 plan)) (qcrel (nth 2 plan)) (ne (nth 3 plan))
         (ws (plist-get model :wte-spec)) (dim (plist-get model :dim))
         (packed (nth 0 ws)) (beta (nth 1 ws)) (fcount (nth 2 ws))
         (x (photon-tensor (list 1 dim) (nl-llm-bitnet-unpack-row packed beta fcount token dim)))
         (bl (plist-get model :rblocks)) (pl (plist-get model :pools)))
    (while bl
      (setq x (nl-llm-integrated-gpattn--blk x (car bl) (car pl) model wbase entv ne qcrel))
      (setq bl (cdr bl) pl (cdr pl)))
    (nl-llm-rmsnorm x lnfg)))

(defun nl-llm-integrated-gpattn--book (model)
  (nl-llm-spcache-new (plist-get model :nsink) (plist-get model :win)
                      (plist-get model :dim) (plist-get model :heads) (plist-get model :kvh) (plist-get model :bs)))

;;;###autoload
(defun nl-llm-integrated-gpattn-spec-greedy (prompt nsteps model lnfg w2 b2)
  "Self-speculative greedy decode with attention ON the GPU.  Same lossless output
as the CPU-attention path; returns (TOKENS . ROUNDS)."
  (let* ((vocab (plist-get model :vocab)) (head (plist-get model :head))
         (book (nl-llm-integrated-gpattn--book model)) (h nil) (out nil) (n 0) (rounds 0))
    (dolist (tk prompt) (setq h (nl-llm-integrated-gpattn-h tk model book lnfg)))
    (while (< n nsteps)
      (setq rounds (1+ rounds))
      (let ((t1 (nl-llm-spec-argmax (nl-llm-bitnet--run-fused-res h head) 0 vocab))
            (d  (nl-llm--head-argmax h w2 b2 vocab)))
        (push t1 out) (setq n (1+ n))
        (let* ((h1 (nl-llm-integrated-gpattn-h t1 model book lnfg))
               (true2 (nl-llm-spec-argmax (nl-llm-bitnet--run-fused-res h1 head) 0 vocab)))
          (if (and (= d true2) (< n nsteps))
              (progn (push d out) (setq n (1+ n))
                     (setq h (nl-llm-integrated-gpattn-h d model book lnfg)))
            (setq h h1)))))
    (cons (nreverse out) rounds)))

(provide 'nl-llm-integrated)
;;; nl-llm-integrated.el ends here
