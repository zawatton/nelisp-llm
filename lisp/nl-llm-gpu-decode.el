;;; nl-llm-gpu-decode.el --- on-GPU KV-cache incremental decode  -*- lexical-binding: t; -*-

;; Runs generation on the GPU: the model weights and a per-block key/value cache
;; stay resident, and each token is decoded by ONE fused command buffer (compiled
;; once, re-submitted per step) that computes only the new position -- projections
;; + RoPE at the runtime position, an append into the resident cache, and a
;; single-query attention over the cache (length read from a resident POS buffer).
;; Only the token index and POS are refreshed per step.  Numerically the same
;; model as the CPU decode (nl-llm-decode) / prefill forward; verified in
;; test/gpu-decode-test.el.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)      ; server + bin path
(require 'nl-llm-gpu-ag)   ; builder: nlga ops, slots, compile/step
(require 'nl-llm-decode)   ; CPU dcache / decode-step / decode-h (context prefill)
(require 'nl-llm-spec)     ; argmax helpers for chain-draft speculative decode

(defun nl-llm-gpu--block-consts (b blk)
  "Wrap a block tensor plist BLK into a plist of resident const rts."
  (let ((o nil) (kv blk))
    (while kv (push (car kv) o) (push (nlga-const b (cadr kv)) o) (setq kv (cddr kv)))
    (nreverse o)))

(defun nl-llm-gpu--cache (b max-seq kvdim)
  "Allocate a resident (max-seq x kvdim) cache rt (persists across steps)."
  (let ((h (nelisp-gpu-server-upload (make-vector (* max-seq kvdim) 0.0))))
    (nlga-rt--make :slot (nlga--slot b (list 'res h (* max-seq kvdim)))
                   :rows max-seq :cols kvdim :handle h)))

(defun nl-llm-gpu--rope1 (b x cosr sinr sign pos cols heads)
  "RoPE one row X (1 x cols) at runtime POS; return a new (1 x cols) rt."
  (let ((os (nlga--tmp b cols)))
    (nlga--d b (list 'decode-rope
                     (list (nlga-rt-slot x) (nlga-rt-slot cosr) (nlga-rt-slot sinr)
                           (nlga-rt-slot sign) (nlga-rt-slot pos) os)
                     (list cols heads) (nlga--g (/ cols 2))))
    (nlga-rt--make :slot os :rows 1 :cols cols)))

(defun nl-llm-gpu--cache-append (b src pos cache kvdim)
  "Append SRC (1 x kvdim) into CACHE at row POS (in place, resident)."
  (nlga--d b (list 'cache-append (list (nlga-rt-slot src) (nlga-rt-slot pos) (nlga-rt-slot cache))
                   (list kvdim) (nlga--g kvdim))))

(defun nl-llm-gpu--attn1 (b q ck cv pos dim heads kvh)
  "Single-query attention over CK/CV up to POS; return ctx (1 x dim)."
  (let ((os (nlga--tmp b dim)))
    (nlga--d b (list 'decode-attn (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv) (nlga-rt-slot pos) os)
                     (list dim heads kvh) (nlga--g dim)))
    (nlga-rt--make :slot os :rows 1 :cols dim)))

(defun nl-llm-gpu--decode-block (b x blk ck cv pos sign cosr sinr heads kvh dim kvdim)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nl-llm-gpu--rope1 b (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)) cosr sinr sign pos dim heads))
         (k (nl-llm-gpu--rope1 b (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)) cosr sinr sign pos kvdim kvh))
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv))))
    (nl-llm-gpu--cache-append b k pos ck kvdim)
    (nl-llm-gpu--cache-append b v pos cv kvdim)
    (let* ((ctx (nl-llm-gpu--attn1 b q ck cv pos dim heads kvh))
           (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (nlga-add b x attn))
           (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
      (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                  (plist-get blk :wu) (plist-get blk :bu)
                                  (plist-get blk :wd) (plist-get blk :bd))))))

;;;###autoload
(defun nl-llm-gpu-decode-new (wte blocks lnfg bh heads kvh dim vocab max-seq tables)
  "Build + compile an on-GPU KV-cache decoder for a weight-tied model.
WTE (vocab x dim), BLOCKS (list of tensor plists with biases), LNFG, BH, and
TABLES = (cos . sin) RoPE tables (max-seq x hd/2).  Returns a context plist for
`nl-llm-gpu-decode-step'.  The server must be running."
  (let* ((b (nlga-new)) (kvdim (* kvh (/ dim heads)))
         (tok (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (pos (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (sign (nlga-scalar b 1.0)) (one (nlga-scalar b 1.0))
         (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
         (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
         (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
         (caches (mapcar (lambda (_) (cons (nl-llm-gpu--cache b max-seq kvdim)
                                           (nl-llm-gpu--cache b max-seq kvdim))) blocks))
         (x (nlga-embed b tok wter)) (bl bconsts) (cl caches))
    (while bl
      (setq x (nl-llm-gpu--decode-block b x (car bl) (car (car cl)) (cdr (car cl))
                                        pos sign cosr sinr heads kvh dim kvdim))
      (setq bl (cdr bl) cl (cdr cl)))
    (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
      (nlga-compile b)
      (list :b b :tok tok :pos pos :lout lout))))

;;;###autoload
(defun nl-llm-gpu-decode-step (ctx token pos)
  "Decode TOKEN at position POS on the GPU; return the (vocab) logit vector.
Call once per position, in order (the cache grows in place)."
  (nlga-update (plist-get ctx :tok) (photon-tensor '(1) (vector (float token))))
  (nlga-update (plist-get ctx :pos) (photon-tensor '(1) (vector (float pos))))
  (nth (plist-get ctx :lout) (nlga-step (plist-get ctx :b))))

;;;###autoload
(defun nl-llm-gpu-decode-free (ctx)
  "Free the compiled decoder CTX."
  (nlga-free (plist-get ctx :b)))

;; --- batched decode: B sequences in parallel, shared position --------
(defun nl-llm-gpu--rope-b (b x cosr sinr sign pos cols heads bsz)
  (let ((os (nlga--tmp b (* bsz cols))))
    (nlga--d b (list 'decode-rope-b (list (nlga-rt-slot x) (nlga-rt-slot cosr) (nlga-rt-slot sinr)
                                          (nlga-rt-slot sign) (nlga-rt-slot pos) os)
                     (list bsz cols heads) (nlga--g (/ (* bsz cols) 2))))
    (nlga-rt--make :slot os :rows bsz :cols cols)))

(defun nl-llm-gpu--cache-append-b (b src pos cache kvdim bsz maxseq)
  (nlga--d b (list 'cache-append-b (list (nlga-rt-slot src) (nlga-rt-slot pos) (nlga-rt-slot cache))
                   (list bsz kvdim maxseq) (nlga--g (* bsz kvdim)))))

(defun nl-llm-gpu--attn-b (b q ck cv pos dim heads kvh bsz maxseq)
  (let ((os (nlga--tmp b (* bsz dim))))
    (nlga--d b (list 'decode-attn-b (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv) (nlga-rt-slot pos) os)
                     (list bsz dim heads kvh maxseq) (nlga--g (* bsz dim))))
    (nlga-rt--make :slot os :rows bsz :cols dim)))

(defun nl-llm-gpu--cache-b (b bsz maxseq kvdim)
  (let ((h (nelisp-gpu-server-upload (make-vector (* bsz maxseq kvdim) 0.0))))
    (nlga-rt--make :slot (nlga--slot b (list 'res h (* bsz maxseq kvdim)))
                   :rows (* bsz maxseq) :cols kvdim :handle h)))

(defun nl-llm-gpu--decode-block-b (b x blk ck cv pos sign cosr sinr heads kvh dim kvdim bsz maxseq)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nl-llm-gpu--rope-b b (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)) cosr sinr sign pos dim heads bsz))
         (k (nl-llm-gpu--rope-b b (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)) cosr sinr sign pos kvdim kvh bsz))
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv))))
    (nl-llm-gpu--cache-append-b b k pos ck kvdim bsz maxseq)
    (nl-llm-gpu--cache-append-b b v pos cv kvdim bsz maxseq)
    (let* ((ctx (nl-llm-gpu--attn-b b q ck cv pos dim heads kvh bsz maxseq))
           (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (nlga-add b x attn))
           (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
      (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                  (plist-get blk :wu) (plist-get blk :bu)
                                  (plist-get blk :wd) (plist-get blk :bd))))))

;;;###autoload
(defun nl-llm-gpu-decode-batch-new (wte blocks lnfg bh heads kvh dim vocab max-seq bsz tables)
  "Like `nl-llm-gpu-decode-new' but decodes BSZ sequences in parallel (each with
its own resident key/value cache, sharing one position).  Returns a context for
`nl-llm-gpu-decode-batch-step'."
  (let* ((b (nlga-new)) (kvdim (* kvh (/ dim heads)))
         (tok (nlga-const b (photon-tensor (list bsz) (make-vector bsz 0.0))))
         (pos (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (sign (nlga-scalar b 1.0)) (one (nlga-scalar b 1.0))
         (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
         (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
         (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
         (caches (mapcar (lambda (_) (cons (nl-llm-gpu--cache-b b bsz max-seq kvdim)
                                           (nl-llm-gpu--cache-b b bsz max-seq kvdim))) blocks))
         (x (nlga-embed b tok wter)) (bl bconsts) (cl caches))
    (while bl
      (setq x (nl-llm-gpu--decode-block-b b x (car bl) (car (car cl)) (cdr (car cl))
                                          pos sign cosr sinr heads kvh dim kvdim bsz max-seq))
      (setq bl (cdr bl) cl (cdr cl)))
    (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
      (nlga-compile b)
      (list :b b :tok tok :pos pos :lout lout :bsz bsz :vocab vocab))))

;;;###autoload
(defun nl-llm-gpu-decode-batch-step (ctx tokens pos)
  "Decode one token per sequence on the GPU.  TOKENS is a vector of BSZ token ids
\(all at position POS).  Returns the flat (BSZ*vocab) logits; sequence s's logits
are at [s*vocab .. (s+1)*vocab)."
  (let ((bsz (plist-get ctx :bsz)))
    (nlga-update (plist-get ctx :tok)
                 (photon-tensor (list bsz) (let ((v (make-vector bsz 0.0)))
                                             (dotimes (i bsz) (aset v i (float (aref tokens i)))) v)))
    (nlga-update (plist-get ctx :pos) (photon-tensor '(1) (vector (float pos))))
    (nth (plist-get ctx :lout) (nlga-step (plist-get ctx :b)))))

;; --- StreamingLLM bounded decode: sink + window, cache-relative RoPE ----
;; Keys/values are stored RAW in a cap=(nsink+win) ring cache; the attention
;; kernel rotates the query by its cache-relative position and each key by its
;; cache-relative rank, so memory stays bounded at cap while relative offsets
;; stay in distribution.  CPU oracle: nl-llm-stream (docs/design/02).
(defun nl-llm-gpu--cache-append-ring (b src pos cache kvdim nsink win)
  "Append SRC (1 x kvdim, RAW) into the ring CACHE at the slot for POS[0]."
  (nlga--d b (list 'cache-append-ring (list (nlga-rt-slot src) (nlga-rt-slot pos) (nlga-rt-slot cache))
                   (list kvdim nsink win) (nlga--g kvdim))))

(defun nl-llm-gpu--attn-stream (b q ck cv pos cosr sinr dim heads kvh nsink win)
  "Cache-relative-RoPE single-query attention over the ring CK/CV; ctx (1 x dim)."
  (let ((os (nlga--tmp b dim)))
    (nlga--d b (list 'decode-attn-stream
                     (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv) (nlga-rt-slot pos)
                           (nlga-rt-slot cosr) (nlga-rt-slot sinr) os)
                     (list dim heads kvh nsink win) (nlga--g dim)))
    (nlga-rt--make :slot os :rows 1 :cols dim)))

(defun nl-llm-gpu--decode-block-stream (b x blk ck cv pos cosr sinr heads kvh dim kvdim nsink win)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)))   ; RAW; rotated in attn
         (k (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)))   ; RAW; rotated in attn
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv))))
    (nl-llm-gpu--cache-append-ring b k pos ck kvdim nsink win)
    (nl-llm-gpu--cache-append-ring b v pos cv kvdim nsink win)
    (let* ((ctx (nl-llm-gpu--attn-stream b q ck cv pos cosr sinr dim heads kvh nsink win))
           (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (nlga-add b x attn))
           (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
      (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                  (plist-get blk :wu) (plist-get blk :bu)
                                  (plist-get blk :wd) (plist-get blk :bd))))))

;;;###autoload
(defun nl-llm-gpu-stream-new (wte blocks lnfg bh heads kvh dim vocab nsink win tables)
  "On-GPU StreamingLLM decoder: KV cache bounded at NSINK+WIN with cache-relative
RoPE.  Same model as `nl-llm-gpu-decode-new'; TABLES = (cos . sin) RoPE tables
with at least NSINK+WIN rows.  Returns a context for `nl-llm-gpu-stream-step'."
  (let* ((b (nlga-new)) (kvdim (* kvh (/ dim heads))) (cap (+ nsink win))
         (tok (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (pos (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (one (nlga-scalar b 1.0))
         (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
         (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
         (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
         (caches (mapcar (lambda (_) (cons (nl-llm-gpu--cache b cap kvdim)
                                           (nl-llm-gpu--cache b cap kvdim))) blocks))
         (x (nlga-embed b tok wter)) (bl bconsts) (cl caches))
    (while bl
      (setq x (nl-llm-gpu--decode-block-stream b x (car bl) (car (car cl)) (cdr (car cl))
                                               pos cosr sinr heads kvh dim kvdim nsink win))
      (setq bl (cdr bl) cl (cdr cl)))
    (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
      (nlga-compile b)
      (list :b b :tok tok :pos pos :lout lout))))

;;;###autoload
(defun nl-llm-gpu-stream-step (ctx token pos)
  "Decode TOKEN at absolute stream position POS on the GPU (bounded cache).
Call once per position, in order; the ring cache is updated in place."
  (nlga-update (plist-get ctx :tok) (photon-tensor '(1) (vector (float token))))
  (nlga-update (plist-get ctx :pos) (photon-tensor '(1) (vector (float pos))))
  (nth (plist-get ctx :lout) (nlga-step (plist-get ctx :b))))

;;;###autoload
(defalias 'nl-llm-gpu-stream-free 'nl-llm-gpu-decode-free
  "Free a streaming decoder context (see `nl-llm-gpu-decode-free').")

;; --- PagedAttention: block-paged KV pool + per-sequence block table ----
;; B sequences share one resident KV POOL of fixed-size blocks per layer; a
;; layer-independent block TABLE (B x mbps, logical block -> physical block id)
;; is filled on demand by a host allocator, so blocks are assigned only as
;; positions cross block boundaries.  Decode is synchronous (shared POS) and
;; produces logits identical to the contiguous batch decode (docs/design/03).
(defun nl-llm-gpu--cache-append-paged (b src pos table pool kvdim bsz bs mbps)
  (nlga--d b (list 'cache-append-paged
                   (list (nlga-rt-slot src) (nlga-rt-slot pos) (nlga-rt-slot table) (nlga-rt-slot pool))
                   (list bsz kvdim bs mbps) (nlga--g (* bsz kvdim)))))

(defun nl-llm-gpu--attn-paged (b q ck cv pos table dim heads kvh bsz bs mbps)
  (let ((os (nlga--tmp b (* bsz dim))))
    (nlga--d b (list 'decode-attn-paged
                     (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv) (nlga-rt-slot pos) (nlga-rt-slot table) os)
                     (list bsz dim heads kvh bs mbps) (nlga--g (* bsz dim))))
    (nlga-rt--make :slot os :rows bsz :cols dim)))

(defun nl-llm-gpu--decode-block-paged (b x blk ck cv pos table sign cosr sinr heads kvh dim kvdim bsz bs mbps)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nl-llm-gpu--rope-b b (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)) cosr sinr sign pos dim heads bsz))
         (k (nl-llm-gpu--rope-b b (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)) cosr sinr sign pos kvdim kvh bsz))
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv))))
    (nl-llm-gpu--cache-append-paged b k pos table ck kvdim bsz bs mbps)
    (nl-llm-gpu--cache-append-paged b v pos table cv kvdim bsz bs mbps)
    (let* ((ctx (nl-llm-gpu--attn-paged b q ck cv pos table dim heads kvh bsz bs mbps))
           (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (nlga-add b x attn))
           (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
      (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                  (plist-get blk :wu) (plist-get blk :bu)
                                  (plist-get blk :wd) (plist-get blk :bd))))))

;;;###autoload
(defun nl-llm-gpu-paged-new (wte blocks lnfg bh heads kvh dim vocab bs mbps bsz tables)
  "On-GPU PagedAttention batch decoder for BSZ sequences.  KV lives in a shared
per-layer block POOL (block size BS, MBPS blocks per sequence, so max length
BS*MBPS); a host allocator fills the block table on demand.  TABLES = (cos . sin)
RoPE tables with >= BS*MBPS rows.  Returns a context for `nl-llm-gpu-paged-step'."
  (let* ((b (nlga-new)) (kvdim (* kvh (/ dim heads))) (poolrows (* bsz mbps bs))
         (tok (nlga-const b (photon-tensor (list bsz) (make-vector bsz 0.0))))
         (pos (nlga-const b (photon-tensor '(1) (vector 0.0))))
         (table (nlga-const b (photon-tensor (list (* bsz mbps)) (make-vector (* bsz mbps) 0.0))))
         (sign (nlga-scalar b 1.0)) (one (nlga-scalar b 1.0))
         (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
         (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
         (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
         (pools (mapcar (lambda (_) (cons (nl-llm-gpu--cache b poolrows kvdim)
                                          (nl-llm-gpu--cache b poolrows kvdim))) blocks))
         (x (nlga-embed b tok wter)) (bl bconsts) (cl pools))
    (while bl
      (setq x (nl-llm-gpu--decode-block-paged b x (car bl) (car (car cl)) (cdr (car cl))
                                              pos table sign cosr sinr heads kvh dim kvdim bsz bs mbps))
      (setq bl (cdr bl) cl (cdr cl)))
    (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
      (nlga-compile b)
      (list :b b :tok tok :pos pos :table table :lout lout
            :bsz bsz :vocab vocab :bs bs :mbps mbps :nblocks 0))))

;;;###autoload
(defun nl-llm-gpu-paged-step (ctx tokens pos)
  "Decode one token per sequence at shared position POS on the paged decoder.
TOKENS is a vector of BSZ ids.  On a block boundary the host allocator assigns
BSZ fresh physical blocks (interleaved: phys = logical*BSZ + seq) and rewrites
the resident block table.  Returns the flat (BSZ*vocab) logits."
  (let* ((bsz (plist-get ctx :bsz)) (bs (plist-get ctx :bs)) (mbps (plist-get ctx :mbps)))
    (when (= (% pos bs) 0)                          ; crossed into a new logical block
      (let* ((lb (/ pos bs)) (tab (make-vector (* bsz mbps) 0.0)))
        (dotimes (l (1+ lb)) (dotimes (s bsz)        ; on-demand: phys = l*bsz + s (interleaved)
          (aset tab (+ (* s mbps) l) (float (+ (* l bsz) s)))))
        (nlga-update (plist-get ctx :table) (photon-tensor (list (* bsz mbps)) tab))
        (plist-put ctx :nblocks (* (1+ lb) bsz))))
    (nlga-update (plist-get ctx :tok)
                 (photon-tensor (list bsz) (let ((v (make-vector bsz 0.0)))
                                             (dotimes (i bsz) (aset v i (float (aref tokens i)))) v)))
    (nlga-update (plist-get ctx :pos) (photon-tensor '(1) (vector (float pos))))
    (nth (plist-get ctx :lout) (nlga-step (plist-get ctx :b)))))

;;;###autoload
(defalias 'nl-llm-gpu-paged-free 'nl-llm-gpu-decode-free
  "Free a paged decoder context (see `nl-llm-gpu-decode-free').")

;; --- variable-length paged decode: per-sequence positions + free-list -------
;; A host-side block allocator: a free-list of physical block ids, a per-sequence
;; block table (logical -> physical, -1 = unallocated) and per-sequence lengths.
;; Blocks are handed out on demand and returned when a sequence is freed, and a
;; prefix can be SHARED across sequences (the shared blocks are counted once).
(cl-defstruct (nl-llm-paged-alloc (:constructor nl-llm-paged-alloc--make))
  free table lens nblocks bsz mbps bs)

(defun nl-llm-paged-alloc-new (nblocks bsz mbps bs)
  "Allocator over NBLOCKS physical blocks for BSZ sequences, MBPS logical blocks
each, block size BS."
  (let ((free nil) (i nblocks))
    (while (> i 0) (setq i (1- i)) (push i free))   ; free = (0 1 ... nblocks-1)
    (nl-llm-paged-alloc--make
     :free free :table (let ((v (make-vector (* bsz mbps) -1.0))) v) :lens (make-vector bsz 0.0)
     :nblocks nblocks :bsz bsz :mbps mbps :bs bs)))

(defun nl-llm-paged-alloc-used (a)
  "Physical blocks currently in use (allocated, not on the free-list)."
  (- (nl-llm-paged-alloc-nblocks a) (length (nl-llm-paged-alloc-free a))))

(defun nl-llm-paged-alloc--block (a)
  (or (pop (nl-llm-paged-alloc-free a)) (error "nl-llm-paged: out of blocks")))

(defun nl-llm-paged-len (a s) (truncate (aref (nl-llm-paged-alloc-lens a) s)))

(defun nl-llm-paged-ensure (a s)
  "Ensure sequence S has a physical block for its current write position; on a
block boundary this allocates a fresh block from the free-list."
  (let* ((mbps (nl-llm-paged-alloc-mbps a)) (bs (nl-llm-paged-alloc-bs a))
         (tbl (nl-llm-paged-alloc-table a)) (lb (/ (nl-llm-paged-len a s) bs)) (k (+ (* s mbps) lb)))
    (when (< (aref tbl k) 0.0) (aset tbl k (float (nl-llm-paged-alloc--block a))))))

(defun nl-llm-paged-advance (a s)
  "Record that sequence S decoded one token (length += 1)."
  (let ((lens (nl-llm-paged-alloc-lens a))) (aset lens s (+ (aref lens s) 1.0))))

(defun nl-llm-paged-free-seq (a s)
  "Return sequence S's privately-owned blocks to the free-list and reset it.
Blocks shared with another sequence (same physical id elsewhere in the table) are
left for their owner; this frees only ids unique to S."
  (let* ((mbps (nl-llm-paged-alloc-mbps a)) (tbl (nl-llm-paged-alloc-table a)) (lb 0))
    (while (< lb mbps)
      (let ((p (aref tbl (+ (* s mbps) lb))))
        (when (>= p 0.0)
          (let ((shared nil) (k 0) (n (length tbl)))
            (while (< k n) (when (and (/= k (+ (* s mbps) lb)) (= (aref tbl k) p)) (setq shared t)) (setq k (1+ k)))
            (unless shared (push (truncate p) (nl-llm-paged-alloc-free a))))
          (aset tbl (+ (* s mbps) lb) -1.0)))
      (setq lb (1+ lb)))
    (aset (nl-llm-paged-alloc-lens a) s 0.0)))

(defun nl-llm-paged-share-prefix (a dst src nprefix)
  "Point DST's first ceil(NPREFIX/bs) logical blocks at SRC's physical blocks
\(read-only prefix sharing) and set DST's length to NPREFIX.  NPREFIX must be a
multiple of the block size (whole-block sharing; partial-block COW is future)."
  (let* ((mbps (nl-llm-paged-alloc-mbps a)) (bs (nl-llm-paged-alloc-bs a)) (tbl (nl-llm-paged-alloc-table a))
         (nb (/ nprefix bs)) (lb 0))
    (unless (= (% nprefix bs) 0) (error "nl-llm-paged-share-prefix: NPREFIX must be a multiple of bs"))
    (while (< lb nb) (aset tbl (+ (* dst mbps) lb) (aref tbl (+ (* src mbps) lb))) (setq lb (1+ lb)))
    (aset (nl-llm-paged-alloc-lens a) dst (float nprefix))))

(defun nl-llm-gpu--decode-block-pv (b x blk ck cv lens table sign cosr sinr heads kvh dim kvdim bsz bs mbps)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nlga--rope-bv b (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)) cosr sinr sign lens dim heads bsz))
         (k (nlga--rope-bv b (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)) cosr sinr sign lens kvdim kvh bsz))
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv))))
    (nlga--d b (list 'cache-append-paged-v (list (nlga-rt-slot k) (nlga-rt-slot lens) (nlga-rt-slot table) (nlga-rt-slot ck))
                     (list bsz kvdim bs mbps) (nlga--g (* bsz kvdim))))
    (nlga--d b (list 'cache-append-paged-v (list (nlga-rt-slot v) (nlga-rt-slot lens) (nlga-rt-slot table) (nlga-rt-slot cv))
                     (list bsz kvdim bs mbps) (nlga--g (* bsz kvdim))))
    (let* ((os (nlga--tmp b (* bsz dim)))
           (ctx (progn (nlga--d b (list 'decode-attn-paged-v
                                        (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv) (nlga-rt-slot lens) (nlga-rt-slot table) os)
                                        (list bsz dim heads kvh bs mbps) (nlga--g (* bsz dim))))
                       (nlga-rt--make :slot os :rows bsz :cols dim)))
           (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
           (x1 (nlga-add b x attn))
           (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
      (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                  (plist-get blk :wu) (plist-get blk :bu)
                                  (plist-get blk :wd) (plist-get blk :bd))))))

(defun nlga--rope-bv (b x cosr sinr sign lens cols heads bsz)
  "Per-sequence RoPE: B rows X, row b rotated at LENS[b]."
  (let ((os (nlga--tmp b (* bsz cols))))
    (nlga--d b (list 'decode-rope-b-v (list (nlga-rt-slot x) (nlga-rt-slot cosr) (nlga-rt-slot sinr)
                                            (nlga-rt-slot sign) (nlga-rt-slot lens) os)
                     (list bsz cols heads) (nlga--g (/ (* bsz cols) 2))))
    (nlga-rt--make :slot os :rows bsz :cols cols)))

;;;###autoload
(defun nl-llm-gpu-paged-v-new (wte blocks lnfg bh heads kvh dim vocab nblocks bs mbps bsz tables)
  "Variable-length PagedAttention decoder: BSZ sequences at independent positions
over a shared pool of NBLOCKS blocks (size BS, MBPS logical/seq).  Returns a
context for `nl-llm-gpu-paged-v-step'; pair it with a `nl-llm-paged-alloc-new'."
  (let* ((b (nlga-new)) (kvdim (* kvh (/ dim heads))) (poolrows (* nblocks bs))
         (tok (nlga-const b (photon-tensor (list bsz) (make-vector bsz 0.0))))
         (lens (nlga-const b (photon-tensor (list bsz) (make-vector bsz 0.0))))
         (table (nlga-const b (photon-tensor (list (* bsz mbps)) (make-vector (* bsz mbps) 0.0))))
         (sign (nlga-scalar b 1.0)) (one (nlga-scalar b 1.0))
         (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
         (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
         (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
         (pools (mapcar (lambda (_) (cons (nl-llm-gpu--cache b poolrows kvdim)
                                          (nl-llm-gpu--cache b poolrows kvdim))) blocks))
         (x (nlga-embed b tok wter)) (bl bconsts) (cl pools))
    (while bl
      (setq x (nl-llm-gpu--decode-block-pv b x (car bl) (car (car cl)) (cdr (car cl))
                                           lens table sign cosr sinr heads kvh dim kvdim bsz bs mbps))
      (setq bl (cdr bl) cl (cdr cl)))
    (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
      (nlga-compile b)
      (list :b b :tok tok :lens lens :table table :lout lout :bsz bsz :vocab vocab))))

;;;###autoload
(defun nl-llm-gpu-paged-v-step (ctx alloc tokens)
  "Decode one token per sequence with per-sequence positions.  TOKENS is a vector
of BSZ ids.  ALLOC supplies the current per-sequence lengths and block table
\(refresh ALLOC -- ensure blocks, share prefixes -- before calling).  Returns the
flat (BSZ*vocab) logits.  Advancing lengths after the step is the caller's job
\(`nl-llm-paged-advance')."
  (let ((bsz (plist-get ctx :bsz)))
    (nlga-update (plist-get ctx :tok)
                 (photon-tensor (list bsz) (let ((v (make-vector bsz 0.0)))
                                             (dotimes (i bsz) (aset v i (float (aref tokens i)))) v)))
    (nlga-update (plist-get ctx :lens) (photon-tensor (list bsz) (copy-sequence (nl-llm-paged-alloc-lens alloc))))
    (nlga-update (plist-get ctx :table) (photon-tensor (list (* bsz (nl-llm-paged-alloc-mbps alloc)))
                                                       (let ((tb (copy-sequence (nl-llm-paged-alloc-table alloc))))
                                                         (dotimes (i (length tb)) (when (< (aref tb i) 0.0) (aset tb i 0.0))) tb)))
    (nth (plist-get ctx :lout) (nlga-step (plist-get ctx :b)))))

;;;###autoload
(defalias 'nl-llm-gpu-paged-v-free 'nl-llm-gpu-decode-free
  "Free a variable-length paged decoder context.")

;; --- end-to-end tree verify: a whole draft tree in one fused forward ---------
;; Decode the accepted context (CPU prefill) to get each block's K/V cache, then
;; run all M tree nodes through every block in one nlga batch -- per-node Q/K/V +
;; per-node RoPE + tree-attn (context + ancestor mask) + SwiGLU -- and a head per
;; node, so one forward yields the next-token logits at every tree node.
(defun nl-llm-gpu--head-vec (src n)
  (let ((v (make-vector n 0.0))) (dotimes (i n) (aset v i (aref src i))) v))

(defun nl-llm-gpu--tree-attn (b q ck cv nk nv parent lenb dim heads kvh m maxdepth)
  (let ((os (nlga--tmp b (* m dim))))
    (nlga--d b (list 'tree-attn (list (nlga-rt-slot q) (nlga-rt-slot ck) (nlga-rt-slot cv)
                                      (nlga-rt-slot nk) (nlga-rt-slot nv) (nlga-rt-slot parent) (nlga-rt-slot lenb) os)
                     (list m dim heads kvh maxdepth) (nlga--g (* m dim))))
    (nlga-rt--make :slot os :rows m :cols dim)))

(defun nl-llm-gpu--tree-block (b x blk ckb cvb parent lenb sign cosr sinr posns heads kvh dim kvdim m maxdepth)
  (let* ((a (nlga-rmsnorm b x (plist-get blk :ln1g)))
         (q (nlga--rope-bv b (nlga-linear b a (plist-get blk :wq) (plist-get blk :bq)) cosr sinr sign posns dim heads m))
         (k (nlga--rope-bv b (nlga-linear b a (plist-get blk :wk) (plist-get blk :bk)) cosr sinr sign posns kvdim kvh m))
         (v (nlga-linear b a (plist-get blk :wv) (plist-get blk :bv)))
         (ctx (nl-llm-gpu--tree-attn b q ckb cvb k v parent lenb dim heads kvh m maxdepth))
         (attn (nlga-linear b ctx (plist-get blk :wo) (plist-get blk :bo)))
         (x1 (nlga-add b x attn))
         (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g))))
    (nlga-add b x1 (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                                (plist-get blk :wu) (plist-get blk :bu)
                                (plist-get blk :wd) (plist-get blk :bd)))))

;;;###autoload
(defun nl-llm-gpu-tree-verify (blocks wte lnfg bh heads kvh dim vocab ctx-tokens nodes parents positions tables maxdepth)
  "One-forward tree verify.  CTX-TOKENS is the accepted sequence; NODES is a list
of M draft-tree token ids with PARENTS (parent index per node, sentinel = M for a
root) and per-node POSITIONS.  Returns a list of M (vocab) logit vectors -- the
next-token logits at every tree node -- computed in a single fused GPU batch where
each node attends the shared context plus its ancestor chain (tree-attn).  TABLES
= (cos . sin) with > max(POSITIONS) rows.  The server must be running."
  (let* ((m (length nodes)) (kvdim (* kvh (/ dim heads))) (L (length ctx-tokens))
         (cdc (mapcar (lambda (_) (nl-llm-dcache-new (max 1 L) dim heads kvh)) blocks)))
    (dolist (tk ctx-tokens) (nl-llm-decode-step tk blocks cdc wte lnfg bh dim))  ; CPU prefill -> per-block K/V
    (let* ((b (nlga-new))
           (tok (nlga-const b (photon-tensor (list m) (let ((v (make-vector m 0.0)) (i 0))
                                                        (dolist (tk nodes) (aset v i (float tk)) (setq i (1+ i))) v))))
           (posns (nlga-const b (photon-tensor (list m) (apply #'vector (mapcar #'float positions)))))
           (parent (nlga-const b (photon-tensor (list m) (apply #'vector (mapcar #'float parents)))))
           (lenb (nlga-const b (photon-tensor '(1) (vector (float L)))))
           (sign (nlga-scalar b 1.0)) (one (nlga-scalar b 1.0))
           (wter (nlga-const b wte)) (lnfgr (nlga-const b lnfg)) (bhr (nlga-const b bh))
           (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
           (bconsts (mapcar (lambda (blk) (nl-llm-gpu--block-consts b blk)) blocks))
           (cks (mapcar (lambda (dc)
                          (cons (nlga-const b (photon-tensor (list L kvdim) (nl-llm-gpu--head-vec (nl-llm-dcache-k dc) (* L kvdim))))
                                (nlga-const b (photon-tensor (list L kvdim) (nl-llm-gpu--head-vec (nl-llm-dcache-v dc) (* L kvdim)))))) cdc))
           (x (nlga-embed b tok wter)) (bl bconsts) (cl cks))
      (while bl
        (setq x (nl-llm-gpu--tree-block b x (car bl) (car (car cl)) (cdr (car cl))
                                        parent lenb sign cosr sinr posns heads kvh dim kvdim m maxdepth))
        (setq bl (cdr bl) cl (cdr cl)))
      (let ((lout (nlga-keep b (nlga-linear b (nlga-rmsnorm b x lnfgr) wter bhr) one)))
        (nlga-compile b)
        (let ((flat (nth lout (nlga-step b))) (res nil))
          (nlga-free b)
          (dotimes (i m) (push (let ((v (make-vector vocab 0.0)))
                                 (dotimes (j vocab) (aset v j (aref flat (+ (* i vocab) j)))) v) res))
          (nreverse res))))))

;;;###autoload
(defun nl-llm-gpu-spec-chain-decode (prompt nsteps blocks wte lnfg bh mtp-heads heads kvh dim vocab maxseq tables maxdepth)
  "Chain-draft speculative decode with multiple MTP heads + GPU tree verify.
MTP-HEADS is a list of (W . B) for head2, head3, ...; the draft depth is
1 + (length MTP-HEADS).  Each round drafts a depth-long chain from the current
hidden (head_k drafts the token k ahead), verifies the whole chain in ONE
`nl-llm-gpu-tree-verify' forward, accepts the longest greedily-matching prefix and
emits accepted + 1 bonus token.  Returns (TOKENS . FORWARDS); TOKENS is exactly
plain greedy (lossless) and length(TOKENS)/FORWARDS is the mean tokens per verify."
  (let* ((depth (1+ (length mtp-heads))) (acc (append prompt nil)) (out nil) (forwards 0))
    (while (< (length out) nsteps)
      (setq forwards (1+ forwards))
      (let* ((cdc (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)) (h nil) (L (length acc)))
        (dolist (tk acc) (setq h (nl-llm-decode-h tk blocks cdc wte lnfg dim)))   ; hidden of last accepted token
        (let* ((drafts (cons (nl-llm--head-argmax h wte bh vocab)                 ; head1 (tied) + MTP heads
                             (mapcar (lambda (wb) (nl-llm--head-argmax h (car wb) (cdr wb) vocab)) mtp-heads)))
               (parents (let ((p nil) (k 0)) (while (< k depth) (push (float (if (= k 0) depth (1- k))) p) (setq k (1+ k))) (nreverse p)))
               (positions (let ((p nil) (k 0)) (while (< k depth) (push (+ L k) p) (setq k (1+ k))) (nreverse p)))
               (pa (nl-llm-gpu-tree-verify blocks wte lnfg bh heads kvh dim vocab acc drafts parents positions tables maxdepth))
               (a 1))                                                             ; node0 (= argmax head1) always accepted
          (while (and (< a depth) (= (nth a drafts) (nl-llm-spec-argmax (nth (1- a) pa) 0 vocab))) (setq a (1+ a)))
          (let ((emit (append (cl-subseq drafts 0 a) (list (nl-llm-spec-argmax (nth (1- a) pa) 0 vocab)))))
            (dolist (tk emit) (when (< (length out) nsteps) (push tk out) (setq acc (append acc (list tk)))))))))
    (cons (nreverse out) forwards)))

(provide 'nl-llm-gpu-decode)
;;; nl-llm-gpu-decode.el ends here
