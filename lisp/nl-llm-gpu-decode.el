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

(provide 'nl-llm-gpu-decode)
;;; nl-llm-gpu-decode.el ends here
