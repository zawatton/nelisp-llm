;;; bench-longctx.el --- long-context benchmark: tok/forward + memory  -*- lexical-binding: t; -*-
;; Drives the 4-technique integrated decode (nl-llm-integrated.el) over a long
;; generation and measures the two things long context cares about:
;;   * tokens / target forward -- the speculative speedup, with the MTP draft head
;;     fit on a short self-rollout of THIS model (lossless: spec stream == greedy).
;;   * memory -- (a) BitNet shrinks every weight byte 8x (constant in length), and
;;     (b) StreamingLLM + PagedAttention hold the KV in a sink+window block pool
;;     whose size is CONSTANT in sequence length, vs a naive cache that grows O(L).
;;     The win therefore grows without bound as context lengthens; we decode a real
;;     long run to show the cache stays pinned, then project the KV bytes out to
;;     128k tokens.
;; Needs a Vulkan device (the ternary matmuls run on the GPU).
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/bench-longctx.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-bitnet)
(require 'nl-llm-integrated)
(require 'nl-llm-gpu)

(defun blc--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun blc--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun blc--mkblk (dim kvdim ff s0)
  (list :ln1g (blc--ones dim) :wq (blc--t (list dim dim) (+ s0 1) 0.4) :bq (blc--t (list dim) (+ s0 11) 0.1)
        :wk (blc--t (list kvdim dim) (+ s0 2) 0.4) :bk (blc--t (list kvdim) (+ s0 12) 0.1)
        :wv (blc--t (list kvdim dim) (+ s0 3) 0.4) :bv (blc--t (list kvdim) (+ s0 13) 0.1)
        :wo (blc--t (list dim dim) (+ s0 4) 0.4) :bo (blc--t (list dim) (+ s0 14) 0.1) :ln2g (blc--ones dim)
        :wg (blc--t (list ff dim) (+ s0 5) 0.4) :bg (blc--t (list ff) (+ s0 15) 0.1)
        :wu (blc--t (list ff dim) (+ s0 6) 0.4) :bu (blc--t (list ff) (+ s0 16) 0.1)
        :wd (blc--t (list dim ff) (+ s0 7) 0.4) :bd (blc--t (list dim) (+ s0 17) 0.1)))

(defun blc--rollout (prompt nsteps pblocks caches wspec lnfg bh dim vocab)
  "Greedy rollout returning (TOKENS . PER-STEP-HIDDENS) for MTP fitting."
  (let* ((fns (nl-llm-integrated--ternary-fns wspec dim)) (embed (car fns)) (linfn (cdr fns))
         (h nil) (toks nil) (hs nil))
    (dolist (tk prompt) (setq h (nl-llm-integrated-h tk pblocks caches embed linfn lnfg dim)))
    (dotimes (_ nsteps)
      (push (copy-sequence (photon-tensor-data h)) hs)
      (let ((g (nl-llm-spec-argmax (nl-llm-bitnet--run1 h wspec bh) 0 vocab)))
        (push g toks) (setq h (nl-llm-integrated-h g pblocks caches embed linfn lnfg dim))))
    (cons (nreverse toks) (nreverse hs))))

(defun blc--fit-mtp (hs toks dim vocab epochs lr)
  "Fit a linear MTP head to predict the token two ahead (closed-form CE gradient)."
  (let ((w (make-vector (* vocab dim) 0.0)) (b (make-vector vocab 0.0)) (pairs nil))
    (dotimes (k (1- (length toks))) (push (cons (nth k hs) (nth (1+ k) toks)) pairs))
    (setq pairs (nreverse pairs))
    (dotimes (_ epochs)
      (dolist (pr pairs)
        (let* ((hv (car pr)) (tgt (cdr pr)) (lg (make-vector vocab 0.0)) (mx -1e30) (s 0.0))
          (dotimes (o vocab) (let ((acc (aref b o)) (base (* o dim)))
            (dotimes (j dim) (setq acc (+ acc (* (aref w (+ base j)) (aref hv j))))) (aset lg o acc) (when (> acc mx) (setq mx acc))))
          (dotimes (o vocab) (aset lg o (exp (- (aref lg o) mx))) (setq s (+ s (aref lg o))))
          (dotimes (o vocab) (let* ((p (/ (aref lg o) s)) (gd (- p (if (= o tgt) 1.0 0.0))) (base (* o dim)))
            (dotimes (j dim) (aset w (+ base j) (- (aref w (+ base j)) (* lr gd (aref hv j)))))
            (aset b o (- (aref b o) (* lr gd))))))))
    (cons (photon-tensor (list vocab dim) w) (photon-tensor (list vocab) b))))

(defun blc--mb (bytes) (/ bytes 1048576.0))

(let* ((dim 64) (heads 8) (kvh 4) (ff 128) (vocab 48) (hd (/ dim heads)) (kvdim (* kvh hd))
       (nblk 4) (nsink 4) (win 124) (cap (+ nsink win)) (bs 32)
       (gen 512) (fit-steps 96)
       (wte (blc--t (list vocab dim) 1 0.4)) (lnfg (blc--ones dim)) (bh (blc--t (list vocab) 19 0.1))
       (blocks (cl-loop for i below nblk collect (blc--mkblk dim kvdim ff (* 100 (1+ i)))))
       (prompt '(3 1 4 1 5 9 2 6 5 3 5)))
  (princ "=== long-context benchmark: tokens/forward + memory (4-technique decode) ===\n")
  (unless (nl-llm-gpu-enable)
    (princ "SKIP: no Vulkan device (the ternary matmuls run on the GPU)\n") (kill-emacs 0))
  (let* ((pblocks (mapcar #'nl-llm-bitnet-pack-block blocks))
         (wspec (nl-llm-bitnet-pack-wte wte))
         (mkc (lambda () (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks)))
         ;; fit the MTP draft head on a short rollout
         (roll (blc--rollout prompt fit-steps pblocks (funcall mkc) wspec lnfg bh dim vocab))
         (mtp (blc--fit-mtp (cdr roll) (car roll) dim vocab 60 0.5)) (w2 (car mtp)) (b2 (cdr mtp))
         ;; (A) timed speculative decode -- NON-FUSED: 7 dispatches/block, weights
         ;;     re-uploaded every token (the plain --run1 path).
         (sc (funcall mkc)) (t0 (float-time))
         (sp (nl-llm-integrated-spec-greedy prompt gen pblocks sc wspec lnfg bh w2 b2 dim vocab))
         (dt-nf (- (float-time) t0)) (spec (car sp)) (rounds (cdr sp))
         ;; (B) timed speculative decode -- FUSED RESIDENT: Q|K|V and gate|up each
         ;;     one dispatch (4/block), weights uploaded once and kept resident.
         (rmodel (nl-llm-integrated-resident-model blocks wte bh dim))
         (fc (funcall mkc)) (t1 (float-time))
         (fsp (nl-llm-integrated-fused-spec-greedy prompt gen rmodel fc lnfg w2 b2))
         (dt-f (- (float-time) t1)) (fspec (car fsp)) (frounds (cdr fsp))
         ;; (C) timed speculative decode -- GPU ATTENTION: KV pools resident, RoPE +
         ;;     softmax in a kernel, no q/k/v read-back and no CPU attention loop.
         (gmodel (nl-llm-integrated-gpattn-model blocks wte bh dim heads kvh nsink win bs))
         (t2 (float-time))
         (gsp (nl-llm-integrated-gpattn-spec-greedy prompt gen gmodel lnfg w2 b2))
         (dt-g (- (float-time) t2)) (gspec (car gsp))
         ;; dispatches per block forward: non-fused / fused-CPU-attn / GPU-attn
         (disp-nf (* 7 nblk)) (disp-f (* 4 nblk)) (disp-g (* 6 nblk))
         ;; weight memory (whole model incl. tied wte): f32 vs packed
         (wf32 (* vocab dim 4)) (wpk (* vocab (/ (+ dim (1- nl-llm-bitnet-pk)) nl-llm-bitnet-pk) 4))
         ;; per-layer per-token KV bytes (K+V, f32)
         (kv-per-tok (* kvdim 2 4 nblk)))
    (nl-llm-integrated-free-model rmodel)
    (nl-llm-integrated-gpattn-free gmodel)
    (dolist (blk blocks) (let ((bb (nl-llm-bitnet-block-bytes blk))) (setq wf32 (+ wf32 (car bb)) wpk (+ wpk (cdr bb)))))
    (nl-llm-gpu-disable)
    (princ (format "\nmodel: dim=%d heads=%d/%d kv ff=%d vocab=%d layers=%d | sink=%d win=%d (cap=%d) blk=%d\n"
                   dim heads kvh ff vocab nblk nsink win cap bs))
    (princ (format "decoded: prompt %d + generated %d = %d positions\n\n"
                   (length prompt) gen (+ (length prompt) gen)))

    (princ "-- dispatch reduction + GPU attention --\n")
    (princ (format "  non-fused : %2d disp/block (7 linears, weight re-upload every token), CPU attention\n" disp-nf))
    (princ (format "  fused     : %2d disp/block (QKV, O, gate|up, D; weights resident), CPU attention\n" disp-f))
    (princ (format "  gpu-attn  : %2d disp/block (+ append + in-kernel RoPE/softmax attention; KV resident)\n" disp-g))
    (princ (format "  wall: non-fused %.1fs (%.1f tok/s) | fused %.1fs (%.1f tok/s, %.2fx) | gpu-attn %.1fs (%.1f tok/s, %.2fx)\n"
                   dt-nf (/ gen dt-nf) dt-f (/ gen dt-f) (/ dt-nf dt-f) dt-g (/ gen dt-g) (/ dt-nf dt-g)))
    (princ (format "  identical output (non-fused == fused == gpu-attn == greedy): %s\n\n"
                   (if (and (equal fspec spec) (equal gspec spec)) "YES" "*** NO ***")))

    (princ "-- tokens / forward (speculative MTP) --\n")
    (princ (format "  %d tokens in %d target forwards = %.2f tok/forward (%.0f%% drafts accepted)\n\n"
                   gen rounds (/ (float gen) rounds) (* 100.0 (/ (float (- gen rounds)) gen))))

    (princ "-- weight memory (BitNet, constant in context length) --\n")
    (princ (format "  f32 %.2f MB -> ternary-packed %.2f MB (%.1fx smaller)\n\n" (blc--mb wf32) (blc--mb wpk) (/ (float wf32) wpk)))

    (princ "-- KV cache memory (StreamingLLM + PagedAttention vs naive growing cache) --\n")
    (let ((c (car sc)))
      (princ (format "  at this run (%d positions): naive %.2f MB -> bounded %.2f MB (fill=%d slots, %d blocks, %.1fx)\n"
                     (+ (length prompt) gen)
                     (blc--mb (* (+ (length prompt) gen) kv-per-tok))
                     (blc--mb (* cap kv-per-tok))
                     (nl-llm-spcache-fill c) (nl-llm-spcache-used-blocks c)
                     (/ (float (* (+ (length prompt) gen) kv-per-tok)) (* cap kv-per-tok)))))
    (princ "  projection (KV stays pinned at cap; naive grows O(L)):\n")
    (princ (format "    %-12s %14s %14s %10s\n" "context L" "naive KV" "bounded KV" "savings"))
    (dolist (L '(1024 4096 16384 65536 262144))
      (let ((naive (* L kv-per-tok)) (bound (* cap kv-per-tok)))
        (princ (format "    %-12d %11.2f MB %11.2f MB %8.0fx\n" L (blc--mb naive) (blc--mb bound) (/ (float naive) bound)))))
    (princ (format "\n  total resident @ 262144 ctx: naive %.1f MB (f32 W + growing KV) -> integrated %.2f MB (ternary W + pinned KV)\n"
                   (blc--mb (+ wf32 (* 262144 kv-per-tok))) (blc--mb (+ wpk (* cap kv-per-tok)))))
    (kill-emacs (if (and (equal fspec spec) (equal gspec spec)) 0 1))))
;;; bench-longctx.el ends here
