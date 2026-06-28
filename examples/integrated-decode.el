;;; integrated-decode.el --- end-to-end decode with ALL FOUR techniques at once  -*- lexical-binding: t; -*-
;; A single decode loop that runs the four GIGAZINE techniques together, each at
;; its own layer:
;;   BitNet b1.58   -- weights ternary+packed, every matmul on the GPU (no f32 W)
;;   StreamingLLM   -- KV bounded at NSINK sink + WIN window, cache-relative RoPE
;;   PagedAttention -- that KV stored in a block pool behind a non-identity table
;;   Speculative    -- an MTP look-ahead head drafts +2; correct drafts land two
;;                     tokens per target forward; output is byte-identical greedy
;;
;; The MTP draft head is fit in-script on a short rollout of THIS model (a few SGD
;; steps -- closed-form CE gradient on a linear head), purely to show a realistic
;; acceptance rate; the lossless guarantee holds for any head.  Needs a Vulkan
;; device (the ternary kernel runs on the GPU).
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/integrated-decode.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-bitnet)
(require 'nl-llm-integrated)
(require 'nl-llm-gpu)

(defun idemo--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun idemo--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun idemo--mkblk (dim kvdim ff s0)
  (list :ln1g (idemo--ones dim) :wq (idemo--t (list dim dim) (+ s0 1) 0.4) :bq (idemo--t (list dim) (+ s0 11) 0.1)
        :wk (idemo--t (list kvdim dim) (+ s0 2) 0.4) :bk (idemo--t (list kvdim) (+ s0 12) 0.1)
        :wv (idemo--t (list kvdim dim) (+ s0 3) 0.4) :bv (idemo--t (list kvdim) (+ s0 13) 0.1)
        :wo (idemo--t (list dim dim) (+ s0 4) 0.4) :bo (idemo--t (list dim) (+ s0 14) 0.1) :ln2g (idemo--ones dim)
        :wg (idemo--t (list ff dim) (+ s0 5) 0.4) :bg (idemo--t (list ff) (+ s0 15) 0.1)
        :wu (idemo--t (list ff dim) (+ s0 6) 0.4) :bu (idemo--t (list ff) (+ s0 16) 0.1)
        :wd (idemo--t (list dim ff) (+ s0 7) 0.4) :bd (idemo--t (list dim) (+ s0 17) 0.1)))

;; greedy rollout that also returns the per-step hidden vectors (for MTP fitting)
(defun idemo--rollout (prompt nsteps pblocks caches wspec lnfg bh dim vocab)
  (let* ((fns (nl-llm-integrated--ternary-fns wspec dim)) (embed (car fns)) (linfn (cdr fns))
         (h nil) (toks nil) (hs nil))
    (dolist (tk prompt) (setq h (nl-llm-integrated-h tk pblocks caches embed linfn lnfg dim)))
    (dotimes (_ nsteps)
      (push (copy-sequence (photon-tensor-data h)) hs)
      (let ((g (nl-llm-spec-argmax (nl-llm-bitnet--run1 h wspec bh) 0 vocab)))
        (push g toks) (setq h (nl-llm-integrated-h g pblocks caches embed linfn lnfg dim))))
    (cons (nreverse toks) (nreverse hs))))

;; fit a linear MTP head (vocab x dim) to predict the token two ahead, by SGD with
;; the closed-form cross-entropy gradient (softmax - onehot) (x) hidden.
(defun idemo--fit-mtp (hs toks dim vocab epochs lr)
  (let ((w (make-vector (* vocab dim) 0.0)) (b (make-vector vocab 0.0))
        (pairs nil))
    (dotimes (k (1- (length toks))) (push (cons (nth k hs) (nth (1+ k) toks)) pairs))  ; (h_k, token_{k+1})
    (setq pairs (nreverse pairs))
    (dotimes (_ epochs)
      (dolist (pr pairs)
        (let* ((hv (car pr)) (tgt (cdr pr)) (lg (make-vector vocab 0.0)) (mx -1e30) (s 0.0))
          (dotimes (o vocab) (let ((acc (aref b o)) (base (* o dim)))
            (dotimes (j dim) (setq acc (+ acc (* (aref w (+ base j)) (aref hv j))))) (aset lg o acc) (when (> acc mx) (setq mx acc))))
          (dotimes (o vocab) (aset lg o (exp (- (aref lg o) mx))) (setq s (+ s (aref lg o))))
          (dotimes (o vocab) (let* ((p (/ (aref lg o) s)) (gd (- p (if (= o tgt) 1.0 0.0))) (base (* o dim)))
            (dotimes (j dim) (aset w (+ base j) (- (aref w (+ base j)) (* lr gd (aref hv j)))))
            (aset b o (- (aref b o) (* lr gd))))))) )
    (cons (photon-tensor (list vocab dim) w) (photon-tensor (list vocab) b))))

(let* ((dim 32) (heads 4) (kvh 2) (ff 48) (vocab 24) (hd (/ dim heads)) (kvdim (* kvh hd))
       (nblk 2) (nsink 4) (win 16) (bs 8)
       (wte (idemo--t (list vocab dim) 1 0.4)) (lnfg (idemo--ones dim)) (bh (idemo--t (list vocab) 19 0.1))
       (blocks (list (idemo--mkblk dim kvdim ff 100) (idemo--mkblk dim kvdim ff 200)))
       (prompt '(3 1 4 1 5 9 2 6)) (nsteps 40))
  (princ "=== 4-technique integrated decode (BitNet + StreamingLLM + PagedAttention + Speculative) ===\n")
  (unless (nl-llm-gpu-enable)
    (princ "SKIP: no Vulkan device (the ternary matmuls run on the GPU)\n") (kill-emacs 0))
  (let* ((pblocks (mapcar #'nl-llm-bitnet-pack-block blocks))
         (wspec (nl-llm-bitnet-pack-wte wte))
         (mkc (lambda () (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks)))
         ;; --- (1) plain greedy + collect hiddens, then fit the MTP draft head
         (roll (idemo--rollout prompt nsteps pblocks (funcall mkc) wspec lnfg bh dim vocab))
         (greedy (car roll))
         (mtp (idemo--fit-mtp (cdr roll) greedy dim vocab 60 0.5)) (w2 (car mtp)) (b2 (cdr mtp))
         ;; --- (2) speculative decode with the fitted MTP head (lossless)
         (sc (funcall mkc))
         (sp (nl-llm-integrated-spec-greedy prompt nsteps pblocks sc wspec lnfg bh w2 b2 dim vocab))
         (spec (car sp)) (rounds (cdr sp))
         ;; weight VRAM (whole model, f32 vs packed incl. wte)
         (f32 (* vocab dim 4)) (pk (* vocab (/ (+ dim (1- nl-llm-bitnet-pk)) nl-llm-bitnet-pk) 4)))
    (dolist (blk blocks) (let ((bb (nl-llm-bitnet-block-bytes blk))) (setq f32 (+ f32 (car bb)) pk (+ pk (cdr bb)))))
    (nl-llm-gpu-disable)
    (princ (format "\nmodel: dim=%d heads=%d/%d kv ff=%d vocab=%d blocks=%d\n" dim heads kvh ff vocab nblk))
    (princ (format "prompt=%S, generated %d tokens\n\n" prompt nsteps))
    (princ "[BitNet b1.58]   weights ternary+packed, every matmul on GPU (no f32 W matrix)\n")
    (princ (format "                 whole-model weight VRAM %.1f KB -> %.1f KB (%.1fx smaller)\n"
                   (/ f32 1024.0) (/ pk 1024.0) (/ (float f32) pk)))
    (princ "[StreamingLLM]   KV bounded: keep sink + window, cache-relative RoPE\n")
    (let ((c (car sc)))
      (princ (format "                 cap = %d sink + %d window = %d tokens; decoded %d positions, fill=%d (bounded)\n"
                     nsink win (+ nsink win) (+ (length prompt) nsteps) (nl-llm-spcache-fill c))))
    (princ "[PagedAttention] that KV stored in a block pool behind a per-cache table\n")
    (let ((c (car sc)))
      (princ (format "                 block size %d, %d blocks allocated, table[0..]=%S (non-identity)\n"
                     bs (nl-llm-spcache-used-blocks c)
                     (append (cl-subseq (nl-llm-spcache-table c) 0 (min 4 (nl-llm-spcache-nlblk c))) nil))))
    (princ "[Speculative]    MTP head drafts +2; correct drafts -> 2 tokens / forward\n")
    (princ (format "                 %d tokens in %d target forwards = %.2f tokens/forward (%.0f%% drafts accepted)\n"
                   nsteps rounds (/ (float nsteps) rounds) (* 100.0 (/ (float (- nsteps rounds)) nsteps))))
    (princ (format "\nlossless check: speculative stream == plain greedy : %s\n" (if (equal spec greedy) "IDENTICAL" "*** DIFFER ***")))
    (princ (format "first 16 tokens: %S\n" (cl-subseq spec 0 (min 16 (length spec)))))
    (kill-emacs (if (equal spec greedy) 0 1))))
;;; integrated-decode.el ends here
