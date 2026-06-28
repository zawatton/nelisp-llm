;;; integrated-test.el --- 4-technique end-to-end decode losslessness  -*- lexical-binding: t; -*-
;; Pins the two transparency properties of the integrated decode:
;;   1. paged storage reproduces the flat streaming cache to f32 (CPU, always) --
;;      so PagedAttention is a transparent storage layer under StreamingLLM, even
;;      once the window evicts.
;;   2. MTP-speculative greedy == plain greedy on the identical ternary/streaming/
;;      paged model (needs a Vulkan device for the packed kernel) -- so the
;;      speculative generation layer is lossless over the full BitNet stack.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/integrated-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-stream)
(require 'nl-llm-bitnet)
(require 'nl-llm-integrated)
(require 'nl-llm-gpu)

(defvar it--fail 0)
(defun it--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq it--fail (1+ it--fail)) "FAIL")) (or extra ""))))
(defun it--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun it--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun it--mkblk (mkt dim kvdim ff s0)
  (list :ln1g (it--ones dim) :wq (funcall mkt (list dim dim) (+ s0 1) 0.4) :bq (funcall mkt (list dim) (+ s0 11) 0.1)
        :wk (funcall mkt (list kvdim dim) (+ s0 2) 0.4) :bk (funcall mkt (list kvdim) (+ s0 12) 0.1)
        :wv (funcall mkt (list kvdim dim) (+ s0 3) 0.4) :bv (funcall mkt (list kvdim) (+ s0 13) 0.1)
        :wo (funcall mkt (list dim dim) (+ s0 4) 0.4) :bo (funcall mkt (list dim) (+ s0 14) 0.1) :ln2g (it--ones dim)
        :wg (funcall mkt (list ff dim) (+ s0 5) 0.4) :bg (funcall mkt (list ff) (+ s0 15) 0.1)
        :wu (funcall mkt (list ff dim) (+ s0 6) 0.4) :bu (funcall mkt (list ff) (+ s0 16) 0.1)
        :wd (funcall mkt (list dim ff) (+ s0 7) 0.4) :bd (funcall mkt (list dim) (+ s0 17) 0.1)))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd))
       (nsink 2) (win 4) (bs 3) (seq '(3 1 4 1 5 9 2 6 8 7 0 5 2 9))
       (wte (it--t (list vocab dim) 1 0.4)) (lnfg (it--ones dim)) (bh (it--t (list vocab) 19 0.1))
       (blocks (list (it--mkblk #'it--t dim kvdim ff 100) (it--mkblk #'it--t dim kvdim ff 200))))

  ;; --- Check 1 (CPU): paged streaming == flat streaming, through window eviction
  (let* ((scaches (mapcar (lambda (_) (nl-llm-scache-new nsink win dim heads kvh)) blocks))
         (pcaches (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks))
         (embed-fn (nl-llm-integrated-embed-f32 wte dim)) (maxrel 0.0))
    (dolist (tk seq)
      ;; flat streaming oracle
      (let ((flat (nl-llm-stream-step tk blocks scaches wte lnfg bh dim))
            ;; integrated paged streaming, f32 linfn -> logits via the f32 tied head
            (h (let ((x (photon-tensor (list 1 dim) (funcall embed-fn tk))) (bl blocks) (cl pcaches))
                 (while bl (setq x (nl-llm-integrated--blk x (car bl) (car cl) #'nl-llm-integrated-linfn-f32))
                   (setq bl (cdr bl) cl (cdr cl)))
                 (nl-llm-rmsnorm x lnfg))))
        (let ((paged (photon-tensor-data (photon-tensor-linear h wte bh))) (j 0))
          (while (< j vocab) (setq maxrel (max maxrel (/ (abs (- (aref flat j) (aref paged j))) (max 1e-3 (abs (aref flat j)))))) (setq j (1+ j))))))
    (it--ck "paged streaming == flat streaming (f32, w/ eviction)" (< maxrel 1e-3) (format "maxrel=%.2e over %d toks" maxrel (length seq)))
    ;; the cache really stayed bounded and really used >1 physical block
    (let ((c (car pcaches)))
      (it--ck "KV bounded at sink+win and genuinely paged"
              (and (<= (nl-llm-spcache-fill c) (+ nsink win)) (>= (nl-llm-spcache-used-blocks c) 2))
              (format "fill=%d cap=%d blocks=%d" (nl-llm-spcache-fill c) (+ nsink win) (nl-llm-spcache-used-blocks c)))))

  ;; --- Check 2 (GPU): MTP-speculative greedy == plain greedy, ternary model
  (if (not (nl-llm-gpu-enable))
      (it--ck "speculative == greedy (ternary) [SKIPPED: no GPU]" t)
    (let* ((pblocks (mapcar #'nl-llm-bitnet-pack-block blocks))
           (wspec (nl-llm-bitnet-pack-wte wte))
           (w2 (it--t (list vocab dim) 71 0.4)) (b2 (it--t (list vocab) 73 0.1))
           (prompt '(3 1 4 1 5)) (nsteps 12)
           (g (nl-llm-integrated-greedy prompt nsteps pblocks
                (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks)
                wspec lnfg bh dim vocab))
           (sp (nl-llm-integrated-spec-greedy prompt nsteps pblocks
                 (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks)
                 wspec lnfg bh w2 b2 dim vocab))
           ;; Check 3: fused resident decode (QKV/gate|up one dispatch, weights resident)
           (rmodel (nl-llm-integrated-resident-model blocks wte bh dim))
           (fsp (nl-llm-integrated-fused-spec-greedy prompt nsteps rmodel
                  (mapcar (lambda (_) (nl-llm-spcache-new nsink win dim heads kvh bs)) blocks) lnfg w2 b2))
           ;; Check 4: attention ON the GPU (KV resident, RoPE+softmax in-kernel)
           (gmodel (nl-llm-integrated-gpattn-model blocks wte bh dim heads kvh nsink win bs))
           (gsp (nl-llm-integrated-gpattn-spec-greedy prompt nsteps gmodel lnfg w2 b2)))
      (nl-llm-integrated-free-model rmodel)
      (nl-llm-integrated-gpattn-free gmodel)
      (nl-llm-gpu-disable)
      (it--ck "speculative greedy == plain greedy (ternary)" (equal (car sp) g)
              (format "%d toks, %d rounds = %.2f tok/forward" nsteps (cdr sp) (/ (float nsteps) (cdr sp))))
      (it--ck "fused resident decode == non-fused decode" (equal (car fsp) g)
              (format "4 disp/block vs 7; %d rounds" (cdr fsp)))
      (it--ck "GPU-attention decode == non-fused decode" (equal (car gsp) g)
              (format "KV resident, RoPE+softmax in-kernel; %d rounds" (cdr gsp)))))

  (princ (format "NL-LLM-INTEGRATED %s (%d failures)\n" (if (= it--fail 0) "ALL-PASS" "HAS-FAILURES") it--fail))
  (kill-emacs (if (= it--fail 0) 0 1)))
;;; integrated-test.el ends here
