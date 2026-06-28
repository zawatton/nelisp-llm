;;; gpu-bitnet-wte-test.el --- packed (tied) embedding/head: fully-ternary model  -*- lexical-binding: t; -*-
;; Packs the tied embedding/head too, so no f32 weight matrix remains: the
;; embedding is an unpacked ternary row and the head is a packed linear.  Checks
;; the fully-packed decode against a CPU reference whose WTE is also ternarized
;; (gather + tied head), and reports the whole-model weight VRAM now that WTE is
;; packed.  Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-bitnet-wte-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-bitnet)
(require 'nl-llm-gpu)

(defvar bw--fail 0)
(defun bw--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" (progn (setq bw--fail (1+ bw--fail)) "FAIL")) (or extra ""))))
(defun bw--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun bw--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-BITNET-WTE SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((dim 32) (heads 4) (kvh 2) (ff 64) (nblocks 2) (vocab 40) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4) (maxseq 16)
       (wte (bw--t (list vocab dim) 99 sc)) (lnfg (bw--ones dim)) (bh (bw--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (bw--ones dim) :wq (bw--t (list dim dim) (+ s0 1) sc) :bq (bw--t (list dim) (+ s0 11) 0.1)
                                 :wk (bw--t (list kvdim dim) (+ s0 2) sc) :bk (bw--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (bw--t (list kvdim dim) (+ s0 3) sc) :bv (bw--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (bw--t (list dim dim) (+ s0 4) sc) :bo (bw--t (list dim) (+ s0 14) 0.1) :ln2g (bw--ones dim)
                                 :wg (bw--t (list ff dim) (+ s0 5) sc) :bg (bw--t (list ff) (+ s0 15) 0.1)
                                 :wu (bw--t (list ff dim) (+ s0 6) sc) :bu (bw--t (list ff) (+ s0 16) 0.1)
                                 :wd (bw--t (list dim ff) (+ s0 7) sc) :bd (bw--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (prompt '(1 5 2 9 3 7)) (gpu nil) (cpu nil))
  ;; fully-packed decode: packed blocks + packed tied embedding/head
  (let ((pblocks (mapcar #'nl-llm-bitnet-pack-block blocks))
        (wspec (nl-llm-bitnet-pack-wte wte))
        (gc (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)))
    (dolist (tk prompt) (push (copy-sequence (nl-llm-bitnet-decode-step-fullpacked tk pblocks gc wspec lnfg bh dim)) gpu)))
  (nl-llm-gpu-disable)
  (setq gpu (nreverse gpu))
  ;; CPU reference: ternarize blocks AND wte, then plain decode (gather + tied head)
  (let ((tblocks (mapcar #'nl-llm-bitnet-ternarize-block blocks))
        (twte (nl-llm-bitnet-ternarize wte))
        (cc (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)))
    (dolist (tk prompt) (push (copy-sequence (nl-llm-decode-step tk tblocks cc twte lnfg bh dim)) cpu)))
  (setq cpu (nreverse cpu))
  (let ((maxrel 0.0) (p 0))
    (dotimes (i (length gpu))
      (let ((g (nth i gpu)) (c (nth i cpu)) (j 0))
        (while (< j vocab) (setq maxrel (max maxrel (/ (abs (- (aref g j) (aref c j))) (max 1e-3 (abs (aref c j)))))) (setq j (1+ j))))
      (setq p (1+ p)))
    (bw--ck "fully-packed decode == CPU ternary (blocks + wte)" (< maxrel 5e-3) (format "maxrel=%.2e over %d toks" maxrel p)))
  ;; VRAM: whole-model weights, all-f32 vs all-packed (blocks + wte)
  (let ((f32 (* vocab dim 4)) (pk (* vocab (/ (+ dim (1- nl-llm-bitnet-pk)) nl-llm-bitnet-pk) 4)))
    (dolist (blk blocks) (let ((b (nl-llm-bitnet-block-bytes blk))) (setq f32 (+ f32 (car b)) pk (+ pk (cdr b)))))
    (bw--ck "fully-packed weight VRAM ~8x smaller (wte packed too)" (<= pk (/ f32 6))
            (format "%.1f KB -> %.1f KB (%.1fx)" (/ f32 1024.0) (/ pk 1024.0) (/ (float f32) pk))))
  (princ (format "NL-LLM-GPU-BITNET-WTE %s (%d failures)\n" (if (= bw--fail 0) "ALL-PASS" "HAS-FAILURES") bw--fail))
  (kill-emacs (if (= bw--fail 0) 0 1)))
;;; gpu-bitnet-wte-test.el ends here
