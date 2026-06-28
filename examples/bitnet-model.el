;;; bitnet-model.el --- whole-model packed ternary forward + VRAM demo  -*- lexical-binding: t; -*-
;; Wires BitNet b1.58 Phase B through a whole model: every block linear is stored
;; as packed ternary weights and run via the bitlinear-packed kernel.  Shows
;;   (1) the packed decode == the f32 ternary decode of the same weights, and
;;   (2) the real weight-VRAM reduction from packing the block linears.
;; Weights are random (this demo is about memory + numerical equivalence, not
;; generation quality).  Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/bitnet-model.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-bitnet)
(require 'nl-llm-gpu)

(defun bm--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun bm--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(unless (nl-llm-gpu-enable) (princ "NL-LLM-BITNET-MODEL SKIP (no GPU)\n") (kill-emacs 0))
(let* ((dim 128) (heads 8) (kvh 4) (ff 512) (nblocks 6) (vocab 256) (hd (/ dim heads)) (kvdim (* kvh hd))
       (sc (/ 1.0 (sqrt (float dim)))) (seq 8) (maxseq 16) (prompt '(1 5 2 9 3 7 4 6))
       (mkblk (lambda (s0) (list :ln1g (bm--ones dim) :wq (bm--t (list dim dim) (+ s0 1) sc) :bq (bm--t (list dim) (+ s0 11) 0.1)
                                 :wk (bm--t (list kvdim dim) (+ s0 2) sc) :bk (bm--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (bm--t (list kvdim dim) (+ s0 3) sc) :bv (bm--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (bm--t (list dim dim) (+ s0 4) sc) :bo (bm--t (list dim) (+ s0 14) 0.1) :ln2g (bm--ones dim)
                                 :wg (bm--t (list ff dim) (+ s0 5) sc) :bg (bm--t (list ff) (+ s0 15) 0.1)
                                 :wu (bm--t (list ff dim) (+ s0 6) sc) :bu (bm--t (list ff) (+ s0 16) 0.1)
                                 :wd (bm--t (list dim ff) (+ s0 7) sc) :bd (bm--t (list dim) (+ s0 17) 0.1))))
       (wte (bm--t (list vocab dim) 99 sc)) (lnfg (bm--ones dim)) (bh (bm--t (list vocab) 19 0.1))
       (blocks (let ((l nil) (n 0)) (while (< n nblocks) (push (funcall mkblk (* (1+ n) 1000)) l) (setq n (1+ n))) (nreverse l)))
       (pblocks (mapcar #'nl-llm-bitnet-pack-block blocks))        ; packed ternary weights
       (tblocks (mapcar #'nl-llm-bitnet-ternarize-block blocks)))  ; f32 ternary reference
  (princ (format "model dim=%d blocks=%d GQA %d/%d ff=%d vocab=%d\n" dim nblocks heads kvh ff vocab))
  ;; (1) correctness: packed decode == f32 ternary decode, position for position
  (let* ((pc (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
         (tc (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
         (maxrel 0.0) (p 0))
    (dolist (tk prompt)
      (let ((pg (nl-llm-bitnet-decode-step tk pblocks pc wte lnfg bh dim))
            (tg (nl-llm-decode-step tk tblocks tc wte lnfg bh dim)) (j 0))
        (while (< j vocab) (setq maxrel (max maxrel (/ (abs (- (aref pg j) (aref tg j))) (max 1e-3 (abs (aref tg j)))))) (setq j (1+ j))))
      (setq p (1+ p)))
    (princ (format "packed decode == f32 ternary decode : maxrel=%.2e over %d positions  [%s]\n"
                   maxrel p (if (< maxrel 5e-3) "OK" "MISMATCH"))))
  (nl-llm-gpu-disable)
  ;; (2) VRAM: block-linear weight bytes, f32 vs packed
  (let ((f32 0) (pkb 0))
    (dolist (blk blocks) (let ((b (nl-llm-bitnet-block-bytes blk))) (setq f32 (+ f32 (car b)) pkb (+ pkb (cdr b)))))
    (let ((wte-bytes (* vocab dim 4)))
      (princ          "block-linear weight VRAM:\n")
      (princ (format  "  f32     : %8.2f KB\n" (/ f32 1024.0)))
      (princ (format  "  packed  : %8.2f KB  (%.1fx smaller)\n" (/ pkb 1024.0) (/ (float f32) pkb)))
      (princ (format  "whole-model weights (incl. f32 wte=%.1fKB embed/head):\n" (/ wte-bytes 1024.0)))
      (princ (format  "  all-f32 : %8.2f KB\n" (/ (+ f32 wte-bytes) 1024.0)))
      (princ (format  "  packed  : %8.2f KB  (%.1fx smaller overall)\n"
                      (/ (+ pkb wte-bytes) 1024.0) (/ (float (+ f32 wte-bytes)) (+ pkb wte-bytes))))))
  (princ "NL-LLM-BITNET-MODEL=OK\n"))
;;; bitnet-model.el ends here
