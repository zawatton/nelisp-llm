;;; gpu-paged-test.el --- on-GPU PagedAttention decode == per-sequence decode  -*- lexical-binding: t; -*-
;; Checks the paged batch decoder (nl-llm-gpu-paged) -- KV in a shared pool of
;; fixed-size blocks addressed through a per-sequence block table, blocks assigned
;; on demand by the host allocator -- produces, for each sequence, the same
;; per-position logits as decoding it alone with the CPU KV-cache decoder.
;; The physical block layout is interleaved (not contiguous per sequence), so this
;; exercises the table indirection end-to-end.  Skips (exit 0) without Vulkan.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-paged-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)

(defvar pg--fail 0)
(defun pg--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" (progn (setq pg--fail (1+ pg--fail)) "FAIL")) (or extra ""))))
(defun pg--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun pg--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-PAGED SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((seq 6) (bsz 3) (dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd))
       (bs 2) (mbps 3) (maxseq (* bs mbps)) (sc 0.4)        ; block size 2, 3 blocks/seq -> maxlen 6
       (toks (vector (vector 0 1 2 3 4 5) (vector 6 7 8 9 10 11) (vector 0 2 4 6 8 10)))
       (wte (pg--t (list vocab dim) 1 sc)) (lnfg (pg--ones dim)) (bh (pg--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (pg--ones dim) :wq (pg--t (list dim dim) (+ s0 1) sc) :bq (pg--t (list dim) (+ s0 11) 0.1)
                                 :wk (pg--t (list kvdim dim) (+ s0 2) sc) :bk (pg--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (pg--t (list kvdim dim) (+ s0 3) sc) :bv (pg--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (pg--t (list dim dim) (+ s0 4) sc) :bo (pg--t (list dim) (+ s0 14) 0.1) :ln2g (pg--ones dim)
                                 :wg (pg--t (list ff dim) (+ s0 5) sc) :bg (pg--t (list ff) (+ s0 15) 0.1)
                                 :wu (pg--t (list ff dim) (+ s0 6) sc) :bu (pg--t (list ff) (+ s0 16) 0.1)
                                 :wd (pg--t (list dim ff) (+ s0 7) sc) :bd (pg--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (tables (nl-llm-gpu-rope-tables maxseq hd))
       (gpu nil) (cpu nil) (ctx (nl-llm-gpu-paged-new wte blocks lnfg bh heads kvh dim vocab bs mbps bsz tables)) (p 0))
  ;; paged GPU decode (synchronous, shared position)
  (while (< p seq)
    (let ((row (make-vector bsz 0))) (dotimes (s bsz) (aset row s (aref (aref toks s) p)))
      (push (copy-sequence (nl-llm-gpu-paged-step ctx row p)) gpu))
    (setq p (1+ p)))
  (let ((nblk (plist-get ctx :nblocks))) (nl-llm-gpu-paged-free ctx) (nl-llm-gpu-disable)
    (setq gpu (nreverse gpu))
    ;; CPU per-sequence reference
    (dotimes (s bsz)
      (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new seq dim heads kvh)) blocks)) (row nil) (q 0))
        (while (< q seq) (push (copy-sequence (nl-llm-decode-step (aref (aref toks s) q) blocks caches wte lnfg bh dim)) row) (setq q (1+ q)))
        (push (nreverse row) cpu)))
    (setq cpu (nreverse cpu))
    (let ((maxrel 0.0) (q 0))
      (while (< q seq)
        (let ((g (nth q gpu)) (s 0)) (while (< s bsz)
          (let ((c (nth q (nth s cpu))) (j 0)) (while (< j vocab)
            (setq maxrel (max maxrel (/ (abs (- (aref g (+ (* s vocab) j)) (aref c j))) (max 1e-3 (abs (aref c j)))))) (setq j (1+ j))))
          (setq s (1+ s))))
        (setq q (1+ q)))
      (pg--ck "paged decode == per-sequence decode" (< maxrel 5e-3) (format "maxrel=%.2e" maxrel)))
    ;; on-demand allocation: blocks used = ceil(seq/bs)*bsz, < the worst-case pool
    (pg--ck "on-demand block allocation count"
            (= nblk (* (/ (+ seq bs -1) bs) bsz))
            (format "used %d blocks (pool budget %d = bsz*mbps)" nblk (* bsz mbps))))
  (princ (format "NL-LLM-GPU-PAGED %s (%d failures)\n" (if (= pg--fail 0) "ALL-PASS" "HAS-FAILURES") pg--fail))
  (kill-emacs (if (= pg--fail 0) 0 1)))
;;; gpu-paged-test.el ends here
