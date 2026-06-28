;;; gpu-paged-cow-test.el --- partial-block prefix sharing + copy-on-write  -*- lexical-binding: t; -*-
;; PagedAttention prefix sharing where the prefix ends MID-BLOCK: the partial last
;; block is shared read-only, and the first write into it copy-on-writes (a GPU
;; block-copy in every layer pool) so the sharer gets a private block without
;; corrupting the owner.  Verifies the sharer's continuation logits match a
;; standalone decode of the same full sequence.  Skips (exit 0) without Vulkan.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-paged-cow-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)

(defvar cw--fail 0)
(defun cw--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" (progn (setq cw--fail (1+ cw--fail)) "FAIL")) (or extra ""))))
(defun cw--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun cw--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun cw--mr (g off c vocab) (let ((m 0.0) (j 0))
  (while (< j vocab) (setq m (max m (/ (abs (- (aref g (+ off j)) (aref c j))) (max 1e-3 (abs (aref c j)))))) (setq j (1+ j))) m))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-PAGED-COW SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4)
       (bs 2) (mbps 8) (maxseq (* bs mbps)) (nblocks 24) (bsz 2)
       (wte (cw--t (list vocab dim) 1 sc)) (lnfg (cw--ones dim)) (bh (cw--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (cw--ones dim) :wq (cw--t (list dim dim) (+ s0 1) sc) :bq (cw--t (list dim) (+ s0 11) 0.1)
                                 :wk (cw--t (list kvdim dim) (+ s0 2) sc) :bk (cw--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (cw--t (list kvdim dim) (+ s0 3) sc) :bv (cw--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (cw--t (list dim dim) (+ s0 4) sc) :bo (cw--t (list dim) (+ s0 14) 0.1) :ln2g (cw--ones dim)
                                 :wg (cw--t (list ff dim) (+ s0 5) sc) :bg (cw--t (list ff) (+ s0 15) 0.1)
                                 :wu (cw--t (list ff dim) (+ s0 6) sc) :bu (cw--t (list ff) (+ s0 16) 0.1)
                                 :wd (cw--t (list dim ff) (+ s0 7) sc) :bd (cw--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (tables (nl-llm-gpu-rope-tables maxseq hd))
       (np 5) (ncont 4)                              ; prefix 5 = blocks [0,1][2,3][4_]; block 2 partial
       (seqtoks (vector 3 1 4 1 5 9 2 6 8)) (gpu nil) (cpu nil)
       (alloc (nl-llm-paged-alloc-new nblocks bsz mbps bs))
       (ctx (nl-llm-gpu-paged-v-new wte blocks lnfg bh heads kvh dim vocab nblocks bs mbps bsz tables))
       (pools (plist-get ctx :pools)))
  ;; phase 1: seq 0 (owner) decodes the prompt; seq 1 idle
  (let ((p 0)) (while (< p np)
    (nl-llm-paged-ensure alloc 0)
    (nl-llm-gpu-paged-v-step ctx alloc (vector (aref seqtoks p) 0))
    (nl-llm-paged-advance alloc 0) (setq p (1+ p))))
  ;; seq 1 shares seq 0's 5-token prefix (block 2 partial + shared)
  (nl-llm-paged-share-prefix alloc 1 0 np)
  ;; phase 2: seq 1 continues; COW the shared partial block on first write
  (let ((p np)) (while (< p (+ np ncont))
    (let ((cowed (nl-llm-paged-cow-block alloc pools 1 bs kvdim)))
      (when (= p np) (cw--ck "COW fired on the shared partial block" cowed)))
    (nl-llm-paged-ensure alloc 1)
    (push (copy-sequence (nl-llm-gpu-paged-v-step ctx alloc (vector (aref seqtoks p) (aref seqtoks p)))) gpu)
    (nl-llm-paged-advance alloc 1) (setq p (1+ p))))
  (nl-llm-gpu-paged-v-free ctx) (nl-llm-gpu-disable)
  (setq gpu (nreverse gpu))
  ;; reference: standalone decode of the full sequence (owns every block)
  (let ((caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)) (p 0))
    (while (< p (+ np ncont))
      (let ((lg (nl-llm-decode-step (aref seqtoks p) blocks caches wte lnfg bh dim)))
        (when (>= p np) (push (copy-sequence lg) cpu)))
      (setq p (1+ p))))
  (setq cpu (nreverse cpu))
  (let ((maxrel 0.0) (i 0))
    (while (< i ncont)
      (setq maxrel (max maxrel (cw--mr (nth i gpu) vocab (nth i cpu) vocab)))   ; seq 1 logits at offset vocab
      (setq i (1+ i)))
    (cw--ck "shared partial-prefix + COW == standalone decode" (< maxrel 5e-3) (format "maxrel=%.2e" maxrel)))
  (princ (format "NL-LLM-GPU-PAGED-COW %s (%d failures)\n" (if (= cw--fail 0) "ALL-PASS" "HAS-FAILURES") cw--fail))
  (kill-emacs (if (= cw--fail 0) 0 1)))
;;; gpu-paged-cow-test.el ends here
