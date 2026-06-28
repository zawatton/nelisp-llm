;;; gpu-paged-v-test.el --- variable-length paged decode + free-list + prefix share  -*- lexical-binding: t; -*-
;; Exercises the variable-length PagedAttention extension:
;;  (1) sequences of DIFFERENT lengths decode together, each matching its own CPU
;;      KV-cache decode (per-sequence positions LENS[s] + per-sequence block table);
;;  (2) the host free-list recycles a freed sequence's physical blocks;
;;  (3) two sequences SHARE a prompt-prefix's blocks -> identical continuation
;;      logits at fewer total blocks than separate copies.
;; Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-paged-v-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)

(defvar pv--fail 0)
(defun pv--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" (progn (setq pv--fail (1+ pv--fail)) "FAIL")) (or extra ""))))
(defun pv--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun pv--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun pv--mr (g off vocab c) (let ((m 0.0) (j 0))
  (while (< j vocab) (setq m (max m (/ (abs (- (aref g (+ off j)) (aref c j))) (max 1e-3 (abs (aref c j)))))) (setq j (1+ j))) m))

;; --- (2) free-list allocator: recycle a freed sequence's blocks (no GPU) ---
(let ((a (nl-llm-paged-alloc-new 4 2 4 2)))   ; 4 blocks, 2 seqs, 4 logical, bs 2
  (nl-llm-paged-ensure a 0) (nl-llm-paged-advance a 0) (nl-llm-paged-advance a 0) ; len 2
  (nl-llm-paged-ensure a 0)                                                        ; cross boundary -> 2 blocks
  (let ((used2 (nl-llm-paged-alloc-used a)))
    (nl-llm-paged-free-seq a 0)                                                    ; return both
    (let ((used0 (nl-llm-paged-alloc-used a)))
      (nl-llm-paged-ensure a 1)                                                    ; reuse a freed block
      (pv--ck "free-list: blocks recycled after free"
              (and (= used2 2) (= used0 0) (= (nl-llm-paged-alloc-used a) 1))
              (format "used 2->%d->%d after free/realloc" used0 (nl-llm-paged-alloc-used a))))))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-PAGED-V SKIP (no GPU server / Vulkan device)\n") (kill-emacs (if (= pv--fail 0) 0 1)))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4)
       (bs 2) (mbps 6) (maxseq (* bs mbps)) (nblocks 24)
       (wte (pv--t (list vocab dim) 1 sc)) (lnfg (pv--ones dim)) (bh (pv--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (pv--ones dim) :wq (pv--t (list dim dim) (+ s0 1) sc) :bq (pv--t (list dim) (+ s0 11) 0.1)
                                 :wk (pv--t (list kvdim dim) (+ s0 2) sc) :bk (pv--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (pv--t (list kvdim dim) (+ s0 3) sc) :bv (pv--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (pv--t (list dim dim) (+ s0 4) sc) :bo (pv--t (list dim) (+ s0 14) 0.1) :ln2g (pv--ones dim)
                                 :wg (pv--t (list ff dim) (+ s0 5) sc) :bg (pv--t (list ff) (+ s0 15) 0.1)
                                 :wu (pv--t (list ff dim) (+ s0 6) sc) :bu (pv--t (list ff) (+ s0 16) 0.1)
                                 :wd (pv--t (list dim ff) (+ s0 7) sc) :bd (pv--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (tables (nl-llm-gpu-rope-tables maxseq hd)))
  ;; --- (1) variable-length batch == per-sequence CPU decode -----------------
  (let* ((bsz 3) (toks (vector (vector 0 1 2 3) (vector 6 7 8 9 10 11) (vector 0 2 4 6 8 10 1 3)))
         (lens (vector 4 6 8)) (maxL 8)
         (alloc (nl-llm-paged-alloc-new nblocks bsz mbps bs))
         (ctx (nl-llm-gpu-paged-v-new wte blocks lnfg bh heads kvh dim vocab nblocks bs mbps bsz tables))
         (cpu (make-vector bsz nil)) (maxrel 0.0) (p 0))
    (dotimes (s bsz) (aset cpu s (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks)))
    (while (< p maxL)
      (let ((row (make-vector bsz 0)))
        (dotimes (s bsz)
          (let ((act (< p (aref lens s))))
            (aset row s (aref (aref toks s) (if act p (1- (aref lens s)))))
            (nl-llm-paged-ensure alloc s)))
        (let ((logits (nl-llm-gpu-paged-v-step ctx alloc row)))
          (dotimes (s bsz)
            (when (< p (aref lens s))
              (let ((c (nl-llm-decode-step (aref (aref toks s) p) blocks (aref cpu s) wte lnfg bh dim)))
                (setq maxrel (max maxrel (pv--mr logits (* s vocab) vocab c)))
                (nl-llm-paged-advance alloc s))))))
      (setq p (1+ p)))
    (nl-llm-gpu-paged-v-free ctx)
    (pv--ck "variable-length batch == per-sequence CPU decode" (< maxrel 5e-3) (format "maxrel=%.2e (lens %S)" maxrel lens)))
  ;; --- (3) prefix sharing: seqB shares seqA's prompt blocks -----------------
  (let* ((bsz 2) (np 4) (ncont 4)                       ; prefix 4 (=2 blocks), continue 4
         (seqtoks (vector 5 1 9 3 7 2 4 6))             ; full token stream A and B both follow
         (alloc (nl-llm-paged-alloc-new nblocks bsz mbps bs))
         (ctx (nl-llm-gpu-paged-v-new wte blocks lnfg bh heads kvh dim vocab nblocks bs mbps bsz tables))
         (la (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))   ; CPU ref for A
         (maxrel 0.0) (p 0))
    ;; phase 1: A decodes the prompt; B idle
    (while (< p np)
      (nl-llm-paged-ensure alloc 0)
      (let ((row (vector (aref seqtoks p) 0)))
        (nl-llm-gpu-paged-v-step ctx alloc row)
        (nl-llm-decode-step (aref seqtoks p) blocks la wte lnfg bh dim)
        (nl-llm-paged-advance alloc 0))
      (setq p (1+ p)))
    (let ((blocks-after-prefix (nl-llm-paged-alloc-used alloc)))
      ;; B shares A's prefix blocks (no new prompt blocks for B)
      (nl-llm-paged-share-prefix alloc 1 0 np)
      (pv--ck "prefix share allocates no new prefix blocks"
              (= (nl-llm-paged-alloc-used alloc) blocks-after-prefix)
              (format "used stays %d after sharing %d-token prefix" blocks-after-prefix np))
      ;; phase 2: A and B continue with the SAME tokens; logits must match
      (setq p np)
      (while (< p (+ np ncont))
        (nl-llm-paged-ensure alloc 0) (nl-llm-paged-ensure alloc 1)
        (let* ((tk (aref seqtoks p)) (row (vector tk tk))
               (logits (nl-llm-gpu-paged-v-step ctx alloc row)))
          (setq maxrel (max maxrel (pv--mr logits 0 vocab (let ((g (make-vector vocab 0.0)))
                                                            (dotimes (j vocab) (aset g j (aref logits (+ vocab j)))) g))))
          (nl-llm-paged-advance alloc 0) (nl-llm-paged-advance alloc 1))
        (setq p (1+ p)))
      (nl-llm-gpu-paged-v-free ctx) (nl-llm-gpu-disable)
      (pv--ck "shared-prefix B == A continuation logits" (< maxrel 5e-3) (format "maxrel=%.2e" maxrel))))
  (princ (format "NL-LLM-GPU-PAGED-V %s (%d failures)\n" (if (= pv--fail 0) "ALL-PASS" "HAS-FAILURES") pv--fail))
  (kill-emacs (if (= pv--fail 0) 0 1)))
;;; gpu-paged-v-test.el ends here
