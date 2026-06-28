;;; gpu-tree-verify-test.el --- end-to-end one-forward tree verify  -*- lexical-binding: t; -*-
;; Wires tree-attn through the whole decode loop: given an accepted context and M
;; draft-tree nodes, nl-llm-gpu-tree-verify returns every node's next-token logits
;; in ONE fused forward (each node attends the shared context + its ancestor
;; chain).  Checks each node against decoding its full path with the CPU decoder.
;; Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-tree-verify-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)

(defvar tv--fail 0)
(defun tv--ck (name ok &optional extra)
  (princ (format "%-48s %s  %s\n" name (if ok "PASS" (progn (setq tv--fail (1+ tv--fail)) "FAIL")) (or extra ""))))
(defun tv--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun tv--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-TREE-VERIFY SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((dim 16) (heads 4) (kvh 2) (ff 24) (vocab 12) (hd (/ dim heads)) (kvdim (* kvh hd)) (sc 0.4) (maxseq 24)
       (wte (tv--t (list vocab dim) 1 sc)) (lnfg (tv--ones dim)) (bh (tv--t (list vocab) 19 0.1))
       (mkblk (lambda (s0) (list :ln1g (tv--ones dim) :wq (tv--t (list dim dim) (+ s0 1) sc) :bq (tv--t (list dim) (+ s0 11) 0.1)
                                 :wk (tv--t (list kvdim dim) (+ s0 2) sc) :bk (tv--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (tv--t (list kvdim dim) (+ s0 3) sc) :bv (tv--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (tv--t (list dim dim) (+ s0 4) sc) :bo (tv--t (list dim) (+ s0 14) 0.1) :ln2g (tv--ones dim)
                                 :wg (tv--t (list ff dim) (+ s0 5) sc) :bg (tv--t (list ff) (+ s0 15) 0.1)
                                 :wu (tv--t (list ff dim) (+ s0 6) sc) :bu (tv--t (list ff) (+ s0 16) 0.1)
                                 :wd (tv--t (list dim ff) (+ s0 7) sc) :bd (tv--t (list dim) (+ s0 17) 0.1))))
       (blocks (list (funcall mkblk 100) (funcall mkblk 200)))
       (tables (nl-llm-gpu-rope-tables maxseq hd))
       ;; accepted context, and a depth-2 draft tree:
       ;;   node0 = t1 (parent = root, pos L), node1,node2 children of 0 (pos L+1),
       ;;   node3 child of node1 (pos L+2)
       (ctx '(3 1 4 1 5 9)) (L (length ctx)) (maxdepth 4)
       (nodes (list 2 7 8 6)) (parents (list 4 0 0 1)) (positions (list L (1+ L) (1+ L) (+ L 2)))
       (gpu (nl-llm-gpu-tree-verify blocks wte lnfg bh heads kvh dim vocab ctx nodes parents positions tables maxdepth)))
  (nl-llm-gpu-disable)
  ;; CPU reference: decode each node's full root->node path, compare its last logits
  (let ((paths (vector (list 0) (list 0 1) (list 0 2) (list 0 1 3)))  ; node index chains root..node
        (maxrel 0.0) (m (length nodes)))
    (dotimes (i m)
      (let* ((chain (aref paths i))
             (toks (append ctx (mapcar (lambda (ni) (nth ni nodes)) chain)))
             (caches (mapcar (lambda (_) (nl-llm-dcache-new maxseq dim heads kvh)) blocks))
             (last nil))
        (dolist (tk toks) (setq last (nl-llm-decode-step tk blocks caches wte lnfg bh dim)))
        (let ((g (nth i gpu)) (j 0))
          (while (< j vocab) (setq maxrel (max maxrel (/ (abs (- (aref g j) (aref last j))) (max 1e-3 (abs (aref last j)))))) (setq j (1+ j))))))
    (tv--ck "tree-verify == CPU per-path decode (1 forward)" (< maxrel 5e-3) (format "maxrel=%.2e (M=%d, L=%d)" maxrel m L)))
  (princ (format "NL-LLM-GPU-TREE-VERIFY %s (%d failures)\n" (if (= tv--fail 0) "ALL-PASS" "HAS-FAILURES") tv--fail))
  (kill-emacs (if (= tv--fail 0) 0 1)))
;;; gpu-tree-verify-test.el ends here
