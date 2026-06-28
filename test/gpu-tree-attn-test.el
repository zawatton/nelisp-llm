;;; gpu-tree-attn-test.el --- batched tree-attention verify primitive  -*- lexical-binding: t; -*-
;; The core of a one-forward tree verify: M draft-tree nodes each attend over a
;; shared context cache + their ancestor chain among the nodes (parent array,
;; sentinel = M).  Checks the single-dispatch tree-attn kernel against a CPU
;; reference that, per node, attends to the context plus the walked parent chain.
;; Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-tree-attn-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-gpu)
(require 'nelisp-gpu-server)

(defvar ta--fail 0)
(defun ta--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS" (progn (setq ta--fail (1+ ta--fail)) "FAIL")) (or extra ""))))
(defun ta--vec (n seed sc) (let ((v (make-vector n 0.0)) (i 0))
  (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-TREE-ATTN SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((M 4) (dim 16) (heads 4) (kvh 2) (hd (/ dim heads)) (kvdim (* kvh hd)) (grp (/ heads kvh))
       (L 5) (maxdepth 4) (coef (/ 1.0 (sqrt (float hd))))
       ;; tree: 0 root, 1<-0, 2<-0, 3<-1  (parent array; sentinel M = root)
       (par (vector (float M) 0.0 0.0 1.0))
       (q (ta--vec (* M dim) 1 0.4)) (ck (ta--vec (* L kvdim) 2 0.4)) (cv (ta--vec (* L kvdim) 3 0.4))
       (nk (ta--vec (* M kvdim) 4 0.4)) (nv (ta--vec (* M kvdim) 5 0.4))
       (res (nelisp-gpu-server-run 'tree-attn
              (list q ck cv nk nv par (vector (float L)) (make-vector (* M dim) 0.0))
              (vector M dim heads kvh maxdepth) (/ (+ (* M dim) 63) 64)))
       (gpu (nth 7 res)) (maxrel 0.0))
  (nl-llm-gpu-disable)
  ;; CPU reference: per (node i, component c=h*hd+t), attend context 0..L-1 + chain(i)
  (dotimes (i M)
    (dotimes (h heads)
      (let* ((c0q (* i dim) ) (kvhh (/ h grp)) (mx -1.0e30) (scores nil) (slots nil))
        (setq c0q (+ (* i dim) (* h hd)))
        ;; collect scores: context then chain
        (dotimes (j L)
          (let ((kb (+ (* j kvdim) (* kvhh hd))) (acc 0.0))
            (dotimes (tt hd) (setq acc (+ acc (* (aref q (+ c0q tt)) (aref ck (+ kb tt))))))
            (push (* acc coef) scores) (push (cons 'ctx j) slots)))
        (let ((cur i) (d 0))
          (while (and (< d maxdepth) (< cur M))
            (let ((kb (+ (* cur kvdim) (* kvhh hd))) (acc 0.0))
              (dotimes (tt hd) (setq acc (+ acc (* (aref q (+ c0q tt)) (aref nk (+ kb tt))))))
              (push (* acc coef) scores) (push (cons 'node cur) slots))
            (setq cur (truncate (aref par cur)) d (1+ d))))
        (setq scores (nreverse scores) slots (nreverse slots))
        (dolist (s scores) (when (> s mx) (setq mx s)))
        ;; softmax + weighted V, per component t
        (dotimes (tt hd)
          (let ((z 0.0) (ctx 0.0) (sl slots) (sc scores))
            (while sc
              (let* ((e (exp (- (car sc) mx))) (slot (car sl))
                     (kb (if (eq (car slot) 'ctx) (+ (* (cdr slot) kvdim) (* kvhh hd) tt)
                           (+ (* (cdr slot) kvdim) (* kvhh hd) tt)))
                     (vv (if (eq (car slot) 'ctx) (aref cv kb) (aref nv kb))))
                (setq z (+ z e) ctx (+ ctx (* e vv)) sc (cdr sc) sl (cdr sl))))
            (let ((want (/ ctx z)) (got (aref gpu (+ (* i dim) (* h hd) tt))))
              (setq maxrel (max maxrel (/ (abs (- want got)) (max 1e-3 (abs want)))))))))))
  (ta--ck "tree-attn (context+ancestor mask) == CPU" (< maxrel 5e-4) (format "maxrel=%.2e" maxrel))
  (princ (format "NL-LLM-GPU-TREE-ATTN %s (%d failures)\n" (if (= ta--fail 0) "ALL-PASS" "HAS-FAILURES") ta--fail))
  (kill-emacs (if (= ta--fail 0) 0 1)))
;;; gpu-tree-attn-test.el ends here
