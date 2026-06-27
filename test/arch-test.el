;;; arch-test.el --- tests for nl-llm modern transformer primitives  -*- lexical-binding: t; -*-
;; Run from the nelisp-llm root:
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/arch-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-arch)

(defvar a--fail 0)
(defun a--ck (name ok &optional extra)
  (princ (format "%-40s %s  %s\n" name (if ok "PASS"
                                         (progn (setq a--fail (1+ a--fail)) "FAIL"))
                 (or extra ""))))
(defun a--close (x y &optional tol) (< (abs (- x y)) (or tol 1.0e-4)))

;; 1. RMSNorm: each output row has RMS ~= 1 (gamma=1, eps=0), known values
(let* ((x (photon-tensor '(2 4) (vector 3.0 4.0 0.0 0.0  1.0 1.0 1.0 1.0)))
       (g (photon-tensor '(4) (vector 1.0 1.0 1.0 1.0)))
       (y (nl-llm-rmsnorm x g 0.0))
       (d (photon-tensor-data y))
       (rms0 (sqrt (/ (+ (* (aref d 0)(aref d 0)) (* (aref d 1)(aref d 1))
                         (* (aref d 2)(aref d 2)) (* (aref d 3)(aref d 3))) 4.0)))
       (rms1 (sqrt (/ (+ (* (aref d 4)(aref d 4)) (* (aref d 5)(aref d 5))
                         (* (aref d 6)(aref d 6)) (* (aref d 7)(aref d 7))) 4.0))))
  (a--ck "rmsnorm: row RMS ~= 1" (and (a--close rms0 1.0) (a--close rms1 1.0)))
  (a--ck "rmsnorm: known values [1.2 1.6 ..]"
         (and (a--close (aref d 0) 1.2) (a--close (aref d 1) 1.6))))

;; 2. RoPE: position 0 unchanged; per-pair norm preserved
(let* ((seq 3) (dim 4)
       (x (photon-tensor (list seq dim)
                         (vector 1.0 2.0 3.0 4.0  0.5 -1.0 2.0 0.3  -2.0 1.0 0.0 1.0)))
       (xd (photon-tensor-data x))
       (y (nl-llm-rope x))
       (d (photon-tensor-data y))
       (id-ok t) (norm-ok t))
  (dotimes (j dim) (unless (a--close (aref d j) (aref xd j)) (setq id-ok nil)))
  (dotimes (p seq)
    (dotimes (i (/ dim 2))
      (let* ((b (* p dim)) (i0 (+ b (* 2 i))) (i1 (+ b (* 2 i) 1))
             (nin (sqrt (+ (* (aref xd i0)(aref xd i0)) (* (aref xd i1)(aref xd i1)))))
             (nout (sqrt (+ (* (aref d i0)(aref d i0)) (* (aref d i1)(aref d i1))))))
        (unless (a--close nin nout 1.0e-4) (setq norm-ok nil)))))
  (a--ck "rope: position 0 is identity" id-ok)
  (a--ck "rope: per-pair norm preserved" norm-ok))

;; 3. SiLU: silu(0)=0; matches x*sigmoid(x)
(let* ((x (photon-tensor '(1 3) (vector 0.0 1.0 -2.0)))
       (d (photon-tensor-data (nl-llm-silu x)))
       (s1 (/ 1.0 (+ 1.0 (exp -1.0)))) (s2 (/ -2.0 (+ 1.0 (exp 2.0)))))
  (a--ck "silu: silu(0)=0, matches x*sigmoid(x)"
         (and (a--close (aref d 0) 0.0) (a--close (aref d 1) s1) (a--close (aref d 2) s2))))

;; 4. SwiGLU: matches a fully hand-computed tiny case
(let* ((x (photon-tensor '(1 2) (vector 1.0 2.0)))
       (wg (photon-tensor '(3 2) (vector 1.0 0.0  0.0 1.0  1.0 1.0)))
       (wu (photon-tensor '(3 2) (vector 1.0 0.0  0.0 1.0  0.0 0.0)))
       (wd (photon-tensor '(2 3) (vector 1.0 0.0 0.0  0.0 1.0 0.0)))
       (out (photon-tensor-data (nl-llm-swiglu x wg wu wd)))
       (sg1 (/ 1.0 (+ 1.0 (exp -1.0)))) (sg2 (/ 2.0 (+ 1.0 (exp -2.0))))
       (e0 (* sg1 1.0)) (e1 (* sg2 2.0)))
  (a--ck "swiglu: matches hand computation"
         (and (a--close (aref out 0) e0) (a--close (aref out 1) e1))
         (format "got=[%.4f %.4f] want=[%.4f %.4f]" (aref out 0) (aref out 1) e0 e1)))

(princ (format "NL-LLM-ARCH %s (%d failures)\n"
               (if (= a--fail 0) "ALL-PASS" "HAS-FAILURES") a--fail))
(kill-emacs (if (= a--fail 0) 0 1))
;;; arch-test.el ends here
