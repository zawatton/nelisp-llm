;;; nl-llm-autograd.el --- autograd ops for the modern transformer block  -*- lexical-binding: t; -*-

;; Reverse-mode autograd ops for the modern primitives (RMSNorm, SiLU,
;; elementwise mul, RoPE, SwiGLU) on photon-autograd's tape, so a modern
;; (Llama/Qwen-style) block can be trained directly -- not only run forward.
;; Each op records a backward closure; correctness is checked numerically by
;; test/autograd-test.el.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)

;;;###autoload
(defun nl-llm-ag-rmsnorm (x gamma &optional eps)
  "Autograd row-wise RMSNorm of X (m x n) with trainable GAMMA (n), no mean sub."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (m (car sh)) (n (nth 1 sh))
         (e (or eps 1.0e-6)) (xd (photon-tensor-data xv)) (gd (photon-tensor-data (pav-value gamma)))
         (istd (make-vector m 0.0)) (od (make-vector (* m n) 0.0)) (ninv (/ 1.0 (float n))) (i 0))
    (while (< i m)
      (let ((base (* i n)) (ss 0.0) (j 0))
        (while (< j n) (let ((v (aref xd (+ base j)))) (setq ss (+ ss (* v v)))) (setq j (1+ j)))
        (let ((is (/ 1.0 (sqrt (+ (* ss ninv) e)))) (j2 0))
          (aset istd i is)
          (while (< j2 n) (aset od (+ base j2) (* (aref xd (+ base j2)) is (aref gd j2))) (setq j2 (1+ j2)))))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (list m n) od)
     (lambda (g)
       (let* ((ggd (photon-tensor-data g)) (dx (make-vector (* m n) 0.0))
              (dgamma (make-vector n 0.0)) (i2 0))
         (while (< i2 m)
           (let ((base (* i2 n)) (is (aref istd i2)) (a 0.0) (j 0))
             (while (< j n)
               (setq a (+ a (* (aref ggd (+ base j)) (aref gd j) (aref xd (+ base j)))))
               (setq j (1+ j)))
             (setq j 0)
             (while (< j n)
               (let ((xk (aref xd (+ base j))) (dyk (aref ggd (+ base j))) (gk (aref gd j)))
                 (aset dgamma j (+ (aref dgamma j) (* dyk xk is)))
                 (aset dx (+ base j) (- (* is dyk gk) (* is is is xk ninv a))))
               (setq j (1+ j))))
           (setq i2 (1+ i2)))
         (photon-autograd--addgrad x (photon-tensor (list m n) dx))
         (photon-autograd--addgrad gamma (photon-tensor (list n) dgamma)))))))

;;;###autoload
(defun nl-llm-ag-silu (x)
  "Autograd SiLU / swish: x * sigmoid(x)."
  (let* ((xv (pav-value x)) (xd (photon-tensor-data xv)) (n (length xd))
         (od (make-vector n 0.0)) (i 0))
    (while (< i n)
      (let* ((v (aref xd i)) (sg (/ 1.0 (+ 1.0 (exp (- v)))))) (aset od i (* v sg)))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (photon-tensor-shape xv) od)
     (lambda (g)
       (let* ((gd (photon-tensor-data g)) (dx (make-vector n 0.0)) (j 0))
         (while (< j n)
           (let* ((v (aref xd j)) (sg (/ 1.0 (+ 1.0 (exp (- v)))))
                  (d (* sg (+ 1.0 (* v (- 1.0 sg))))))
             (aset dx j (* (aref gd j) d)))
           (setq j (1+ j)))
         (photon-autograd--addgrad x (photon-tensor (photon-tensor-shape xv) dx)))))))

;;;###autoload
(defun nl-llm-ag-mul (a b)
  "Autograd elementwise product of same-shape A and B."
  (let ((out (photon-tensor-hadamard (pav-value a) (pav-value b))))
    (photon-autograd--record
     out
     (lambda (g)
       (photon-autograd--addgrad a (photon-tensor-hadamard g (pav-value b)))
       (photon-autograd--addgrad b (photon-tensor-hadamard g (pav-value a)))))))

(defun nl-llm--rope-apply (src seq dim heads base transpose)
  "Return a fresh copy of float vector SRC with per-head RoPE applied.
TRANSPOSE non-nil applies the inverse rotation (Jacobian transpose)."
  (let* ((hd (/ dim heads)) (half (/ hd 2)) (out (copy-sequence src)) (p 0))
    (while (< p seq)
      (let ((h 0))
        (while (< h heads)
          (let ((bb (+ (* p dim) (* h hd))) (m 0))
            (while (< m half)
              (let* ((theta (/ (float p) (expt base (/ (* 2.0 m) (float hd)))))
                     (c (cos theta)) (s (if transpose (- (sin theta)) (sin theta)))
                     (i0 (+ bb (* 2 m))) (i1 (+ bb (* 2 m) 1))
                     (a0 (aref src i0)) (a1 (aref src i1)))
                (aset out i0 (- (* a0 c) (* a1 s)))
                (aset out i1 (+ (* a0 s) (* a1 c))))
              (setq m (1+ m))))
          (setq h (1+ h))))
      (setq p (1+ p)))
    out))

;;;###autoload
(defun nl-llm-ag-rope (x heads &optional base)
  "Autograd per-head RoPE on X (seq x dim).  RoPE is an orthogonal rotation,
so the backward pass is the inverse rotation of the upstream gradient."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (seq (car sh)) (dim (nth 1 sh))
         (b (or base 10000.0)))
    (photon-autograd--record
     (photon-tensor (list seq dim)
                    (nl-llm--rope-apply (photon-tensor-data xv) seq dim heads b nil))
     (lambda (g)
       (photon-autograd--addgrad
        x (photon-tensor (list seq dim)
                         (nl-llm--rope-apply (photon-tensor-data g) seq dim heads b t)))))))

;;;###autoload
(defun nl-llm-ag-swiglu (x wg bg wu bu wd bd)
  "Autograd SwiGLU FFN over X: (silu(X.Wg^T+bg) (*) (X.Wu^T+bu)) . Wd^T + bd."
  (photon-autograd-linear
   (nl-llm-ag-mul (nl-llm-ag-silu (photon-autograd-linear x wg bg))
                  (photon-autograd-linear x wu bu))
   wd bd))

(provide 'nl-llm-autograd)
;;; nl-llm-autograd.el ends here
