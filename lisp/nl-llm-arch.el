;;; nl-llm-arch.el --- modern transformer primitives over photon-tensor  -*- lexical-binding: t; -*-

;; Architectural building blocks shared by current strong open-weight models
;; (RMSNorm, RoPE, SiLU/SwiGLU) implemented on top of nelisp-photon's
;; photon-tensor core.  This is the experiment layer; photon-tensor stays the
;; stable substrate.  Pure-elisp row-major [SHAPE DATA] tensors.

;;; Code:

(require 'photon-tensor)

(defun nl-llm-mul (a b)
  "Elementwise product of two same-shape tensors A and B."
  (let* ((da (photon-tensor-data a)) (db (photon-tensor-data b))
         (n (length da)) (out (make-vector n 0.0)) (i 0))
    (while (< i n) (aset out i (* (aref da i) (aref db i))) (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

;;;###autoload
(defun nl-llm-rmsnorm (x gamma &optional eps)
  "Row-wise RMSNorm of 2D tensor X, scaled by GAMMA (length = cols).
Unlike LayerNorm there is no mean subtraction: x / sqrt(mean(x^2)+eps) * g."
  (let* ((sh (photon-tensor-shape x)) (m (car sh)) (n (nth 1 sh))
         (d (photon-tensor-data x)) (g (photon-tensor-data gamma))
         (out (make-vector (* m n) 0.0)) (e (or eps 1.0e-6)) (i 0))
    (while (< i m)
      (let ((base (* i n)) (ss 0.0) (j 0))
        (while (< j n)
          (let ((v (aref d (+ base j)))) (setq ss (+ ss (* v v))))
          (setq j (1+ j)))
        (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) e)))) (j2 0))
          (while (< j2 n)
            (aset out (+ base j2) (* (aref d (+ base j2)) inv (aref g j2)))
            (setq j2 (1+ j2)))))
      (setq i (1+ i)))
    (photon-tensor (list m n) out)))

;;;###autoload
(defun nl-llm-rope (x &optional base)
  "Apply rotary position embedding (RoPE) to 2D tensor X (seq x dim).
DIM must be even; pairs (2i, 2i+1) are rotated by angle p / BASE^(2i/dim) at
position p.  Norm-preserving per pair; position 0 is the identity."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (d (photon-tensor-data x)) (out (copy-sequence d))
         (b (or base 10000.0)) (half (/ dim 2)) (p 0))
    (while (< p seq)
      (let ((rb (* p dim)) (i 0))
        (while (< i half)
          (let* ((theta (/ (float p) (expt b (/ (* 2.0 i) (float dim)))))
                 (c (cos theta)) (s (sin theta))
                 (i0 (+ rb (* 2 i))) (i1 (+ rb (* 2 i) 1))
                 (a0 (aref d i0)) (a1 (aref d i1)))
            (aset out i0 (- (* a0 c) (* a1 s)))
            (aset out i1 (+ (* a0 s) (* a1 c))))
          (setq i (1+ i))))
      (setq p (1+ p)))
    (photon-tensor (list seq dim) out)))

;;;###autoload
(defun nl-llm-silu (x)
  "Elementwise SiLU / swish: x * sigmoid(x)."
  (let* ((d (photon-tensor-data x)) (n (length d)) (out (make-vector n 0.0)) (i 0))
    (while (< i n)
      (let ((v (aref d i))) (aset out i (/ v (+ 1.0 (exp (- v))))))
      (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape x) out)))

;;;###autoload
(defun nl-llm-swiglu (x w-gate w-up w-down)
  "SwiGLU feed-forward over X (seq x dim).
W-GATE and W-UP are (ff x dim); W-DOWN is (dim x ff).  Computes
\(silu(X.Wg^T) (*) (X.Wu^T)) . Wd^T, the GLU variant used by Llama/Qwen."
  (photon-tensor-linear
   (nl-llm-mul (nl-llm-silu (photon-tensor-linear x w-gate))
               (photon-tensor-linear x w-up))
   w-down))

(provide 'nl-llm-arch)
;;; nl-llm-arch.el ends here
