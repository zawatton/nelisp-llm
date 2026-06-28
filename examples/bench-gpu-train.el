;;; bench-gpu-train.el --- on-device vs host-driven training timing  -*- lexical-binding: t; -*-
;; Times one training step of an FFN-shaped 2-layer MLP (dim -> 4*dim -> dim,
;; MSE) three ways, at a few sizes:
;;   - CPU         : photon-autograd on the pure-elisp backend
;;   - host GPU    : photon-autograd on the GPU backend; in-place SGD forces a
;;                   weight re-upload each step (nl-llm-gpu-invalidate)
;;   - on-device   : nl-llm-gpu-mlp-train -- weights stay resident and are
;;                   updated on the GPU; no weight round-trip per step
;; The on-device per-step time is measured by differencing (2S vs S steps) to
;; remove the one-time upload/readback.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/bench-gpu-train.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-train)

(defun bt--vec (n seed sc)
  (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n)
      (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536))
                                65536.0) 0.5)))
      (setq i (1+ i)))
    v))
(defun bt--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
                               (photon-tensor shape (bt--vec n seed sc))))
(defun bt--copy (tn) (photon-tensor (photon-tensor-shape tn) (copy-sequence (photon-tensor-data tn))))
(defmacro bt--ms (&rest body) `(let ((t0 (float-time))) ,@body (* 1000.0 (- (float-time) t0))))

(defun bt--host-steps (x w1 b1 w2 b2 target lr steps invalidate)
  "Run STEPS host-autograd MLP SGD steps on the current backend."
  (let* ((seq (car (photon-tensor-shape x))) (out (car (photon-tensor-shape w2)))
         (ntot (* seq out)) (invn (/ 1.0 (float ntot))) (td (photon-tensor-data target))
         (xc (photon-autograd-const x))
         (w1c (photon-autograd-const w1)) (b1c (photon-autograd-const b1))
         (w2c (photon-autograd-const w2)) (b2c (photon-autograd-const b2))
         (params (list w1c b1c w2c b2c)) (s 0))
    (while (< s steps)
      (photon-autograd-reset-tape)
      (let* ((y (photon-autograd-linear (photon-autograd-gelu (photon-autograd-linear xc w1c b1c))
                                        w2c b2c))
             (yd (photon-tensor-data (pav-value y))) (gd (photon-tensor-data (pav-grad y))) (i 0))
        (while (< i ntot)
          (aset gd i (* (- (aref yd i) (aref td i)) invn)) (setq i (1+ i)))
        (photon-autograd-zero-grad params)
        (dolist (v photon-autograd--tape) (when (pav-backward v) (funcall (pav-backward v) (pav-grad v))))
        (photon-autograd-sgd params lr)
        (when invalidate (nl-llm-gpu-invalidate params)))
      (setq s (1+ s)))))

(let ((have (nl-llm-gpu-enable)) (lr 0.02) (S 6) (seq 16))
  (photon-tensor-use-cpu-backend)
  (unless have (princ "(no GPU -- CPU only)\n"))
  (princ (format "%-14s %10s %10s %12s %10s %10s\n"
                 "FFN dim" "CPU ms" "hGPU ms" "ondev ms" "vs CPU" "vs hGPU"))
  (dolist (dim '(64 128 256))
    (let* ((in dim) (out dim) (hid (* 4 dim)) (sc (/ 1.0 (sqrt (float dim))))
           (x (bt--t (list seq in) 1 sc)) (target (bt--t (list seq out) 9 sc))
           (W1 (bt--t (list hid in) 2 sc)) (b1 (bt--t (list hid) 3 sc))
           (W2 (bt--t (list out hid) 4 sc)) (b2 (bt--t (list out) 5 sc)))
      ;; CPU
      (photon-tensor-use-cpu-backend)
      (let ((cpu (/ (bt--ms (bt--host-steps (bt--copy x) (bt--copy W1) (bt--copy b1)
                                            (bt--copy W2) (bt--copy b2) (bt--copy target) lr S nil)) S)))
        (if (not have)
            (princ (format "%-14s %10.1f %10s %12s %10s %10s\n" (format "%d->%d->%d" dim hid dim) cpu "-" "-" "-" "-"))
          ;; host GPU (re-upload each step)
          (photon-tensor-use-gpu-backend)
          (let ((hgpu (/ (bt--ms (bt--host-steps (bt--copy x) (bt--copy W1) (bt--copy b1)
                                                 (bt--copy W2) (bt--copy b2) (bt--copy target) lr S t)) S)))
            ;; on-device: difference 2S vs S to drop fixed upload/readback
            (let* ((t1 (bt--ms (nl-llm-gpu-mlp-train (bt--copy x) (bt--copy W1) (bt--copy b1)
                                                     (bt--copy W2) (bt--copy b2) (bt--copy target) lr S)))
                   (t2 (bt--ms (nl-llm-gpu-mlp-train (bt--copy x) (bt--copy W1) (bt--copy b1)
                                                     (bt--copy W2) (bt--copy b2) (bt--copy target) lr (* 2 S))))
                   (ondev (/ (- t2 t1) S)))
              (princ (format "%-14s %10.1f %10.1f %12.1f %9.2fx %9.2fx\n"
                             (format "%d->%d->%d" dim hid dim) cpu hgpu ondev
                             (/ cpu ondev) (/ hgpu ondev)))))))))
  (when have (nl-llm-gpu-disable)))
(princ "BENCH-GPU-TRAIN done\n")
;;; bench-gpu-train.el ends here
