;;; gpu-bitpack-test.el --- BitNet b1.58 Phase B: packed ternary matmul  -*- lexical-binding: t; -*-
;; Checks the packed-weight forward (nl-llm-bitnet-linear, kernel bitlinear-packed)
;; equals the f32 ternary linear X . (beta*ternary(W))^T + bias it encodes, and
;; reports the weight-memory reduction from base-4 packing.  Skips without Vulkan.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-bitpack-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nl-llm-bitnet)

(defvar bp--fail 0)
(defun bp--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" (progn (setq bp--fail (1+ bp--fail)) "FAIL")) (or extra ""))))
(defun bp--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-BITPACK SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((seq 5) (in 19) (out 7) (sc 0.5) (pk nl-llm-bitnet-pk)         ; in not a multiple of pk
       (x (bp--t (list seq in) 1 sc)) (w (bp--t (list out in) 2 sc)) (bias (bp--t (list out) 3 0.1))
       (xd (photon-tensor-data x)) (wd (photon-tensor-data w)) (bd (photon-tensor-data bias))
       (gpu (nl-llm-bitnet-linear x w bias)))
  (nl-llm-gpu-disable)
  ;; CPU reference: y = X . (beta*ternary(W))^T + bias
  (let* ((n (* out in)) (acc 0.0))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let* ((beta (/ acc (float n))) (wq (make-vector n 0.0)) (maxrel 0.0))
      (dotimes (i n) (let ((q (if (> beta 0.0) (/ (aref wd i) beta) 0.0)))
        (aset wq i (* beta (cond ((>= q 0.5) 1.0) ((<= q -0.5) -1.0) (t 0.0))))))
      (dotimes (s seq) (dotimes (o out)
        (let ((cpu (aref bd o)) (i 0)) (while (< i in)
          (setq cpu (+ cpu (* (aref xd (+ (* s in) i)) (aref wq (+ (* o in) i))))) (setq i (1+ i)))
          (let ((g (aref gpu (+ (* s out) o))))
            (setq maxrel (max maxrel (/ (abs (- g cpu)) (max 1e-3 (abs cpu)))))))))
      (bp--ck "packed ternary linear == f32 ternary linear" (< maxrel 1e-5) (format "maxrel=%.2e" maxrel))))
  ;; memory: packed floats vs dense f32 weights
  (let* ((fcount (/ (+ in pk -1) pk)) (dense (* out in)) (packed (* out fcount)))
    (bp--ck "packed weight buffer is smaller" (< packed dense)
            (format "%d -> %d floats (%.1fx, pk=%d)" dense packed (/ (float dense) packed) pk)))
  (princ (format "NL-LLM-GPU-BITPACK %s (%d failures)\n" (if (= bp--fail 0) "ALL-PASS" "HAS-FAILURES") bp--fail))
  (kill-emacs (if (= bp--fail 0) 0 1)))
;;; gpu-bitpack-test.el ends here
