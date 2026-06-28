;;; bench-dp4a.el --- DP4A int8 matmul vs f32, weights resident  -*- lexical-binding: t; -*-
;; Times the same out x in matmul three ways with the WEIGHTS RESIDENT (uploaded
;; once, as in real inference), so the timing reflects kernel compute + activation
;; streaming, not weight upload:
;;   linear           f32 dense matmul
;;   bitlinear-packed f32 ternary weights (base-4 unpack, `in' f32 MACs/row)
;;   bitlinear-dp4a   int8 activations + ternary weights via hardware OpSDot
;;                    (`in'/4 DP4A instructions/row -- the compute win)
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/bench-dp4a.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nelisp-gpu-server)
(require 'nl-llm-bitnet)

(defun bd--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun bd--time (reps thunk) (let ((t0 (float-time))) (dotimes (_ reps) (funcall thunk)) (/ (* 1000.0 (- (float-time) t0)) reps)))

(unless (nl-llm-gpu-enable) (princ "NL-LLM-BENCH-DP4A SKIP (no GPU)\n") (kill-emacs 0))
(let* ((seq 16) (in 1024) (out 1024) (reps 40) (pk nl-llm-bitnet-pk) (sc 0.4)
       (x (bd--t (list seq in) 1 sc)) (w (bd--t (list out in) 2 sc)) (bias (bd--t (list out) 3 0.1))
       (xd (photon-tensor-data x)) (bd (photon-tensor-data bias))
       ;; f32 dense weight resident
       (hW (nelisp-gpu-server-upload (photon-tensor-data w))) (hB (nelisp-gpu-server-upload bd))
       ;; f32 ternary packed weight resident
       (pk3 (nl-llm-bitnet-pack w pk)) (packed (nth 0 pk3)) (betap (nth 1 pk3)) (fcount (nth 2 pk3))
       (hWpk (nelisp-gpu-server-upload packed)) (hBeta (nelisp-gpu-server-upload (vector betap)))
       ;; dp4a int8: ternary weight resident, activations packed per call
       (pw (nl-llm-bitnet-pack-i8-w w)) (wlo (nth 0 pw)) (whi (nth 1 pw)) (betad (nth 2 pw)) (ng (nth 3 pw))
       (hWlo (nelisp-gpu-server-upload wlo)) (hWhi (nelisp-gpu-server-upload whi)) (hBetad (nelisp-gpu-server-upload (vector betad)))
       (pa (nl-llm-bitnet-pack-i8-act x)) (alo (nth 0 pa)) (ahi (nth 1 pa)) (gamma (nth 2 pa))
       (groups (/ (+ (* seq out) 63) 64)) (on (* seq out)))
  (princ (format "matmul seq=%d in=%d out=%d, weights resident, %d reps\n" seq in out reps))
  (cl-flet ((lin () (nelisp-gpu-server-run2 'linear
                      (list (cons 'in xd) (list 'res hW (* out in)) (list 'res hB out) (cons 'out on))
                      (list seq in out) groups))
            (pkd () (nelisp-gpu-server-run2 'bitlinear-packed
                      (list (cons 'in xd) (list 'res hWpk (* out fcount)) (list 'res hB out) (list 'res hBeta 1) (cons 'out on))
                      (list seq in out pk fcount) groups))
            (dp4 () (nelisp-gpu-server-run2 'bitlinear-dp4a
                      (list (cons 'in alo) (cons 'in ahi) (list 'res hWlo (* out ng)) (list 'res hWhi (* out ng))
                            (list 'res hB out) (list 'res hBetad 1) (cons 'in gamma) (cons 'out on))
                      (list seq out ng) groups)))
    (lin) (pkd) (dp4)                       ; warm up
    (let ((tl (bd--time reps #'lin)) (tp (bd--time reps #'pkd)) (td (bd--time reps #'dp4)))
      (nl-llm-gpu-disable)
      (princ (format "  linear (f32 dense)        : %7.3f ms/call\n" tl))
      (princ (format "  bitlinear-packed (f32 ter): %7.3f ms/call\n" tp))
      (princ (format "  bitlinear-dp4a   (int8)   : %7.3f ms/call\n" td))
      (princ (format "  => DP4A is %.2fx vs f32 dense, %.2fx vs f32 ternary\n" (/ tl td) (/ tp td)))
      (princ (format "  (inner loop: dense/ternary do %d MACs/row, DP4A does %d OpSDot/row)\n" in ng))))
  (princ "NL-LLM-BENCH-DP4A=OK\n"))
;;; bench-dp4a.el ends here
