;;; ternary-gpu-test.el --- the two-bit path on the device -*- lexical-binding: t -*-

;; `ternary-rows' and its transpose are checked against their own arithmetic
;; in nelisp-gpu.  What this checks is the wiring above them: that a linear
;; read out of the exported table reaches the kernel with the right word
;; count, the right block count and the right scale buffer, none of which the
;; kernel can know it has been given wrongly.
;;
;; The forward is compared row by row against the CPU path on the same
;; linear.  The transpose is held by <W.x, g> = <x, W'.g>, which needs no
;; reference and which no index swap survives.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-weights)
(require 'nl-llm-weights-gpu)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defvar tg-pass 0)
(defvar tg-fail 0)
(defvar tg-seed 99)
(defvar tg-table (or (getenv "NL_TERNARY_TABLE") "build/bonsai/wts-t2.bin"))

(defun tg-check (name ok fmt &rest args)
  (if ok (setq tg-pass (1+ tg-pass)) (setq tg-fail (1+ tg-fail)))
  (message "%-50s %s  %s" name (if ok "PASS" "FAIL") (apply #'format fmt args)))

(defun tg-rnd ()
  (setq tg-seed (mod (+ (* tg-seed 1103515245) 12345) 2147483648))
  (- (/ (float tg-seed) 1073741824.0) 1.0))

(defun tg-run ()
  (unless (file-readable-p tg-table)
    (message "SKIP: no %s -- run `make ternary-export'" tg-table)
    (kill-emacs 0))
  (let* ((wts (nl-llm-weights-open tg-table))
         (lin (nl-llm-weights-linear wts :wout 0))
         (rows (nl-llm-weights-lin-rows lin))
         (cols (nl-llm-weights-lin-cols lin))
         (tn (nl-llm-weights-tensor wts :wout 0))
         (sc (nl-llm-weights-scales wts tn))
         (x (make-vector cols 0.0))
         (g (make-vector rows 0.0)))
    (tg-check "the linear is ternary" (nl-llm-weights-lin-ternary lin)
              "%d x %d, %d words a row, %d columns a scale"
              rows cols (nl-llm-weights-lin-words lin)
              (nl-llm-weights-lin-block lin))
    (dotimes (i cols) (aset x i (tg-rnd)))
    (dotimes (o rows) (aset g o (tg-rnd)))
    (nelisp-gpu-server-start)
    (unwind-protect
        (let ((h (nl-llm-wgpu-upload-lin lin)))
          (unwind-protect
              (progn (tg-forward lin h wts tn sc x rows cols)
                     (tg-transpose lin h x g rows cols))
            (dolist (k '(:w :s :b)) (nelisp-gpu-server-free (plist-get h k)))))
      (nelisp-gpu-server-stop)))
  (message "ternary-gpu: %d passed, %d failed" tg-pass tg-fail)
  (when (> tg-fail 0) (kill-emacs 1)))

(defun tg-forward (lin h wts tn sc x rows cols)
  (let ((y (nl-llm-wgpu-apply lin h x))
        (worst 0.0) (scale 0.0) (probed 0))
    (dolist (o (list 0 1 (/ rows 3) (1- rows)))
      (let ((row (nl-llm-weights-row wts tn o sc)) (acc 0.0))
        (dotimes (i cols) (setq acc (+ acc (* (aref row i) (aref x i)))))
        (setq scale (max scale (abs acc))
              worst (max worst (abs (- acc (aref y o))))
              probed (1+ probed))))
    (tg-check "forward agrees with the dequantized rows"
              (< (/ worst (max 1.0e-12 scale)) 1.0e-4)
              "worst %.3e of the row scale over %d rows"
              (/ worst (max 1.0e-12 scale)) probed)
    (let ((nonzero (cl-some (lambda (v) (/= v 0.0)) (append y nil))))
      (tg-check "control: the forward is not all zero" nonzero
                (if nonzero "it is not" "every output came back zero")))))

(defun tg-transpose (lin h x g rows cols)
  (let* ((y (nl-llm-wgpu-apply lin h x))
         (xt (nl-llm-wgpu-apply-t lin h g))
         (l 0.0) (r 0.0))
    (dotimes (o rows) (setq l (+ l (* (aref y o) (aref g o)))))
    (dotimes (i cols) (setq r (+ r (* (aref x i) (aref xt i)))))
    (tg-check "transpose: <W.x, g> = <x, W'.g> on the device"
              (< (/ (abs (- l r)) (max 1.0e-8 (abs l))) 1.0e-4)
              "%.6f against %.6f" l r)
    (let ((nonzero (cl-some (lambda (v) (/= v 0.0)) (append xt nil))))
      (tg-check "control: the transpose is not all zero" nonzero
                (if nonzero "it is not" "every output came back zero")))))

(tg-run)
