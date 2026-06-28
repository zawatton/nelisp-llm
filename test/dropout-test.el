;;; dropout-test.el --- inverted-dropout mask correctness  -*- lexical-binding: t; -*-
;; Checks nl-llm-dropout-mask: ~ (1-p) fraction survive at value 1/(1-p), the rest
;; are 0, and the mean is ~1 (so dropout needs no separate eval-time rescale).
;; Dropout itself is nlga-dropout = elementwise mul by this mask (already a
;; gradient-checked op).  Pure CPU, seeded.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/dropout-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-arch) (require 'nl-llm-attn) (require 'nl-llm-moe) (require 'nl-llm-block)

(defvar dr--fail 0)
(defun dr--ck (name ok &optional extra)
  (princ (format "%-44s %s  %s\n" name (if ok "PASS" (progn (setq dr--fail (1+ dr--fail)) "FAIL")) (or extra ""))))

(random "nl-llm-dropout-seed")

(let* ((p 0.5) (keep 0.5) (inv 2.0) (n 4000)
       (m (photon-tensor-data (nl-llm-dropout-mask (list n) p)))
       (nz 0) (sum 0.0) (badval nil) (i 0))
  (while (< i n)
    (let ((x (aref m i)))
      (cond ((= x 0.0) nil)
            ((< (abs (- x inv)) 1e-6) (setq nz (1+ nz)))
            (t (setq badval t)))
      (setq sum (+ sum x)))
    (setq i (1+ i)))
  (dr--ck "survivors are 0 or 1/(1-p)" (not badval))
  (dr--ck "keep fraction ~ 1-p" (< (abs (- (/ (float nz) n) keep)) 0.05) (format "keep=%.3f" (/ (float nz) n)))
  (dr--ck "mask mean ~ 1 (inverted dropout)" (< (abs (- (/ sum n) 1.0)) 0.05) (format "mean=%.3f" (/ sum n))))

;; p=0 -> all-ones (eval mask)
(let* ((m (photon-tensor-data (nl-llm-dropout-mask (list 50) 0.0))) (ok t) (i 0))
  (while (< i 50) (unless (= (aref m i) 1.0) (setq ok nil)) (setq i (1+ i)))
  (dr--ck "p=0 mask is all ones" ok))

;; applying the mask zeroes the dropped positions (dropout = elementwise mul)
(let* ((x (photon-tensor (list 6) (vector 1.0 1.0 1.0 1.0 1.0 1.0)))
       (m (nl-llm-dropout-mask (list 6) 0.5))
       (y (photon-tensor-hadamard x m)) (xd (photon-tensor-data y)) (md (photon-tensor-data m)) (ok t) (i 0))
  (while (< i 6) (unless (= (aref xd i) (aref md i)) (setq ok nil)) (setq i (1+ i)))
  (dr--ck "dropout applies the mask elementwise" ok))

(princ (format "NL-LLM-DROPOUT %s (%d failures)\n" (if (= dr--fail 0) "ALL-PASS" "HAS-FAILURES") dr--fail))
(kill-emacs (if (= dr--fail 0) 0 1))
;;; dropout-test.el ends here
