;;; block-test.el --- modern transformer block/model forward tests  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/block-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-block)

(defvar bl--fail 0)
(defun bl--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS"
                                         (progn (setq bl--fail (1+ bl--fail)) "FAIL"))
                 (or extra ""))))
(defun bl--mk (rows cols seed)
  (let ((v (make-vector (* rows cols) 0.0)) (i 0))
    (while (< i (* rows cols))
      (aset v i (* 0.1 (- (mod (+ (* (1+ i) 7) seed) 13) 6))) (setq i (1+ i)))
    (photon-tensor (list rows cols) v)))
(defun bl--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(let* ((vocab 12) (dim 8) (heads 2) (kvh 1) (hd (/ dim heads)) (kvdim (* kvh hd)) (ff 16) (ne 2)
       (b0 (list :ln1g (bl--ones dim) :ln2g (bl--ones dim)
                 :wq (bl--mk dim dim 1) :wk (bl--mk kvdim dim 2)
                 :wv (bl--mk kvdim dim 3) :wo (bl--mk dim dim 4)
                 :wg (bl--mk ff dim 5) :wu (bl--mk ff dim 6) :wd (bl--mk dim ff 7)))
       (b1 (list :ln1g (bl--ones dim) :ln2g (bl--ones dim)
                 :wq (bl--mk dim dim 11) :wk (bl--mk kvdim dim 12)
                 :wv (bl--mk kvdim dim 13) :wo (bl--mk dim dim 14)
                 :router (bl--mk ne dim 15) :top-k 1
                 :experts (list (list :wg (bl--mk ff dim 21) :wu (bl--mk ff dim 22) :wd (bl--mk dim ff 23))
                                (list :wg (bl--mk ff dim 31) :wu (bl--mk ff dim 32) :wd (bl--mk dim ff 33)))))
       (model (list :dim dim :heads heads :kv-heads kvh
                    :wte (bl--mk vocab dim 41) :lnf (bl--ones dim)
                    :head (bl--mk vocab dim 42) :blocks (list b0 b1)))
       (tokens '(1 2 3 0 4))
       (seq (length tokens))
       (logits (nl-llm-model-forward model tokens))
       (d (photon-tensor-data logits)))
  (bl--ck "model forward shape (seq x vocab)"
          (equal (photon-tensor-shape logits) (list seq vocab)))
  (let ((fin t) (i 0) (n (length d)))
    (while (< i n)
      (let ((val (aref d i)))
        (unless (and (= val val) (< (abs val) 1.0e30)) (setq fin nil)))
      (setq i (1+ i)))
    (bl--ck "model logits finite (no NaN/Inf)" fin))
  (let* ((d2 (photon-tensor-data (nl-llm-model-forward model tokens))) (m 0.0) (i 0) (n (length d)))
    (while (< i n) (let ((e (abs (- (aref d i) (aref d2 i))))) (when (> e m) (setq m e))) (setq i (1+ i)))
    (bl--ck "model forward deterministic" (= m 0.0) (format "maxdiff=%.2e" m)))
  (let ((bx (nl-llm-block (bl--mk seq dim 9) b1 heads kvh)))
    (bl--ck "block preserves shape (seq x dim)"
            (equal (photon-tensor-shape bx) (list seq dim)))))

(princ (format "NL-LLM-BLOCK %s (%d failures)\n"
               (if (= bl--fail 0) "ALL-PASS" "HAS-FAILURES") bl--fail))
(kill-emacs (if (= bl--fail 0) 0 1))
;;; block-test.el ends here
