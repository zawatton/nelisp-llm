;;; moe-test.el --- MoE routing correctness tests  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/moe-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-arch)
(require 'nl-llm-moe)

(defvar mo--fail 0)
(defun mo--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS"
                                         (progn (setq mo--fail (1+ mo--fail)) "FAIL"))
                 (or extra ""))))
(defun mo--mk (rows cols seed)
  (let ((v (make-vector (* rows cols) 0.0)) (i 0))
    (while (< i (* rows cols))
      (aset v i (* 0.1 (- (mod (+ (* (1+ i) 7) seed) 13) 6))) (setq i (1+ i)))
    (photon-tensor (list rows cols) v)))
(defun mo--close (a b &optional tol) (< (abs (- a b)) (or tol 1.0e-5)))

(let* ((dim 6) (ff 8) (seq 4) (ne 3)
       (router (mo--mk ne dim 5))
       (experts (list (list :wg (mo--mk ff dim 11) :wu (mo--mk ff dim 12) :wd (mo--mk dim ff 13))
                      (list :wg (mo--mk ff dim 21) :wu (mo--mk ff dim 22) :wd (mo--mk dim ff 23))
                      (list :wg (mo--mk ff dim 31) :wu (mo--mk ff dim 32) :wd (mo--mk dim ff 33))))
       (x (mo--mk seq dim 9))
       (logits (photon-tensor-data (photon-tensor-linear x router)))
       (yes (mapcar (lambda (ex) (photon-tensor-data
                                  (nl-llm-swiglu x (plist-get ex :wg) (plist-get ex :wu)
                                                 (plist-get ex :wd))))
                    experts)))
  (let* ((got (photon-tensor-data (nl-llm-moe x router experts 1))) (ok t))
    (dotimes (i seq)
      (let ((be 0) (bv (aref logits (* i ne))) (e 1))
        (while (< e ne) (when (> (aref logits (+ (* i ne) e)) bv)
                          (setq bv (aref logits (+ (* i ne) e)) be e))
               (setq e (1+ e)))
        (dotimes (t0 dim)
          (unless (mo--close (aref got (+ (* i dim) t0)) (aref (nth be yes) (+ (* i dim) t0)))
            (setq ok nil)))))
    (mo--ck "moe top-1 == argmax expert per row" ok))

  (let* ((got (photon-tensor-data (nl-llm-moe x router experts ne))) (ok t))
    (dotimes (i seq)
      (let ((base (* i ne)) (mx -1.0e30) (probs (make-vector ne 0.0)) (sm 0.0))
        (dotimes (e ne) (when (> (aref logits (+ base e)) mx) (setq mx (aref logits (+ base e)))))
        (dotimes (e ne) (let ((p (exp (- (aref logits (+ base e)) mx)))) (aset probs e p) (setq sm (+ sm p))))
        (dotimes (e ne) (aset probs e (/ (aref probs e) sm)))
        (dotimes (t0 dim)
          (let ((mix 0.0) (e 0))
            (while (< e ne) (setq mix (+ mix (* (aref probs e) (aref (nth e yes) (+ (* i dim) t0))))) (setq e (1+ e)))
            (unless (mo--close (aref got (+ (* i dim) t0)) mix) (setq ok nil))))))
    (mo--ck "moe top-k=E == full softmax mixture" ok))

  (mo--ck "moe output shape (seq x dim)"
          (equal (photon-tensor-shape (nl-llm-moe x router experts 2)) (list seq dim))))

(princ (format "NL-LLM-MOE %s (%d failures)\n"
               (if (= mo--fail 0) "ALL-PASS" "HAS-FAILURES") mo--fail))
(kill-emacs (if (= mo--fail 0) 0 1))
;;; moe-test.el ends here
