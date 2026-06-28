;;; sample-test.el --- temperature / top-k sampling correctness  -*- lexical-binding: t; -*-
;; Deterministic checks of nl-llm-sample (seeded RNG): top-k=1 and temp->0 reduce
;; to argmax, top-k restricts the support, and a uniform distribution spreads.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/sample-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-arch) (require 'nl-llm-attn) (require 'nl-llm-moe) (require 'nl-llm-block)

(defvar sm--fail 0)
(defun sm--ck (name ok &optional extra)
  (princ (format "%-44s %s  %s\n" name (if ok "PASS" (progn (setq sm--fail (1+ sm--fail)) "FAIL")) (or extra ""))))

(random "nl-llm-sample-seed")   ; deterministic

(let ((lg (vector 1.0 5.0 2.0 0.0)) (v 4))   ; argmax = index 1
  ;; top-k=1 -> always argmax
  (let ((ok t) (n 0)) (while (< n 30) (unless (= (nl-llm-sample lg 0 v 1.0 1) 1) (setq ok nil)) (setq n (1+ n)))
    (sm--ck "top-k=1 == argmax" ok))
  ;; very low temperature -> argmax
  (let ((ok t) (n 0)) (while (< n 30) (unless (= (nl-llm-sample lg 0 v 0.01 0) 1) (setq ok nil)) (setq n (1+ n)))
    (sm--ck "temp->0 == argmax" ok))
  ;; top-k=2 restricts support to the two largest (indices 1 and 2)
  (let ((ok t) (n 0)) (while (< n 60) (let ((s (nl-llm-sample lg 0 v 1.0 2))) (unless (or (= s 1) (= s 2)) (setq ok nil))) (setq n (1+ n)))
    (sm--ck "top-k=2 stays in top-2 support" ok)))

;; uniform logits -> all classes appear (temperature 1, no top-k)
(let* ((lg (vector 0.0 0.0 0.0 0.0)) (v 4) (seen (make-vector 4 0)) (n 0))
  (while (< n 200) (let ((s (nl-llm-sample lg 0 v 1.0 0))) (aset seen s (1+ (aref seen s)))) (setq n (1+ n)))
  (sm--ck "uniform spreads over all classes"
          (and (> (aref seen 0) 0) (> (aref seen 1) 0) (> (aref seen 2) 0) (> (aref seen 3) 0))
          (format "counts=%S" (append seen nil))))

;; peaked distribution: index 2 strongly favored -> most-sampled is 2
(let* ((lg (vector 0.0 0.0 4.0 0.0)) (v 4) (cnt (make-vector 4 0)) (n 0))
  (while (< n 200) (let ((s (nl-llm-sample lg 0 v 1.0 0))) (aset cnt s (1+ (aref cnt s)))) (setq n (1+ n)))
  (sm--ck "peaked logit is the mode"
          (and (> (aref cnt 2) (aref cnt 0)) (> (aref cnt 2) (aref cnt 1)) (> (aref cnt 2) (aref cnt 3)))
          (format "counts=%S" (append cnt nil))))

(princ (format "NL-LLM-SAMPLE %s (%d failures)\n" (if (= sm--fail 0) "ALL-PASS" "HAS-FAILURES") sm--fail))
(kill-emacs (if (= sm--fail 0) 0 1))
;;; sample-test.el ends here
