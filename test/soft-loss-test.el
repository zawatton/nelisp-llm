;;; soft-loss-test.el --- tests for the distillation soft-target loss  -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-soft-loss)

(defvar sl--fail 0)
(defvar sl--pass 0)

(defun sl--ck (name ok &optional detail)
  (if ok (setq sl--pass (1+ sl--pass)) (setq sl--fail (1+ sl--fail)))
  (princ (format "%-60s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(defun sl--finite-p (value)
  (and (floatp value) (= value value)
       (< value 1.0e+300) (> value -1.0e+300)))

(defun sl--max-error (a b)
  (let ((worst 0.0) (at nil))
    (dotimes (i (length a))
      (let* ((x (aref a i)) (y (aref b i))
             (absolute (abs (- x y)))
             (relative (/ absolute (max (abs x) (abs y) 1.0e-30))))
        (when (> relative worst)
          (setq worst relative at i))))
    (list worst at)))

(let* ((targets '((0 . -0.2) (1 . -0.7) (2 . -1.4)))
       (logits (vconcat (mapcar #'cdr targets)))
       (perturbed (copy-sequence logits)))
  (sl--ck "KL of teacher distribution against itself is zero"
          (< (abs (nl-llm-soft-loss-kl logits targets)) 1.0e-14))
  (aset perturbed 0 (+ (aref perturbed 0) 0.5))
  (sl--ck "control: perturbing one logit is strictly positive"
          (> (nl-llm-soft-loss-kl perturbed targets) 0.0)))

(let* ((targets '((0 . -0.2) (1 . -0.7) (2 . -1.4)))
       (logits (vconcat (mapcar #'cdr targets)))
       (shifted (copy-sequence logits)))
  (dotimes (i (length shifted)) (aset shifted i (+ (aref shifted i) 800.0)))
  (sl--ck "KL is invariant to adding a constant to every logit"
          (< (abs (- (nl-llm-soft-loss-kl logits targets)
                     (nl-llm-soft-loss-kl shifted targets)))
             1.0e-14)))

(let* ((logits (vconcat '(0.17 -0.31 0.43 -0.59 0.71 -0.83
                          0.29 -0.47 0.61 -0.13 0.37 -0.67)))
       (targets '((1 . -0.2) (4 . -0.8) (7 . -1.3) (10 . -0.5)))
       (gradient (nl-llm-soft-loss-kl-grad logits targets))
       (h 1.0e-5) (finite (make-vector (length logits) 0.0)))
  (dotimes (i (length logits))
    (let ((saved (aref logits i)))
      (aset logits i (+ saved h))
      (let ((up (nl-llm-soft-loss-kl logits targets)))
        (aset logits i (- saved h))
        (let ((down (nl-llm-soft-loss-kl logits targets)))
          (aset logits i saved)
          (aset finite i (/ (- up down) (* 2.0 h)))))))
  (let* ((error (sl--max-error gradient finite)) (relative (car error)))
    (sl--ck "finite differences agree with analytic KL gradient"
            (< relative 1.0e-6)
            (format "worst relative %.3e at %S" relative (cadr error)))
    (princ (format "finite-difference worst relative error: %.17g\n" relative)))
  (let ((wrong (make-vector (length gradient) 0.0)))
    (dotimes (i (length gradient)) (aset wrong i (- (aref gradient i))))
    (let* ((error (sl--max-error wrong finite)) (relative (car error)))
      (sl--ck "control: deliberately wrong gradient fails"
              (> relative 0.01)
              (format "wrong-gradient relative error %.3e" relative))
      (princ (format "wrong-gradient failure: %.17g\n" relative)))))

(let* ((logits (vconcat '(0.2 -0.3 0.4 0.1 0.6)))
       (targets '((1 . -0.2) (3 . -0.8)))
       (gradient (nl-llm-soft-loss-kl-grad logits targets))
       (outside '(0 2 4))
       (zero (cl-every (lambda (i) (= (aref gradient i) 0.0)) outside)))
  (sl--ck "gradient is exactly zero outside target ids" zero
          (format "%S" (mapcar (lambda (i) (aref gradient i)) outside))))

(let* ((logits (vconcat '(800.0 799.0 798.0 797.0)))
       (targets '((0 . -0.2) (1 . -0.8) (2 . -1.3)))
       (loss (nl-llm-soft-loss-kl logits targets))
       (gradient (nl-llm-soft-loss-kl-grad logits targets)))
  (sl--ck "large logits produce finite loss and gradient"
          (and (sl--finite-p loss)
               (cl-every #'sl--finite-p gradient))))

(let ((position (list :token "a"
                      :top (list (list :token "a" :logprob -0.1)
                                 (list :token "b" :logprob -0.4)
                                 (list :token "c" :logprob -1.0))))
      (table (make-hash-table :test #'equal)))
  (puthash "a" 4 table)
  (puthash "c" 9 table)
  (sl--ck "targets drops a missing alternative but keeps the rest"
          (equal (nl-llm-soft-loss-targets position table)
                 '((4 . -0.1) (9 . -1.0)))))

(let ((position (list :token "missing"
                      :top (list (list :token "a" :logprob -0.1))))
      (table (make-hash-table :test #'equal)))
  (puthash "a" 4 table)
  (sl--ck "targets is nil when sampled token is missing"
          (null (nl-llm-soft-loss-targets position table))))

(condition-case nil
    (progn (nl-llm-soft-loss-kl (vector 0.0) nil)
           (sl--ck "empty targets signal an error" nil))
  (error (sl--ck "empty targets signal an error" t)))

(let ((path (expand-file-name "build/distilled-soft.eld")))
  (if (not (file-readable-p path))
      (princ "real distilled-soft dataset: skip (file absent)\n")
    (with-temp-buffer
      (insert-file-contents path)
      (let* ((data (read (current-buffer)))
             (table (make-hash-table :test #'equal))
             (vocabulary (plist-get data :vocabulary))
             (example (aref (plist-get data :examples) 0))
             (position (car (plist-get example :tokens))))
        (dotimes (i (length vocabulary))
          (let ((entry (aref vocabulary i)))
            (puthash (car entry) (hash-table-count table) table)))
        (let ((targets (nl-llm-soft-loss-targets position table)))
          (sl--ck "real distilled-soft position has finite KL loss"
                  (and targets
                       (sl--finite-p
                        (nl-llm-soft-loss-kl
                         (make-vector (1+ (hash-table-count table)) 0.0)
                         targets)))))))))

(princ (format "\nsoft-loss: %d passed, %d failed\n" sl--pass sl--fail))
(kill-emacs (if (= sl--fail 0) 0 1))

;;; soft-loss-test.el ends here
