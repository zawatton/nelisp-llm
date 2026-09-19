;;; deltanet-test.el --- the gated delta rule  -*- lexical-binding: t; -*-

;; Two questions, and they need different instruments.
;;
;; Is the forward the *right* recurrence?  Only a comparison against an
;; independent transcription can say, so tools/deltanet-ref.py writes a fixture
;; and this checks against it.  Two paths through one implementation would
;; agree about a misreading; two transcriptions of the reference do not.
;;
;; Is the backward the gradient *of that forward*?  Finite differences, which
;; need no second implementation and cannot be fooled by a shared misreading.
;;
;; The tolerance on the first is 1e-9, not 1e-5, and that matters: an earlier
;; version omitted the reference's division by sqrt(d_k) before the L2 norm, on
;; the reasoning that normalising discards scale.  It agreed to 6e-07 -- close,
;; and wrong.  The epsilon in x / (|x| + 1e-6) does not scale with x, so the
;; pre-division survives it.  A 1e-5 tolerance would have shipped that.

(require 'cl-lib)
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'nl-llm-deltanet)
(require 'json)

(defvar dn--fail 0)
(defvar dn--fixture (expand-file-name "build/deltanet-fixture.json"))

(defun dn--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name
                 (if ok "PASS" (progn (setq dn--fail (1+ dn--fail)) "FAIL"))
                 (or extra ""))))

(defun dn--flat (rows n)
  (let ((out (make-vector (* (length rows) n) 0.0)) (i 0))
    (dolist (r rows) (dolist (x r) (aset out i (float x)) (setq i (1+ i))))
    out))

(if (not (file-readable-p dn--fixture))
    (princ (format "SKIP: no fixture\n  %s\n  regenerate with:  \
python3 tools/deltanet-ref.py\n" dn--fixture))

  ;; --- the forward, against the independent reference --------------------
  (let* ((blob (with-temp-buffer
                 (insert-file-contents dn--fixture)
                 (json-parse-buffer :object-type 'plist :array-type 'list)))
         (seq (plist-get blob :seq)) (dk (plist-get blob :dk))
         (dv (plist-get blob :dv))
         (q (dn--flat (plist-get blob :q) dk))
         (k (dn--flat (plist-get blob :k) dk))
         (v (dn--flat (plist-get blob :v) dv))
         (a (vconcat (mapcar #'float (plist-get blob :a))))
         (b (vconcat (mapcar #'float (plist-get blob :b))))
         (want (dn--flat (plist-get blob :out) dv))
         (got (nth 0 (nl-llm-dn-forward q k v a b
                                        (float (plist-get blob :a_log))
                                        (float (plist-get blob :dt_bias))
                                        seq dk dv)))
         (m 0.0) (sc 0.0))
    (dotimes (i (* seq dv))
      (setq sc (max sc (abs (aref want i))))
      (setq m (max m (abs (- (aref want i) (aref got i))))))
    (dn--ck "forward == the independent reference"
            (< (/ m (max sc 1.0e-30)) 1.0e-9)
            (format "rel %.3e (seq %d, dk %d, dv %d)"
                    (/ m (max sc 1.0e-30)) seq dk dv))
    ;; A control for the tolerance above: normalising *without* the
    ;; pre-division is the mistake this check exists to catch, and it has to
    ;; land outside 1e-9 or the check is decorative.
    (let* ((scaled (make-vector (* seq dk) 0.0))
           (f (sqrt (float dk))))
      (dotimes (i (* seq dk)) (aset scaled i (* f (aref q i))))
      (let* ((other (nth 0 (nl-llm-dn-forward scaled k v a b
                                              (float (plist-get blob :a_log))
                                              (float (plist-get blob :dt_bias))
                                              seq dk dv)))
             (d 0.0))
        (dotimes (i (* seq dv))
          (setq d (max d (abs (- (aref want i) (aref other i))))))
        (dn--ck "control: the pre-division is not a no-op"
                (> (/ d (max sc 1.0e-30)) 1.0e-9)
                (format "rel %.3e when q is pre-scaled" (/ d (max sc 1.0e-30)))))))

  ;; --- the backward, against finite differences ---------------------------
  (let* ((seq 5) (dk 6) (dv 4)
         (q (make-vector (* seq dk) 0.0)) (k (make-vector (* seq dk) 0.0))
         (v (make-vector (* seq dv) 0.0))
         (a (make-vector seq 0.0)) (b (make-vector seq 0.0))
         (a-log 0.2) (dt-bias -0.1)
         (w (make-vector (* seq dv) 0.0))
         (h 1.0e-6))
    (dotimes (i (* seq dk))
      (aset q i (* 0.3 (- (mod (* (1+ i) 7919) 17) 8)))
      (aset k i (* 0.3 (- (mod (* (1+ i) 5387) 19) 9))))
    (dotimes (i (* seq dv)) (aset v i (* 0.3 (- (mod (* (1+ i) 3319) 13) 6))))
    (dotimes (i seq)
      (aset a i (* 0.4 (- (mod (* (1+ i) 11) 7) 3)))
      (aset b i (* 0.5 (- (mod (* (1+ i) 13) 5) 2))))
    (dotimes (i (* seq dv)) (aset w i (* 0.1 (- (mod (* (1+ i) 6151) 11) 5))))
    (let* ((loss (lambda (al db)
                   (let ((o (nth 0 (nl-llm-dn-forward q k v a b al db seq dk dv)))
                         (acc 0.0))
                     (dotimes (i (* seq dv))
                       (setq acc (+ acc (* (aref w i) (aref o i)))))
                     acc)))
           (fw (nl-llm-dn-forward q k v a b a-log dt-bias seq dk dv))
           (gr (nl-llm-dn-backward q k v a b a-log dt-bias seq dk dv
                                   (nth 1 fw) w)))
      (dolist (spec (list (list "dL/dq" q (plist-get gr :dq))
                          (list "dL/dk" k (plist-get gr :dk))
                          (list "dL/dv" v (plist-get gr :dv))
                          (list "dL/da" a (plist-get gr :da))
                          (list "dL/db" b (plist-get gr :db))))
        (let ((vec (nth 1 spec)) (ana (nth 2 spec)) (worst 0.0))
          (dotimes (i (length vec))
            (let ((o (aref vec i)))
              (aset vec i (+ o h))
              (let ((lp (funcall loss a-log dt-bias)))
                (aset vec i (- o h))
                (let* ((lm (funcall loss a-log dt-bias))
                       (num (/ (- lp lm) (* 2 h))))
                  (aset vec i o)
                  (setq worst (max worst
                                   (/ (abs (- num (aref ana i)))
                                      (max (abs num) (abs (aref ana i))
                                           1.0e-8))))))))
          (dn--ck (format "%s through the scan" (nth 0 spec))
                  (< worst 1.0e-5)
                  (format "worst rel %.2e over %d" worst (length vec)))))
      (dolist (spec (list (list "dL/dA_log" :da-log
                                (lambda (d) (funcall loss (+ a-log d) dt-bias)))
                          (list "dL/ddt_bias" :ddt-bias
                                (lambda (d) (funcall loss a-log (+ dt-bias d))))))
        (let* ((num (/ (- (funcall (nth 2 spec) h) (funcall (nth 2 spec) (- h)))
                       (* 2 h)))
               (ana (plist-get gr (nth 1 spec)))
               (rel (/ (abs (- num ana)) (max (abs num) (abs ana) 1.0e-8))))
          (dn--ck (nth 0 spec) (< rel 1.0e-5)
                  (format "worst rel %.2e (%.6g vs %.6g)" rel num ana))))))

  (princ (format "\n%s: %d failure(s)\n"
                 (if (zerop dn--fail) "deltanet OK" "deltanet") dn--fail))
  (when (> dn--fail 0) (kill-emacs 1)))
