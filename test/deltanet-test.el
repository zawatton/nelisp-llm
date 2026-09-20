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

(defun dn--fd (loss vec ana h)
  "Worst relative disagreement between ANA and central differences of LOSS."
  (let ((worst 0.0))
    (dotimes (i (length vec))
      (let ((o (aref vec i)))
        (aset vec i (+ o h))
        (let ((lp (funcall loss)))
          (aset vec i (- o h))
          (let* ((lm (funcall loss)) (num (/ (- lp lm) (* 2 h))))
            (aset vec i o)
            (setq worst (max worst (/ (abs (- num (aref ana i)))
                                      (max (abs num) (abs (aref ana i))
                                           1.0e-8))))))))
    worst))

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

  ;; --- the block around it: the convolution and the gated norm ----------
  ;;
  ;; Gradients alone would pass a convolution that reads the future, so the
  ;; conv carries a causality control: perturbing the last position must leave
  ;; every earlier output bit-identical.  And the gated norm's order is checked
  ;; only in the sense that its gradient matches its own forward -- the order
  ;; itself (normalise, weight, then gate) came from the reference, and all
  ;; three orders type-check.
  (let* ((seq 7) (ch 5) (kern 4) (h 1.0e-6)
         (x (make-vector (* seq ch) 0.0))
         (w (make-vector (* ch kern) 0.0))
         (bias (make-vector ch 0.0))
         (wt (make-vector (* seq ch) 0.0)))
    (dotimes (i (* seq ch))
      (aset x i (* 0.4 (- (mod (* (1+ i) 7919) 11) 5)))
      (aset wt i (* 0.2 (- (mod (* (1+ i) 6151) 9) 4))))
    (dotimes (i (* ch kern)) (aset w i (* 0.3 (- (mod (* (1+ i) 5387) 7) 3))))
    (dotimes (i ch) (aset bias i (* 0.1 (- (mod (* (1+ i) 13) 5) 2))))
    (let* ((loss (lambda ()
                   (let ((y (nth 0 (nl-llm-dn-conv x w bias seq ch kern)))
                         (acc 0.0))
                     (dotimes (i (* seq ch))
                       (setq acc (+ acc (* (aref wt i) (aref y i)))))
                     acc)))
           (fw (nl-llm-dn-conv x w bias seq ch kern))
           (gr (nl-llm-dn-conv-vjp x w (nth 1 fw) wt seq ch kern)))
      (dolist (sp (list (list "conv dL/dx" x (plist-get gr :dx))
                        (list "conv dL/dw" w (plist-get gr :dw))
                        (list "conv dL/dbias" bias (plist-get gr :dbias))))
        (let ((worst (dn--fd loss (nth 1 sp) (nth 2 sp) h)))
          (dn--ck (nth 0 sp) (< worst 1.0e-5)
                  (format "worst rel %.2e" worst)))))
    (let* ((y0 (nth 0 (nl-llm-dn-conv x w bias seq ch kern)))
           (x2 (copy-sequence x)) (early 0.0) (late 0.0))
      (dotimes (c ch) (aset x2 (+ (* (1- seq) ch) c) 99.0))
      (let ((y1 (nth 0 (nl-llm-dn-conv x2 w bias seq ch kern))))
        (dotimes (tt (1- seq))
          (dotimes (c ch)
            (setq early (max early (abs (- (aref y0 (+ (* tt ch) c))
                                           (aref y1 (+ (* tt ch) c))))))))
        (dotimes (c ch)
          (setq late (max late (abs (- (aref y0 (+ (* (1- seq) ch) c))
                                       (aref y1 (+ (* (1- seq) ch) c)))))))
        (dn--ck "control: the convolution is causal"
                (and (= early 0.0) (> late 1.0e-6))
                (format "earlier moved %.3e, last moved %.3e" early late)))))

  (let* ((n 9) (h 1.0e-6) (eps 1.0e-6)
         (x (make-vector n 0.0)) (gate (make-vector n 0.0))
         (weight (make-vector n 0.0)) (wt (make-vector n 0.0)))
    (dotimes (i n)
      (aset x i (* 0.5 (- (mod (* (1+ i) 7919) 13) 6)))
      (aset gate i (* 0.4 (- (mod (* (1+ i) 5387) 11) 5)))
      (aset weight i (+ 0.7 (* 0.05 (mod i 7))))
      (aset wt i (* 0.3 (- (mod (* (1+ i) 3319) 9) 4))))
    (let* ((loss (lambda ()
                   (let ((o (nth 0 (nl-llm-dn-norm-gated x gate weight n eps)))
                         (acc 0.0))
                     (dotimes (i n) (setq acc (+ acc (* (aref wt i) (aref o i)))))
                     acc)))
           (fw (nl-llm-dn-norm-gated x gate weight n eps))
           (gr (nl-llm-dn-norm-gated-vjp x gate weight (nth 1 fw) wt n eps)))
      (dolist (sp (list (list "gated norm dL/dx" x (plist-get gr :dx))
                        (list "gated norm dL/dgate" gate (plist-get gr :dgate))
                        (list "gated norm dL/dweight" weight
                              (plist-get gr :dweight))))
        (let ((worst (dn--fd loss (nth 1 sp) (nth 2 sp) h)))
          (dn--ck (nth 0 sp) (< worst 1.0e-5)
                  (format "worst rel %.2e" worst))))))

  ;; --- the whole block ---------------------------------------------------
  ;;
  ;; Projections, convolution, the recurrence per head, the gated norm and the
  ;; output projection, checked end to end against finite differences for
  ;; every input and every weight.
  ;;
  ;; nv/nk is 2 here on purpose.  With one key head per value head the repeat
  ;; is an identity and an implementation that forgot to *accumulate* dq and dk
  ;; across the value heads sharing them would pass; with two it does not.
  (let* ((seq 4) (hidden 6) (nk 2) (nv 4) (hd 3) (kern 4)
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd))
         (cfg (list :seq seq :hidden hidden :nk nk :nv nv :hd hd :kern kern
                    :eps 1.0e-6))
         (det (lambda (v mul md sc)
                (dotimes (i (length v))
                  (aset v i (* sc (- (mod (* (1+ i) mul) md) (/ md 2)))))
                v))
         (x (funcall det (make-vector (* seq hidden) 0.0) 7919 11 0.4))
         (wts (list :wqkvz (funcall det (make-vector (* hidden (+ cd vd)) 0.0)
                                    5387 9 0.15)
                    :wba (funcall det (make-vector (* hidden 2 nv) 0.0) 3319 7 0.2)
                    :conv-w (funcall det (make-vector (* cd kern) 0.0) 6151 7 0.25)
                    :conv-b (funcall det (make-vector cd 0.0) 13 5 0.1)
                    :a-log (funcall det (make-vector nv 0.0) 17 5 0.2)
                    :dt-bias (funcall det (make-vector nv 0.0) 19 5 0.15)
                    :norm-w (let ((w (make-vector hd 0.0)))
                              (dotimes (i hd) (aset w i (+ 0.8 (* 0.1 i)))) w)
                    :wout (funcall det (make-vector (* vd hidden) 0.0) 4271 9 0.18)))
         (wt (funcall det (make-vector (* seq hidden) 0.0) 2749 9 0.3))
         (h 1.0e-6))
    (let* ((loss (lambda ()
                   (let ((o (nth 0 (nl-llm-dn-block x cfg wts))) (acc 0.0))
                     (dotimes (i (* seq hidden))
                       (setq acc (+ acc (* (aref wt i) (aref o i)))))
                     acc)))
           (fw (nl-llm-dn-block x cfg wts))
           (gr (nl-llm-dn-block-backward x cfg wts (nth 1 fw) wt)))
      (dolist (sp (list (list "block dL/dx" x (plist-get gr :dx))
                        (list "block dL/dwqkvz" (plist-get wts :wqkvz)
                              (plist-get gr :dwqkvz))
                        (list "block dL/dwba" (plist-get wts :wba)
                              (plist-get gr :dwba))
                        (list "block dL/dconv-w" (plist-get wts :conv-w)
                              (plist-get gr :dconv-w))
                        (list "block dL/dconv-b" (plist-get wts :conv-b)
                              (plist-get gr :dconv-b))
                        (list "block dL/da-log" (plist-get wts :a-log)
                              (plist-get gr :da-log))
                        (list "block dL/ddt-bias" (plist-get wts :dt-bias)
                              (plist-get gr :ddt-bias))
                        (list "block dL/dnorm-w" (plist-get wts :norm-w)
                              (plist-get gr :dnorm-w))
                        (list "block dL/dwout" (plist-get wts :wout)
                              (plist-get gr :dwout))))
        (let ((worst (dn--fd loss (nth 1 sp) (nth 2 sp) h)))
          (dn--ck (nth 0 sp) (< worst 1.0e-5)
                  (format "worst rel %.2e over %d" worst (length (nth 1 sp))))))))

  ;; --- which side of the norm the gate falls on ---------------------------
  ;;
  ;; Both orders run, both are stable, and both give a plausible residual
  ;; stream, so the only thing that separates them is the model's output.
  ;; What these checks pin is that the variable is load-bearing, that each
  ;; order's gradient is its own, and what the difference actually is.
  (let* ((n 16) (eps 1.0e-6)
         (x (make-vector n 0.0)) (g (make-vector n 0.0)) (wt (make-vector n 0.0))
         (dout (make-vector n 0.0)) (seed 20260920))
    (dotimes (i n)
      (setq seed (mod (+ (* seed 1103515245) 12345) 2147483648))
      (aset x i (- (/ (float seed) 1073741824.0) 1.0))
      (setq seed (mod (+ (* seed 1103515245) 12345) 2147483648))
      (aset g i (* 2.0 (- (/ (float seed) 1073741824.0) 1.0)))
      (setq seed (mod (+ (* seed 1103515245) 12345) 2147483648))
      (aset wt i (+ 0.8 (* 0.4 (/ (float seed) 2147483648.0))))
      (setq seed (mod (+ (* seed 1103515245) 12345) 2147483648))
      (aset dout i (- (/ (float seed) 1073741824.0) 1.0)))
    (let (outs)
      (dolist (order '(t nil))
        (let* ((nl-llm-dn-gate-before-norm order)
               (fw (nl-llm-dn-norm-gated x g wt n eps))
               (vj (nl-llm-dn-norm-gated-vjp x g wt (nth 1 fw) dout n eps))
               (loss (lambda ()
                       (let ((o (nth 0 (nl-llm-dn-norm-gated x g wt n eps)))
                             (acc 0.0))
                         (dotimes (i n) (setq acc (+ acc (* (aref o i) (aref dout i)))))
                         acc))))
          (push (copy-sequence (nth 0 fw)) outs)
          (dolist (probe (list (cons "dx" (cons x (plist-get vj :dx)))
                               (cons "dgate" (cons g (plist-get vj :dgate)))
                               (cons "dweight" (cons wt (plist-get vj :dweight)))))
            (let ((worst (dn--fd loss (car (cdr probe)) (cdr (cdr probe)) 1.0e-5)))
              (dn--ck (format "gated norm %s, gate %s norm" (car probe)
                              (if order "before" "after"))
                      (< worst 1.0e-5) (format "worst rel %.2e" worst))))))
      ;; a control: if the two orders agreed, every check above would be
      ;; checking one function twice
      (let ((worst 0.0))
        (dotimes (i n)
          (setq worst (max worst (abs (- (aref (nth 0 outs) i) (aref (nth 1 outs) i))))))
        (dn--ck "control: the two gate orders differ" (> worst 1.0e-6)
                (format "worst |difference| %.3e" worst))))
    ;; and what the difference is: gating before the norm bounds the output
    ;; however large the gate grows, gating after does not
    (let ((big (make-vector n 0.0)) (norms nil))
      (dotimes (i n) (aset big i 40.0))
      (dolist (order '(t nil))
        (let* ((nl-llm-dn-gate-before-norm order)
               (o (nth 0 (nl-llm-dn-norm-gated x big wt n eps)))
               (ss 0.0))
          (dotimes (i n) (setq ss (+ ss (* (aref o i) (aref o i)))))
          (push (sqrt (/ ss (float n))) norms)))
      (dn--ck "gate before the norm bounds the output"
              (< (nth 1 norms) (* 0.1 (nth 0 norms)))
              (format "rms %.3f before vs %.3f after, at gate 40"
                      (nth 1 norms) (nth 0 norms)))))

  (princ (format "\n%s: %d failure(s)\n"
                 (if (zerop dn--fail) "deltanet OK" "deltanet") dn--fail))
  (when (> dn--fail 0) (kill-emacs 1)))
