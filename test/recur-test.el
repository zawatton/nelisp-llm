;;; recur-test.el --- correctness + gradient checks for recurrent depth  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/recur-test.el
;; Six groups (see docs/design/07-recurrent-depth.org "Verification"):
;;   1. shape / determinism
;;   2. recurrence has effect
;;   3. gradients through the recurrence (adapter, core, prelude weights)
;;   4. truncation is exact (maxdiff 0 vs a hand-built detached reference)
;;   5. adaptive-exit bounds
;;   6. training smoke (sum-mod task)
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-recur)

(random "nl-llm-recur-test-seed")   ; deterministic run: seed before any use of `random'

(defvar rt--fail 0)
(defun rt--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS"
                                         (progn (setq rt--fail (1+ rt--fail)) "FAIL"))
                 (or extra ""))))

(defun rt--loss-val (l) (aref (photon-tensor-data (pav-value l)) 0))

(defun rt--maxdiff (a b)
  "Max abs elementwise difference between two same-length float vectors."
  (let ((n (length a)) (i 0) (md 0.0))
    (while (< i n)
      (let ((d (abs (- (aref a i) (aref b i)))))
        (when (or (/= d d) (> d md)) (setq md d)))
      (setq i (1+ i)))
    md))

(defun rt--finite-nonneg-p (x) (and (>= x 0.0) (< x 1.0e30)))

(defun rt--const-randn (seq dim sigma seed)
  "An explicit S0 pav (seq x dim), deterministic from SEED -- for tests where
bit-identical reproducibility matters."
  (photon-autograd-const (photon-tensor (list seq dim) (nl-llm-recur-randn (* seq dim) sigma seed))))

(defun rt--targets-next (tokens)
  "Trivial next-token TARGETS vector for TOKENS (last position repeats the
last token) -- good enough to drive a scalar loss for gradchecks."
  (let* ((n (length tokens)) (v (make-vector n 0)) (i 0))
    (while (< i n)
      (aset v i (nth (min (1- n) (1+ i)) tokens))
      (setq i (1+ i)))
    v))

(defun rt--gradcheck (name model tokens targets r k s0 target)
  "Central-difference check of `nl-llm-recur-loss' MODEL TOKENS TARGETS R :K K
:S0 S0 w.r.t. every element of TARGET's value, against the analytic backward
grad accumulated into TARGET.  Prints a PASS/FAIL row."
  (let* ((eps 1.0e-4) (tol 1.0e-3)
         (xd (photon-tensor-data (pav-value target))) (nx (length xd))
         (ana (progn
                (photon-autograd-zero-grad (nl-llm-recur-params model))
                (photon-autograd-reset-tape)
                (photon-autograd-backward (nl-llm-recur-loss model tokens targets r :k k :s0 s0))
                (copy-sequence (photon-tensor-data (pav-grad target)))))
         (maxrel 0.0) (i 0))
    (while (< i nx)
      (let ((orig (aref xd i)))
        (aset xd i (+ orig eps)) (photon-autograd-reset-tape)
        (let ((lp (rt--loss-val (nl-llm-recur-loss model tokens targets r :k k :s0 s0))))
          (aset xd i (- orig eps)) (photon-autograd-reset-tape)
          (let* ((lm (rt--loss-val (nl-llm-recur-loss model tokens targets r :k k :s0 s0)))
                 (num (/ (- lp lm) (* 2.0 eps)))
                 (den (max 1.0e-4 (abs num) (abs (aref ana i))))
                 (rel (/ (abs (- num (aref ana i))) den)))
            (when (or (/= rel rel) (> rel maxrel)) (setq maxrel rel))))
        (aset xd i orig))
      (setq i (1+ i)))
    (rt--ck name (< maxrel tol) (format "maxrel=%.2e" maxrel))))

;; ---- 1. shape / determinism ------------------------------------------

(let* ((dim 4) (m (nl-llm-recur-model-new :vocab 11 :dim dim :heads 2 :kv-heads 1 :ff 6
                                           :n-prelude 1 :n-core 1 :n-coda 1 :seed 2 :sigma 1.0))
       (toks '(1 2 3 10)) (seq (length toks))
       (s0 (rt--const-randn seq dim 1.0 99)))
  (photon-autograd-reset-tape)
  (let* ((f1 (nl-llm-recur-forward m toks 2 :s0 s0))
         (lg1v (pav-value (plist-get f1 :logits)))
         (lg1 (photon-tensor-data lg1v)))
    (rt--ck "shape: logits is (seq x vocab)" (equal (photon-tensor-shape lg1v) (list seq 11)))
    (photon-autograd-reset-tape)
    (let* ((f2 (nl-llm-recur-forward m toks 2 :s0 s0))
           (lg2 (photon-tensor-data (pav-value (plist-get f2 :logits)))))
      (rt--ck "determinism: same S0 -> bit-identical logits"
              (= (rt--maxdiff lg1 lg2) 0.0) (format "maxdiff=%.3e" (rt--maxdiff lg1 lg2)))))

  ;; ---- 2. recurrence has effect ---------------------------------------
  (photon-autograd-reset-tape)
  (let* ((f-r1 (nl-llm-recur-forward m toks 1 :s0 s0))
         (lg-r1 (photon-tensor-data (pav-value (plist-get f-r1 :logits)))))
    (photon-autograd-reset-tape)
    (let* ((f-r2 (nl-llm-recur-forward m toks 2 :s0 s0))
           (lg-r2 (photon-tensor-data (pav-value (plist-get f-r2 :logits))))
           (d12 (rt--maxdiff lg-r1 lg-r2)))
      (rt--ck "recurrence has effect: r=1 vs r=2 logits differ" (> d12 1.0e-9)
              (format "maxdiff=%.3e" d12)))
    (photon-autograd-reset-tape)
    (let* ((s0b (rt--const-randn seq dim 1.0 12345))
           (f-r1b (nl-llm-recur-forward m toks 1 :s0 s0b))
           (lg-r1b (photon-tensor-data (pav-value (plist-get f-r1b :logits))))
           (dab (rt--maxdiff lg-r1 lg-r1b)))
      (rt--ck "state is used: r=1 with a different S0 differs" (> dab 1.0e-9)
              (format "maxdiff=%.3e" dab))))

  ;; ---- 3. gradients through the recurrence ----------------------------
  (let* ((targets (rt--targets-next toks)))
    (rt--gradcheck "grad d/d(adapter wa), r=2 k=2" m toks targets 2 2 s0 (plist-get m :wa))
    (rt--gradcheck "grad d/d(core block wq), r=2 k=2" m toks targets 2 2 s0
                   (plist-get (car (plist-get m :core)) :wq))
    (rt--gradcheck "grad d/d(prelude block wq), r=2 k=2" m toks targets 2 2 s0
                   (plist-get (car (plist-get m :prelude)) :wq))))

;; ---- 4. truncation is exact --------------------------------------------

(let* ((dim 4) (m (nl-llm-recur-model-new :vocab 11 :dim dim :heads 2 :kv-heads 1 :ff 6
                                           :n-prelude 1 :n-core 1 :n-coda 1 :seed 5 :sigma 0.7))
       (toks '(2 4 6 10)) (seq (length toks))
       (s0 (rt--const-randn seq dim 0.7 42))
       (targets (rt--targets-next toks))
       (params (nl-llm-recur-params m)))
  (cl-flet ((grads-for (r k s0v)
              (photon-autograd-zero-grad params)
              (photon-autograd-reset-tape)
              (photon-autograd-backward (nl-llm-recur-loss m toks targets r :k k :s0 s0v))
              (mapcar (lambda (v) (copy-sequence (photon-tensor-data (pav-grad v)))) params))
            (grads-maxdiff (gs1 gs2)
              (let ((md 0.0) (rest2 gs2))
                (dolist (g1 gs1)
                  (let ((d (rt--maxdiff g1 (car rest2))))
                    (when (or (/= d d) (> d md)) (setq md d)))
                  (setq rest2 (cdr rest2)))
                md)))
    (let ((g-k3 (grads-for 3 3 s0))    ; full backprop, r=3
          (g-k1 (grads-for 3 1 s0)))   ; truncated: only the last iteration on the tape
      (let* ((s2-value (nl-llm-ag-no-grad
                        (nth 1 (plist-get (nl-llm-recur-forward m toks 2 :s0 s0) :states))))
             (s0-ref (photon-autograd-const s2-value)))
        (photon-autograd-reset-tape)
        (let* ((g-ref (grads-for 1 1 s0-ref))
               (md-ref (grads-maxdiff g-k1 g-ref))
               (md-vs-full (grads-maxdiff g-k1 g-k3)))
          (rt--ck "truncation exact: k=1 grads == detached-S0 reference"
                  (= md-ref 0.0) (format "maxdiff=%.3e" md-ref))
          (rt--ck "truncation matters: k=1 grads != k=3 (full) grads"
                  (> md-vs-full 1.0e-9) (format "maxdiff=%.3e" md-vs-full)))))))

;; ---- 5+6. train on the sum-mod task, then adaptive-exit bounds ---------

(let* ((dim 16) (kdigits 4) (nsteps 200) (lr 0.3)
       (m (nl-llm-recur-model-new :vocab 11 :dim dim :heads 2 :kv-heads 1 :ff 32
                                   :n-prelude 1 :n-core 1 :n-coda 1 :seed 7 :sigma 1.0))
       (params (nl-llm-recur-params m))
       (train-data (nl-llm-recur-task-sum-mod kdigits 64 11))
       (ntrain (length train-data))
       (losses nil) (step 0))
  (while (< step nsteps)
    (let* ((ex (nth (% step ntrain) train-data)))
      (photon-autograd-zero-grad params)
      (photon-autograd-reset-tape)
      (let* ((r (nl-llm-recur-sample-r 3))
             (loss (nl-llm-recur-loss m (car ex) (cdr ex) r :k 2)))
        (push (rt--loss-val loss) losses)
        (photon-autograd-backward loss)
        (photon-autograd-sgd params lr)))
    (setq step (1+ step)))
  (setq losses (nreverse losses))
  (let* ((first50 (/ (apply #'+ (cl-subseq losses 0 50)) 50.0))
         (last50 (/ (apply #'+ (last losses 50)) 50.0)))
    (princ (format "training smoke: first50=%.4f last50=%.4f\n" first50 last50))
    (rt--ck "training smoke: last-50 mean loss < first-50 mean" (< last50 first50)
            (format "first50=%.4f last50=%.4f" first50 last50)))

  ;; ---- 5. adaptive-exit bounds (on this briefly trained model) --------
  (let* ((probe-tok (car (car (nl-llm-recur-task-sum-mod kdigits 1 4242))))
         (seq (length probe-tok))
         (s0-a (rt--const-randn seq dim 1.0 555)))
    (let ((ad-huge (nl-llm-recur-forward-adaptive m probe-tok 8 1.0e6 :s0 s0-a)))
      (rt--ck "adaptive: EPS huge -> r-used = 1" (= (plist-get ad-huge :r-used) 1)
              (format "r-used=%d" (plist-get ad-huge :r-used))))
    (let ((ad-zero (nl-llm-recur-forward-adaptive m probe-tok 8 0.0 :s0 s0-a)))
      (rt--ck "adaptive: EPS = 0 -> r-used = RMAX" (= (plist-get ad-zero :r-used) 8)
              (format "r-used=%d" (plist-get ad-zero :r-used))))
    (let* ((ad-mid (nl-llm-recur-forward-adaptive m probe-tok 8 1.0e-2 :s0 s0-a))
           (ru (plist-get ad-mid :r-used)) (kls (plist-get ad-mid :kls)))
      (rt--ck "adaptive: 1 <= r-used <= RMAX (mid EPS)" (and (>= ru 1) (<= ru 8))
              (format "r-used=%d" ru))
      (rt--ck "adaptive: returned KLs are finite and >= 0"
              (let ((ok t)) (dolist (kl kls) (unless (rt--finite-nonneg-p kl) (setq ok nil))) ok)
              (format "kls=%S" kls))))

  ;; ---- 6. held-out loss at r=1 and r=4 (report only, not a pass/fail
  ;; comparison -- "more depth helps" is a demo metric at this scale) -----
  (let* ((eval-data (nl-llm-recur-task-sum-mod kdigits 20 999))
         (s0-eval (rt--const-randn (length (car (car eval-data))) dim 1.0 777)))
    (cl-flet ((mean-loss-at (r)
                (/ (apply #'+ (mapcar (lambda (ex) (rt--loss-val
                                                    (nl-llm-recur-loss m (car ex) (cdr ex) r :k r :s0 s0-eval)))
                                      eval-data))
                   (float (length eval-data)))))
      (let ((l1 (mean-loss-at 1)) (l4 (mean-loss-at 4)))
        (princ (format "held-out mean loss: r=1 -> %.4f   r=4 -> %.4f\n" l1 l4))
        (rt--ck "held-out loss @ r=1 is finite" (rt--finite-nonneg-p l1) (format "%.4f" l1))
        (rt--ck "held-out loss @ r=4 is finite" (rt--finite-nonneg-p l4) (format "%.4f" l4))))))

(princ (format "NL-LLM-RECUR %s (%d failures)\n"
               (if (= rt--fail 0) "ALL-PASS" "HAS-FAILURES") rt--fail))
(kill-emacs (if (= rt--fail 0) 0 1))
;;; recur-test.el ends here
