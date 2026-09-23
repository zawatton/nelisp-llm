;;; recur-demo.el --- recurrent depth: train, then show what R buys  -*- lexical-binding: t; -*-
;; Trains a tiny recurrent-depth model (docs/design/07-recurrent-depth.org) on
;; the sum-mod-10 synthetic task (K digits, no intermediate steps -- the model
;; must do the additions somewhere inside the recurrence) and reports:
;;   - held-out loss for R in {1, 2, 4, 8, 16} (does more test-time depth help?)
;;   - the mean iterations used by zero-shot adaptive exit at a few EPS
;;   - a path-independence number at R=16 (do different S0 converge together?)
;;   emacs -Q --batch -l examples/recur-demo.el
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-recur)

(random "nl-llm-recur-demo-seed")

(defun rd--loss-val (l) (aref (photon-tensor-data (pav-value l)) 0))
(defun rd--finite-nonneg-p (x) (and (numberp x) (>= x 0.0) (< x 1.0e30)))
(defun rd--const-randn (seq dim sigma seed)
  (photon-autograd-const (photon-tensor (list seq dim) (nl-llm-recur-randn (* seq dim) sigma seed))))
(defun rd--l2 (a b)
  (let ((n (length a)) (i 0) (s 0.0))
    (while (< i n) (let ((d (- (aref a i) (aref b i)))) (setq s (+ s (* d d)))) (setq i (1+ i)))
    (sqrt s)))
(defun rd--norm (a) (rd--l2 a (make-vector (length a) 0.0)))

;; ---- train ---------------------------------------------------------------

(defconst rd--dim 16) (defconst rd--kdigits 4) (defconst rd--nsteps 300) (defconst rd--lr 0.3)

(let* ((m (nl-llm-recur-model-new :vocab 11 :dim rd--dim :heads 2 :kv-heads 1 :ff 32
                                   :n-prelude 1 :n-core 1 :n-coda 1 :seed 7 :sigma 1.0))
       (params (nl-llm-recur-params m))
       (train-data (nl-llm-recur-task-sum-mod rd--kdigits 64 11))
       (ntrain (length train-data))
       (losses nil) (step 0))
  (princ (format "recurrent depth: dim=%d K=%d digits vocab=11 prelude=1 core=1 coda=1 steps=%d k=2\n"
                 rd--dim rd--kdigits rd--nsteps))
  (while (< step rd--nsteps)
    (let* ((ex (nth (% step ntrain) train-data))
           (r (nl-llm-recur-sample-r 3)))
      (photon-autograd-zero-grad params)
      (photon-autograd-reset-tape)
      (let ((loss (nl-llm-recur-loss m (car ex) (cdr ex) r :k 2)))
        (push (rd--loss-val loss) losses)
        (photon-autograd-backward loss)
        (photon-autograd-sgd params rd--lr)))
    (when (or (= (% step 50) 0) (= step (1- rd--nsteps)))
      (princ (format "step %3d  loss=%.4f\n" step (car losses))))
    (setq step (1+ step)))
  (setq losses (nreverse losses))
  (let* ((first50 (/ (apply #'+ (cl-subseq losses 0 50)) 50.0))
         (last50 (/ (apply #'+ (last losses 50)) 50.0))
         (trained-ok (< last50 first50)))
    (princ (format "mean loss: first50=%.4f last50=%.4f\n" first50 last50))

    ;; ---- held-out loss vs R ------------------------------------------
    (let* ((eval-data (nl-llm-recur-task-sum-mod rd--kdigits 20 999))
           (eval-s0s (let ((i 0) (out nil))
                       (dolist (ex0 eval-data)
                         (push (rd--const-randn (length (car ex0)) rd--dim 1.0 (+ 7000 i)) out)
                         (setq i (1+ i)))
                       (nreverse out)))
           (rvals '(1 2 4 8 16))
           (row-ok t))
      (princ "\nheld-out loss vs test-time depth R (same S0 per example across R):\n")
      (princ "| R  | mean loss |\n|----+-----------|\n")
      (dolist (r rvals)
        (let* ((n 0) (tot 0.0))
          (let ((exs eval-data) (s0s eval-s0s))
            (while exs
              (let* ((ex (car exs)) (s0 (car s0s)))
                (setq tot (+ tot (rd--loss-val (nl-llm-recur-loss m (car ex) (cdr ex) r :k r :s0 s0))))
                (setq n (1+ n)))
              (setq exs (cdr exs)) (setq s0s (cdr s0s))))
          (let ((ml (/ tot (float n))))
            (unless (rd--finite-nonneg-p ml) (setq row-ok nil))
            (princ (format "| %-2d | %9.4f |\n" r ml)))))

      ;; ---- adaptive exit: mean r-used at a few EPS ---------------------
      (princ "\nadaptive exit (RMAX=16): mean iterations used vs EPS:\n")
      (princ "| EPS       | mean r-used | mean final KL |\n|-----------+-------------+----------------|\n")
      (dolist (eps '(1.0 0.1 0.01 0.001 0.0001))
        (let* ((n 0) (tot-r 0.0) (tot-kl 0.0))
          (let ((exs eval-data) (s0s eval-s0s))
            (while exs
              (let* ((ex (car exs)) (s0 (car s0s))
                     (ad (nl-llm-recur-forward-adaptive m (car ex) 16 eps :s0 s0))
                     (kls (plist-get ad :kls)))
                (setq tot-r (+ tot-r (plist-get ad :r-used)))
                (setq tot-kl (+ tot-kl (or (car (last kls)) 0.0)))
                (setq n (1+ n)))
              (setq exs (cdr exs)) (setq s0s (cdr s0s))))
          (unless (rd--finite-nonneg-p (/ tot-r (float n))) (setq row-ok nil))
          (princ (format "| %-9.4f | %11.2f | %14.6f |\n" eps (/ tot-r (float n)) (/ tot-kl (float n))))))

      ;; ---- path independence at R=16 ------------------------------------
      (princ "\npath independence at R=16 (two different S0 per example):\n")
      (let* ((n 0) (tot-rel 0.0))
        (dolist (ex eval-data)
          (let* ((seq (length (car ex)))
                 (s0a (rd--const-randn seq rd--dim 1.0 (+ 9000 n)))
                 (s0b (rd--const-randn seq rd--dim 1.0 (+ 9500 n)))
                 (sa (nl-llm-ag-no-grad
                      (car (last (plist-get (nl-llm-recur-forward m (car ex) 16 :s0 s0a) :states)))))
                 (sb (nl-llm-ag-no-grad
                      (car (last (plist-get (nl-llm-recur-forward m (car ex) 16 :s0 s0b) :states)))))
                 (dist (rd--l2 (photon-tensor-data sa) (photon-tensor-data sb)))
                 (norm (max 1.0e-6 (rd--norm (photon-tensor-data sa)))))
            (setq tot-rel (+ tot-rel (/ dist norm)))
            (setq n (1+ n))))
        (let ((mean-rel (/ tot-rel (float n))))
          (unless (rd--finite-nonneg-p mean-rel) (setq row-ok nil))
          (princ (format "mean relative L2 distance between S_16 from two S0's: %.4f\n" mean-rel))))

      (princ (format "RECUR-DEMO=%s\n" (if (and trained-ok row-ok) "PASS" "FAIL")))
      (kill-emacs (if (and trained-ok row-ok) 0 1)))))
;;; recur-demo.el ends here
