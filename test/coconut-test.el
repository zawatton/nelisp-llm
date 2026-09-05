;;; coconut-test.el --- verify continuous latent reasoning on CPU  -*- lexical-binding: t; -*-

;; This deterministic script checks Coconut's row autograd operations, exact
;; zero-thought equivalence, latent sequence bookkeeping, feedback gradients,
;; thought ablation, and the complete staged chain-add training curriculum.
;; Run from the repository root with the lisp and nelisp-photon load paths.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-coconut)

(random "fixed-seed")

(defvar coc--fail 0)

(defun coc--ck (name ok &optional extra)
  "Print one check NAME with OK status and optional EXTRA detail."
  (princ (format "%-46s %s  %s\n"
                 name
                 (if ok "PASS"
                   (progn (setq coc--fail (1+ coc--fail)) "FAIL"))
                 (or extra ""))))

(defun coc--matrix (rows cols seed)
  "Return a deterministic ROWS by COLS autograd matrix using SEED."
  (let ((data (make-vector (* rows cols) 0.0))
        (i 0))
    (while (< i (* rows cols))
      (aset data i (* 0.17 (- (mod (+ (* (1+ i) 7) seed) 13) 6)))
      (setq i (1+ i)))
    (photon-autograd-const (photon-tensor (list rows cols) data))))

(defun coc--mask (n)
  "Return a deterministic length-N weighting vector for scalarized checks."
  (let ((data (make-vector n 0.0))
        (i 0))
    (while (< i n)
      (aset data i (- (mod (* (1+ i) 11) 9) 4.0))
      (setq i (1+ i)))
    data))

(defun coc--weighted-value (out weights)
  "Return the dot product of autograd OUT's value and WEIGHTS."
  (let* ((data (photon-tensor-data (pav-value out)))
         (n (length data))
         (sum 0.0)
         (i 0))
    (while (< i n)
      (setq sum (+ sum (* (aref data i) (aref weights i))))
      (setq i (1+ i)))
    sum))

(defun coc--gradcheck-op (name inputs forward input-index)
  "Numerically check one INPUT-INDEX of FORWARD and print NAME."
  (let ((eps 1.0e-5)
        (tolerance 1.0e-3))
    (photon-autograd-zero-grad inputs)
    (photon-autograd-reset-tape)
    (let* ((out (funcall forward))
           (out-data (photon-tensor-data (pav-value out)))
           (weights (coc--mask (length out-data)))
           (out-grad (photon-tensor-data (pav-grad out)))
           (i 0))
      (while (< i (length out-data))
        (aset out-grad i (aref weights i))
        (setq i (1+ i)))
      (dolist (var photon-autograd--tape)
        (when (pav-backward var)
          (funcall (pav-backward var) (pav-grad var))))
      (let* ((target (nth input-index inputs))
             (analytic (copy-sequence
                        (photon-tensor-data (pav-grad target))))
             (values (photon-tensor-data (pav-value target)))
             (count (length values))
             (maxrel 0.0)
             (k 0))
        (while (< k count)
          (let ((original (aref values k)))
            (aset values k (+ original eps))
            (photon-autograd-reset-tape)
            (let ((plus (coc--weighted-value (funcall forward) weights)))
              (aset values k (- original eps))
              (photon-autograd-reset-tape)
              (let* ((minus (coc--weighted-value (funcall forward) weights))
                     (numeric (/ (- plus minus) (* 2.0 eps)))
                     (denom (max 1.0e-4 (abs numeric)
                                 (abs (aref analytic k))))
                     (relative (/ (abs (- numeric (aref analytic k))) denom)))
                (when (> relative maxrel)
                  (setq maxrel relative)))
              (aset values k original)))
          (setq k (1+ k)))
        (coc--ck name (< maxrel tolerance)
                 (format "maxrel=%.2e" maxrel))))))

(defun coc--loss-gradcheck (name model parameter question c suffix)
  "Numerically check Coconut loss gradient for PARAMETER and print NAME."
  (let* ((eps 1.0e-5)
         (tolerance 1.0e-3)
         (params (nl-llm-coconut-params model)))
    (photon-autograd-zero-grad params)
    (let ((loss (nl-llm-coconut-loss model question c suffix)))
      (photon-autograd-backward loss))
    (let* ((analytic (copy-sequence
                      (photon-tensor-data (pav-grad parameter))))
           (values (photon-tensor-data (pav-value parameter)))
           (count (length values))
           (maxrel 0.0)
           (i 0))
      (while (< i count)
        (let ((original (aref values i)))
          (aset values i (+ original eps))
          (let ((plus (nl-llm-coconut--loss-value
                       (nl-llm-coconut-loss model question c suffix))))
            (aset values i (- original eps))
            (let* ((minus (nl-llm-coconut--loss-value
                           (nl-llm-coconut-loss model question c suffix)))
                   (numeric (/ (- plus minus) (* 2.0 eps)))
                   (denom (max 1.0e-4 (abs numeric)
                               (abs (aref analytic i))))
                   (relative (/ (abs (- numeric (aref analytic i))) denom)))
              (when (> relative maxrel)
                (setq maxrel relative)))
            (aset values i original)))
        (setq i (1+ i)))
      (coc--ck name (< maxrel tolerance)
               (format "maxrel=%.2e" maxrel)))))

(defun coc--maxdiff (a b)
  "Return the maximum absolute element difference between tensors A and B."
  (let* ((ad (photon-tensor-data a))
         (bd (photon-tensor-data b))
         (n (length ad))
         (maximum 0.0)
         (i 0))
    (while (< i n)
      (let ((difference (abs (- (aref ad i) (aref bd i)))))
        (when (> difference maximum)
          (setq maximum difference)))
      (setq i (1+ i)))
    maximum))

(defun coc--maxdiff-from-row (a b first-row cols)
  "Return maximum A/B difference from FIRST-ROW onward for COLS columns."
  (let* ((ad (photon-tensor-data a))
         (bd (photon-tensor-data b))
         (start (* first-row cols))
         (n (length ad))
         (maximum 0.0)
         (i start))
    (while (< i n)
      (let ((difference (abs (- (aref ad i) (aref bd i)))))
        (when (> difference maximum)
          (setq maximum difference)))
      (setq i (1+ i)))
    maximum))

(defun coc--plain-forward (model tokens)
  "Run the literal plain autograd transformer stack on MODEL and TOKENS."
  (photon-autograd-reset-tape)
  (let ((x (photon-autograd-embedding
            (plist-get model :wte) tokens (plist-get model :dim))))
    (dolist (block (plist-get model :blocks))
      (setq x (nl-llm-ag-block
               x block (plist-get model :heads)
               (plist-get model :kv-heads))))
    (setq x (nl-llm-ag-rmsnorm x (plist-get model :lnfg)))
    (photon-autograd-linear
     x (plist-get model :wh) (plist-get model :bh))))

(defun coc--answer-accuracy (model data stage c)
  "Return greedy final-answer accuracy of MODEL over DATA at STAGE and C."
  (let ((correct 0)
        (count 0))
    (dolist (item data)
      (let* ((example (nl-llm-coconut-stage-example item stage c))
             (generated (nl-llm-coconut-generate
                         model (car example) (* stage c)
                         (length (cdr example))))
             (answer (and generated (car (last generated)))))
        (when (and answer (= answer (plist-get item :a)))
          (setq correct (1+ correct)))
        (setq count (1+ count))))
    (/ (float correct) count)))

;; Group 1: row slice and concatenation numerical gradients.
(let ((x (coc--matrix 4 3 1)))
  (coc--gradcheck-op "rows: slice d/dx" (list x)
                     (lambda () (nl-llm-ag-slice-rows x 1 2)) 0))
(let ((a (coc--matrix 2 3 2))
      (b (coc--matrix 3 3 3)))
  (coc--gradcheck-op "rows: concat d/da" (list a b)
                     (lambda () (nl-llm-ag-concat-rows (list a b))) 0)
  (coc--gradcheck-op "rows: concat d/db" (list a b)
                     (lambda () (nl-llm-ag-concat-rows (list a b))) 1))

;; Group 2: zero thoughts are exactly the literal plain forward.
(let* ((model (nl-llm-coconut-model-new
               :vocab 12 :dim 8 :heads 2 :kv-heads 1 :ff 12
               :nblocks 1 :seed 17))
       (question '(3 1 4))
       (suffix '(5 9))
       (tokens (append question
                       (list (plist-get model :bot) (plist-get model :eot))
                       suffix))
       (coconut (pav-value
                 (nl-llm-coconut-forward model question 0 suffix)))
       (plain (pav-value (coc--plain-forward model tokens)))
       (maxdiff (coc--maxdiff coconut plain)))
  (coc--ck "c=0: literal-forward equivalence" (= maxdiff 0.0)
           (format "maxdiff=%.1f" maxdiff)))

;; Group 3: latent row bookkeeping and deterministic repeated forward.
(let* ((model (nl-llm-coconut-model-new
               :vocab 12 :dim 8 :heads 2 :kv-heads 1 :ff 12
               :nblocks 1 :seed 23))
       (question '(8 6 7))
       (suffix '(5 3))
       (first (pav-value
               (nl-llm-coconut-forward model question 2 suffix)))
       (second (pav-value
                (nl-llm-coconut-forward model question 2 suffix)))
       (expected-rows (+ (length question) 1 2 1 (length suffix)))
       (shape (photon-tensor-shape first))
       (maxdiff (coc--maxdiff first second)))
  (coc--ck "c=2: shape and deterministic logits"
           (and (equal shape (list expected-rows 12)) (= maxdiff 0.0))
           (format "shape=%S maxdiff=%.1f" shape maxdiff)))

;; Group 4: finite differences through two chained thoughts.
(let* ((model (nl-llm-coconut-model-new
               :vocab 12 :dim 8 :heads 2 :kv-heads 1 :ff 16
               :nblocks 1 :seed 31))
       (question '(1 4 7))
       (suffix '(2 9))
       (block (car (plist-get model :blocks))))
  (coc--loss-gradcheck "thought-grad: loss d/dwte"
                       model (plist-get model :wte) question 2 suffix)
  (coc--loss-gradcheck "thought-grad: loss d/dwq"
                       model (plist-get block :wq) question 2 suffix))

;; Group 5: ablating the continuous rows changes language-mode logits.
(let* ((model (nl-llm-coconut-model-new
               :vocab 12 :dim 8 :heads 2 :kv-heads 1 :ff 12
               :nblocks 1 :seed 41))
       (question '(2 7 1))
       (suffix '(8 2))
       (normal (pav-value
                (nl-llm-coconut-forward model question 2 suffix)))
       (zeroed (pav-value
                (nl-llm-coconut--forward model question 2 suffix t)))
       (first-language-row (+ (length question) 2 1))
       (maxdiff (coc--maxdiff-from-row normal zeroed first-language-row 12)))
  (coc--ck "thought-ablation: suffix logits change" (> maxdiff 0.0)
           (format "maxdiff=%.3e" maxdiff)))

;; Group 6: the full three-stage chain-add curriculum learns at every stage.
(let* ((data (nl-llm-coconut-task-chain-add 4 64 2026))
       (model (nl-llm-coconut-model-new
               :vocab 12 :dim 16 :heads 2 :kv-heads 1 :ff 16
               :nblocks 2 :seed 53))
       (untrained (coc--answer-accuracy model data 2 1))
       (trajectories (nl-llm-coconut-train
                      model data :stages 2 :epochs 3 :lr 0.3 :c 1))
       (trained (coc--answer-accuracy model data 2 1))
       (stage 0))
  (dolist (trajectory trajectories)
    (coc--ck (format "curriculum: stage %d loss decreases" stage)
             (< (car (last trajectory)) (car trajectory))
             (format "%s"
                     (mapconcat (lambda (value) (format "%.4f" value))
                                trajectory " -> ")))
    (setq stage (1+ stage)))
  (coc--ck "curriculum: final accuracy above chance"
           (> trained 0.1)
           (format "accuracy=%.3f" trained))
  (coc--ck "curriculum: final accuracy beats untrained"
           (> trained untrained)
           (format "untrained=%.3f trained=%.3f" untrained trained)))

(princ (format "NL-LLM-COCONUT %s (%d failures)\n"
               (if (= coc--fail 0) "ALL-PASS" "HAS-FAILURES") coc--fail))
(kill-emacs (if (= coc--fail 0) 0 1))
;;; coconut-test.el ends here
