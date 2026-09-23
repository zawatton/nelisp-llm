;;; agent-recur-supervised-test.el --- recurrent completion supervision -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'ert)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-agent-supervised)

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here)))
(require 'nl-llm-agent-recur-supervised)

(declare-function nl-llm-agent-recur-supervised-loss
                  "nl-llm-agent-recur-supervised" (model examples &rest keys))
(declare-function nl-llm-agent-recur-supervised-train
                  "nl-llm-agent-recur-supervised" (model examples &rest keys))
(declare-function nl-llm-agent-recur-supervised--s0
                  "nl-llm-agent-recur-supervised" (geometry tokens seed))

(defconst nl-llm-agent-recur-supervised-test--examples
  [(:prompt "Q: a → " :completion "b\n")
   (:prompt "Q: 日本 " :completion "語\n")
   (:prompt "Q: xy? " :completion "z!\n")])

(defun nl-llm-agent-recur-supervised-test--model (&optional tokenizer)
  (let ((model (nl-llm-recur-model-new
                :vocab 256 :dim 4 :heads 1 :kv-heads 1 :ff 8
                :n-prelude 1 :n-core 1 :n-coda 1 :seed 17 :sigma 0.2)))
    (when tokenizer
      (setq model (plist-put model :tokenizer tokenizer)))
    model))

(defun nl-llm-agent-recur-supervised-test--snapshot (model)
  (mapcar (lambda (parameter)
            (list (copy-sequence (photon-tensor-shape (pav-value parameter)))
                  (copy-sequence (photon-tensor-data (pav-value parameter)))))
          (nl-llm-recur-params model)))

(defun nl-llm-agent-recur-supervised-test--loss-value (value)
  (if (numberp value)
      value
    (aref (photon-tensor-data (pav-value value)) 0)))

(defun nl-llm-agent-recur-supervised-test--direct-ce
    (model geometry tokens targets start r k seed)
  "Compute completion-only CE directly from recurrent logits.

This intentionally does not call the adapter's loss function: it applies the
target-row mask and stable log-sum-exp arithmetic to the direct forward."
  (let* ((photon-autograd--tape nil)
         ;; Build the deterministic initial state here rather than calling
         ;; the adapter's private helper; this keeps the numerical oracle
         ;; independent of the implementation under test.
         (s0 (photon-autograd-const
              (photon-tensor
               (list (length tokens) (plist-get geometry :dim))
               (nl-llm-recur-randn
                (* (length tokens) (plist-get geometry :dim))
                (plist-get geometry :sigma) seed))))
         (logits
         (plist-get
          (nl-llm-recur-forward
           model tokens r :k k :s0 s0)
          :logits))
        (data nil) (vocab 256) (rows (length targets)) (weighted 0.0)
        (row 0))
    (setq data (photon-tensor-data (pav-value logits)))
    (while (< row rows)
      (when (>= (1+ row) start)
        (let ((base (* row vocab)) (column 0) (maximum -1.0e30)
              (target (aref targets row)) (total 0.0))
          (while (< column vocab)
            (setq maximum (max maximum (aref data (+ base column))))
            (setq column (1+ column)))
          (setq column 0)
          (while (< column vocab)
            (setq total (+ total (exp (- (aref data (+ base column)) maximum))))
            (setq column (1+ column)))
          (setq weighted
                (+ weighted
                   (- (+ maximum (log total))
                      (aref data (+ base target)))))))
      (setq row (1+ row)))
    weighted))

(ert-deftest nl-llm-agent-recur-supervised-loss-is-deterministic-and-read-only ()
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (examples nl-llm-agent-recur-supervised-test--examples)
         (before (nl-llm-agent-recur-supervised-test--snapshot model))
         (encoded (nl-llm-agent-supervised-encode
                   examples "utf8-byte-v1"))
         (expected-random nil) (actual-random nil) loss1 loss2)
    (random "recur-supervised-random-sentinel")
    (setq expected-random (random 1000000))
    (random "recur-supervised-random-sentinel")
    (let ((photon-autograd--tape (list 'caller-tape)))
      (setq loss1
            (nl-llm-agent-recur-supervised-loss
             model examples :r 2 :k 1 :s0-seed 104729
             :tokenizer "utf8-byte-v1"))
      (should (equal photon-autograd--tape '(caller-tape))))
    (setq actual-random (random 1000000))
    (setq loss2
          (nl-llm-agent-recur-supervised-loss
           model examples :r 2 :k 1 :s0-seed 104729
           :tokenizer "utf8-byte-v1"))
    (should (= (nl-llm-agent-recur-supervised-test--loss-value loss1)
               (nl-llm-agent-recur-supervised-test--loss-value loss2)))
    (should (= (length (plist-get encoded :trajectories)) 3))
    (should (= (plist-get encoded :completion-tokens) 9))
    (should (stringp (plist-get encoded :dataset-sha256)))
    (should (= expected-random actual-random))
    (should (equal before
                   (nl-llm-agent-recur-supervised-test--snapshot model)))
    (should (numberp (nl-llm-agent-recur-supervised-test--loss-value loss1)))))

(ert-deftest nl-llm-agent-recur-supervised-loss-matches-weighted-completion-oracle ()
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (examples nl-llm-agent-recur-supervised-test--examples)
         (encoded (nl-llm-agent-supervised-encode examples "utf8-byte-v1"))
         (total (plist-get encoded :completion-tokens))
         (trajectories (plist-get encoded :trajectories))
         (starts (plist-get encoded :loss-starts))
         (geometry (list :dim 4 :sigma 0.2))
         (weighted 0.0)
         (index 0))
    (dolist (trajectory trajectories)
      (setq weighted
            (+ weighted
               (nl-llm-agent-recur-supervised-test--direct-ce
                model geometry (butlast trajectory)
                (apply #'vector (cdr trajectory))
                (aref starts index) 2 2 7)))
      (setq index (1+ index)))
    (setq weighted (/ weighted (float total)))
    (let ((whole
           (nl-llm-agent-recur-supervised-loss
            model examples :r 2 :k 2 :s0-seed 7
            :tokenizer "utf8-byte-v1")))
      (should (< (abs (- weighted
                         (nl-llm-agent-recur-supervised-test--loss-value whole)))
                 1.0e-7)))))

(ert-deftest nl-llm-agent-recur-supervised-forwards-r-k-and-s0 ()
  (let ((calls nil)
        (model (nl-llm-agent-recur-supervised-test--model)))
    (cl-letf (((symbol-function 'nl-llm-recur-forward)
               (lambda (model tokens r &rest keys)
                 (push (list model tokens r keys) calls)
                 (let ((logits (photon-autograd-const
                                (photon-tensor (list (length tokens) 256)
                                               (make-vector (* (length tokens) 256)
                                                            0.0)))))
                   (list :logits logits :states nil :e nil :r r)))))
      (should (numberp
               (nl-llm-agent-recur-supervised-loss
                model nl-llm-agent-recur-supervised-test--examples
                :r 3 :k 2 :s0-seed 99 :tokenizer "utf8-byte-v1"))))
    (should (= (length calls) 3))
    (dolist (call calls)
      (should (= (nth 2 call) 3))
      (should (= (plist-get (nth 3 call) :k) 2))
      (let ((s0 (plist-get (nth 3 call) :s0)))
        (should (pav-p s0))
        (should (equal (photon-tensor-shape (pav-value s0))
                       (list (length (nth 1 call)) 4)))))))

(ert-deftest nl-llm-agent-recur-supervised-s0-is-prefix-stable ()
  (let* ((geometry '(:dim 4 :sigma 0.2))
         (short (nl-llm-agent-recur-supervised--s0 geometry '(1 2) 99))
         (long (nl-llm-agent-recur-supervised--s0 geometry '(1 2 3) 99))
         (short-data (photon-tensor-data (pav-value short)))
         (long-data (photon-tensor-data (pav-value long))))
    (should (equal (photon-tensor-shape (pav-value short)) '(2 4)))
    (should (equal (photon-tensor-shape (pav-value long)) '(3 4)))
    (should (equal (cl-subseq (append long-data nil) 0 (* 2 4))
                   (append short-data nil)))))

(ert-deftest nl-llm-agent-recur-supervised-train-changes-finite-model ()
  (let* ((examples [(:prompt "Q: " :completion "A\n")])
         (model (nl-llm-agent-recur-supervised-test--model))
         (before (nl-llm-agent-recur-supervised-test--snapshot model))
         (loss-before
          (nl-llm-agent-recur-supervised-test--loss-value
           (nl-llm-agent-recur-supervised-loss
            model examples :r 1 :k 1 :s0-seed 1)))
         (result
          (nl-llm-agent-recur-supervised-train
           model examples :r 1 :k 1 :s0-seed 1 :lr 0.001 :epochs 3))
         (loss-after
          (nl-llm-agent-recur-supervised-test--loss-value
           (nl-llm-agent-recur-supervised-loss
            model examples :r 1 :k 1 :s0-seed 1))))
    (should (eq (plist-get result :backend) 'cpu))
    (should (eq (plist-get result :model-family) 'recurrent-depth))
    (should (= (plist-get result :r) 1))
    (should (= (plist-get result :k) 1))
    (should (= (plist-get result :s0-seed) 1))
    (should (= (plist-get result :steps) 3))
    (should (= (plist-get result :examples) 1))
    (should (= (plist-get result :completion-tokens) 2))
    (should (equal (plist-get result :dataset-sha256)
                   (plist-get
                    (nl-llm-agent-supervised-encode examples "utf8-byte-v1")
                    :dataset-sha256)))
    (should (numberp (plist-get result :loss-before)))
    (should (numberp (plist-get result :loss-after)))
    (should (< loss-after loss-before))
    (should-not (equal before
                       (nl-llm-agent-recur-supervised-test--snapshot model)))
    (dolist (parameter (nl-llm-recur-params model))
      (dolist (value (append (photon-tensor-data (pav-value parameter)) nil))
        (should (and (= value value) (< (abs value) 1.0e30)))))))

(ert-deftest nl-llm-agent-recur-supervised-rejects-invalid-before-mutation ()
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (before (nl-llm-agent-recur-supervised-test--snapshot model))
         (examples nl-llm-agent-recur-supervised-test--examples))
    (dolist (keys '((:r 0) (:r 33) (:r 2 :k 0) (:r 2 :k 3)
                    (:r 2 :s0-seed -1) (:r 2 :s0-seed #x100000000)
                    (:r 2 :unknown 1) (:r 2 :k 1 :k 2)))
      (should-error
       (apply #'nl-llm-agent-recur-supervised-train
              model examples (append keys '(:epochs 1)))))
    (should-error
     (nl-llm-agent-recur-supervised-train
      model (vector (aref examples 0) 'bad)
      :r 2 :k 1 :epochs 1))
    (should-error
     (nl-llm-agent-recur-supervised-loss
      model examples :r 2 :k 1 :tokenizer "ascii-char-v1"))
    (should (equal before
                   (nl-llm-agent-recur-supervised-test--snapshot model)))))

(ert-deftest nl-llm-agent-recur-supervised-preserves-caller-data-and-validates-shape ()
  (let* ((examples (copy-tree nl-llm-agent-recur-supervised-test--examples t))
         (before (prin1-to-string examples))
         (model (nl-llm-agent-recur-supervised-test--model)))
    (nl-llm-agent-recur-supervised-loss
     model examples :r 2 :k 1 :s0-seed 5)
    (should (equal before (prin1-to-string examples)))
    (should-error
     (nl-llm-agent-recur-supervised-loss
      (plist-put (copy-tree model t) :dim 5) examples :r 2))
    (should-error
     (nl-llm-agent-recur-supervised-loss
      (nl-llm-agent-recur-supervised-test--model "ascii-char-v1")
      examples :r 2 :tokenizer "utf8-byte-v1"))))

(ert-deftest nl-llm-agent-recur-supervised-preserves-gradient-sentinel ()
  "LOSS must not clear or overwrite gradients belonging to the caller."
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (parameter (car (nl-llm-recur-params model)))
         (gradient (pav-grad parameter))
         (data (photon-tensor-data gradient))
         (sentinel 123.456789))
    (aset data 0 sentinel)
    (let ((before (copy-sequence data)))
      (should (numberp
               (nl-llm-agent-recur-supervised-loss
                model nl-llm-agent-recur-supervised-test--examples
                :r 2 :k 1 :s0-seed 5)))
      (should (equal before (photon-tensor-data (pav-grad parameter)))))))

(ert-deftest nl-llm-agent-recur-supervised-accepts-zero-sigma-and-empty-stacks ()
  (let ((model (nl-llm-agent-recur-supervised-test--model)))
    (setf (plist-get model :sigma) 0.0)
    (setf (plist-get model :prelude) nil)
    (setf (plist-get model :coda) nil)
    (should (numberp
             (nl-llm-agent-recur-supervised-loss
              model nl-llm-agent-recur-supervised-test--examples
              :r 1 :k 1 :s0-seed 0)))))

(ert-deftest nl-llm-agent-recur-supervised-rejects-active-gpu-alias ()
  "The CPU adapter must fail closed when photon operations are GPU-swapped."
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (before (nl-llm-agent-recur-supervised-test--snapshot model))
         (cpu-linear (symbol-function 'photon-tensor-linear))
         (calls 0)
         (gpu-linear (lambda (&rest _args)
                       (setq calls (1+ calls))
                       (error "synthetic GPU backend")))
         (saved (list (cons 'photon-tensor-linear cpu-linear))))
    ;; Keep the backend's saved-function metadata present while checking that
    ;; a restored CPU dispatch is accepted; no GPU is started by this test.
    (cl-progv '(photon-tensor-gpu--saved) (list saved)
      (cl-letf (((symbol-function 'photon-tensor-linear) gpu-linear)
                ((symbol-function 'photon-tensor-linear-gpu) gpu-linear))
        (should (eq (indirect-function 'photon-tensor-linear)
                    (indirect-function 'photon-tensor-linear-gpu)))
        (let ((loss-error nil)
              (train-error nil))
          (condition-case err
              (nl-llm-agent-recur-supervised-loss
               model nl-llm-agent-recur-supervised-test--examples :r 1)
            (error (setq loss-error err)))
          (condition-case err
              (nl-llm-agent-recur-supervised-train
               model nl-llm-agent-recur-supervised-test--examples
               :r 1 :epochs 1)
            (error (setq train-error err)))
          (should loss-error)
          (should train-error)
          (should (string-match-p "refuses an active GPU"
                                  (error-message-string loss-error)))
          (should (string-match-p "refuses an active GPU"
                                  (error-message-string train-error)))
          (should (= calls 0))))
      (should (equal (symbol-value 'photon-tensor-gpu--saved) saved))
      (should (eq (symbol-function 'photon-tensor-linear) cpu-linear))
      ;; Restoring CPU dispatch with saved backend metadata remains usable.
      (should (numberp
               (nl-llm-agent-recur-supervised-loss
                model nl-llm-agent-recur-supervised-test--examples :r 1)))
      (should (equal before
                     (nl-llm-agent-recur-supervised-test--snapshot model))))))

(ert-deftest nl-llm-agent-recur-supervised-rejects-cyclic-and-bad-model-metadata ()
  (let* ((examples nl-llm-agent-recur-supervised-test--examples)
         (model (nl-llm-agent-recur-supervised-test--model))
         (cyclic-example (list :prompt "Q" :completion "A\n"))
         (cyclic-block (car (plist-get model :core)))
         ;; One dangling key exercises odd plist length before unknown-key
         ;; handling; the duplicate case below separately covers duplicates.
         (odd-model (append (copy-tree model t) '(:unexpected)))
         (duplicate-model (append (copy-tree model t)
                                  (list :dim (plist-get model :dim)))))
    (setcdr (last cyclic-example) cyclic-example)
    (should-error
     (nl-llm-agent-recur-supervised-loss
      model (vector cyclic-example) :r 1))
    (setcdr (last cyclic-block) cyclic-block)
    (should-error
     (nl-llm-agent-recur-supervised-loss model examples :r 1))
    (should-error
     (nl-llm-agent-recur-supervised-loss odd-model examples :r 1))
    (should-error
     (nl-llm-agent-recur-supervised-loss duplicate-model examples :r 1))))

(ert-deftest nl-llm-agent-recur-supervised-rejects-explicit-nil-options-before-mutation ()
  (let* ((model (nl-llm-agent-recur-supervised-test--model))
         (examples nl-llm-agent-recur-supervised-test--examples)
         (before (nl-llm-agent-recur-supervised-test--snapshot model)))
    (dolist (keys '((:r nil) (:k nil) (:s0-seed nil) (:tokenizer nil)
                    (:backend nil) (:optimizer nil)))
      (should-error
       (apply #'nl-llm-agent-recur-supervised-train
              model examples (append keys '(:epochs 1)))))
    (should (equal before
                   (nl-llm-agent-recur-supervised-test--snapshot model)))))

(ert-run-tests-batch-and-exit)

;;; agent-recur-supervised-test.el ends here
