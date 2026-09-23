;;; agent-completion-resume-gpu-test.el --- completion checkpoint GPU resume -*- lexical-binding: t; -*-

;; This is an explicit Vulkan test.  It is intentionally not part of the
;; ordinary CPU test pass because it owns the process-wide GPU lifetime.

(require 'ert)
(require 'cl-lib)
(defconst nl-llm-agent-completion-resume-gpu-test--here
  (file-name-directory (or load-file-name buffer-file-name)))
(add-to-list 'load-path
             (expand-file-name "../lisp"
                               nl-llm-agent-completion-resume-gpu-test--here))
(add-to-list 'load-path
             (expand-file-name "../../nelisp-photon/lisp"
                               nl-llm-agent-completion-resume-gpu-test--here))

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-completion-plan)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-agent-training-checkpoint)
(require 'nl-llm-evolve)
(require 'nl-llm-gpu)

(defconst nl-llm-agent-completion-resume-gpu-test--tokens
  (vector (vector 65 66 67 68)
          (vector 65 66 67 68 69)
          (vector 65 66 67 68 69 70)))

(defconst nl-llm-agent-completion-resume-gpu-test--starts [1 2 3])

(defconst nl-llm-agent-completion-resume-gpu-test--masks
  (vector [0 1 0 1]
          [0 0 1 0 1]
          [0 0 0 1 0 1]))

(defconst nl-llm-agent-completion-resume-gpu-test--shuffle-seed 104729)

(defun nl-llm-agent-completion-resume-gpu-test--plan (transfer optimizer)
  "Return the frozen variable-length sparse completion PLAN."
  (nl-llm-agent-completion-plan-make
   nl-llm-agent-completion-resume-gpu-test--tokens
   nl-llm-agent-completion-resume-gpu-test--starts
   :tokenizer "utf8-byte-v1" :sequence 8 :learning-rate 0.01 :epochs 3
   :optimizer optimizer :transfer-mode transfer
   :loss-masks nl-llm-agent-completion-resume-gpu-test--masks
   :shuffle-seed nl-llm-agent-completion-resume-gpu-test--shuffle-seed))

(defun nl-llm-agent-completion-resume-gpu-test--model ()
  "Return the deterministic small UTF-8 byte PAV model."
  (nl-llm-agent-improve-model 4 4 256 1 2 "utf8-byte-v1"))

(defun nl-llm-agent-completion-resume-gpu-test--parameters (model)
  "Return detached flattened parameter values from MODEL."
  (apply #'vconcat
         (mapcar
          (lambda (parameter)
            (copy-sequence (photon-tensor-data (pav-value parameter))))
          (nl-llm-agent--p5-params model))))

(defun nl-llm-agent-completion-resume-gpu-test--max-diff (left right)
  "Return maximum finite absolute difference between equal vectors."
  (unless (= (length left) (length right))
    (error "completion resume parameter lengths differ"))
  (let ((maximum 0.0)
        (index 0))
    (while (< index (length left))
      (let ((a (aref left index))
            (b (aref right index)))
        (unless (and (= a a) (= b b)
                     (< (abs (float a)) 1.0e300)
                     (< (abs (float b)) 1.0e300))
          (error "completion resume produced a non-finite parameter"))
        (setq maximum (max maximum (abs (- a b)))))
      (setq index (1+ index)))
    maximum))

(defun nl-llm-agent-completion-resume-gpu-test--optimizer-diff (left right)
  "Return maximum finite difference between two Adam state lists."
  (cond
   ((and (null left) (null right)) 0.0)
   ((or (null left) (null right)
        (/= (length left) (length right)))
    (error "completion resume optimizer-state shapes differ"))
   (t
    (let ((maximum 0.0))
      (cl-mapc
       (lambda (left-pair right-pair)
         (setq maximum
               (max maximum
                    (nl-llm-agent-completion-resume-gpu-test--max-diff
                     (photon-tensor-data (car left-pair))
                     (photon-tensor-data (car right-pair)))
                    (nl-llm-agent-completion-resume-gpu-test--max-diff
                     (photon-tensor-data (cdr left-pair))
                     (photon-tensor-data (cdr right-pair))))))
       left right)
      maximum))))

(defun nl-llm-agent-completion-resume-gpu-test--train-keys (plan)
  "Return the explicit plan inputs accepted by the bound context."
  (list :loss-starts
        (plist-get plan :loss-starts)
        :loss-masks
        (plist-get plan :loss-masks)
        :shuffle-seed
        (plist-get plan :shuffle-seed)))

(defun nl-llm-agent-completion-resume-gpu-test--payload (plan)
  "Return a detached transaction payload for PLAN checkpoint binding."
  (list :examples (copy-tree (plist-get plan :trajectories) t)
        :lr (plist-get plan :learning-rate)
        :epochs (plist-get plan :epochs)))

(defun nl-llm-agent-completion-resume-gpu-test--run-case
    (base plan optimizer transfer split)
  "Compare uninterrupted and split training for PLAN at SPLIT."
  (let* ((sequence (plist-get plan :sequence))
         (lr (plist-get plan :learning-rate))
         (epochs (plist-get plan :epochs))
         (trajs (append (plist-get plan :trajectories) nil))
         (total (* epochs (length trajs)))
         (payload (nl-llm-agent-completion-resume-gpu-test--payload plan))
         (baseline-model (nl-llm-evolve-copy-model base))
         (partial-model (nl-llm-evolve-copy-model base))
         (resumed-model (nl-llm-evolve-copy-model base))
         (baseline-context nil)
         (partial-context nil)
         (resumed-context nil)
         (baseline-callbacks nil)
         (partial-callbacks nil)
         (resumed-callbacks nil)
         baseline-values baseline-state checkpoint)
    (unwind-protect
        (progn
          (setq baseline-context
                (nl-llm-agent-ondevice-from-model
                 baseline-model sequence lr :optimizer optimizer
                 :loss-mode 'completion :transfer-mode transfer
                 :completion-plan plan))
          (apply #'nl-llm-agent-ondevice-train
                 baseline-context trajs epochs
                 :after-step
                 (lambda (_ctx completed _planned)
                   (push completed baseline-callbacks))
                 (nl-llm-agent-completion-resume-gpu-test--train-keys plan))
          (nl-llm-agent-ondevice-sync baseline-context)
          (setq baseline-values
                (nl-llm-agent-completion-resume-gpu-test--parameters
                 baseline-model)
                baseline-state
                (nl-llm-agent-ondevice-optimizer-state baseline-context))
          (should (= (plist-get baseline-context :step) total))
          (should (equal (nreverse baseline-callbacks)
                         (number-sequence 1 total)))

          (setq partial-context
                (nl-llm-agent-ondevice-from-model
                 partial-model sequence lr :optimizer optimizer
                 :loss-mode 'completion :transfer-mode transfer
                 :completion-plan plan))
          (should
           (eq
            (catch 'nl-llm-agent-completion-resume-interrupted
              (apply #'nl-llm-agent-ondevice-train
                     partial-context trajs epochs
                     :after-step
                     (lambda (_ctx completed _planned)
                       (push completed partial-callbacks)
                       (when (= completed split)
                         (throw
                          'nl-llm-agent-completion-resume-interrupted
                          'interrupted)))
                     (nl-llm-agent-completion-resume-gpu-test--train-keys plan))
              'completed)
            'interrupted))
          (should (= (plist-get partial-context :step) split))
          (should (equal (nreverse partial-callbacks)
                         (number-sequence 1 split)))
          (let* ((snapshot (nl-llm-agent-ondevice-snapshot partial-context))
                 (snapshot-model
                  (nl-llm-agent-artifact-export-pav
                   (plist-get snapshot :model) split))
                 (directory (make-temp-file "nl-completion-resume-gpu-" t))
                 (file (expand-file-name "checkpoint.sexp" directory)))
            (should (equal (plist-get snapshot :completion-plan) plan))
            (unwind-protect
                (progn
                  (nl-llm-agent-training-checkpoint-save
                   file :job-id "completion-gpu-job" :payload payload
                   :scope "completion-gpu-scope" :parent-generation 0
                   :parent-score 0.0 :sequence sequence :optimizer optimizer
                   :completed-steps split :total-steps total
                   :optimizer-step split :model snapshot-model
                   :optimizer-state (plist-get snapshot :optimizer-state)
                   :completion-plan plan)
                  (setq checkpoint
                        (nl-llm-agent-training-checkpoint-load
                         file :job-id "completion-gpu-job" :payload payload
                         :scope "completion-gpu-scope" :parent-generation 0
                         :parent-score 0.0 :sequence sequence
                         :optimizer optimizer :total-steps total
                         :completion-plan plan)))
              (delete-directory directory t)))
          ;; The original resident graph is deliberately freed before the
          ;; restored graph is built, exercising the real checkpoint boundary.
          (nl-llm-agent-ondevice-free partial-context)
          (setq partial-context nil)
          (nl-llm-agent-training-checkpoint-restore-model
           resumed-model checkpoint)
          (setq resumed-context
                (nl-llm-agent-ondevice-from-model
                 resumed-model sequence lr :optimizer optimizer
                 :loss-mode 'completion :transfer-mode transfer
                 :completion-plan plan))
          (nl-llm-agent-ondevice-restore-training-state
           resumed-context split (plist-get checkpoint :optimizer-state) plan)
          (apply #'nl-llm-agent-ondevice-train
                 resumed-context trajs epochs :start-step split
                 :after-step
                 (lambda (_ctx completed _planned)
                   (push completed resumed-callbacks))
                 (nl-llm-agent-completion-resume-gpu-test--train-keys plan))
          (nl-llm-agent-ondevice-sync resumed-context)
          (should (= (plist-get resumed-context :step) total))
          (should (equal (nreverse resumed-callbacks)
                         (number-sequence (1+ split) total)))
          (let ((resumed-values
                 (nl-llm-agent-completion-resume-gpu-test--parameters
                  resumed-model))
                (resumed-state
                 (nl-llm-agent-ondevice-optimizer-state resumed-context)))
            (princ
             (format "completion resume optimizer=%s transfer=%s split=%d weight-diff=%.3g optimizer-diff=%.3g\n"
                     optimizer transfer split
                     (nl-llm-agent-completion-resume-gpu-test--max-diff
                      baseline-values resumed-values)
                     (nl-llm-agent-completion-resume-gpu-test--optimizer-diff
                      baseline-state resumed-state)))
            (let ((weight-diff
                   (nl-llm-agent-completion-resume-gpu-test--max-diff
                    baseline-values resumed-values))
                  (optimizer-diff
                   (nl-llm-agent-completion-resume-gpu-test--optimizer-diff
                    baseline-state resumed-state)))
              (should (< weight-diff 1.0e-5))
              (should (< optimizer-diff 1.0e-5))
              (should (equal baseline-values resumed-values))
              (should (equal baseline-state resumed-state)))
            (should (> (nl-llm-agent-completion-resume-gpu-test--max-diff
                        (nl-llm-agent-completion-resume-gpu-test--parameters
                         base)
                        baseline-values)
                       1.0e-10))
            ;; A final-step resume is a no-op and cannot invoke a callback.
            (let ((before (copy-sequence resumed-values))
                  (before-state resumed-state)
                  (callback-count (length resumed-callbacks)))
              (apply #'nl-llm-agent-ondevice-train
                     resumed-context trajs epochs :start-step total
                     :after-step
                     (lambda (&rest _args)
                       (error "final-step resume unexpectedly trained"))
                     (nl-llm-agent-completion-resume-gpu-test--train-keys plan))
              (nl-llm-agent-ondevice-sync resumed-context)
              (should (= (plist-get resumed-context :step) total))
              (should (= callback-count (length resumed-callbacks)))
              (should (equal before
                             (nl-llm-agent-completion-resume-gpu-test--parameters
                              resumed-model)))
              (should (equal before-state
                             (nl-llm-agent-ondevice-optimizer-state
                              resumed-context))))))
      (when baseline-context
        (ignore-errors (nl-llm-agent-ondevice-free baseline-context)))
      (when partial-context
        (ignore-errors (nl-llm-agent-ondevice-free partial-context)))
      (when resumed-context
        (ignore-errors (nl-llm-agent-ondevice-free resumed-context))))))

(ert-deftest nl-llm-agent-completion-resume-real-gpu-sparse-shuffle ()
  (unless (nl-llm-gpu-enable)
    (ert-skip "No Vulkan device; completion checkpoint resume requires GPU"))
  (unwind-protect
      (let ((base (nl-llm-agent-completion-resume-gpu-test--model)))
        (dolist (case '((sgd nil) (sgd compact) (adam nil) (adam compact)))
          (let* ((optimizer (nth 0 case))
                 (transfer (nth 1 case))
                 (plan
                  (nl-llm-agent-completion-resume-gpu-test--plan
                   transfer optimizer)))
            ;; 2 is inside the first epoch, 3 is its boundary, and 5 is in the
            ;; second epoch, exercising replay of the shuffle state further on.
            (dolist (split '(2 3 5))
              (nl-llm-agent-completion-resume-gpu-test--run-case
               base plan optimizer transfer split)))))
    (nl-llm-gpu-disable)))

(provide 'agent-completion-resume-gpu-test)

(ert-run-tests-batch-and-exit)

;;; agent-completion-resume-gpu-test.el ends here
