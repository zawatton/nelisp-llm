;;; agent-unicode-pipeline-test.el --- versioned tokenizer pipeline tests -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(setq load-prefer-newer t)
(require 'ert)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-evolve)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-agent-training-checkpoint)

(defun nl-llm-agent-unicode-test--model (&optional tokenizer)
  "Return a tiny P5 model for TOKENIZER."
  (let ((id (nl-llm-agent-tokenizer-id tokenizer)))
    (nl-llm-agent-improve-model
     2 2 (nl-llm-agent-tokenizer-vocab id) 1 1 id)))

(defun nl-llm-agent-unicode-test--parameter-data (model)
  "Return detached flattened parameter values from MODEL."
  (apply
   #'vconcat
   (mapcar
    (lambda (parameter)
      (copy-sequence (photon-tensor-data (pav-value parameter))))
    (nl-llm-agent--p5-params model))))

(ert-deftest nl-llm-agent-unicode-artifact-preserves-explicit-tokenizer ()
  (let* ((directory (make-temp-file "nl-llm-unicode-artifact-" t))
         (catalog (expand-file-name "catalog.json" directory))
         (model
          (nl-llm-agent-unicode-test--model
           nl-llm-agent-tokenizer-utf8))
         (checkpoint (nl-llm-agent-artifact-export-pav model 3)))
    (unwind-protect
        (progn
          (should
           (equal
            (plist-get (plist-get checkpoint :config) :tokenizer)
            nl-llm-agent-tokenizer-utf8))
          (nl-llm-agent-artifact-publish
           catalog checkpoint :id "unicode-g1" :generation 1 :score 1.0
           :grammar '(:type "done" :length 1 :allow "日本語"))
          (let ((loaded
                 (nl-llm-agent-artifact-load-pav catalog "unicode-g1")))
            (should (= (plist-get loaded :vocab) 256))
            (should
             (equal (plist-get loaded :tokenizer)
                    nl-llm-agent-tokenizer-utf8))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-unicode-legacy-artifact-default-is-narrow ()
  (let* ((legacy
          (nl-llm-agent-artifact-export-pav
           (nl-llm-agent-improve-model 2 2 96 1 1)))
         (config (plist-get legacy :config)))
    ;; Simulate an immutable pre-tokenizer checkpoint without rewriting it.
    (setq config (copy-sequence config))
    (setq config (plist-put config :tokenizer nil))
    (let* ((raw (plist-put (copy-sequence legacy) :config config))
           (normalized
            (nl-llm-agent-artifact--checkpoint-model raw "legacy")))
      (should
       (equal (plist-get normalized :tokenizer)
              nl-llm-agent-tokenizer-ascii))
      (should-not (plist-get (plist-get raw :config) :tokenizer)))
    (let* ((wrong (copy-tree legacy t))
           (wrong-config (plist-get wrong :config)))
      (setf (plist-get wrong-config :tokenizer) nil
            (plist-get wrong-config :vocab) 256)
      (should-error
       (nl-llm-agent-artifact--checkpoint-model wrong "missing-utf8")))))

(ert-deftest nl-llm-agent-unicode-evolve-uses-bounded-encoded-tokens ()
  (let ((text "日本語\nagent"))
    (should-error
     (nl-llm-agent-evolve--validate-finetune
      (list :examples (vector text) :lr 0.1 :epochs 1)))
    (should
     (condition-case nil
         (progn
           (nl-llm-agent-evolve--validate-finetune
            (list :examples (vector text) :lr 0.1 :epochs 1)
            nl-llm-agent-tokenizer-utf8)
           t)
       (error nil)))
    (should
     (equal
      (car
       (nl-llm-agent-evolve--tokens
        (vector text) nl-llm-agent-tokenizer-utf8))
      (nl-llm-agent-tokenizer-encode
       text nl-llm-agent-tokenizer-utf8)))
    ;; 1366 characters are bounded as text but require 4098 UTF-8 byte tokens.
    (should-error
     (nl-llm-agent-evolve--validate-finetune
      (list :examples (vector (make-string 1366 ?日))
            :lr 0.1 :epochs 1)
      nl-llm-agent-tokenizer-utf8))))

(ert-deftest nl-llm-agent-unicode-cpu-evaluator-and-queue-bind-tokenizer ()
  (let* ((directory (make-temp-file "nl-llm-unicode-evolve-" t))
         (catalog (expand-file-name "catalog.json" directory))
         (model
          (nl-llm-agent-unicode-test--model
           nl-llm-agent-tokenizer-utf8))
         (evaluate
          (nl-llm-agent-evolve-p5-evaluator
           ["日本"] nl-llm-agent-tokenizer-utf8))
         (queue
          (nl-llm-agent-evolve-p5-queue
           model evaluate catalog
           '(:type "done" :length 1 :allow "日本")
           :id-prefix "unicode" :min-delta 1000000.0)))
    (unwind-protect
        (progn
          (should (numberp (funcall evaluate model)))
          (nl-llm-evolve-queue-submit
           queue "trajectory-finetune"
           '(:examples ["日本"] :lr 0.01 :epochs 1)
           :id "unicode-cpu")
          (should
           (memq
            (plist-get
             (nl-llm-evolve-queue-run queue "unicode-cpu") :status)
            '(rejected promoted))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-unicode-checkpoint-rejects-tokenizer-mismatch-atomically ()
  (let* ((ascii (nl-llm-agent-unicode-test--model))
         (before (nl-llm-agent-unicode-test--parameter-data ascii))
         (utf8
          (nl-llm-agent-unicode-test--model
           nl-llm-agent-tokenizer-utf8))
         (checkpoint
          (list
           :format nl-llm-agent-training-checkpoint-format
           :job-id "unicode-job"
           :payload-digest
           (nl-llm-agent-training-checkpoint-payload-digest
            '(:examples ["日本"] :lr 0.1 :epochs 1))
           :scope "unicode-test" :parent-generation 0 :parent-score 0.0
           :sequence 6 :optimizer 'sgd :completed-steps 0 :total-steps 1
           :optimizer-step 0
           :model (nl-llm-agent-artifact-export-pav utf8 0)
           :optimizer-state nil)))
    (should-error
     (nl-llm-agent-training-checkpoint-restore-model ascii checkpoint))
    (should
     (equal before (nl-llm-agent-unicode-test--parameter-data ascii)))))

(ert-deftest nl-llm-agent-unicode-rollout-samples-shared-byte-prefixes ()
  (let ((model
         (nl-llm-agent-unicode-test--model
          nl-llm-agent-tokenizer-utf8))
        (histories nil)
        (grammar
         (lambda (emitted)
           (pcase (length emitted)
             (0 '(:force ?語))
             (1 '(:allow "日本"))
             (_ :stop)))))
    (cl-letf (((symbol-function 'nl-llm-agent--p5-last-logits)
               (lambda (_model tokens)
                 (setq histories
                       (append histories (list (copy-sequence tokens))))
                 (make-vector 256 0.0)))
              ((symbol-function 'nl-llm-agent--sample-among)
               (lambda (_logits ids _temp)
                 (if (memq #x9c ids) #x9c (car ids)))))
      (let* ((result (nl-llm-agent-p5-rollout model grammar 1.0 "P"))
             (expected
              (nl-llm-agent-tokenizer-encode
               "P語本" nl-llm-agent-tokenizer-utf8)))
        (should (equal (car result) "語本"))
        (should (equal (cdr result) expected))
        ;; The forced character performs no needless model forward.  Sampling
        ;; the selected three-byte character sees the full prefix each time.
        (should (= (length histories) 3))
        (should
         (equal
          histories
          (list (butlast expected 3)
                (butlast expected 2)
                (butlast expected 1))))))))

(ert-deftest nl-llm-agent-unicode-gpu-padding-keeps-space-semantics ()
  (let* ((vocab 256)
         (onehot
          (nl-llm-agent--onehot-pad '(65) 3 vocab 32))
         (data (photon-tensor-data onehot)))
    (should (= (aref data 65) 1.0))
    (should (= (aref data (+ vocab 32)) 1.0))
    (should (= (aref data (+ (* 2 vocab) 32)) 1.0))
    (should (equal (append (nl-llm-agent--shift-pad '(65 66) 3 32) nil)
                   '(66 32 32)))))

(ert-deftest nl-llm-agent-unicode-gpu-adam-resume-equivalence ()
  (if (not (nl-llm-gpu-enable))
      (ert-skip "No Vulkan device")
    (let* ((tokens
            (nl-llm-agent-tokenizer-encode
             "日本" nl-llm-agent-tokenizer-utf8))
           (base
            (nl-llm-agent-unicode-test--model
             nl-llm-agent-tokenizer-utf8))
           (continuous (nl-llm-evolve-copy-model base))
           (partial (nl-llm-evolve-copy-model base))
           (resumed (nl-llm-evolve-copy-model base))
           continuous-context partial-context resumed-context snapshot)
      (unwind-protect
          (progn
            (setq continuous-context
                  (nl-llm-agent-ondevice-from-model
                   continuous 6 0.01 :optimizer 'adam)
                  partial-context
                  (nl-llm-agent-ondevice-from-model
                   partial 6 0.01 :optimizer 'adam))
            (should (= (plist-get continuous-context :pad-token) 32))
            (nl-llm-agent-ondevice-train continuous-context (list tokens) 2)
            (nl-llm-agent-ondevice-train partial-context (list tokens) 1)
            (setq snapshot
                  (nl-llm-agent-ondevice-snapshot partial-context))
            (let ((checkpoint
                   (list
                    :format nl-llm-agent-training-checkpoint-format
                    :job-id "unicode-resume"
                    :payload-digest
                    (nl-llm-agent-training-checkpoint-payload-digest
                     '(:examples ["日本"] :lr 0.01 :epochs 2))
                    :scope "unicode-gpu" :parent-generation 0
                    :parent-score 0.0 :sequence 6 :optimizer 'adam
                    :completed-steps 1 :total-steps 2 :optimizer-step 1
                    :model
                    (nl-llm-agent-artifact-export-pav
                     (plist-get snapshot :model) 1)
                    :optimizer-state (plist-get snapshot :optimizer-state))))
              (nl-llm-agent-training-checkpoint-restore-model resumed checkpoint)
              (setq resumed-context
                    (nl-llm-agent-ondevice-from-model
                     resumed 6 0.01 :optimizer 'adam))
              (nl-llm-agent-ondevice-restore-training-state
               resumed-context 1 (plist-get checkpoint :optimizer-state))
              (nl-llm-agent-ondevice-train
               resumed-context (list tokens) 2 :start-step 1)
              (nl-llm-agent-ondevice-sync continuous-context)
              (nl-llm-agent-ondevice-sync resumed-context)
              (should
               (equal
                (nl-llm-agent-unicode-test--parameter-data continuous)
                (nl-llm-agent-unicode-test--parameter-data resumed)))))
        (dolist (context
                 (list continuous-context partial-context resumed-context))
          (when context
            (ignore-errors (nl-llm-agent-ondevice-free context))))))))

(provide 'agent-unicode-pipeline-test)

(ert-run-tests-batch-and-exit)

;;; agent-unicode-pipeline-test.el ends here
