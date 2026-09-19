;;; agent-supervised-evolve-test.el --- supervised evolution adapter tests -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)

(defconst nl-llm-agent-supervised-evolve-test--here
  (file-name-directory (or load-file-name buffer-file-name)))

(let ((here nl-llm-agent-supervised-evolve-test--here))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here)))

(require 'nl-llm-agent-supervised-evolve)

(defun nl-llm-agent-supervised-evolve-test--score (_model)
  "Use a fixed score so adapter tests exercise queue plumbing only."
  0.0)

(defun nl-llm-agent-supervised-evolve-test--fixture ()
  "Return (QUEUE . DIRECTORY) with the legacy handler already installed."
  (let* ((directory (make-temp-file "nl-llm-supervised-evolve-" t))
         (catalog (expand-file-name "catalog.json" directory))
         (model (nl-llm-agent-improve-model 2 2 96 1 1))
         (queue
          (nl-llm-agent-evolve-p5-queue
           model #'nl-llm-agent-supervised-evolve-test--score catalog
           '(:type "done" :length 4 :allow "ab ")
           :maxseq 64)))
    (cons queue directory)))

(defmacro nl-llm-agent-supervised-evolve-test--with-queue
    (queue &rest body)
  (declare (indent 1) (debug t))
  `(let* ((fixture (nl-llm-agent-supervised-evolve-test--fixture))
          (,queue (car fixture))
          (directory (cdr fixture)))
     (unwind-protect
         (progn ,@body)
       (delete-directory directory t))))

(ert-deftest nl-llm-agent-supervised-evolve-register-adds-kind-only ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (let ((legacy
           (nl-llm-evolve-queue--handler queue "trajectory-finetune"))
          (before (nl-llm-evolve-queue-catalog queue)))
      (should (eq
               (nl-llm-agent-supervised-evolve-register queue)
               queue))
      (should (equal
               (car (nl-llm-evolve-queue-catalog queue))
               (car before)))
      (should (eq
               (nl-llm-evolve-queue-handler-train-fn
                (nl-llm-evolve-queue--handler queue "trajectory-finetune"))
               (nl-llm-evolve-queue-handler-train-fn legacy)))
      (should (equal
               (mapcar (lambda (entry) (plist-get entry :kind))
                       (nl-llm-evolve-queue-catalog queue))
               '("trajectory-finetune" "supervised-finetune"))))))

(ert-deftest nl-llm-agent-supervised-evolve-rejects-invalid-before-queueing ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register queue)
    (let ((before (nl-llm-evolve-queue-status queue)))
      (dolist (payload
               (list
                '(:examples ["Q:A"])
                '(:examples [(:prompt "Q" :completion "A" :extra t)])
                '(:examples [(:prompt "Q" :completion "A")] :unknown t)
                '(:examples [(:prompt "Q" :completion "A")] :lr 0.0)
                '(:examples [(:prompt "Q" :completion "A")] :epochs 33)))
        (should-error
         (nl-llm-evolve-queue-submit
          queue "supervised-finetune" payload)))
      (should (equal before (nl-llm-evolve-queue-status queue))))))

(ert-deftest nl-llm-agent-supervised-evolve-validates-gpu-sequence-at-submit ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register
     queue :training-backend 'gpu :training-sequence 2 :optimizer 'adam)
    (should-error
     (nl-llm-evolve-queue-submit
      queue "supervised-finetune"
      '(:examples [(:prompt "Q:" :completion "A")])))))

(ert-deftest nl-llm-agent-supervised-evolve-forwards-cpu-without-sequence ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register queue)
    (let (calls)
      (cl-letf (((symbol-function 'nl-llm-agent-supervised-train)
                 (lambda (&rest arguments)
                   (setq calls arguments)
                   :trained)))
        (nl-llm-evolve-queue-submit
         queue "supervised-finetune"
         '(:examples [(:prompt "Q" :completion "A")]
                     :lr 0.2 :epochs 3)
         :id "cpu")
        (nl-llm-evolve-queue-run queue "cpu"))
      (should (equal (cadr calls)
                     [(:prompt "Q" :completion "A")]))
      (should (equal (cddr calls)
                     '(:backend cpu :lr 0.2 :epochs 3 :optimizer sgd)))
      (should-not (memq :sequence calls)))))

(ert-deftest nl-llm-agent-supervised-evolve-forwards-gpu-sequence-and-optimizer ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register
     queue :training-backend 'gpu :training-sequence 8 :optimizer 'adam)
    (let (calls)
      (cl-letf (((symbol-function 'nl-llm-agent-supervised-train)
                 (lambda (&rest arguments)
                   (setq calls arguments)
                   :trained)))
        (nl-llm-evolve-queue-submit
         queue "supervised-finetune"
         '(:examples [(:prompt "Q" :completion "A")])
         :id "gpu")
        (nl-llm-evolve-queue-run queue "gpu"))
      (should (equal (cddr calls)
                     '(:backend gpu :lr 0.05 :epochs 1
                       :optimizer adam :sequence 8))))))

(ert-deftest nl-llm-agent-supervised-evolve-rejects-duplicate-without-mutation ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register queue)
    (let ((before (nl-llm-evolve-queue-catalog queue)))
      (should-error (nl-llm-agent-supervised-evolve-register queue))
      (should (equal before (nl-llm-evolve-queue-catalog queue))))))

(ert-deftest nl-llm-agent-supervised-evolve-has-no-midpoint-resume-callback ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register queue)
    (should-not
     (nl-llm-evolve-queue-handler-resume-fn
      (nl-llm-evolve-queue--handler queue "supervised-finetune")))))

(ert-deftest nl-llm-agent-supervised-evolve-rejects-candidate-tokenizer-mismatch ()
  (nl-llm-agent-supervised-evolve-test--with-queue queue
    (nl-llm-agent-supervised-evolve-register queue)
    (let ((calls 0))
      (cl-letf (((symbol-function
                  'nl-llm-agent-supervised-evolve--validate-model)
                 (lambda (_model)
                   (setq calls (1+ calls))
                   "utf8-byte-v1")))
        (nl-llm-evolve-queue-submit
         queue "supervised-finetune"
         '(:examples [(:prompt "Q" :completion "A")])
         :id "mismatch")
        (let ((result (nl-llm-evolve-queue-run queue "mismatch")))
          (should (eq (plist-get result :status) 'error))
          (should (= (nl-llm-evolution-generation
                      (nl-llm-evolve-queue-evolution queue))
                     0))
          (should (= calls 1)))))))

(ert-run-tests-batch-and-exit)
