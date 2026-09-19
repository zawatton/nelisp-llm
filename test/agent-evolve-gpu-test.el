;;; agent-evolve-gpu-test.el --- isolated resident GPU challenger  -*- lexical-binding: t; -*-

;; This is an integration test.  It exercises the actual Vulkan/nlga path and
;; skips successfully when no supported device is available.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-agent-evolve)
(require 'nl-llm-gpu)
(require 'nl-llm-agent-ondevice)

(defvar agent-evolve-gpu--fail 0)

(defun agent-evolve-gpu--ck (name ok &optional extra)
  (princ (format "%-70s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-evolve-gpu--fail
                         (1+ agent-evolve-gpu--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-evolve-gpu--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(defun agent-evolve-gpu--parameter-data (model)
  "Return detached flattened parameter values for MODEL."
  (apply #'vconcat
         (mapcar
          (lambda (parameter)
            (copy-sequence
             (photon-tensor-data (pav-value parameter))))
          (nl-llm-agent--p5-params model))))

(defun agent-evolve-gpu--maxdiff (left right)
  "Return maximum absolute element difference between LEFT and RIGHT vectors."
  (let ((maximum 0.0))
    (dotimes (index (length left))
      (setq maximum
            (max maximum
                 (abs (- (aref left index) (aref right index))))))
    maximum))

(if (not (nl-llm-gpu-enable))
    (agent-evolve-gpu--ck
     "resident GPU challenger [SKIPPED: no Vulkan device]" t)
  (let* ((directory (make-temp-file "nl-llm-agent-evolve-gpu-" t))
         (catalog-file (expand-file-name "catalog.json" directory))
         (model
          (nl-llm-agent-improve-model
           2 2 nl-llm-agent-char-vocab 1 1))
         (evaluate (nl-llm-agent-evolve-p5-evaluator [" a"]))
         (queue
          (nl-llm-agent-evolve-p5-queue
           model evaluate catalog-file
           '(:type "done" :length 4 :allow "ab ")
           :id-prefix "gpu-self" :min-delta 0.0 :maxseq 128
           :training-backend 'gpu :training-sequence 2
           :optimizer 'sgd))
         (evolution (nl-llm-evolve-queue-evolution queue))
         (old-champion (nl-llm-evolution-champion evolution))
         (old-values (agent-evolve-gpu--parameter-data old-champion)))
    (unwind-protect
        (progn
          (agent-evolve-gpu--ck
           "GPU queue advertises only its bounded resident challenger handler"
           (equal
            (nl-llm-evolve-queue-catalog queue)
            '((:kind "trajectory-finetune"
               :description
               "Fine-tune an isolated GPU-resident P5 challenger and read back once"))))
          (agent-evolve-gpu--ck
           "trajectory exceeding the compiled GPU sequence is rejected before upload"
           (agent-evolve-gpu--error-p
            (lambda ()
              (nl-llm-evolve-queue-submit
               queue "trajectory-finetune"
               '(:examples [" ab"] :lr 0.1 :epochs 1)
               :id "too-long"))))
          (nl-llm-evolve-queue-submit
           queue "trajectory-finetune"
           '(:examples [" a"] :lr 0.1 :epochs 8)
           :id "learn-a-on-gpu")
          (let* ((result
                  (nl-llm-evolve-queue-run queue "learn-a-on-gpu"))
                 (new-champion
                  (nl-llm-evolution-champion evolution))
                 (new-values
                  (agent-evolve-gpu--parameter-data new-champion)))
            (agent-evolve-gpu--ck
             "actual resident GPU training passes the fixed gate and publishes g1"
             (and (eq (plist-get result :status) 'promoted)
                  (= (nl-llm-evolution-generation evolution) 1)
                  (equal
                   (plist-get
                    (car (nl-llm-agent-artifact-catalog catalog-file)) :id)
                   "gpu-self-g1")))
            (agent-evolve-gpu--ck
             "GPU readback changes only the promoted isolated challenger"
             (and (equal
                   old-values
                   (agent-evolve-gpu--parameter-data old-champion))
                  (not (equal old-values new-values)))))
          (let* ((adam-model
                  (nl-llm-agent-improve-model
                   2 2 nl-llm-agent-char-vocab 1 1))
                 (adam-queue
                  (nl-llm-agent-evolve-p5-queue
                   adam-model evaluate catalog-file
                   '(:type "done" :length 4 :allow "ab ")
                   :id-prefix "gpu-adam" :min-delta 0.0 :maxseq 128
                   :training-backend 'gpu :training-sequence 2
                   :optimizer 'adam)))
            (nl-llm-evolve-queue-submit
             adam-queue "trajectory-finetune"
             '(:examples [" a"] :lr 0.01 :epochs 8)
             :id "learn-a-with-resident-adam")
            (let ((adam-result
                   (nl-llm-evolve-queue-run
                    adam-queue "learn-a-with-resident-adam")))
              (agent-evolve-gpu--ck
               "resident Adam state executes through the same guarded GPU handler"
               (and (eq (plist-get adam-result :status) 'promoted)
                    (= (nl-llm-evolution-generation
                        (nl-llm-evolve-queue-evolution adam-queue))
                       1)))))
          (let* ((resume-root (expand-file-name "resume" directory))
                 (training-directory
                  (expand-file-name "training" resume-root))
                 (queue-file (expand-file-name "queue.sexp" resume-root))
                 (resume-catalog
                  (expand-file-name "catalog.json" resume-root))
                 (continuous-catalog
                  (expand-file-name "continuous.json" resume-root))
                 (scope "gpu-resume-test-v1")
                 (payload '(:examples [" a"] :lr 0.01 :epochs 8))
                 (base
                  (nl-llm-agent-improve-model
                   2 2 nl-llm-agent-char-vocab 1 1))
                 (resume-evaluate
                  (nl-llm-agent-evolve-p5-evaluator [" a"]))
                 (new-resume-queue
                  (lambda ()
                    (nl-llm-agent-evolve-p5-queue
                     base resume-evaluate resume-catalog
                     '(:type "done" :length 4 :allow "ab ")
                     :id-prefix "resumed" :min-delta 0.0 :maxseq 128
                     :training-backend 'gpu :training-sequence 2
                     :optimizer 'adam
                     :training-checkpoint-directory training-directory
                     :checkpoint-every 2 :checkpoint-scope scope
                     :checkpoint-file queue-file)))
                 (interrupted-queue (funcall new-resume-queue))
                 (interrupted-evolution
                  (nl-llm-evolve-queue-evolution interrupted-queue))
                 (partial
                  (nl-llm-evolve-copy-model
                   (nl-llm-evolution-champion interrupted-evolution)))
                 (examples
                  (nl-llm-agent-evolve--tokens
                   (plist-get payload :examples)))
                 (checkpoint-path
                  (nl-llm-agent-evolve--training-checkpoint-path
                   training-directory scope "resume-job"))
                 (partial-context nil))
            (make-directory resume-root t)
            (nl-llm-evolve-queue-submit
             interrupted-queue "trajectory-finetune" payload
             :id "resume-job")
            (unwind-protect
                (progn
                  (setq partial-context
                        (nl-llm-agent-ondevice-from-model
                         partial 2 0.01 :optimizer 'adam))
                  ;; Four of the planned eight flattened steps complete before
                  ;; the simulated process loss.
                  (nl-llm-agent-ondevice-train
                   partial-context examples 4)
                  (let* ((snapshot
                          (nl-llm-agent-ondevice-snapshot partial-context))
                         (step (plist-get snapshot :step)))
                    (nl-llm-agent-training-checkpoint-save
                     checkpoint-path :job-id "resume-job"
                     :payload payload :scope scope
                     :parent-generation 0
                     :parent-score
                     (nl-llm-evolution-champion-score interrupted-evolution)
                     :sequence 2 :optimizer 'adam
                     :completed-steps 4 :total-steps 8
                     :optimizer-step step
                     :model
                     (nl-llm-agent-artifact-export-pav
                      (plist-get snapshot :model) step)
                     :optimizer-state
                     (plist-get snapshot :optimizer-state))))
              (when partial-context
                (nl-llm-agent-ondevice-free partial-context)))
            (setf
             (nl-llm-evolve-queue-job-status
              (car (nl-llm-evolve-queue-jobs interrupted-queue)))
             'running)
            (nl-llm-evolve-queue-save interrupted-queue)
            (let ((restored (funcall new-resume-queue)))
              (nl-llm-evolve-queue-restore restored)
              (agent-evolve-gpu--ck
               "restart exposes interrupted GPU work without automatic replay"
               (let ((status (nl-llm-evolve-queue-status restored)))
                 (and (= (plist-get status :interrupted) 1)
                      (= (logand (file-modes checkpoint-path) #o777) #o600)
                      (file-regular-p checkpoint-path))))
              (let* ((resumed-result
                      (nl-llm-evolve-queue-resume restored "resume-job"))
                     (resumed-values
                      (agent-evolve-gpu--parameter-data
                       (nl-llm-evolution-champion
                        (nl-llm-evolve-queue-evolution restored))))
                     (continuous
                      (nl-llm-agent-evolve-p5-queue
                       base resume-evaluate continuous-catalog
                       '(:type "done" :length 4 :allow "ab ")
                       :id-prefix "continuous" :min-delta 0.0 :maxseq 128
                       :training-backend 'gpu :training-sequence 2
                       :optimizer 'adam)))
                (nl-llm-evolve-queue-submit
                 continuous "trajectory-finetune" payload
                 :id "continuous-job")
                (nl-llm-evolve-queue-run continuous "continuous-job")
                (let* ((continuous-values
                        (agent-evolve-gpu--parameter-data
                         (nl-llm-evolution-champion
                          (nl-llm-evolve-queue-evolution continuous))))
                       (difference
                        (agent-evolve-gpu--maxdiff
                         resumed-values continuous-values)))
                  (agent-evolve-gpu--ck
                   "explicit resume matches continuous Adam training and cleans recovery state"
                   (and (eq (plist-get resumed-result :status) 'promoted)
                        (plist-get
                         (plist-get
                          (plist-get resumed-result :result) :metadata)
                         :resumed)
                        (not (file-exists-p checkpoint-path))
                        (< difference 1.0e-4))
                   (format "maxdiff=%.2e" difference)))))))
      (ignore-errors (nl-llm-gpu-disable))
      (delete-directory directory t))))

(princ (format "NL-LLM-AGENT-EVOLVE-GPU %s (%d failures)\n"
               (if (= agent-evolve-gpu--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-evolve-gpu--fail))
(kill-emacs (if (= agent-evolve-gpu--fail 0) 0 1))

;;; agent-evolve-gpu-test.el ends here
