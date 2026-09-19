;;; agent-training-checkpoint-test.el --- durable P5 recovery state  -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-agent-training-checkpoint)

(defvar agent-training-checkpoint--fail 0)

(defun agent-training-checkpoint--ck (name ok)
  (princ (format "%-72s %s\n" name
                 (if ok "PASS"
                   (setq agent-training-checkpoint--fail
                         (1+ agent-training-checkpoint--fail))
                   "FAIL"))))

(defun agent-training-checkpoint--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(defun agent-training-checkpoint--zero-like (tensor)
  "Return a detached zero tensor shaped like TENSOR."
  (photon-tensor
   (copy-sequence (photon-tensor-shape tensor))
   (make-vector (length (photon-tensor-data tensor)) 0.0)))

(defun agent-training-checkpoint--adam-state (checkpoint)
  "Return shape-correct zero Adam state for CHECKPOINT."
  (mapcar
   (lambda (parameter)
     (cons (agent-training-checkpoint--zero-like parameter)
           (agent-training-checkpoint--zero-like parameter)))
   (reverse
    (nl-llm-agent-training-checkpoint--model-parameters checkpoint))))

(defun agent-training-checkpoint--parameter-data (model)
  "Return flattened detached P5 parameter values for MODEL."
  (apply #'vconcat
         (mapcar
          (lambda (parameter)
            (copy-sequence (photon-tensor-data (pav-value parameter))))
          (nl-llm-agent--p5-params model))))

(let* ((directory (make-temp-file "nl-llm-training-checkpoint-" t))
       (file (expand-file-name "resume.nltrain" directory))
       (payload '(:examples [" a"] :lr 0.01 :epochs 4))
       (model
        (nl-llm-agent-improve-model
         2 2 nl-llm-agent-char-vocab 1 1))
       (model-checkpoint (nl-llm-agent-artifact-export-pav model 2))
       (optimizer-state
        (agent-training-checkpoint--adam-state model-checkpoint)))
  (unwind-protect
      (progn
        (nl-llm-agent-training-checkpoint-save
         file :job-id "job-1" :payload payload :scope "scope-v1"
         :parent-generation 0 :parent-score -4.5 :sequence 2
         :optimizer 'adam :completed-steps 2 :total-steps 4
         :optimizer-step 2 :model model-checkpoint
         :optimizer-state optimizer-state)
        (agent-training-checkpoint--ck
         "checkpoint is atomically persisted as private mode-0600 state"
         (and (file-regular-p file)
              (= (logand (file-modes file) #o777) #o600)))
        (let ((loaded
               (nl-llm-agent-training-checkpoint-load
                file :job-id "job-1" :payload payload :scope "scope-v1"
                :parent-generation 0 :parent-score -4.5 :sequence 2
                :optimizer 'adam :total-steps 4)))
          (agent-training-checkpoint--ck
           "exact transaction bindings load weights, Adam tensors, and progress"
           (and (= (plist-get loaded :completed-steps) 2)
                (= (plist-get loaded :optimizer-step) 2)
                (= (length (plist-get loaded :optimizer-state))
                   (length (nl-llm-agent--p5-params model)))))
          (let* ((candidate
                  (nl-llm-agent-improve-model
                   2 2 nl-llm-agent-char-vocab 1 1))
                 (first
                  (car (nl-llm-agent--p5-params candidate))))
            (aset (photon-tensor-data (pav-value first)) 0 99.0)
            (nl-llm-agent-training-checkpoint-restore-model
             candidate loaded)
            (agent-training-checkpoint--ck
             "validated checkpoint restores only the supplied isolated candidate"
             (equal (agent-training-checkpoint--parameter-data candidate)
                    (agent-training-checkpoint--parameter-data model)))))
        (agent-training-checkpoint--ck
         "payload, scope, parent generation, or training plan mismatch is rejected"
         (cl-every
          #'identity
          (list
           (agent-training-checkpoint--error-p
            (lambda ()
              (nl-llm-agent-training-checkpoint-load
               file :job-id "job-1" :payload '(:examples [" b"])
               :scope "scope-v1" :parent-generation 0 :parent-score -4.5
               :sequence 2 :optimizer 'adam :total-steps 4)))
           (agent-training-checkpoint--error-p
            (lambda ()
              (nl-llm-agent-training-checkpoint-load
               file :job-id "job-1" :payload payload :scope "other"
               :parent-generation 0 :parent-score -4.5 :sequence 2
               :optimizer 'adam :total-steps 4)))
           (agent-training-checkpoint--error-p
            (lambda ()
              (nl-llm-agent-training-checkpoint-load
               file :job-id "job-1" :payload payload :scope "scope-v1"
               :parent-generation 1 :parent-score -4.5 :sequence 2
               :optimizer 'adam :total-steps 4))))))
        (let* ((bad-model (copy-tree model-checkpoint))
               (data
                (photon-tensor-data
                 (car
                  (nl-llm-agent-training-checkpoint--model-parameters
                   bad-model)))))
          (aset data 0 (read "0.0e+NaN"))
          (agent-training-checkpoint--ck
           "non-finite weights are rejected before candidate mutation"
           (agent-training-checkpoint--error-p
            (lambda ()
              (nl-llm-agent-training-checkpoint-save
               file :job-id "job-1" :payload payload :scope "scope-v1"
               :parent-generation 0 :parent-score -4.5 :sequence 2
               :optimizer 'adam :completed-steps 2 :total-steps 4
               :optimizer-step 2 :model bad-model
               :optimizer-state optimizer-state))))))
        (let ((checkpoint
               (nl-llm-agent-training-checkpoint--read file))
              (cases
               (list
                (lambda (value)
                  (let ((copy (copy-tree value)))
                    (plist-put copy :parent-score (read "0.0e+NaN"))
                    copy))
                (lambda (value)
                  (let ((copy (copy-tree value)))
                    (plist-put copy :parent-score 1.0e300)
                    copy))
                (lambda (value)
                  (let ((copy (copy-tree value)))
                    (plist-put copy :optimizer-state nil)
                    copy))
                (lambda (value)
                  (let ((copy (copy-tree value)))
                    (plist-put copy :optimizer-step 3)
                    copy)))))
          (agent-training-checkpoint--ck
           "invalid parent scores, Adam state, and progress are rejected before candidate mutation"
           (let ((ok t))
             (dolist (transform cases ok)
               (let* ((candidate
                       (nl-llm-agent-improve-model
                        2 2 nl-llm-agent-char-vocab 1 1))
                      (first (car (nl-llm-agent--p5-params candidate))))
                 (aset (photon-tensor-data (pav-value first)) 0 77.0)
                 (let ((before (agent-training-checkpoint--parameter-data
                                candidate)))
                   (unless
                       (and
                        (agent-training-checkpoint--error-p
                         (lambda ()
                           (nl-llm-agent-training-checkpoint-restore-model
                            candidate (funcall transform checkpoint))))
                        (equal before
                               (agent-training-checkpoint--parameter-data
                                candidate)))
                     (setq ok nil))))))))
    (delete-directory directory t)))

(princ (format "NL-LLM-AGENT-TRAINING-CHECKPOINT %s (%d failures)\n"
               (if (= agent-training-checkpoint--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-training-checkpoint--fail))
(kill-emacs (if (= agent-training-checkpoint--fail 0) 0 1))

;;; agent-training-checkpoint-test.el ends here
