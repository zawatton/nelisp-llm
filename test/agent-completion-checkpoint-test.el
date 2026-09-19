;;; agent-completion-checkpoint-test.el --- completion checkpoint contract -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)

(defconst nl-llm-agent-completion-checkpoint-test--here
  (file-name-directory (or load-file-name buffer-file-name)))
(add-to-list 'load-path
             (expand-file-name "../lisp"
                               nl-llm-agent-completion-checkpoint-test--here))
(add-to-list 'load-path
             (expand-file-name "../../nelisp-photon/lisp"
                               nl-llm-agent-completion-checkpoint-test--here))

(require 'nl-llm-agent-training-checkpoint)
(require 'nl-llm-agent-completion-plan)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)

(defun nl-llm-agent-completion-checkpoint-test--plan
    (&optional optimizer learning-rate loss-masks shuffle-seed)
  "Return one small canonical completion plan."
  (nl-llm-agent-completion-plan-make
   (vector (vector ?A ?B ?C)) [1]
   :tokenizer "ascii-char-v1" :sequence 4
   :learning-rate (or learning-rate 0.01)
   :epochs 2 :optimizer (or optimizer 'sgd) :transfer-mode 'compact
   :loss-masks loss-masks :shuffle-seed shuffle-seed))

(defun nl-llm-agent-completion-checkpoint-test--model (&optional tokenizer step)
  "Return a detached small CPU PAV model checkpoint."
  (nl-llm-agent-artifact-export-pav
   (nl-llm-agent-improve-model
   2 2 nl-llm-agent-char-vocab 1 1 (or tokenizer "ascii-char-v1"))
   step))

(defun nl-llm-agent-completion-checkpoint-test--zero-like (tensor)
  "Return a zero tensor with TENSOR's exact shape."
  (photon-tensor
   (copy-sequence (photon-tensor-shape tensor))
   (make-vector (length (photon-tensor-data tensor)) 0.0)))

(defun nl-llm-agent-completion-checkpoint-test--adam-state (model)
  "Return a shape-correct detached Adam state for MODEL."
  (mapcar
   (lambda (parameter)
     (cons
      (nl-llm-agent-completion-checkpoint-test--zero-like parameter)
      (nl-llm-agent-completion-checkpoint-test--zero-like parameter)))
   (reverse
    (nl-llm-agent-training-checkpoint--model-parameters model))))

(defun nl-llm-agent-completion-checkpoint-test--save-args
    (file model plan &optional total sequence optimizer payload)
  "Save one completion checkpoint with overridable binding fields."
  (nl-llm-agent-training-checkpoint-save
   file :job-id "completion-job"
   :payload (or payload '(:examples ["abc"] :lr 0.01 :epochs 2))
   :scope "completion-scope" :parent-generation 0 :parent-score -1.0
   :sequence (or sequence (plist-get plan :sequence))
   :optimizer (or optimizer (plist-get plan :optimizer))
   :completed-steps 1 :total-steps (or total 2)
   :optimizer-step 1 :model model :optimizer-state nil
   :completion-plan plan))

(ert-deftest nl-llm-agent-completion-checkpoint-save-loads-canonical-v1 ()
  (let* ((directory (make-temp-file "nl-completion-checkpoint-" t))
         (file (expand-file-name "nested/completion.sexp" directory))
         (plan
          (nl-llm-agent-completion-checkpoint-test--plan
           'sgd 0.01 (vector [0 0 1]) 12345))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1)))
    (unwind-protect
        (progn
          (nl-llm-agent-completion-checkpoint-test--save-args
           file model plan)
          (should (file-regular-p file))
          (should (= (logand (file-modes file) #o777) #o600))
          (let ((loaded
                 (nl-llm-agent-training-checkpoint-load
                  file :job-id "completion-job"
                  :payload '(:examples ["abc"] :lr 0.01 :epochs 2)
                  :scope "completion-scope" :parent-generation 0
                  :parent-score -1.0 :sequence 4 :optimizer 'sgd
                  :total-steps 2 :completion-plan plan)))
            (should
             (equal (plist-get loaded :format)
                    nl-llm-agent-training-checkpoint-completion-format))
            (should (equal (plist-get loaded :completion-plan) plan))
            (should (equal (plist-get (plist-get loaded :completion-plan)
                                      :loss-masks)
                           (vector [0 0 1])))
            (should (= (plist-get (plist-get loaded :completion-plan)
                                  :shuffle-seed)
                       12345))
            (should (= (plist-get loaded :completed-steps) 1))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-fixes-completion-float-serialization ()
  (let* ((directory (make-temp-file "nl-completion-float-" t))
         (file (expand-file-name "completion.sexp" directory))
         (plan
          (nl-llm-agent-completion-checkpoint-test--plan 'sgd 0.01234567))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1)))
    (unwind-protect
        (progn
          (let ((float-output-format "%.2f"))
            (nl-llm-agent-completion-checkpoint-test--save-args
             file model plan nil nil nil
             '(:examples ["abc"] :lr 0.01234567 :epochs 2)))
          (let ((loaded
                 (nl-llm-agent-training-checkpoint-load
                  file :job-id "completion-job"
                  :payload '(:examples ["abc"] :lr 0.01234567 :epochs 2)
                  :scope "completion-scope" :parent-generation 0
                  :parent-score -1.0 :sequence 4 :optimizer 'sgd
                  :total-steps 2 :completion-plan plan)))
            (should (equal (plist-get loaded :completion-plan) plan))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-legacy-v1-remains-byte-contract ()
  (let* ((directory (make-temp-file "nl-legacy-checkpoint-" t))
         (file (expand-file-name "legacy.sexp" directory))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1))
         (payload '(:examples ["abc"] :lr 0.01 :epochs 1)))
    (unwind-protect
        (progn
          (nl-llm-agent-training-checkpoint-save
           file :job-id "legacy-job" :payload payload :scope "legacy-scope"
           :parent-generation 0 :parent-score -1.0 :sequence 4
           :optimizer 'sgd :completed-steps 1 :total-steps 1
           :optimizer-step 1 :model model :optimizer-state nil)
          (let ((raw (nl-llm-agent-training-checkpoint--read file)))
            (should (equal (plist-get raw :format)
                           nl-llm-agent-training-checkpoint-format))
            (should-not (plist-member raw :completion-plan)))
          (should
           (equal
            (plist-get
             (nl-llm-agent-training-checkpoint-load
              file :job-id "legacy-job" :payload payload
              :scope "legacy-scope" :parent-generation 0 :parent-score -1.0
              :sequence 4 :optimizer 'sgd :total-steps 1)
             :format)
            nl-llm-agent-training-checkpoint-format)))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-legacy-keeps-ambient-float-format ()
  (let* ((directory (make-temp-file "nl-legacy-float-" t))
         (file (expand-file-name "legacy.sexp" directory))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1))
         (payload '(:examples ["abc"] :lr 0.01234567 :epochs 1)))
    (unwind-protect
        (let ((float-output-format "%.2f"))
          (nl-llm-agent-training-checkpoint-save
           file :job-id "legacy-job" :payload payload :scope "legacy-scope"
           :parent-generation 0 :parent-score -1.0 :sequence 4
           :optimizer 'sgd :completed-steps 1 :total-steps 1
           :optimizer-step 1 :model model :optimizer-state nil)
          (let* ((raw (nl-llm-agent-training-checkpoint--read file))
                 (expected
                  (let ((print-length nil)
                        (print-level nil)
                        (print-circle nil))
                    (prin1-to-string raw)))
                 (actual
                  (with-temp-buffer
                    (insert-file-contents file)
                    (buffer-string))))
            (should (equal actual expected))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-rejects-cross-format-and-plan-change ()
  (let* ((directory (make-temp-file "nl-completion-cross-" t))
         (completion-file (expand-file-name "completion.sexp" directory))
         (legacy-file (expand-file-name "legacy.sexp" directory))
         (plan (nl-llm-agent-completion-checkpoint-test--plan))
         (changed
          (nl-llm-agent-completion-plan-make
           (vector (vector ?A ?B ?D)) [1]
           :tokenizer "ascii-char-v1" :sequence 4 :learning-rate 0.01
           :epochs 2 :optimizer 'sgd :transfer-mode 'compact))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1))
         (payload '(:examples ["abc"] :lr 0.01 :epochs 2)))
    (unwind-protect
        (progn
          (nl-llm-agent-completion-checkpoint-test--save-args
           completion-file model plan)
          (nl-llm-agent-training-checkpoint-save
           legacy-file :job-id "legacy-job" :payload payload
           :scope "legacy-scope" :parent-generation 0 :parent-score -1.0
           :sequence 4 :optimizer 'sgd :completed-steps 1 :total-steps 1
           :optimizer-step 1 :model model :optimizer-state nil)
          (should-error
           (nl-llm-agent-training-checkpoint-load
            completion-file :job-id "completion-job" :payload payload
            :scope "completion-scope" :parent-generation 0 :parent-score -1.0
            :sequence 4 :optimizer 'sgd :total-steps 2))
          (should-error
           (nl-llm-agent-training-checkpoint-load
            legacy-file :job-id "legacy-job" :payload payload
            :scope "legacy-scope" :parent-generation 0 :parent-score -1.0
            :sequence 4 :optimizer 'sgd :total-steps 1
            :completion-plan plan))
          ;; Recomputing the changed plan's own digest must not make it match
          ;; the checkpoint's detached canonical plan.
          (should-error
           (nl-llm-agent-training-checkpoint-load
            completion-file :job-id "completion-job" :payload payload
            :scope "completion-scope" :parent-generation 0 :parent-score -1.0
            :sequence 4 :optimizer 'sgd :total-steps 2
            :completion-plan changed)))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-rejects-binding-mismatches ()
  (let* ((directory (make-temp-file "nl-completion-bindings-" t))
         (plan (nl-llm-agent-completion-checkpoint-test--plan))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1))
         (payload '(:examples ["abc"] :lr 0.01 :epochs 2)))
    (unwind-protect
        (progn
          (dolist (case
                  '((:sequence 5)
                    (:optimizer adam)
                    (:total 1)))
            (should-error
             (nl-llm-agent-completion-checkpoint-test--save-args
              (expand-file-name
               (format "bad-%s.sexp" (car case)) directory)
              model plan (plist-get case :total)
              (plist-get case :sequence) (plist-get case :optimizer))))
          (should-error
           (nl-llm-agent-completion-checkpoint-test--save-args
            (expand-file-name "bad-tokenizer.sexp" directory)
            (nl-llm-agent-completion-checkpoint-test--model "utf8-byte-v1" 1)
            plan))
          (should-error
           (nl-llm-agent-training-checkpoint-load
            (let ((file (expand-file-name "valid.sexp" directory)))
              (nl-llm-agent-completion-checkpoint-test--save-args
               file model plan)
              file)
            :job-id "completion-job" :payload payload
            :scope "completion-scope" :parent-generation 0 :parent-score -1.0
            :sequence 5 :optimizer 'sgd :total-steps 2
            :completion-plan plan)))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-strict-adam-progress-validation ()
  (let* ((directory (make-temp-file "nl-completion-adam-" t))
         (file (expand-file-name "completion.sexp" directory))
         (plan (nl-llm-agent-completion-checkpoint-test--plan 'adam))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1))
         (state (nl-llm-agent-completion-checkpoint-test--adam-state model)))
    (unwind-protect
        (progn
          (nl-llm-agent-training-checkpoint-save
           file :job-id "completion-job" :payload
           '(:examples ["abc"] :lr 0.01 :epochs 2)
           :scope "completion-scope" :parent-generation 0 :parent-score -1.0
           :sequence 4 :optimizer 'adam :completed-steps 1 :total-steps 2
           :optimizer-step 1 :model model :optimizer-state state
           :completion-plan plan)
          (let ((raw (nl-llm-agent-training-checkpoint--read file)))
            (let ((short (copy-tree raw t)))
              (plist-put short :optimizer-state (butlast state))
              (should-error
               (nl-llm-agent-training-checkpoint--validate short)))
            (let* ((short-moment (copy-tree raw t))
                   (short-state (plist-get short-moment :optimizer-state))
                   (pair (car short-state))
                   (moment (car pair))
                   (short-moment-tensor
                    (photon-tensor
                     (copy-sequence (photon-tensor-shape moment))
                     (vconcat
                      (butlast (append (photon-tensor-data moment) nil))))))
              (setcar pair short-moment-tensor)
              (should-error
               (nl-llm-agent-training-checkpoint--validate short-moment)))
            (let ((wrong-step (copy-tree raw t))
                  (wrong-model (copy-tree (plist-get raw :model) t)))
              (plist-put wrong-model :step 0)
              (plist-put wrong-step :model wrong-model)
              (should-error
               (nl-llm-agent-training-checkpoint--validate wrong-step)))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-completion-checkpoint-invalid-plan-does-not-create-directory ()
  (let* ((parent (make-temp-file "nl-completion-invalid-" t))
         (missing (expand-file-name "not-created/checkpoint.sexp" parent))
         (plan (nl-llm-agent-completion-checkpoint-test--plan))
         (bad (copy-tree plan t))
         (model (nl-llm-agent-completion-checkpoint-test--model nil 1)))
    (unwind-protect
        (progn
          (plist-put bad :digest (make-string 64 ?0))
          (should-error
           (nl-llm-agent-completion-checkpoint-test--save-args
            missing model bad))
          (should-not (file-exists-p (file-name-directory missing))))
      (delete-directory parent t))))

(provide 'agent-completion-checkpoint-test)

(ert-run-tests-batch-and-exit)

;;; agent-completion-checkpoint-test.el ends here
