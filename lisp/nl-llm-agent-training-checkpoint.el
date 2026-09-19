;;; nl-llm-agent-training-checkpoint.el --- durable challenger state  -*- lexical-binding: t; -*-

;; A training checkpoint is private recovery state, not a published model.  It
;; binds candidate weights and optimizer tensors to one queue job, proposal,
;; parent generation, and trusted configuration scope.  Loading validates every
;; binding before mutating a fresh isolated candidate.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-completion-plan)

(defvar read-eval)

(defconst nl-llm-agent-training-checkpoint-format
  "nl-llm-agent-training-v1"
  "Format tag for private resumable challenger checkpoints.")

(defconst nl-llm-agent-training-checkpoint-completion-format
  "nl-llm-agent-training-completion-v1"
  "Format tag for completion-only resumable challenger checkpoints.")

(defconst nl-llm-agent-training-checkpoint-max-bytes (* 256 1024 1024)
  "Maximum private challenger checkpoint size accepted by the lightweight host.")

(defconst nl-llm-agent-training-checkpoint-max-sequence 4096
  "Maximum fixed sequence accepted in a lightweight training checkpoint.")

(defconst nl-llm-agent-training-checkpoint--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd)
  "Ordered dense P5 block parameters stored in a checkpoint.")

(defconst nl-llm-agent-training-checkpoint--keys
  '(:format :job-id :payload-digest :scope :parent-generation
    :parent-score :sequence :optimizer :completed-steps :total-steps
    :optimizer-step :model :optimizer-state)
  "Legacy v1 checkpoint keys, retained for byte-format compatibility.")

(defconst nl-llm-agent-training-checkpoint--completion-keys
  (append nl-llm-agent-training-checkpoint--keys '(:completion-plan))
  "Exact keys required by completion-only checkpoints.")

(defun nl-llm-agent-training-checkpoint--keys (value allowed where)
  "Validate plist VALUE keys against ALLOWED for WHERE."
  (unless (and (listp value) (= (% (length value) 2) 0))
    (error "%s must be a plist" where))
  (let ((tail value))
    (while tail
      (unless (memq (car tail) allowed)
        (error "%s contains unknown key %S" where (car tail)))
      (setq tail (cddr tail))))
  value)

(defun nl-llm-agent-training-checkpoint--exact-keys (value required where)
  "Validate exact REQUIRED keys in plist VALUE for WHERE.

Unlike the historical v1 validator, the completion format rejects duplicate
keys and missing fields so a legacy checkpoint cannot be upgraded implicitly."
  (unless (and (listp value) (= (% (length value) 2) 0))
    (error "%s must be a plist" where))
  (let ((tail value) (seen nil))
    (while tail
      (let ((key (pop tail)))
        (pop tail)
        (unless (memq key required)
          (error "%s contains unknown key %S" where key))
        (when (memq key seen)
          (error "%s contains duplicate key %S" where key))
        (push key seen)))
    (unless (= (length seen) (length required))
      (error "%s is missing a required key" where)))
  value)

(defun nl-llm-agent-training-checkpoint--identifier (value where)
  "Return bounded identifier VALUE for WHERE."
  (unless (and (stringp value)
               (<= 1 (length value)) (<= (length value) 128)
               (string-match-p "\\`[A-Za-z0-9_.-]+\\'" value))
    (error "%s has invalid identifier %S" where value))
  value)

(defun nl-llm-agent-training-checkpoint--scope (value)
  "Return bounded trusted configuration scope VALUE."
  (unless (and (stringp value) (<= 1 (length value))
               (<= (length value) 256))
    (error "training checkpoint requires a bounded configuration scope"))
  value)

(defun nl-llm-agent-training-checkpoint-payload-digest (payload)
  "Return a stable SHA-256 binding for detached proposal PAYLOAD."
  (let ((print-length nil)
        (print-level nil)
        (print-circle t))
    (secure-hash 'sha256 (prin1-to-string payload))))

(defun nl-llm-agent-training-checkpoint--completion-payload-digest (payload)
  "Return the completion payload digest with canonical float serialization.

The legacy digest intentionally retains its ambient `float-output-format'
behavior.  Completion checkpoints need a fixed representation because their
payload is loaded outside the dynamic binding used by the writer."
  (let ((float-output-format "%.17g")
        (print-escape-nonascii t))
    (nl-llm-agent-training-checkpoint-payload-digest payload)))

(defun nl-llm-agent-training-checkpoint--model-parameters (checkpoint)
  "Return ordered tensor parameters from P5 CHECKPOINT."
  (append
   (list (plist-get checkpoint :wte))
   (cl-loop
    for block in (plist-get checkpoint :blocks)
    append
    (mapcar
     (lambda (key) (plist-get block key))
     nl-llm-agent-training-checkpoint--block-keys))
   (list (plist-get checkpoint :lnfg)
         (plist-get checkpoint :wh)
         (plist-get checkpoint :bh))))

(defun nl-llm-agent-training-checkpoint--finite-tensor-p (tensor shape)
  "Return non-nil when TENSOR has SHAPE and bounded finite numeric data."
  (condition-case nil
      (and (equal (photon-tensor-shape tensor) shape)
           (vectorp (photon-tensor-data tensor))
           (cl-every
            (lambda (value)
              (and (numberp value) (= value value)
                   (< (abs (float value)) 1.0e300)))
            (append (photon-tensor-data tensor) nil)))
    (error nil)))

(defun nl-llm-agent-training-checkpoint--optimizer-state
    (optimizer state model-checkpoint)
  "Validate OPTIMIZER STATE against MODEL-CHECKPOINT."
  (if (eq optimizer 'sgd)
      (when state
        (error "SGD training checkpoint cannot contain optimizer tensors"))
    (let ((parameters
           (reverse
            (nl-llm-agent-training-checkpoint--model-parameters
             model-checkpoint))))
      (unless (and (listp state) (= (length state) (length parameters)))
        (error "Adam training checkpoint has the wrong parameter count"))
      (cl-mapc
       (lambda (pair parameter)
         (let ((shape (photon-tensor-shape parameter)))
           (unless (and (consp pair)
                        (nl-llm-agent-training-checkpoint--finite-tensor-p
                         (car pair) shape)
                        (nl-llm-agent-training-checkpoint--finite-tensor-p
                         (cdr pair) shape))
             (error "Adam training checkpoint has incompatible tensors"))))
       state parameters)))
  state)

(defun nl-llm-agent-training-checkpoint--completion-optimizer-state
    (optimizer state model-checkpoint)
  "Validate completion OPTIMIZER STATE with exact tensor lengths.

The legacy validator intentionally keeps its historical shape-only behavior;
the completion format has a stricter wire contract."
  (if (eq optimizer 'sgd)
      (when state
        (error "SGD completion checkpoint cannot contain optimizer tensors"))
    (let ((parameters
           (reverse
            (nl-llm-agent-training-checkpoint--model-parameters
             model-checkpoint))))
      (unless (and (listp state) (= (length state) (length parameters)))
        (error "Adam completion checkpoint has the wrong parameter count"))
      (cl-mapc
       (lambda (pair parameter)
         (let ((shape (photon-tensor-shape parameter)))
           (unless (and (consp pair)
                        (nl-llm-agent-artifact--tensor-p (car pair))
                        (nl-llm-agent-artifact--tensor-p (cdr pair))
                        (nl-llm-agent-training-checkpoint--finite-tensor-p
                         (car pair) shape)
                        (nl-llm-agent-training-checkpoint--finite-tensor-p
                         (cdr pair) shape))
             (error "Adam completion checkpoint has incompatible tensors"))))
       state parameters)))
  state)

(defun nl-llm-agent-training-checkpoint--validate (value)
  "Validate training checkpoint VALUE and return it."
  (let* ((format (plist-get value :format))
         (completionp
          (equal format nl-llm-agent-training-checkpoint-completion-format)))
    (if completionp
        (nl-llm-agent-training-checkpoint--exact-keys
         value nl-llm-agent-training-checkpoint--completion-keys
         "completion training checkpoint")
      ;; Preserve the permissive duplicate-key behavior of legacy v1 files.
      (nl-llm-agent-training-checkpoint--keys
       value nl-llm-agent-training-checkpoint--keys
       "training checkpoint"))
    (unless (or (equal format nl-llm-agent-training-checkpoint-format)
                completionp)
      (error "unsupported training checkpoint format %S" format))
  (nl-llm-agent-training-checkpoint--identifier
   (plist-get value :job-id) "training checkpoint job")
  (let ((digest (plist-get value :payload-digest))
        (generation (plist-get value :parent-generation))
        (score (plist-get value :parent-score))
        (sequence (plist-get value :sequence))
        (optimizer (plist-get value :optimizer))
        (completed (plist-get value :completed-steps))
        (total (plist-get value :total-steps))
        (optimizer-step (plist-get value :optimizer-step))
        (model (plist-get value :model))
        (completion-plan (plist-get value :completion-plan)))
    (nl-llm-agent-training-checkpoint--scope (plist-get value :scope))
    (unless (and (stringp digest)
                 (string-match-p "\\`[a-f0-9]\\{64\\}\\'" digest))
      (error "training checkpoint has invalid payload digest"))
    (unless (and (integerp generation) (>= generation 0))
      (error "training checkpoint has invalid parent generation"))
    (unless (and (numberp score) (= score score)
                 (< (abs (float score)) 1.0e300))
      (error "training checkpoint has invalid parent score"))
    (unless (and (integerp sequence) (<= 2 sequence)
                 (<= sequence nl-llm-agent-training-checkpoint-max-sequence))
      (error "training checkpoint has invalid sequence"))
    (unless (memq optimizer '(sgd adam))
      (error "training checkpoint has invalid optimizer"))
    (unless (and (integerp total) (> total 0)
                 (integerp completed) (<= 0 completed) (<= completed total)
                 (integerp optimizer-step) (= optimizer-step completed))
      (error "training checkpoint has invalid progress counters"))
    (nl-llm-agent-artifact--checkpoint-model model "training checkpoint")
    (unless (plist-get model :wh)
      (error "training checkpoint requires an independent P5 output head"))
    (dolist (parameter
             (nl-llm-agent-training-checkpoint--model-parameters model))
      (unless (nl-llm-agent-training-checkpoint--finite-tensor-p
               parameter (photon-tensor-shape parameter))
        (error "training checkpoint contains non-finite model weights")))
    (if completionp
        (let* ((plan
                (nl-llm-agent-completion-plan-validate completion-plan))
               (config (plist-get model :config))
               (model-tokenizer
                (nl-llm-agent-tokenizer-id (plist-get config :tokenizer)))
               (model-vocab (plist-get config :vocab))
               (expected-total
                (* (plist-get plan :epochs)
                   (length (plist-get plan :trajectories))))
               (model-step (plist-get model :step)))
          ;; The plan validator returns a detached canonical plist.  Require
          ;; that exact representation on disk rather than silently upgrading
          ;; a legacy or caller-mutated plan.
          (unless (equal plan completion-plan)
            (error "completion checkpoint plan is not canonical"))
          (unless (= sequence (plist-get plan :sequence))
            (error "completion checkpoint sequence differs from plan"))
          (unless (eq optimizer (plist-get plan :optimizer))
            (error "completion checkpoint optimizer differs from plan"))
          (unless (and (equal model-tokenizer (plist-get plan :tokenizer))
                       (= model-vocab (plist-get plan :vocab)))
            (error "completion checkpoint model tokenizer or vocab differs from plan"))
          (unless (= total expected-total)
            (error "completion checkpoint total steps differ from plan"))
          (unless (and (integerp model-step)
                       (= model-step completed)
                       (= model-step optimizer-step))
            (error "completion checkpoint model step differs from progress"))
          (nl-llm-agent-training-checkpoint--completion-optimizer-state
           optimizer (plist-get value :optimizer-state) model))
      (nl-llm-agent-training-checkpoint--optimizer-state
       optimizer (plist-get value :optimizer-state) model)))
  value))

;;;###autoload
(cl-defun nl-llm-agent-training-checkpoint-save
    (file &key job-id payload scope parent-generation parent-score sequence
          optimizer completed-steps total-steps optimizer-step model
          optimizer-state completion-plan)
  "Atomically save one private resumable challenger checkpoint to FILE."
  (let* ((canonical-plan
          (when completion-plan
            (nl-llm-agent-completion-plan-validate completion-plan)))
         (path (expand-file-name file))
         (directory (file-name-directory path))
         (value
          (append
           (list
            :format (if canonical-plan
                        nl-llm-agent-training-checkpoint-completion-format
                      nl-llm-agent-training-checkpoint-format)
           :job-id job-id
           :payload-digest
           (if canonical-plan
               (nl-llm-agent-training-checkpoint--completion-payload-digest
                payload)
             (nl-llm-agent-training-checkpoint-payload-digest payload))
            :scope scope
            :parent-generation parent-generation
            :parent-score parent-score
            :sequence sequence
            :optimizer optimizer
            :completed-steps completed-steps
            :total-steps total-steps
            :optimizer-step optimizer-step
            :model model
            :optimizer-state optimizer-state)
           (when canonical-plan
             (list :completion-plan canonical-plan))))
         (temporary nil)
         text)
    ;; Do not create the destination directory until every format-specific
    ;; binding, including the completion plan, has been validated.
    (nl-llm-agent-training-checkpoint--validate value)
    (setq text
          (let ((print-length nil)
                (print-level nil)
                (print-circle nil))
            ;; Completion plans include a floating learning rate and their
            ;; digest uses a fixed %.17g representation.  Do not bind
            ;; `float-output-format' at all for legacy v1 serialization: nil
            ;; would incorrectly override its historical ambient behavior.
            (if canonical-plan
                (let ((float-output-format "%.17g"))
                  (prin1-to-string value))
              (prin1-to-string value))))
    (when (> (string-bytes text)
             nl-llm-agent-training-checkpoint-max-bytes)
      (error "training checkpoint exceeds %d bytes"
             nl-llm-agent-training-checkpoint-max-bytes))
    (make-directory directory t)
    (setq temporary
          (make-temp-file
           (expand-file-name ".training-checkpoint-" directory)))
    (unwind-protect
        (progn
          (let ((coding-system-for-write 'utf-8))
            (write-region text nil temporary nil 'silent))
          (set-file-modes temporary #o600)
          (rename-file temporary path t)
          (setq temporary nil)
          path)
      (when (and temporary (file-exists-p temporary))
        (delete-file temporary)))))

(defun nl-llm-agent-training-checkpoint--read (path)
  "Read bounded data from training checkpoint PATH."
  (unless (file-regular-p path)
    (error "training checkpoint does not exist: %s" path))
  (when (> (file-attribute-size (file-attributes path))
           nl-llm-agent-training-checkpoint-max-bytes)
    (error "training checkpoint exceeds %d bytes"
           nl-llm-agent-training-checkpoint-max-bytes))
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8))
      (insert-file-contents path))
    (let* ((text (buffer-string))
           (read-eval nil)
           (parsed (read-from-string text))
           (value (car parsed))
           (trailing (substring text (cdr parsed))))
      (unless (string-match-p "\\`[ \t\r\n]*\\'" trailing)
        (error "training checkpoint contains trailing data"))
      (nl-llm-agent-training-checkpoint--validate value))))

;;;###autoload
(cl-defun nl-llm-agent-training-checkpoint-load
    (file &key job-id payload scope parent-generation parent-score sequence
          optimizer total-steps completion-plan)
  "Load FILE only when every expected job and training binding matches."
  (let ((value
         (nl-llm-agent-training-checkpoint--read
          (expand-file-name file))))
    (let ((stored-plan (plist-get value :completion-plan))
          (completion-format-p
           (equal (plist-get value :format)
                  nl-llm-agent-training-checkpoint-completion-format)))
      (cond
       (completion-format-p
        (unless completion-plan
          (error "completion checkpoint requires an expected completion plan"))
        (let ((expected
               (nl-llm-agent-completion-plan-validate completion-plan)))
          (unless (equal stored-plan expected)
            (error "completion checkpoint plan does not match"))))
       (completion-plan
        (error "legacy checkpoint cannot load with a completion plan")))
    (unless (and
             (equal (plist-get value :job-id) job-id)
             (equal (plist-get value :payload-digest)
                    (if completion-format-p
                        (nl-llm-agent-training-checkpoint--completion-payload-digest
                         payload)
                      (nl-llm-agent-training-checkpoint-payload-digest
                       payload)))
             (equal (plist-get value :scope) scope)
             (= (plist-get value :parent-generation) parent-generation)
             (= (plist-get value :parent-score) parent-score)
             (= (plist-get value :sequence) sequence)
             (eq (plist-get value :optimizer) optimizer)
             (= (plist-get value :total-steps) total-steps))
      (error "training checkpoint does not match this proposal transaction"))
    value)))

;;;###autoload
(defun nl-llm-agent-training-checkpoint-restore-model (candidate checkpoint)
  "Replace isolated P5 CANDIDATE weights from validated CHECKPOINT."
  (setq checkpoint
        (nl-llm-agent-training-checkpoint--validate checkpoint))
  (let* ((source-model (plist-get checkpoint :model))
         (source-config (plist-get source-model :config))
         (source-tokenizer
          (nl-llm-agent-tokenizer-id
           (plist-get source-config :tokenizer)))
         (candidate-tokenizer
          (nl-llm-agent-tokenizer-id (plist-get candidate :tokenizer))))
    (unless (and
             (= (plist-get source-config :vocab)
                (nl-llm-agent-tokenizer-vocab source-tokenizer))
             (= (plist-get candidate :vocab)
                (nl-llm-agent-tokenizer-vocab candidate-tokenizer))
             (equal source-tokenizer candidate-tokenizer))
      (error "training checkpoint tokenizer differs from candidate")))
  (let* ((source
          (nl-llm-agent-training-checkpoint--model-parameters
           (plist-get checkpoint :model)))
         (destination (nl-llm-agent--p5-params candidate)))
    (unless (= (length source) (length destination))
      (error "training checkpoint architecture differs from candidate"))
    (cl-mapc
     (lambda (tensor parameter)
       (unless (equal (photon-tensor-shape tensor)
                      (photon-tensor-shape (pav-value parameter)))
         (error "training checkpoint tensor shape differs from candidate")))
     source destination)
    (cl-mapc
     (lambda (tensor parameter)
       (let ((source-data (photon-tensor-data tensor))
             (destination-data
              (photon-tensor-data (pav-value parameter))))
         (dotimes (index (length source-data))
           (aset destination-data index (aref source-data index)))))
     source destination))
  candidate)

(provide 'nl-llm-agent-training-checkpoint)
;;; nl-llm-agent-training-checkpoint.el ends here
