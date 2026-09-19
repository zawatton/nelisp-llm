;;; nl-llm-agent-evolve.el --- guarded P5 evolution pipeline  -*- lexical-binding: t; -*-

;; Compose the generic evaluated queue, a P5-compatible trainable model, and
;; immutable service artifacts.  A model may supply trajectory text and bounded
;; hyperparameters; only this trusted handler chooses the training function, and
;; the caller-supplied evaluator still decides whether publication is allowed.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-training-checkpoint)
(require 'nl-llm-evolve-queue)

(declare-function nl-llm-gpu-enable "nl-llm-gpu" ())
(declare-function nl-llm-agent-ondevice-from-model
                  "nl-llm-agent-ondevice"
                  (model seq lr &rest keys))
(declare-function nl-llm-agent-ondevice-train
                  "nl-llm-agent-ondevice"
                  (context trajectories epochs &rest keys))
(declare-function nl-llm-agent-ondevice-sync
                  "nl-llm-agent-ondevice" (context))
(declare-function nl-llm-agent-ondevice-snapshot
                  "nl-llm-agent-ondevice" (context))
(declare-function nl-llm-agent-ondevice-restore-training-state
                  "nl-llm-agent-ondevice"
                  (context step optimizer-state))
(declare-function nl-llm-agent-ondevice-free
                  "nl-llm-agent-ondevice" (context))

(defconst nl-llm-agent-evolve-max-examples 128
  "Maximum trajectories accepted by one fine-tune proposal.")

(defconst nl-llm-agent-evolve-max-example-length 4096
  "Maximum characters accepted in one fine-tune trajectory.")

(defconst nl-llm-agent-evolve-max-example-tokens 4096
  "Maximum encoded tokens accepted in one fine-tune trajectory.")

(defconst nl-llm-agent-evolve-max-total-characters 65536
  "Maximum total trajectory characters accepted by one proposal.")

(defconst nl-llm-agent-evolve-max-training-sequence 4096
  "Maximum fixed GPU sequence accepted by an evolution queue.")

(defun nl-llm-agent-evolve--keys (value allowed where)
  "Validate plist VALUE keys against ALLOWED for WHERE."
  (unless (and (listp value) (= (% (length value) 2) 0))
    (error "%s must be a plist" where))
  (let ((tail value))
    (while tail
      (unless (memq (car tail) allowed)
        (error "%s contains unknown key %S" where (car tail)))
      (setq tail (cddr tail))))
  value)

(defun nl-llm-agent-evolve--model-tokenizer (model where)
  "Return MODEL's canonical tokenizer id after validation for WHERE."
  (let ((tokenizer
         (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
        (vocab (plist-get model :vocab)))
    (unless (and (integerp vocab)
                 (= vocab (nl-llm-agent-tokenizer-vocab tokenizer)))
      (error "%s vocab does not match tokenizer %s" where tokenizer))
    tokenizer))

(defun nl-llm-agent-evolve--validate-finetune (payload &optional tokenizer)
  "Validate a data-only trajectory fine-tune PAYLOAD."
  (setq tokenizer (nl-llm-agent-tokenizer-id tokenizer))
  (nl-llm-agent-evolve--keys
   payload '(:examples :lr :epochs) "trajectory fine-tune proposal")
  (let ((examples (plist-get payload :examples))
        (lr (or (plist-get payload :lr) 0.05))
        (epochs (or (plist-get payload :epochs) 1))
        (total 0))
    (unless (and (vectorp examples)
                 (<= 1 (length examples))
                 (<= (length examples) nl-llm-agent-evolve-max-examples))
      (error "trajectory fine-tune requires 1..%d examples"
             nl-llm-agent-evolve-max-examples))
    (dolist (example (append examples nil))
      (unless (and (stringp example)
                   (<= 2 (length example))
                   (<= (length example)
                       nl-llm-agent-evolve-max-example-length))
        (error "trajectory fine-tune example length is invalid"))
      (let ((tokens (nl-llm-agent-tokenizer-encode example tokenizer)))
        (when (> (length tokens) nl-llm-agent-evolve-max-example-tokens)
          (error "trajectory fine-tune example exceeds %d encoded tokens"
                 nl-llm-agent-evolve-max-example-tokens)))
      (setq total (+ total (length example))))
    (when (> total nl-llm-agent-evolve-max-total-characters)
      (error "trajectory fine-tune proposal exceeds %d total characters"
             nl-llm-agent-evolve-max-total-characters))
    (unless (and (numberp lr) (= lr lr) (> lr 0.0) (<= lr 1.0))
      (error "trajectory fine-tune :lr must be in (0, 1]"))
    (unless (and (integerp epochs) (<= 1 epochs) (<= epochs 32))
      (error "trajectory fine-tune :epochs must be in [1, 32]"))))

(defun nl-llm-agent-evolve--tokens (examples &optional tokenizer)
  "Return EXAMPLES as detached token-id lists for TOKENIZER."
  (setq tokenizer (nl-llm-agent-tokenizer-id tokenizer))
  (mapcar
   (lambda (example)
     (nl-llm-agent-tokenizer-encode example tokenizer))
   (append examples nil)))

(defun nl-llm-agent-evolve--training-backend (value)
  "Return normalized trusted training backend VALUE."
  (let ((backend (if (stringp value) (intern value) value)))
    (unless (memq backend '(cpu gpu))
      (error "unsupported P5 evolution training backend %S" value))
    backend))

(defun nl-llm-agent-evolve--optimizer (value backend)
  "Return normalized optimizer VALUE supported by BACKEND."
  (let ((optimizer (if (stringp value) (intern value) value)))
    (unless (memq optimizer '(sgd adam))
      (error "unsupported P5 evolution optimizer %S" value))
    (when (and (eq backend 'cpu) (not (eq optimizer 'sgd)))
      (error "CPU P5 evolution supports only SGD"))
    optimizer))

(defun nl-llm-agent-evolve--validate-gpu-finetune
    (payload sequence &optional tokenizer)
  "Validate GPU fine-tune PAYLOAD against fixed SEQUENCE capacity."
  (setq tokenizer (nl-llm-agent-tokenizer-id tokenizer))
  (nl-llm-agent-evolve--validate-finetune payload tokenizer)
  (dolist (example (append (plist-get payload :examples) nil))
    (let ((tokens (nl-llm-agent-tokenizer-encode example tokenizer)))
      (when (> (length tokens) sequence)
        (error "GPU trajectory token length %d exceeds fixed sequence %d"
               (length tokens) sequence)))))

(defun nl-llm-agent-evolve--training-checkpoint-path
    (directory scope job-id)
  "Return private checkpoint path beneath DIRECTORY for one queue JOB-ID."
  (let ((scope-id (substring (secure-hash 'sha256 scope) 0 16)))
    (expand-file-name
     (format "%s-%s.nltrain" scope-id job-id)
     directory)))

;;;###autoload
(cl-defun nl-llm-agent-evolve-p5-gpu-finetune
    (candidate payload evolution sequence optimizer
               &key checkpoint-directory checkpoint-every checkpoint-scope)
  "Train isolated P5 CANDIDATE as a resident GPU challenger.

PAYLOAD has already crossed the bounded data validator.  Parameters and the
optimizer stay resident for all examples and epochs.  Without recovery, one
final readback updates the isolated CPU candidate.  When CHECKPOINT-EVERY is
positive, bounded interval readbacks update only that isolated candidate and
atomically persist weights, optimizer, and progress under CHECKPOINT-DIRECTORY.
The fixed evaluator and publisher run only after training returns.  An
explicitly resumed job must match CHECKPOINT-SCOPE and the original transaction
bindings."
  (require 'nl-llm-gpu)
  (require 'nl-llm-agent-ondevice)
  (let ((tokenizer
         (nl-llm-agent-evolve--model-tokenizer
          candidate "GPU challenger")))
    (nl-llm-agent-evolve--validate-gpu-finetune
     payload sequence tokenizer))
  (unless (nl-llm-gpu-enable)
    (error "GPU training backend is configured but no Vulkan device is available"))
  (let* ((execution (nl-llm-evolve-queue-current-execution-context))
         (job-id (plist-get execution :job-id))
         (resuming (plist-get execution :resuming))
         (examples
          (nl-llm-agent-evolve--tokens
           (plist-get payload :examples)
           (nl-llm-agent-evolve--model-tokenizer
            candidate "GPU challenger")))
         (epochs (or (plist-get payload :epochs) 1))
         (total (* epochs (length examples)))
         (checkpoint-file
          (and checkpoint-directory
               (nl-llm-agent-evolve--training-checkpoint-path
                checkpoint-directory checkpoint-scope job-id)))
         (loaded
          (when resuming
            (unless checkpoint-file
              (error "GPU challenger resume is not configured"))
            (nl-llm-agent-training-checkpoint-load
             checkpoint-file
             :job-id job-id :payload payload :scope checkpoint-scope
             :parent-generation (nl-llm-evolution-generation evolution)
             :parent-score (nl-llm-evolution-champion-score evolution)
             :sequence sequence :optimizer optimizer :total-steps total)))
         (start-step (or (plist-get loaded :completed-steps) 0))
         (optimizer-step (or (plist-get loaded :optimizer-step) 0))
         (context nil))
    (when loaded
      (nl-llm-agent-training-checkpoint-restore-model candidate loaded))
    (unwind-protect
        (progn
          (setq context
                (nl-llm-agent-ondevice-from-model
                 candidate sequence
                 (or (plist-get payload :lr) 0.05)
                 :optimizer optimizer))
          (when loaded
            (nl-llm-agent-ondevice-restore-training-state
             context optimizer-step (plist-get loaded :optimizer-state)))
          (nl-llm-agent-ondevice-train
           context examples epochs :start-step start-step
           :after-step
           (when checkpoint-file
             (lambda (active completed planned)
               (when (or (= completed planned)
                         (= (% completed checkpoint-every) 0))
                 (let* ((snapshot
                         (nl-llm-agent-ondevice-snapshot active))
                        (step (plist-get snapshot :step)))
                   (nl-llm-agent-training-checkpoint-save
                    checkpoint-file
                    :job-id job-id :payload payload :scope checkpoint-scope
                    :parent-generation
                    (nl-llm-evolution-generation evolution)
                    :parent-score
                    (nl-llm-evolution-champion-score evolution)
                    :sequence sequence :optimizer optimizer
                    :completed-steps completed :total-steps planned
                    :optimizer-step step
                    :model
                    (nl-llm-agent-artifact-export-pav
                     (plist-get snapshot :model) step)
                    :optimizer-state
                    (plist-get snapshot :optimizer-state)))))))
          ;; With checkpointing, the final callback already synchronized the
          ;; weights.  A fully completed resume executes no new callback.
          (when (or (not checkpoint-file) (= start-step total))
            (nl-llm-agent-ondevice-sync context)))
      (when context
        (nl-llm-agent-ondevice-free context)))))

;;;###autoload
(defun nl-llm-agent-evolve-p5-evaluator (examples &optional tokenizer)
  "Return a fixed evaluator built from bounded held-out text EXAMPLES.

The returned function reports mean negative next-token cross-entropy, so larger
scores are better.  EXAMPLES are detached before the closure is returned and
cannot subsequently be changed by the model or caller."
  (setq tokenizer (nl-llm-agent-tokenizer-id tokenizer))
  (nl-llm-agent-evolve--validate-finetune
   (list :examples examples :lr 0.05 :epochs 1) tokenizer)
  (let ((benchmark
         (nl-llm-agent-evolve--tokens
          (copy-sequence examples) tokenizer)))
    (lambda (model)
      (unless (equal
               (nl-llm-agent-evolve--model-tokenizer
                model "P5 evaluator model")
               tokenizer)
        (error "P5 evaluator model tokenizer differs from benchmark"))
      (let ((total 0.0)
            (count 0))
        (dolist (tokens benchmark)
          (let ((loss
                 (nl-llm-agent--p5-forward
                  model (butlast tokens) (apply #'vector (cdr tokens)))))
            (setq total
                  (+ total
                     (aref (photon-tensor-data (pav-value loss)) 0)))
            (setq count (1+ count))))
        (- (/ total count))))))

;;;###autoload
(cl-defun nl-llm-agent-evolve-p5-queue
    (model evaluate catalog-file grammar
           &key (id-prefix "champion") name (min-delta 0.0)
           (maxseq 1024) (max-pending 64) (max-history 256)
           (training-backend 'cpu) (training-sequence 256)
           (optimizer 'sgd)
           training-checkpoint-directory (checkpoint-every 0)
           checkpoint-scope
           checkpoint-file
           promotion-gate
           (initial-generation
            (or (plist-get model :artifact-generation) 0)))
  "Build an evaluated, artifact-publishing P5 improvement queue.

MODEL is a trainable PAV model from `nl-llm-agent-improve-model'.  The trusted
TRAINING-BACKEND is `cpu' or `gpu'; the latter builds a dedicated resident graph
over each isolated challenger and trains without per-step weight transfers.
It reads back once before evaluation unless bounded recovery checkpoints add
interval readbacks.  TRAINING-SEQUENCE bounds that fixed GPU graph.  OPTIMIZER
is `sgd' on either backend or `adam' on GPU.  A positive CHECKPOINT-EVERY plus
TRAINING-CHECKPOINT-DIRECTORY and CHECKPOINT-SCOPE enables transaction-bound GPU
recovery through explicit queue resume.  EVALUATE is the fixed trusted benchmark
score; larger is better.  Accepted generations are published to CATALOG-FILE
under data-only GRAMMAR.  The registered
  `trajectory-finetune' proposal accepts (:examples VECTOR :lr NUMBER :epochs
  INTEGER), all strictly bounded.  INITIAL-GENERATION defaults to MODEL's
  artifact generation.  CHECKPOINT-FILE enables durable queue state; call
  `nl-llm-evolve-queue-restore' after handlers are registered.  PROMOTION-GATE,
  when non-nil, is trusted completion validation receiving detached parent and
  candidate models after the numeric gain gate."
  (setq training-backend
        (nl-llm-agent-evolve--training-backend training-backend))
  (setq optimizer
        (nl-llm-agent-evolve--optimizer optimizer training-backend))
  (let ((tokenizer
         (nl-llm-agent-evolve--model-tokenizer model "P5 evolution model")))
  (unless (and (integerp training-sequence)
               (<= 2 training-sequence)
               (<= training-sequence
                   nl-llm-agent-evolve-max-training-sequence))
    (error "P5 evolution training sequence must be in [2, %d]"
           nl-llm-agent-evolve-max-training-sequence))
  (unless (and (integerp checkpoint-every) (>= checkpoint-every 0)
               (<= checkpoint-every 1000000))
    (error "P5 evolution checkpoint interval must be in [0, 1000000]"))
  (when (and (> checkpoint-every 0) (not (eq training-backend 'gpu)))
    (error "resumable P5 checkpoints require the GPU training backend"))
  (when (> checkpoint-every 0)
    (unless (and (stringp training-checkpoint-directory)
                 (not (string-empty-p training-checkpoint-directory)))
      (error "resumable GPU training requires a checkpoint directory"))
    (unless (and (stringp checkpoint-scope)
                 (<= 1 (length checkpoint-scope))
                 (<= (length checkpoint-scope) 256))
      (error "resumable GPU training requires a bounded checkpoint scope"))
    (setq training-checkpoint-directory
          (expand-file-name training-checkpoint-directory)))
  (when (= checkpoint-every 0)
    (setq training-checkpoint-directory nil
          checkpoint-scope nil))
  (let* ((publisher
          (nl-llm-agent-artifact-publisher
           catalog-file :id-prefix id-prefix :name name :grammar grammar
           :maxseq maxseq :export #'nl-llm-agent-artifact-export-pav))
         (evolution
          (nl-llm-evolution-new
           model evaluate :min-delta min-delta :publish publisher
           :promotion-gate promotion-gate
           :generation initial-generation))
         (queue
          (nl-llm-evolve-queue-new
           evolution :max-pending max-pending :max-history max-history
           :checkpoint-file checkpoint-file)))
    (nl-llm-evolve-queue-register
     queue "trajectory-finetune"
     (lambda (_candidate _payload _state) nil)
     :train
     (lambda (candidate payload state)
       (unless (equal
                (nl-llm-agent-evolve--model-tokenizer
                 candidate "P5 evolution candidate")
                tokenizer)
         (error "P5 evolution candidate tokenizer differs from queue model"))
       (if (eq training-backend 'gpu)
           (nl-llm-agent-evolve-p5-gpu-finetune
            candidate payload state training-sequence optimizer
            :checkpoint-directory training-checkpoint-directory
            :checkpoint-every checkpoint-every
            :checkpoint-scope checkpoint-scope)
         (nl-llm-agent-p5-finetune
          candidate
          (nl-llm-agent-evolve--tokens
           (plist-get payload :examples) tokenizer)
          (or (plist-get payload :lr) 0.05)
          (or (plist-get payload :epochs) 1))))
     :resume
     (when (and (eq training-backend 'gpu) (> checkpoint-every 0))
       (lambda (candidate payload state)
         (nl-llm-agent-evolve-p5-gpu-finetune
          candidate payload state training-sequence optimizer
          :checkpoint-directory training-checkpoint-directory
          :checkpoint-every checkpoint-every
          :checkpoint-scope checkpoint-scope)))
     :finish
     (when (and (eq training-backend 'gpu) (> checkpoint-every 0))
       (lambda (context _result)
         (let ((path
                (nl-llm-agent-evolve--training-checkpoint-path
                 training-checkpoint-directory checkpoint-scope
                 (plist-get context :job-id))))
           (when (file-exists-p path)
             (delete-file path)))))
     :validate
     (if (eq training-backend 'gpu)
         (lambda (payload)
           (nl-llm-agent-evolve--validate-gpu-finetune
            payload training-sequence tokenizer))
       (lambda (payload)
         (nl-llm-agent-evolve--validate-finetune payload tokenizer)))
     :description
     (cond
      ((and (eq training-backend 'gpu) (> checkpoint-every 0))
       "Fine-tune a resumable GPU-resident P5 challenger with bounded checkpoints")
      ((eq training-backend 'gpu)
       "Fine-tune an isolated GPU-resident P5 challenger and read back once")
      (t
       "Fine-tune an isolated P5 challenger on bounded successful trajectory text")))
    queue)))

(provide 'nl-llm-agent-evolve)
;;; nl-llm-agent-evolve.el ends here
