;;; nl-llm-agent-supervised-evolve.el --- supervised queue handler -*- lexical-binding: t; -*-

;; Register completion-only supervised fine-tuning beside the legacy
;; trajectory handler.  The queue continues to own cloning, evaluation,
;; publication, and persistence; this file only adds a trusted trainer.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-agent-evolve)
(require 'nl-llm-agent-supervised)
(require 'nl-llm-evolve)
(require 'nl-llm-evolve-queue)

(defconst nl-llm-agent-supervised-evolve-kind "supervised-finetune"
  "Queue proposal kind registered by the supervised evolution adapter.")

(defun nl-llm-agent-supervised-evolve--sequence (value)
  "Validate and return fixed GPU sequence VALUE."
  (unless (and (integerp value)
               (<= 2 value)
               (<= value nl-llm-agent-evolve-max-training-sequence))
    (error "supervised evolution training sequence must be in [2, %d]"
           nl-llm-agent-evolve-max-training-sequence))
  value)

(defun nl-llm-agent-supervised-evolve--payload
    (payload tokenizer backend sequence optimizer)
  "Validate supervised PAYLOAD and return its detached encoding.

TOKENIZER, BACKEND, SEQUENCE, and OPTIMIZER are trusted queue bindings.  The
payload itself is still treated as hostile data and is checked before any
training callback can run."
  (nl-llm-agent-supervised--keys
   payload '(:examples :lr :epochs) '(:examples)
   "supervised fine-tune proposal")
  (let* ((examples (plist-get payload :examples))
         (lr (if (plist-member payload :lr)
                 (plist-get payload :lr)
               0.05))
         (epochs (if (plist-member payload :epochs)
                     (plist-get payload :epochs)
                   1))
         (plan (nl-llm-agent-supervised-encode examples tokenizer))
         (options
          (list :backend backend :lr lr :epochs epochs
                :optimizer optimizer)))
    (when (eq backend 'gpu)
      (setq options
            (append options (list :sequence sequence))))
    ;; Reuse the supervised trainer's scalar/backend normalization, while the
    ;; queue adapter adds the fixed-sequence check below.
    (nl-llm-agent-supervised--options
     options
     (apply #'max (mapcar #'length (plist-get plan :trajectories))))
    (when (eq backend 'gpu)
      (dolist (trajectory (plist-get plan :trajectories))
        (when (> (length trajectory) sequence)
          (error "supervised GPU trajectory exceeds fixed sequence %d"
                 sequence))))
    plan))

(defun nl-llm-agent-supervised-evolve--validate-model (model)
  "Validate MODEL geometry and return its canonical tokenizer."
  (nl-llm-agent-evolve--model-tokenizer
   model "supervised evolution queue model")
  (nl-llm-agent-supervised--model-tokenizer model))

;;;###autoload
(cl-defun nl-llm-agent-supervised-evolve-register
    (queue &key (training-backend 'cpu) (training-sequence 256)
           (optimizer 'sgd))
  "Register completion-only supervised fine-tuning on existing QUEUE.

QUEUE must already be an evaluated evolution queue, normally made by
`nl-llm-agent-evolve-p5-queue'.  The existing evaluator, publisher, cloning,
and checkpoint persistence remain authoritative.  The new
`supervised-finetune' proposal accepts a strict data plist with `:examples'
(a vector of `(:prompt STRING :completion STRING)' entries), optional `:lr',
and optional `:epochs'.  Prompts remain context and only completion targets
are trained by `nl-llm-agent-supervised-train'.

No GPU is enabled here.  A GPU backend requires the caller to own the enabled
device, and interrupted mid-training jobs deliberately have no resume callback
because supervised training does not expose a midpoint snapshot contract.
Checkpoint restoration therefore requires registering this handler again with
the same trusted backend, sequence, and optimizer settings.  Return QUEUE."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-agent-supervised-evolve-register: invalid queue"))
  (setq training-backend
        (nl-llm-agent-evolve--training-backend training-backend))
  (setq optimizer
        (nl-llm-agent-evolve--optimizer optimizer training-backend))
  (setq training-sequence
        (nl-llm-agent-supervised-evolve--sequence training-sequence))
  (let* ((evolution (nl-llm-evolve-queue-evolution queue))
         (tokenizer
          (nl-llm-agent-supervised-evolve--validate-model
           (nl-llm-evolution-champion evolution)))
         (kind nl-llm-agent-supervised-evolve-kind))
    ;; Check every registration input before touching the handler list.  The
    ;; queue primitive also checks this, but doing it here makes the public
    ;; adapter's duplicate-registration guarantee explicit.
    (when (nl-llm-evolve-queue--handler queue kind)
      (error "supervised evolution handler already registered"))
    (let ((validator
           (lambda (payload)
             (nl-llm-agent-supervised-evolve--payload
              payload tokenizer training-backend training-sequence optimizer)))
          (trainer
           (lambda (candidate payload _state)
             (unless (equal
                      (nl-llm-agent-supervised-evolve--validate-model candidate)
                      tokenizer)
               (error "supervised evolution candidate tokenizer differs from queue model"))
             (let ((keys
                    (list :backend training-backend
                          :lr (if (plist-member payload :lr)
                                  (plist-get payload :lr)
                                0.05)
                          :epochs (if (plist-member payload :epochs)
                                      (plist-get payload :epochs)
                                    1)
                          :optimizer optimizer)))
               (when (eq training-backend 'gpu)
                 (setq keys (append keys (list :sequence training-sequence))))
               (apply #'nl-llm-agent-supervised-train
                      candidate (plist-get payload :examples) keys)))))
      (nl-llm-evolve-queue-register
       queue kind
       (lambda (_candidate _payload _state) nil)
       :train trainer
       :validate validator
       :description
       "Completion-only supervised fine-tune of an isolated evolution challenger")
      queue)))

(provide 'nl-llm-agent-supervised-evolve)
;;; nl-llm-agent-supervised-evolve.el ends here
