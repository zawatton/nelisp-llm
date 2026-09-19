;;; nl-llm-agent-supervised.el --- completion-only P5 supervision -*- lexical-binding: t; -*-

;; Prompts remain attention context while only completion targets contribute to
;; loss and gradients.  Public callers provide examples, never prepared plans.

;;; Code:

(require 'cl-lib)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-tokenizer)

(declare-function nl-llm-agent-ondevice-from-model
                  "nl-llm-agent-ondevice"
                  (model seq lr &rest keys))
(declare-function nl-llm-agent-ondevice-train
                  "nl-llm-agent-ondevice"
                  (ctx trajs epochs &rest keys))
(declare-function nl-llm-agent-ondevice-sync
                  "nl-llm-agent-ondevice" (ctx))
(declare-function nl-llm-agent-ondevice-free
                  "nl-llm-agent-ondevice" (ctx))

(defconst nl-llm-agent-supervised-max-examples 128)
(defconst nl-llm-agent-supervised-max-example-chars 4096)
(defconst nl-llm-agent-supervised-max-example-tokens 4096)
(defconst nl-llm-agent-supervised-max-total-chars 65536)

(defun nl-llm-agent-supervised--keys (value allowed required where)
  "Validate exact plist VALUE keys against ALLOWED and REQUIRED for WHERE."
  (unless (listp value)
    (error "%s must be a plist" where))
  (let ((tail value) seen)
    (while tail
      (unless (and (consp tail) (consp (cdr tail)))
        (error "%s must be a proper plist" where))
      (let ((key (car tail)))
        (unless (memq key allowed)
          (error "%s has unknown key %S" where key))
        (when (memq key seen)
          (error "%s has duplicate key %S" where key))
        (push key seen))
      (setq tail (cddr tail)))
    (dolist (key required)
      (unless (memq key seen)
        (error "%s is missing key %S" where key))))
  value)

(defun nl-llm-agent-supervised--text (value where)
  "Return detached non-empty string VALUE for WHERE."
  (unless (and (stringp value) (> (length value) 0)
               (<= (length value)
                   nl-llm-agent-supervised-max-example-chars))
    (error "%s must be non-empty text of at most %d characters"
           where nl-llm-agent-supervised-max-example-chars))
  (substring-no-properties value))

(defun nl-llm-agent-supervised--dataset-digest (tokenizer encoded)
  "Return canonical digest binding TOKENIZER and boundary-aware ENCODED data."
  (let ((print-length nil)
        (print-level nil)
        (print-circle nil)
        (print-escape-nonascii t)
        (float-output-format nil))
    (secure-hash
     'sha256
     (prin1-to-string
      (list :tokenizer tokenizer :examples encoded)))))

(defun nl-llm-agent-supervised--prepare (examples tokenizer)
  "Validate EXAMPLES for TOKENIZER and return one fresh internal plan."
  (setq tokenizer (nl-llm-agent-tokenizer-id tokenizer))
  (unless (and (vectorp examples)
               (<= 1 (length examples))
               (<= (length examples)
                   nl-llm-agent-supervised-max-examples))
    (error "supervised examples must be a vector containing 1..%d entries"
           nl-llm-agent-supervised-max-examples))
  (let ((trajectories nil)
        (loss-starts (make-vector (length examples) 0))
        (canonical (make-vector (length examples) nil))
        (total-chars 0)
        (completion-tokens 0)
        (max-tokens 0))
    (dotimes (index (length examples))
      (let* ((entry (aref examples index))
             (_keys
              (nl-llm-agent-supervised--keys
               entry '(:prompt :completion) '(:prompt :completion)
               (format "supervised example %d" index)))
             (prompt
              (nl-llm-agent-supervised--text
               (plist-get entry :prompt)
               (format "supervised example %d prompt" index)))
             (completion
              (nl-llm-agent-supervised--text
               (plist-get entry :completion)
               (format "supervised example %d completion" index)))
             (chars (+ (length prompt) (length completion))))
        (when (> chars nl-llm-agent-supervised-max-example-chars)
          (error "supervised example %d exceeds %d combined characters"
                 index nl-llm-agent-supervised-max-example-chars))
        (setq total-chars (+ total-chars chars))
        (when (> total-chars nl-llm-agent-supervised-max-total-chars)
          (error "supervised dataset exceeds %d total characters"
                 nl-llm-agent-supervised-max-total-chars))
        (let* ((prompt-ids
                (nl-llm-agent-tokenizer-encode prompt tokenizer))
               (completion-ids
                (nl-llm-agent-tokenizer-encode completion tokenizer))
               (trajectory (append prompt-ids completion-ids))
               (tokens (length trajectory)))
          (when (> tokens nl-llm-agent-supervised-max-example-tokens)
            (error "supervised example %d exceeds %d combined tokens"
                   index nl-llm-agent-supervised-max-example-tokens))
          (aset loss-starts index (length prompt-ids))
          (aset canonical index
                (list :prompt prompt :completion completion
                      :prompt-tokens (copy-sequence prompt-ids)
                      :completion-tokens (copy-sequence completion-ids)))
          (push trajectory trajectories)
          (setq completion-tokens
                (+ completion-tokens (length completion-ids))
                max-tokens (max max-tokens tokens)))))
    (setq trajectories (nreverse trajectories))
    (list :tokenizer tokenizer
          :trajectories trajectories
          :loss-starts loss-starts
          :completion-tokens completion-tokens
          :dataset-sha256
          (nl-llm-agent-supervised--dataset-digest tokenizer canonical)
          :examples (length examples)
          :max-tokens max-tokens)))

;;;###autoload
(defun nl-llm-agent-supervised-encode (examples &optional tokenizer)
  "Validate and encode supervised EXAMPLES using TOKENIZER.

Return detached data containing canonical :tokenizer, token :trajectories,
completion :loss-starts, total :completion-tokens, and :dataset-sha256."
  (let ((plan (nl-llm-agent-supervised--prepare examples tokenizer)))
    (list :tokenizer (copy-sequence (plist-get plan :tokenizer))
          :trajectories
          (mapcar #'copy-sequence (plist-get plan :trajectories))
          :loss-starts (copy-sequence (plist-get plan :loss-starts))
          :completion-tokens (plist-get plan :completion-tokens)
          :dataset-sha256
          (copy-sequence (plist-get plan :dataset-sha256)))))

(defun nl-llm-agent-supervised--model-tokenizer (model)
  "Validate trainable MODEL and return its canonical tokenizer."
  (let ((tokenizer
         (nl-llm-agent-improve--model-tokenizer
          model "supervised P5 model"))
        (parameters (nl-llm-agent--p5-params model)))
    (unless (and parameters (cl-every #'pav-p parameters))
      (error "supervised P5 model parameters must be trainable PAV values"))
    tokenizer))

(defun nl-llm-agent-supervised--targets-mask (trajectory loss-start)
  "Return (TARGETS . MASK) for TRAJECTORY and completion LOSS-START."
  (let* ((targets (apply #'vector (cdr trajectory)))
         (rows (length targets))
         (mask (make-vector rows 0)))
    (dotimes (row rows)
      (when (>= (1+ row) loss-start)
        (aset mask row 1)))
    (cons targets mask)))

(defun nl-llm-agent-supervised--loss-plan (model plan)
  "Return token-weighted completion loss for MODEL over prepared PLAN."
  (let ((trajectories (plist-get plan :trajectories))
        (loss-starts (plist-get plan :loss-starts))
        (weighted 0.0)
        (index 0))
    (dolist (trajectory trajectories)
      (let* ((loss-start (aref loss-starts index))
             (targets-mask
              (nl-llm-agent-supervised--targets-mask
               trajectory loss-start))
             (active (- (length trajectory) loss-start))
             (loss
              (nl-llm-agent--p5-forward
               model (butlast trajectory)
               (car targets-mask) (cdr targets-mask))))
        (setq weighted
              (+ weighted
                 (* active
                    (aref (photon-tensor-data (pav-value loss)) 0)))
              index (1+ index))))
    (/ weighted (float (plist-get plan :completion-tokens)))))

;;;###autoload
(defun nl-llm-agent-supervised-loss (model examples)
  "Return token-weighted completion cross-entropy for MODEL and EXAMPLES."
  (let* ((tokenizer (nl-llm-agent-supervised--model-tokenizer model))
         (plan (nl-llm-agent-supervised--prepare examples tokenizer)))
    (nl-llm-agent-supervised--loss-plan model plan)))

(defun nl-llm-agent-supervised--options (keys max-tokens)
  "Validate training KEYS using MAX-TOKENS and return normalized options."
  (nl-llm-agent-supervised--keys
   keys '(:backend :lr :epochs :optimizer :sequence) nil
   "supervised training options")
  (let ((backend (or (plist-get keys :backend) 'cpu))
        (lr (if (plist-member keys :lr) (plist-get keys :lr) 0.05))
        (epochs (if (plist-member keys :epochs)
                    (plist-get keys :epochs) 1))
        (optimizer (or (plist-get keys :optimizer) 'sgd))
        (sequence (plist-get keys :sequence)))
    (unless (memq backend '(cpu gpu))
      (error "supervised backend must be cpu or gpu"))
    (unless (and (numberp lr) (= lr lr) (> lr 0.0) (<= lr 1.0))
      (error "supervised learning rate must be finite and in (0, 1]"))
    (unless (and (integerp epochs) (<= 1 epochs) (<= epochs 32))
      (error "supervised epochs must be an integer in 1..32"))
    (unless (memq optimizer '(sgd adam))
      (error "supervised optimizer must be sgd or adam"))
    (when (and (eq backend 'cpu) (not (eq optimizer 'sgd)))
      (error "supervised CPU training supports only SGD"))
    (if (eq backend 'cpu)
        (when (plist-member keys :sequence)
          (error "supervised CPU training does not accept :sequence"))
      (setq sequence (or sequence (max 2 max-tokens)))
      (unless (and (integerp sequence) (<= 2 sequence) (<= sequence 4096)
                   (>= sequence max-tokens))
        (error "supervised GPU sequence must be in [2,4096] and cover all tokens")))
    (list :backend backend :lr lr :epochs epochs
          :optimizer optimizer :sequence sequence)))

(defun nl-llm-agent-supervised--train-cpu (model plan lr epochs)
  "Mutate MODEL with completion-only CPU training over PLAN."
  (let ((parameters (nl-llm-agent--p5-params model))
        (trajectories (plist-get plan :trajectories))
        (loss-starts (plist-get plan :loss-starts)))
    (dotimes (_epoch epochs)
      (let ((index 0))
        (dolist (trajectory trajectories)
          (let* ((targets-mask
                  (nl-llm-agent-supervised--targets-mask
                   trajectory (aref loss-starts index)))
                 (loss
                  (nl-llm-agent--p5-forward
                   model (butlast trajectory)
                   (car targets-mask) (cdr targets-mask))))
            (photon-autograd-zero-grad parameters)
            (photon-autograd-backward loss)
            (photon-autograd-sgd parameters lr))
          (setq index (1+ index)))))))

(defun nl-llm-agent-supervised--train-gpu
    (model plan lr epochs optimizer sequence)
  "Mutate MODEL with completion-only resident GPU training over PLAN."
  (require 'nl-llm-agent-ondevice)
  (let ((context nil))
    (unwind-protect
        (progn
          (setq context
                (nl-llm-agent-ondevice-from-model
                 model sequence lr :optimizer optimizer
                 :loss-mode 'completion :transfer-mode 'compact))
          (nl-llm-agent-ondevice-train
           context (plist-get plan :trajectories) epochs
           :loss-starts (plist-get plan :loss-starts))
          (nl-llm-agent-ondevice-sync context))
      (when context
        (nl-llm-agent-ondevice-free context)))))

;;;###autoload
(defun nl-llm-agent-supervised-train (model examples &rest keys)
  "Train MODEL on completion-only supervised EXAMPLES.

KEYS accepts :backend (`cpu' or `gpu'), :lr, :epochs, :optimizer (`sgd' or
`adam'), and GPU-only :sequence.  Defaults are CPU, learning rate 0.05, one
epoch, and SGD.  GPU training requires an already enabled GPU, as does the
underlying on-device API.  The provided MODEL mutates in place."
  ;; Validate options which do not depend on data, model identity, the complete
  ;; dataset, and finally sequence coverage before any weight mutation or GPU
  ;; allocation.  The resulting plan is private fresh data, not caller input.
  (nl-llm-agent-supervised--keys
   keys '(:backend :lr :epochs :optimizer :sequence) nil
   "supervised training options")
  (let* ((tokenizer (nl-llm-agent-supervised--model-tokenizer model))
         (plan (nl-llm-agent-supervised--prepare examples tokenizer))
         (options
          (nl-llm-agent-supervised--options
           keys (plist-get plan :max-tokens)))
         (backend (plist-get options :backend))
         (lr (plist-get options :lr))
         (epochs (plist-get options :epochs))
         (optimizer (plist-get options :optimizer))
         (loss-before (nl-llm-agent-supervised--loss-plan model plan)))
    (if (eq backend 'cpu)
        (nl-llm-agent-supervised--train-cpu model plan lr epochs)
      (nl-llm-agent-supervised--train-gpu
       model plan lr epochs optimizer (plist-get options :sequence)))
    (list :backend backend
          :steps (* epochs (plist-get plan :examples))
          :examples (plist-get plan :examples)
          :completion-tokens (plist-get plan :completion-tokens)
          :dataset-sha256 (copy-sequence (plist-get plan :dataset-sha256))
          :loss-before loss-before
          :loss-after (nl-llm-agent-supervised--loss-plan model plan))))

(provide 'nl-llm-agent-supervised)
;;; nl-llm-agent-supervised.el ends here
