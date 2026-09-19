;;; nl-llm-weights-train.el --- train a LoRA on an imported model  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 3, the loop.  Composes the
;; verified pieces into completion-only supervised training over an imported
;; int8 model: tokenize with the *donor's* tokenizer, run the stack with a tape,
;; take cross-entropy on the completion positions alone, and push the gradient
;; back through the head and every block into the LoRA.
;;
;; The donor's tokenizer, not the agent's.  `nl-llm-agent-supervised' uses the
;; bounded ascii/utf8 tokenizers, which is right for a model trained from
;; scratch alongside them and wrong here: an imported model's embedding rows are
;; indexed by Qwen's 151936-entry vocabulary, so training it against any other
;; id space would be training against noise.  `nl-llm-qwen-tok-encode' supplies
;; the ids and the prompt length gives the loss boundary, which is the same
;; "prompt stays attention context, only completions carry loss" contract the
;; agent path states -- just in the donor's ids.
;;
;; Nothing here is new machinery.  The linear's backward including the frozen
;; base's W^T.g, every vjp between the linears, and the composition over a whole
;; block are each checked against finite differences elsewhere; this file wires
;; the head and the loss onto them and iterates.  So the check that belongs here
;; is that the *stack* gradient is right -- head plus blocks plus loss -- and
;; that a step reduces the loss it was given.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-lora)
(require 'nl-llm-weights-backward)

;; Declared rather than required: the tokenizer is only needed by
;; `nl-llm-wtrain-encode', and loading its 3.4 MB table is not a cost the
;; gradient paths should pay.  The caller that encodes has already loaded it.
(declare-function nl-llm-qwen-tok-encode "nl-llm-qwen-tokenizer" (tok text))

(defvar nl-llm-wtrain-last-grads nil
  "LoRA gradients from the most recent `nl-llm-wtrain-step'.
Exposed so a gradient check can look at them without the step having to return
two things; the training loop itself only needs the loss.")

;;;###autoload
(defun nl-llm-wtrain-encode (tok example)
  "Encode EXAMPLE, a (:prompt :completion) plist, with the donor tokenizer TOK.
Returns (IDS . LOSS-START): IDS the concatenation, LOSS-START the number of
prompt tokens, so positions before it contribute attention context and no loss."
  (let* ((p (nl-llm-qwen-tok-encode tok (plist-get example :prompt)))
         (c (nl-llm-qwen-tok-encode tok (plist-get example :completion))))
    (unless (and p c)
      (error "nl-llm-wtrain-encode: empty prompt or completion"))
    (cons (append p c) (length p))))

;;;###autoload
(defun nl-llm-wtrain-xent (logits vocab target)
  "Cross-entropy of LOGITS (VOCAB long) against TARGET, and its gradient.
Returns (LOSS . DLOGITS).  Computed through the log-sum-exp shift, so a large
logit does not overflow before it can be normalised."
  (let ((mx -1.0e30) (sum 0.0) (d (make-vector vocab 0.0)))
    (dotimes (j vocab) (when (> (aref logits j) mx) (setq mx (aref logits j))))
    (dotimes (j vocab)
      (let ((e (exp (- (aref logits j) mx))))
        (aset d j e)
        (setq sum (+ sum e))))
    (dotimes (j vocab) (aset d j (/ (aref d j) sum)))
    (let ((loss (- (+ (log sum) mx) (aref logits target))))
      (aset d target (- (aref d target) 1.0))
      (cons loss d))))

;;;###autoload
(defun nl-llm-wtrain-forward (layers head ids cfg loras)
  "Run IDS through LAYERS and return (HIDDEN TAPES EMB).
HIDDEN is the post-final-norm activation, flat SEQ x dim; TAPES the per-layer
tapes in order; EMB the embedding rows, kept because the input gradient of the
first block ends there.  HEAD carries the final RMSNorm gain as :lnf."
  (let* ((dim (plist-get cfg :dim))
         (seq (length ids))
         (x (make-vector (* seq dim) 0.0))
         (i 0) (tapes nil))
    (dolist (id ids)
      (let ((row (nl-llm-weights-row-of head id)))
        (dotimes (t0 dim) (aset x (+ (* i dim) t0) (aref row t0))))
      (setq i (1+ i)))
    (let ((emb (copy-sequence x)))
      (dolist (lay layers)
        (let ((fw (nl-llm-wb-block-forward lay x seq cfg loras)))
          (push (nth 1 fw) tapes)
          (setq x (nth 0 fw))))
      (let ((pre (copy-sequence x))
            (out (make-vector (* seq dim) 0.0))
            (gain (plist-get head :lnf))
            (eps (or (plist-get cfg :rms-eps) 1.0e-6)))
        (dotimes (p seq)
          (let ((r (nl-llm-wf--rmsnorm x (* p dim) dim gain eps)))
            (dotimes (t0 dim) (aset out (+ (* p dim) t0) (aref r t0)))))
        (list out (nreverse tapes) emb pre)))))

;;;###autoload
(defun nl-llm-weights-row-of (head id)
  "Return row ID of HEAD's tied embedding, dequantized."
  (let* ((lin (plist-get head :lin))
         (cols (nl-llm-weights-lin-cols lin))
         (words (nl-llm-weights-lin-words lin))
         (b (nl-llm-weights-lin-bytes lin))
         (scale (aref (nl-llm-weights-lin-scales lin) id))
         (base (* id words 4))
         (out (make-vector cols 0.0)))
    (dotimes (i cols)
      (let ((byte (aref b (+ base i))))
        (aset out i (* (if (> byte 127) (- byte 256) byte) scale))))
    out))

;;;###autoload
(defun nl-llm-wtrain-step (layers head ids loss-start cfg loras &optional lr)
  "One training step over IDS with LORAS attached; returns the mean loss.
Loss is cross-entropy at positions LOSS-START..SEQ-1 predicting the next id,
which is the completion-only contract.  LR nil takes the gradient without
updating, so a caller can check it."
  (let* ((dim (plist-get cfg :dim))
         (seq (length ids))
         (vocab (nl-llm-weights-lin-rows (plist-get head :lin)))
         (fw (nl-llm-wtrain-forward layers head ids cfg loras))
         (hidden (nth 0 fw)) (tapes (nth 1 fw)) (pre (nth 3 fw))
         (gain (plist-get head :lnf))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (dpre (make-vector (* seq dim) 0.0))
         (total 0.0) (count 0)
         (ids-v (vconcat ids)))
    ;; Positions LOSS-START-1 .. SEQ-2 are the ones whose *next* token is a
    ;; completion token, so those are where the loss lives.
    (let ((p (max 0 (1- loss-start))))
      (while (< p (1- seq))
        ;; The head goes through the same two hooks as a block's linears, so
        ;; a caller that has uploaded it gets it on the GPU without this file
        ;; knowing there is one.  It matters more here than anywhere else: the
        ;; head is 151936 x 1024, the largest matrix in the model, and every
        ;; scored position pays it twice.
        (let* ((logits (nl-llm-wb--apply (plist-get head :lin)
                                         hidden (* p dim)))
               (lg (nl-llm-wtrain-xent logits vocab (aref ids-v (1+ p))))
               (dh (nl-llm-wb--transpose (plist-get head :lin) (cdr lg)))
               (dn (nl-llm-wb-rmsnorm-vjp pre (* p dim) dim gain eps dh)))
          (setq total (+ total (car lg)) count (1+ count))
          (dotimes (t0 dim)
            (aset dpre (+ (* p dim) t0)
                  (+ (aref dpre (+ (* p dim) t0)) (aref dn t0)))))
        (setq p (1+ p))))
    (when (zerop count)
      (error "nl-llm-wtrain-step: no completion positions to learn from"))
    ;; Scale by 1/count here so the reported loss and the gradient agree.
    (dotimes (i (length dpre)) (aset dpre i (/ (aref dpre i) (float count))))
    ;; back through the blocks, last to first
    (let ((grads nil) (d dpre))
      (dolist (pair (nreverse (cl-mapcar #'cons layers tapes)))
        (let ((res (nl-llm-wb-block-backward (car pair) (cdr pair) d
                                             (length ids) cfg loras)))
          (setq d (car res))
          (dolist (role '(:wq :wk :wv :wo :wg :wu :wd))
            (let ((g (plist-get (cdr res) role)))
              (when g (setq grads (nl-llm-wb--merge-grads grads role g)))))))
      (when lr
        (dolist (role '(:wq :wk :wv :wo :wg :wu :wd))
          (let ((lora (plist-get loras role)) (g (plist-get grads role)))
            (when (and lora g) (nl-llm-wlora-sgd lora g lr)))))
      (setq nl-llm-wtrain-last-grads grads))
    (/ total (float count))))

;;;###autoload
(defun nl-llm-wtrain-fit (layers head examples tok cfg loras steps lr
                                 &optional progress)
  "Take STEPS passes over EXAMPLES, training LORAS at LR.  Returns the losses.
EXAMPLES is the vector from a distilled dataset.  One example per step, cycling,
because an imported model at this size is slow enough that a step is a unit
worth reporting on its own."
  (let ((losses nil) (n (length examples)))
    (dotimes (step steps)
      (let* ((example (aref examples (mod step n)))
             (enc (nl-llm-wtrain-encode tok example))
             (loss (nl-llm-wtrain-step layers head (car enc) (cdr enc)
                                       cfg loras lr)))
        (push loss losses)
        (when progress (funcall progress step loss (length (car enc))))))
    (nreverse losses)))

(provide 'nl-llm-weights-train)
;;; nl-llm-weights-train.el ends here
