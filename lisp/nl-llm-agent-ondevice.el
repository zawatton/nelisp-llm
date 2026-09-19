;;; nl-llm-agent-ondevice.el --- close the self-improvement loop on the GPU  -*- lexical-binding: t; -*-

;; The final piece of the agent harness (docs/design/05-agent-harness.org): run the
;; WHOLE self-improvement loop on-device by transferring the GPU-trained weights
;; back into the rollout decoder.
;;
;; One model, two views that SHARE the same weight tensors:
;;   * a CPU view (`nl-llm-agent-improve-model', stacked nl-llm-ag-block) used to
;;     ROLLOUT actions; and
;;   * an nlga GPU graph (`nlga-model') built FROM the very same host tensors, used
;;     to TRAIN.
;; The two are the same function (verified: CPU vs nlga logits agree to ~2e-5), so
;; after `nlga-step' trains on the GPU, `nlga-readback' copies the trained weights
;; straight back into the shared host tensors -- the CPU rollout instantly decodes
;; with the GPU-trained weights.  No reformatting, no second copy: the transfer is
;; the readback into the shared tensor objects.
;;
;; Loop: rollout (CPU) -> keep reward>=1 -> train (GPU) -> readback -> repeat.  The
;; rollout success rate rising IS the proof the transfer works -- the only thing
;; that changed the CPU decoder's weights is the GPU training fed back through it.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-completion-plan)

(defconst nl-llm-agent-ondevice-max-sequence 4096
  "Maximum fixed sequence length accepted by the on-device P5 trainer.")

(defconst nl-llm-agent-ondevice-max-completion-trajectories 128
  "Maximum trajectories in one low-level completion-loss training plan.")

(defconst nl-llm-agent-ondevice-max-completion-tokens (* 128 4096)
  "Maximum aggregate tokens in a low-level completion-loss training plan.")

(defconst nl-llm-agent-ondevice--uint32-mask #xffffffff
  "Mask used by the local 32-bit xorshift training-order generator.")

(defconst nl-llm-agent-ondevice--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd)
  "Ordered P5 block parameters consumed by `nlga-model'.")

(defun nl-llm-agent-ondevice--model (model)
  "Validate and return a trainable dense P5 MODEL."
  (unless (listp model)
    (error "on-device challenger must be a P5 model plist"))
  (let ((dim (plist-get model :dim))
        (ff (plist-get model :ff))
        (vocab (plist-get model :vocab))
        (tokenizer
         (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
        (heads (plist-get model :heads))
        (nblocks (plist-get model :nblocks))
        (blocks (plist-get model :blocks)))
    (unless (and (integerp dim) (> dim 0)
                 (integerp ff) (> ff 0)
                 (integerp vocab)
                 (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
                 (integerp heads) (> heads 0) (= (% dim heads) 0)
                 (= (% (/ dim heads) 2) 0)
                 (integerp nblocks) (> nblocks 0)
                 (listp blocks) (= (length blocks) nblocks))
      (error "on-device challenger has incompatible P5 architecture"))
    (unless (cl-every #'pav-p (nl-llm-agent--p5-params model))
      (error "on-device challenger parameters must be trainable PAV values")))
  model)

(defun nl-llm-agent-ondevice--optimizer (optimizer)
  "Return normalized supported OPTIMIZER symbol."
  (let ((value (if (stringp optimizer) (intern optimizer) optimizer)))
    (unless (memq value '(sgd adam))
      (error "unsupported on-device optimizer %S" optimizer))
    value))

(defun nl-llm-agent-ondevice--loss-mode (mode)
  "Return normalized supported loss MODE."
  (unless (memq mode '(nil completion))
    (error "unsupported on-device loss mode %S" mode))
  mode)

(defun nl-llm-agent-ondevice--transfer-mode (mode loss-mode)
  "Return normalized transfer MODE for LOSS-MODE."
  (unless (memq mode '(nil compact))
    (error "unsupported on-device transfer mode %S" mode))
  (when (and mode (not (eq loss-mode 'completion)))
    (error "compact transfer requires completion loss mode"))
  mode)

(defun nl-llm-agent-ondevice--completion-plan-copy (plan)
  "Validate PLAN and return a detached canonical completion plan."
  (unless plan
    (error "bound completion context requires a completion plan"))
  (nl-llm-agent-completion-plan-validate plan))

(defun nl-llm-agent-ondevice--check-completion-plan-geometry
    (plan model seq lr optimizer transfer-mode)
  "Reject PLAN unless it matches MODEL and the resident training geometry."
  (let* ((tokenizer
          (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
         (vocab (plist-get model :vocab))
         (pad-token
          (car (nl-llm-agent-tokenizer-encode " " tokenizer))))
    (unless (and (equal (plist-get plan :tokenizer) tokenizer)
                 (= (plist-get plan :vocab) vocab)
                 (= (plist-get plan :sequence) seq)
                 (= (plist-get plan :learning-rate) (float lr))
                 (eq (plist-get plan :optimizer) optimizer)
                 (equal (plist-get plan :transfer-mode) transfer-mode)
                 (= (plist-get plan :pad-token) pad-token))
      (error "completion plan does not match resident model or training geometry")))
  plan)

(defun nl-llm-agent-ondevice--bound-completion-plan (ctx)
  "Validate the bound completion plan and CTX metadata, returning a copy."
  (let* ((plan
          (nl-llm-agent-ondevice--completion-plan-copy
           (plist-get ctx :completion-plan)))
         (model (plist-get ctx :cpu)))
    (unless (eq (plist-get ctx :loss-mode) 'completion)
      (error "bound completion plan requires completion loss mode"))
    (unless (and (equal (plist-get ctx :tokenizer) (plist-get plan :tokenizer))
                 (= (plist-get ctx :vocab) (plist-get plan :vocab))
                 (= (plist-get ctx :seq) (plist-get plan :sequence))
                 (= (plist-get ctx :learning-rate)
                    (plist-get plan :learning-rate))
                 (eq (plist-get ctx :optimizer) (plist-get plan :optimizer))
                 (equal (plist-get ctx :transfer-mode)
                        (plist-get plan :transfer-mode))
                 (= (plist-get ctx :pad-token) (plist-get plan :pad-token)))
      (error "completion plan does not match context metadata"))
    (unless (and (listp model) model)
      (error "bound completion context requires a CPU model view"))
    (nl-llm-agent-ondevice--check-completion-plan-geometry
     plan model (plist-get ctx :seq) (plist-get ctx :learning-rate)
     (plist-get ctx :optimizer) (plist-get ctx :transfer-mode))
    (let ((step (plist-get ctx :step))
          (total (* (plist-get plan :epochs)
                    (length (plist-get plan :trajectories)))))
      (unless (and (integerp step) (>= step 0) (<= step total))
        (error "bound completion context step is outside the training plan")))
    plan))

(defun nl-llm-agent-ondevice--bound-training-plan
    (ctx trajs epochs loss-starts loss-masks shuffle-seed)
  "Validate caller training metadata against CTX's bound completion plan."
  (let* ((bound (nl-llm-agent-ondevice--bound-completion-plan ctx))
         (expected
          (nl-llm-agent-completion-plan-make
           trajs loss-starts
           :tokenizer (plist-get bound :tokenizer)
           :sequence (plist-get bound :sequence)
           :learning-rate (plist-get bound :learning-rate)
           :epochs epochs :optimizer (plist-get bound :optimizer)
           :transfer-mode (plist-get bound :transfer-mode)
           :loss-masks loss-masks :shuffle-seed shuffle-seed)))
    (unless (and (= epochs (plist-get bound :epochs))
                 (equal expected bound))
      (error "training metadata does not match bound completion plan"))
    bound))

(defun nl-llm-agent-ondevice--finite-tensor-p (tensor shape)
  "Return non-nil when TENSOR has SHAPE and finite numeric data."
  (condition-case nil
      (let ((size 1))
        (dolist (dimension shape)
          (unless (and (integerp dimension) (> dimension 0))
            (error "invalid tensor shape"))
          (setq size (* size dimension)))
        (and (vectorp tensor) (= (length tensor) 2)
             (equal (photon-tensor-shape tensor) shape)
             (vectorp (photon-tensor-data tensor))
             (= (length (photon-tensor-data tensor)) size)
             (cl-every
              (lambda (value)
                (and (numberp value) (= value value)
                     (< (abs (float value)) 1.0e300)))
              (append (photon-tensor-data tensor) nil))))
    (error nil)))

(defun nl-llm-agent-ondevice--validate-adam-state (ctx state)
  "Validate Adam STATE against resident CTX parameters before any writes."
  (let* ((parameters (nlga-params (plist-get ctx :b)))
         (tail state) (seen nil) (count 0))
    (while tail
      (unless (consp tail)
        (error "Adam optimizer state must be a proper list"))
      (when (memq tail seen)
        (error "Adam optimizer state must not be circular"))
      (push tail seen)
      (setq count (1+ count))
      (when (> count (length parameters))
        (error "Adam optimizer state has the wrong parameter count"))
      (setq tail (cdr tail)))
    (unless (= count (length parameters))
      (error "Adam optimizer state has the wrong parameter count"))
    (cl-mapc
     (lambda (pair parameter)
       (let ((shape (photon-tensor-shape (plist-get parameter :tensor))))
         (unless (and (consp pair)
                      (nl-llm-agent-ondevice--finite-tensor-p (car pair) shape)
                      (nl-llm-agent-ondevice--finite-tensor-p (cdr pair) shape))
           (error "Adam optimizer state has incompatible tensors"))))
     state parameters)
    state))

(defun nl-llm-agent--onehot-pad (toks seq vocab &optional pad-id)
  "Onehot TOKS as SEQ by VOCAB rows, padded with PAD-ID or zero."
  (setq pad-id (or pad-id 0))
  (let ((v (make-vector (* seq vocab) 0.0)) (i 0) (cs (append toks nil)))
    (while (and cs (< i seq)) (aset v (+ (* i vocab) (car cs)) 1.0) (setq cs (cdr cs) i (1+ i)))
    (while (< i seq) (aset v (+ (* i vocab) pad-id) 1.0) (setq i (1+ i)))
    (photon-tensor (list seq vocab) v)))

(defun nl-llm-agent--shift-pad (toks seq &optional pad-id)
  "Return next-token targets for TOKS, padded to SEQ with PAD-ID or zero."
  (setq pad-id (or pad-id 0))
  (let ((tv (make-vector seq 0)) (a (apply #'vector (append toks nil))) (n 0))
    (setq n (length a))
    (dotimes (i seq)
      (aset tv i (if (< (1+ i) n) (aref a (1+ i)) pad-id)))
    tv))

(defun nl-llm-agent-ondevice--index-pad (toks seq pad-id)
  "Return TOKS as a SEQ by 1 float index tensor padded with PAD-ID."
  (let ((data (make-vector seq (float pad-id)))
        (rest (append toks nil))
        (index 0))
    (while rest
      (aset data index (float (car rest)))
      (setq rest (cdr rest)
            index (1+ index)))
    (photon-tensor (list seq 1) data)))

(defun nl-llm-agent-ondevice--xorshift32 (state)
  "Advance the local 32-bit xorshift STATE.

This is only a deterministic engineering PRNG for training order; it makes no
cryptographic claims."
  (let ((mask nl-llm-agent-ondevice--uint32-mask))
    (setq state (logand mask (logxor state (ash state 13))))
    (setq state (logand mask (logxor state (ash state -17))))
    (logand mask (logxor state (ash state 5)))))

(defun nl-llm-agent-ondevice--shuffle-epoch (count state)
  "Return a Fisher--Yates permutation of COUNT and the next PRNG STATE.

The permutation is generated from a copied index vector, so the caller's
trajectory and completion-plan containers remain untouched.  The modulo draw
is intentional engineering for this local PRNG and is not a cryptographic
randomness claim."
  (let ((indices (make-vector count 0))
        (i (1- count)))
    (dotimes (index count)
      (aset indices index index))
    (while (> i 0)
      (setq state (nl-llm-agent-ondevice--xorshift32 state))
      (let* ((j (mod state (1+ i)))
             (value (aref indices i)))
        (aset indices i (aref indices j))
        (aset indices j value))
      (setq i (1- i)))
    (cons state indices)))

;;;###autoload
(cl-defun nl-llm-agent-ondevice-from-model
    (model seq lr &key (optimizer 'sgd) loss-mode transfer-mode
           completion-plan)
  "Build an isolated resident GPU training graph over P5 MODEL.

MODEL remains the CPU view and owns the host tensors used for eventual
evaluation and publication.  Its parameter values are uploaded once and stay
resident through every training step.  `nl-llm-agent-ondevice-sync' performs
the single explicit readback into MODEL after training.  SEQ is the fixed
training sequence length and LR the optimizer step.  OPTIMIZER is `sgd' or
`adam'.  LOSS-MODE may be `completion'; this opt-in graph accepts per-example
completion starts and excludes prompt and padding rows from CE.  The default
nil retains legacy full-sequence loss.  TRANSFER-MODE may be `compact' only
with completion loss; it uploads token indices, target indices, and row scales
as SEQ by 1 residents and does not retain training logits for readback.  Its
default nil preserves the dense one-hot transfer path.  Free the returned
context with `nl-llm-agent-ondevice-free'.  Requires an active GPU
(`nl-llm-gpu-enable')."
  (setq model (nl-llm-agent-ondevice--model model))
  (unless (and (integerp seq) (<= 2 seq)
               (<= seq nl-llm-agent-ondevice-max-sequence))
    (error "on-device sequence must be in [2, %d]"
           nl-llm-agent-ondevice-max-sequence))
  (unless (and (numberp lr) (= lr lr) (> lr 0.0) (<= lr 1.0))
    (error "on-device learning rate must be in (0, 1]"))
  (setq optimizer (nl-llm-agent-ondevice--optimizer optimizer))
  (setq loss-mode (nl-llm-agent-ondevice--loss-mode loss-mode))
  ;; Validate the opt-in combination before allocating GPU residents.
  (setq transfer-mode
        (nl-llm-agent-ondevice--transfer-mode transfer-mode loss-mode))
  (setq lr (float lr))
  (let ((canonical-plan
         (when completion-plan
           (unless (eq loss-mode 'completion)
             (error "completion plan requires completion loss mode"))
           (let ((plan
                  (nl-llm-agent-ondevice--completion-plan-copy
                   completion-plan)))
             (nl-llm-agent-ondevice--check-completion-plan-geometry
              plan model seq lr optimizer transfer-mode)
             plan))))
    (let* ((compactp (eq transfer-mode 'compact))
         (dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (vocab (plist-get model :vocab))
         (tokenizer
          (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
         (pad-token
          (car (nl-llm-agent-tokenizer-encode " " tokenizer)))
         (hd (/ dim heads)) (scl (/ 1.0 (sqrt (float hd))))
         (tables (nl-llm-gpu-rope-tables seq hd))
         (mask (unless compactp
                 (let ((md (make-vector (* seq seq) 0.0)) (i 0))
                   (while (< i seq) (let ((j (1+ i))) (while (< j seq) (aset md (+ (* i seq) j) -1.0e30) (setq j (1+ j)))) (setq i (1+ i)))
                   (photon-tensor (list seq seq) md))))
         (b (nlga-new))
         (complete nil))
    (cl-flet ((wp (pav) (nlga-param b (pav-value pav)))
              (wb (blk)
                (let ((out nil))
                  (dolist (key nl-llm-agent-ondevice--block-keys)
                    (setq out
                          (append out
                                  (list key
                                        (nlga-param
                                         b (pav-value (plist-get blk key)))))))
                  out)))
      (unwind-protect
          (let* ((oh
                  (nlga-const
                   b (photon-tensor
                      (if compactp (list seq 1) (list seq vocab))
                      (make-vector (if compactp seq (* seq vocab)) 0.0))))
                 (ohtgt
                  (nlga-const
                   b (photon-tensor
                      (if compactp (list seq 1) (list seq vocab))
                      (make-vector (if compactp seq (* seq vocab)) 0.0))))
                 (loss-scale
                  (when loss-mode
                    (nlga-const
                     b (photon-tensor
                        (if compactp (list seq 1) (list seq vocab))
                        (make-vector (if compactp seq (* seq vocab)) 0.0)))))
                 (wter (wp (plist-get model :wte)))
                 (blks
                  (mapcar
                   (lambda (blk) (wb blk))
                   (plist-get model :blocks)))
                 (lnfgr (wp (plist-get model :lnfg)))
                 (whr (wp (plist-get model :wh)))
                 (bhr (wp (plist-get model :bh)))
                 (cosr (nlga-const b (car tables)))
                 (sinr (nlga-const b (cdr tables)))
                 (sposr (nlga-scalar b 1.0))
                 (snegr (nlga-scalar b -1.0))
                 (sclr (unless compactp (nlga-scalar b scl)))
                 (oner (unless compactp (nlga-scalar b 1.0)))
                 (maskr (unless compactp (nlga-const b mask)))
                 (logits
                  (if compactp
                      (nlga-model-idx
                       b oh wter blks lnfgr whr bhr heads heads
                       cosr sinr sposr snegr nil nil)
                    (nlga-model
                     b oh wter blks lnfgr whr bhr heads heads
                     cosr sinr sposr snegr sclr maskr)))
                 (lout (unless compactp (nlga-keep b logits oner))))
            (cond
             (compactp
              (nlga-seed-ce-idx-masked b logits ohtgt loss-scale))
             (loss-mode
              (nlga-seed-ce-masked b logits ohtgt loss-scale))
             (t
              (nlga-seed-ce b logits ohtgt)))
            (if (eq optimizer 'adam)
                (nlga-finish-adam b lr)
              (nlga-finish b (nlga-scalar b lr)))
            (nlga-compile b)
            (setq complete t)
            (list :cpu model :b b :oh oh :ohtgt ohtgt
                  :loss-scale loss-scale :loss-mode loss-mode
                  :transfer-mode transfer-mode :lout lout
                  :seq seq :vocab vocab :pad-token pad-token
                  :tokenizer (copy-sequence tokenizer) :learning-rate lr
                  :optimizer optimizer :step 0
                  :completion-plan canonical-plan))
        (unless complete
          (nlga-free b)))))))

;;;###autoload
(defun nl-llm-agent-ondevice-new (dim ff heads nblocks seq lr)
  "Build a new P5 model and its resident GPU training context.

This compatibility constructor uses SGD.  Use
`nl-llm-agent-ondevice-from-model' to train an isolated existing challenger or
to select Adam."
  (nl-llm-agent-ondevice-from-model
   (nl-llm-agent-improve-model
    dim ff nl-llm-agent-char-vocab nblocks heads)
   seq lr :optimizer 'sgd))

(cl-defun nl-llm-agent-ondevice-train
    (ctx trajs epochs &key (start-step 0) after-step loss-starts shuffle-seed
         loss-masks)
  "Train resident CTX on TRAJS for EPOCHS, optionally resuming at START-STEP.

START-STEP counts flattened epoch/example steps already completed.  AFTER-STEP,
when non-nil, receives (CTX COMPLETED TOTAL) after each newly completed device
step.  Unbound completion-loss contexts are deliberately ephemeral: their
callback may inspect progress, but snapshot and optimizer-state APIs reject
them.  A context created with COMPLETION-PLAN binds those APIs to the
detached plan and permits a validated nonzero resume.  Such contexts require
LOSS-STARTS, a vector aligned with TRAJS whose element is the token index of
that trajectory's first completion token.  Legacy contexts reject LOSS-STARTS.
SHUFFLE-SEED enables deterministic per-epoch Fisher--Yates ordering for
completion-loss contexts only.  It must be an integer in [1, #xffffffff];
its xorshift32 state carries across epochs for this call and is not a durable
checkpoint field.  LOSS-MASKS is an optional vector of per-trajectory ordinary
binary vectors for completion-loss contexts.  Mask index J selects target token
J, and therefore loss row J-1; selected rows are normalized to SEQ divided by
the number of selected targets.  Nil retains the historical traversal and
completion-row plan exactly.
Return the total completed step count."
  (unless (and (integerp epochs) (> epochs 0))
    (error "on-device epochs must be positive"))
  (let* ((loss-mode (plist-get ctx :loss-mode))
         (completionp (eq loss-mode 'completion))
         (bound-plan nil))
    (when (and loss-mode (not completionp))
      (error "on-device context has unsupported loss mode %S" loss-mode))
    ;; Build and compare a bounded canonical plan before traversing caller
    ;; trajectories or making any resident update.  The detached vectors then
    ;; remain stable if a progress callback mutates the caller's containers.
    (when (and completionp (plist-get ctx :completion-plan))
      (setq bound-plan
            (nl-llm-agent-ondevice--bound-training-plan
             ctx trajs epochs loss-starts loss-masks shuffle-seed))
      (setq trajs (append (plist-get bound-plan :trajectories) nil)
            epochs (plist-get bound-plan :epochs)
            loss-starts (copy-sequence (plist-get bound-plan :loss-starts))
            loss-masks (when (plist-get bound-plan :loss-masks)
                         (copy-tree (plist-get bound-plan :loss-masks) t))
            shuffle-seed (plist-get bound-plan :shuffle-seed)))
    (unless (and (listp trajs) (not (null trajs)))
      (error "on-device trajectories must be a non-empty list"))
    (unless (and (integerp start-step) (>= start-step 0)
                 (<= start-step (* epochs (length trajs))))
      (error "on-device start step is outside the training plan"))
    (when (and after-step (not (functionp after-step)))
      (error "on-device after-step callback must be a function"))
    (if bound-plan
        (unless (and (integerp (plist-get ctx :step))
                     (= (plist-get ctx :step) start-step))
          (error "bound completion context step does not match start step"))
      (when (and completionp (/= start-step 0))
        (error "completion-loss contexts cannot resume from a start step")))
    (when shuffle-seed
      (unless (and (integerp shuffle-seed)
                   (<= 1 shuffle-seed)
                   (<= shuffle-seed nl-llm-agent-ondevice--uint32-mask))
        (error "on-device shuffle seed must be an integer in [1, #xffffffff]"))
      (unless completionp
        (error "on-device shuffle requires completion loss mode")))
    (if completionp
        (unless (and (vectorp loss-starts)
                     (= (length loss-starts) (length trajs)))
          (error "completion loss requires one loss start per trajectory"))
      (when loss-starts
        (error "legacy on-device loss does not accept loss starts")))
    (when loss-masks
      (unless completionp
        (error "on-device loss masks require completion loss mode"))
      (unless (and (vectorp loss-masks)
                   (= (length loss-masks) (length trajs)))
        (error "completion loss masks require one mask per trajectory")))
    ;; Completion plans are detached and fully validated before any resident
    ;; input, optimizer counter, or parameter can be mutated.  Keep only the
    ;; row scale here; expansion across VOCAB happens for the current step.
    (let ((plans nil)
          (index 0)
          (total-tokens 0)
          (seq (plist-get ctx :seq))
          (vocab (plist-get ctx :vocab)))
      (when completionp
        (when (> (length trajs)
                 nl-llm-agent-ondevice-max-completion-trajectories)
          (error "completion plan exceeds %d trajectories"
                 nl-llm-agent-ondevice-max-completion-trajectories))
        (dolist (trajectory trajs)
          (unless (or (proper-list-p trajectory) (vectorp trajectory))
            (error "completion trajectory %d must be a token sequence" index))
          (let* ((n (length trajectory))
                 (start (aref loss-starts index))
                 (mask (when loss-masks (aref loss-masks index))))
            (unless (and (<= 2 n) (<= n seq))
              (error "completion trajectory %d length must be in [2, %d]"
                     index seq))
            (setq total-tokens (+ total-tokens n))
            (when (> total-tokens
                     nl-llm-agent-ondevice-max-completion-tokens)
              (error "completion plan exceeds %d aggregate tokens"
                     nl-llm-agent-ondevice-max-completion-tokens))
            ;; Copy only after the cheap length bound rejects oversized input.
            (let ((tokens (append trajectory nil)))
              (dolist (token tokens)
                (unless (and (integerp token) (<= 0 token) (< token vocab))
                  (error "completion trajectory %d has invalid token %S"
                         index token)))
              (unless (and (integerp start) (<= 1 start) (< start n))
                (error
                 "completion loss start %S for trajectory %d must be in [1, %d]"
                 start index (1- n)))
              (when loss-masks
                (unless (and (vectorp mask) (= (length mask) n))
                  (error "completion loss mask %d must be an ordinary vector of length %d"
                         index n)))
              (let ((rows (make-vector seq 0.0)))
                (if loss-masks
                    (let ((selected 0))
                      ;; Validate every mask element before constructing any
                      ;; executable plan.  Prompt positions are never targets.
                      (dotimes (target-index n)
                        (let ((value (aref mask target-index)))
                          (unless (and (integerp value) (or (= value 0) (= value 1)))
                            (error "completion loss mask %d has invalid value %S"
                                   index value))
                          (when (< target-index start)
                            (when (= value 1)
                              (error "completion loss mask %d selects prompt index %d"
                                     index target-index)))
                          (when (= value 1)
                            (setq selected (1+ selected)))))
                      (unless (> selected 0)
                        (error "completion loss mask %d selects no completion target"
                               index))
                      (let ((factor (/ (float seq) selected)))
                        (dotimes (target-index n)
                          (when (= (aref mask target-index) 1)
                            (aset rows (1- target-index) factor))))
                      ;; The row vector is detached from MASK before the plan
                      ;; can reach a callback or a later shuffled epoch.
                      (setq rows (copy-sequence rows)))
                  (let* ((active (- n start))
                         (factor (/ (float seq) active))
                         (row (1- start)))
                    ;; Preserve the historical completion plan byte-for-byte
                    ;; when LOSS-MASKS is nil.
                    (while (< row (1- n))
                      (aset rows row factor)
                      (setq row (1+ row)))))
                (push (list tokens rows) plans))))
          (setq index (1+ index)))
        (setq plans (nreverse plans)))
      (nl-llm-agent-ondevice--train-plan
       ctx trajs epochs start-step after-step plans shuffle-seed))))

(defun nl-llm-agent-ondevice--expand-row-scales (rows vocab)
  "Expand ROWS across VOCAB columns into a scale tensor."
  (let* ((seq (length rows))
         (data (make-vector (* seq vocab) 0.0))
         (row 0))
    (while (< row seq)
      (let ((value (aref rows row))
            (column 0)
            (base (* row vocab)))
        (while (< column vocab)
          (aset data (+ base column) value)
          (setq column (1+ column))))
      (setq row (1+ row)))
    (photon-tensor (list seq vocab) data)))

(defun nl-llm-agent-ondevice--train-plan
    (ctx trajs epochs start-step after-step completion-plans &optional shuffle-seed)
  "Execute a prevalidated on-device training plan."
  (let ((b (plist-get ctx :b)) (oh (plist-get ctx :oh)) (ohtgt (plist-get ctx :ohtgt))
        (loss-scale (plist-get ctx :loss-scale))
        (seq (plist-get ctx :seq)) (vocab (plist-get ctx :vocab))
        (pad-token (or (plist-get ctx :pad-token) 0))
        (optimizer (plist-get ctx :optimizer))
        (compactp (eq (plist-get ctx :transfer-mode) 'compact))
        (cursor 0)
        (total (* epochs (length trajs)))
        (count (length trajs))
        (shuffled-trajs (and shuffle-seed (vconcat trajs)))
        (shuffled-plans (and shuffle-seed (vconcat completion-plans)))
        (shuffle-state shuffle-seed))
    (cl-labels
        ((train-one
          (tr plan)
          (when (>= cursor start-step)
            (when plan
              (setq tr (car plan))
              (nlga-update
               loss-scale
               (if compactp
                   (photon-tensor (list seq 1) (copy-sequence (cadr plan)))
                 (nl-llm-agent-ondevice--expand-row-scales
                  (cadr plan) vocab))))
            (nlga-update
             oh (if compactp
                    (nl-llm-agent-ondevice--index-pad tr seq pad-token)
                  (nl-llm-agent--onehot-pad tr seq vocab pad-token)))
            (nlga-update
             ohtgt
             (if compactp
                 (nl-llm-agent-ondevice--index-pad
                  (nl-llm-agent--shift-pad tr seq pad-token)
                  seq pad-token)
               (nl-llm-agent--onehot-pad
                (nl-llm-agent--shift-pad tr seq pad-token)
                seq vocab pad-token)))
            (setf (plist-get ctx :step) (1+ (plist-get ctx :step)))
            (when (eq optimizer 'adam)
              (nlga-adam-update-t b (plist-get ctx :step)))
            (nlga-step b)
            (when after-step
              (funcall after-step ctx (1+ cursor) total)))
          (setq cursor (1+ cursor))))
      (dotimes (_ epochs)
        (if shuffle-seed
            (let* ((epoch (nl-llm-agent-ondevice--shuffle-epoch
                           count shuffle-state))
                   (indices (cdr epoch)))
              (setq shuffle-state (car epoch))
              (dotimes (position count)
                (let ((index (aref indices position)))
                  (train-one (aref shuffled-trajs index)
                             (aref shuffled-plans index)))))
          ;; Keep the nil-seed traversal over the original caller containers.
          (cl-loop
           for tr in trajs
           for plan in (or completion-plans (make-list count nil)) do
            (train-one tr plan)))))
    total))

(defun nl-llm-agent-ondevice-optimizer-state (ctx)
  "Read back CTX's resident optimizer state, or nil for SGD."
  (when (eq (plist-get ctx :loss-mode) 'completion)
    (unless (plist-get ctx :completion-plan)
      (error "unbound completion context cannot export optimizer state"))
    (nl-llm-agent-ondevice--bound-completion-plan ctx))
  (when (and (plist-get ctx :loss-mode)
             (not (eq (plist-get ctx :loss-mode) 'completion)))
    (error "on-device context has unsupported loss mode"))
  (when (eq (plist-get ctx :optimizer) 'adam)
    (nlga-adam-state (plist-get ctx :b))))

(defun nl-llm-agent-ondevice-restore-training-state
    (ctx step optimizer-state &optional completion-plan)
  "Restore resident CTX optimizer state and completed STEP count.

COMPLETION-PLAN is required for a bound completion context and rejected for
legacy or unbound completion contexts.  All plan and tensor validation occurs
before resident optimizer buffers or the context step are changed."
  (let ((loss-mode (plist-get ctx :loss-mode)))
    (when (and loss-mode (not (eq loss-mode 'completion)))
      (error "on-device context has unsupported loss mode"))
    (cond
     ((eq loss-mode 'completion)
      (unless (plist-get ctx :completion-plan)
        (error "unbound completion context cannot restore training state"))
      (let* ((bound (nl-llm-agent-ondevice--bound-completion-plan ctx))
             (expected
              (nl-llm-agent-ondevice--completion-plan-copy completion-plan))
             (total (* (plist-get bound :epochs)
                       (length (plist-get bound :trajectories)))))
        (unless (equal expected bound)
          (error "completion restore plan does not match context plan"))
        (unless (and (integerp step) (>= step 0) (<= step total))
          (error "completion restore step is outside the training plan"))
        (if (eq (plist-get ctx :optimizer) 'adam)
            (nl-llm-agent-ondevice--validate-adam-state ctx optimizer-state)
          (when optimizer-state
            (error "SGD completion restore cannot accept optimizer tensors")))
        (when (eq (plist-get ctx :optimizer) 'adam)
          (nlga-adam-restore (plist-get ctx :b) optimizer-state))
        (setf (plist-get ctx :step) step)
        ctx))
     (t
      (when completion-plan
        (error "legacy restore cannot accept a completion plan"))
      (unless (and (integerp step) (>= step 0))
        (error "on-device restore step must be non-negative"))
      (if (eq (plist-get ctx :optimizer) 'adam)
          (progn
            (unless (and (listp optimizer-state) optimizer-state)
              (error "on-device Adam restore requires optimizer tensors"))
            (nlga-adam-restore (plist-get ctx :b) optimizer-state))
        (when optimizer-state
          (error "on-device SGD restore cannot accept optimizer tensors")))
      (setf (plist-get ctx :step) step)
      ctx))))

(defun nl-llm-agent-ondevice-snapshot (ctx)
  "Read back CTX weights and optimizer into a durable host-side snapshot."
  (let ((completionp (eq (plist-get ctx :loss-mode) 'completion))
        (plan nil))
    (when completionp
      (unless (plist-get ctx :completion-plan)
        (error "unbound completion context cannot be snapshotted"))
      (setq plan (nl-llm-agent-ondevice--bound-completion-plan ctx)))
    (when (and (plist-get ctx :loss-mode) (not completionp))
      (error "on-device context has unsupported loss mode"))
    (nl-llm-agent-ondevice-sync ctx)
    (append
     (list :model (plist-get ctx :cpu)
           :step (plist-get ctx :step)
           :optimizer (plist-get ctx :optimizer)
           :optimizer-state
           (nl-llm-agent-ondevice-optimizer-state ctx))
     (when plan (list :completion-plan plan)))))

(defun nl-llm-agent-ondevice-sync (ctx)
  "Transfer the GPU-trained weights back into the shared host tensors -- after this
the CPU rollout decodes with the GPU-trained weights."
  (nlga-readback (plist-get ctx :b)))

(defun nl-llm-agent-ondevice-free (ctx)
  (nlga-free (plist-get ctx :b)))

;;;###autoload
(cl-defun nl-llm-agent-improve-ondevice (ctx grammar reward &key (rounds 5) (rollouts 24) (epochs 4) (temp 1.0) (eval-n 20) trace)
  "Run the whole self-improvement loop on-device: each round, ROLLOUT actions.
The CPU view of CTX keeps REWARD scores >= 1 and trains the shared nlga graph
on the GPU, then reads back the trained weights into the CPU view.  The next
round's rollout therefore uses the GPU-trained weights.  Returns the per-round
success-rate list (initial + one per round)."
  (let ((m (plist-get ctx :cpu)))
    (cl-flet ((rate () (let ((ok 0)) (dotimes (_ eval-n)
                          (when (>= (funcall reward (car (nl-llm-agent-p5-rollout m grammar temp))) 1.0) (setq ok (1+ ok))))
                          (/ (float ok) eval-n))))
      ;; a replay buffer of recent wins keeps the GPU training stable -- training
      ;; only on the current round's wins overfits and the rollout oscillates.
      (let ((rates (list (rate))) (replay nil) (cap 48))
        (dotimes (r rounds)
          (dotimes (_ rollouts)
            (let ((roll (nl-llm-agent-p5-rollout m grammar temp)))
              (when (>= (funcall reward (car roll)) 1.0) (push (cdr roll) replay))))
          (when (> (length replay) cap) (setq replay (cl-subseq replay 0 cap)))
          (when replay
            (nl-llm-agent-ondevice-train ctx replay epochs)
            (nl-llm-agent-ondevice-sync ctx))
          (let ((rt (rate))) (push rt rates) (when trace (funcall trace (1+ r) (length replay) rt))))
        (nreverse rates)))))

(provide 'nl-llm-agent-ondevice)
;;; nl-llm-agent-ondevice.el ends here
