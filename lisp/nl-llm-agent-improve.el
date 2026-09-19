;;; nl-llm-agent-improve.el --- the self-improvement loop (STaR)  -*- lexical-binding: t; -*-

;; Phase 5 of the agent harness (docs/design/05-agent-harness.org): close the loop
;; so the model gets better at acting by training on its OWN successful actions.
;;
;;   model --(constrained sampling)--> actions --(reward)--> keep successes
;;     --(next-token CE fine-tune via the CPU autograd)--> better model --> repeat
;;
;; This is expert-iteration / STaR in miniature: each round the model samples K
;; trajectories under the action grammar, a reward function scores them, the
;; successful ones become supervised next-token training data, and the model is
;; fine-tuned (real backprop, photon-autograd + nl-llm-autograd) to make those
;; actions more likely.  The measured success rate climbs round over round --
;; self-improvement with no external labels, only the model's own wins.
;;
;; Same modern block (RMSNorm + RoPE attention + SwiGLU + tied-ish head) as
;; examples/train-modern.el, so the trained weights are a real nelisp-llm model.
;; CPU; deterministic under a seeded RNG.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-tokenizer)

;; ---- a small trainable model (pav params) ----------------------------------

(defun nl-llm-agent--p5-p (shape seed scale)
  (let ((n 1)) (dolist (d shape) (setq n (* n d)))
    (photon-autograd-const
     (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
       (while (< i n) (aset v i (* scale 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v)))))
(defun nl-llm-agent--p5-c (n val) (photon-autograd-const (photon-tensor (list n) (make-vector n val))))

(defun nl-llm-agent--p5-block (dim ff s0)
  "One stacked-block weight plist (for `nl-llm-ag-block', MHA so kv = heads)."
  (let ((sc (/ 1.0 (sqrt (float dim)))))
    (list :ln1g (nl-llm-agent--p5-c dim 1.0)
          :wq (nl-llm-agent--p5-p (list dim dim) (+ s0 1) sc) :bq (nl-llm-agent--p5-c dim 0.0)
          :wk (nl-llm-agent--p5-p (list dim dim) (+ s0 2) sc) :bk (nl-llm-agent--p5-c dim 0.0)
          :wv (nl-llm-agent--p5-p (list dim dim) (+ s0 3) sc) :bv (nl-llm-agent--p5-c dim 0.0)
          :wo (nl-llm-agent--p5-p (list dim dim) (+ s0 4) sc) :bo (nl-llm-agent--p5-c dim 0.0) :ln2g (nl-llm-agent--p5-c dim 1.0)
          :wg (nl-llm-agent--p5-p (list ff dim) (+ s0 5) sc) :bg (nl-llm-agent--p5-c ff 0.0)
          :wu (nl-llm-agent--p5-p (list ff dim) (+ s0 6) sc) :bu (nl-llm-agent--p5-c ff 0.0)
          :wd (nl-llm-agent--p5-p (list dim ff) (+ s0 7) sc) :bd (nl-llm-agent--p5-c dim 0.0))))

(defun nl-llm-agent-improve--model-tokenizer (model where)
  "Return MODEL's canonical tokenizer id after validation for WHERE."
  (let ((tokenizer
         (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
        (vocab (plist-get model :vocab)))
    (unless (and (integerp vocab)
                 (= vocab (nl-llm-agent-tokenizer-vocab tokenizer)))
      (error "%s vocab does not match tokenizer %s" where tokenizer))
    tokenizer))

;;;###autoload
(defun nl-llm-agent-improve-model
    (&optional dim ff vocab nblocks heads tokenizer)
  "Build a small trainable token model for the self-improvement loop.
NBLOCKS stacked GQA/SwiGLU blocks default to one and HEADS defaults to one.
TOKENIZER defaults to the legacy ASCII tokenizer, and VOCAB must match it.
Weights mutate in place, so one plist serves both rollout and training."
  (let* ((dim (or dim 24)) (ff (or ff dim))
         (tokenizer (nl-llm-agent-tokenizer-id tokenizer))
         (vocab (or vocab (nl-llm-agent-tokenizer-vocab tokenizer)))
         (nblocks (or nblocks 1)) (heads (or heads 1)) (sc (/ 1.0 (sqrt (float dim)))))
    (unless (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
      (error "P5 model vocab does not match tokenizer %s" tokenizer))
    (list :wte (nl-llm-agent--p5-p (list vocab dim) 1 sc)
          :blocks (cl-loop for i below nblocks collect (nl-llm-agent--p5-block dim ff (* 20 (1+ i))))
          :lnfg (nl-llm-agent--p5-c dim 1.0) :wh (nl-llm-agent--p5-p (list vocab dim) 9 sc) :bh (nl-llm-agent--p5-c vocab 0.0)
          :dim dim :ff ff :vocab vocab :heads heads :nblocks nblocks
          :tokenizer tokenizer)))

(defun nl-llm-agent--p5-params (m)
  (append (list (plist-get m :wte))
          (cl-loop for blk in (plist-get m :blocks) append
                   (mapcar (lambda (k) (plist-get blk k))
                           '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd)))
          (list (plist-get m :lnfg) (plist-get m :wh) (plist-get m :bh))))

(defun nl-llm-agent--p5-forward (m toks &optional targets mask)
  "Stacked-block forward over TOKS (a list of ids).  Returns the softmax-CE loss
against TARGETS (a vector) if given, else the logits pav.  When MASK is non-nil,
only rows selected by that binary vector contribute to the mean loss."
  (photon-autograd-reset-tape)
  (let* ((dim (plist-get m :dim)) (heads (plist-get m :heads))
         (x (photon-autograd-embedding (plist-get m :wte) toks dim)))
    (dolist (blk (plist-get m :blocks)) (setq x (nl-llm-ag-block x blk heads heads)))
    (let* ((xf (nl-llm-ag-rmsnorm x (plist-get m :lnfg)))
           (logits (photon-autograd-linear xf (plist-get m :wh) (plist-get m :bh))))
      (if targets
          (if mask
              (nl-llm-ag-masked-softmax-ce logits targets mask)
            (photon-autograd-softmax-ce logits targets))
        logits))))

(defun nl-llm-agent--p5-last-logits (m toks)
  "Logit vector for the position AFTER TOKS (the last row of the forward)."
  (let* ((lg (nl-llm-agent--p5-forward m toks)) (data (photon-tensor-data (pav-value lg)))
         (vocab (plist-get m :vocab)) (base (* (1- (length toks)) vocab)) (out (make-vector vocab 0.0)))
    (dotimes (i vocab) (aset out i (aref data (+ base i)))) out))

;; ---- rollout (constrained SAMPLING), reward, fine-tune ---------------------

(defun nl-llm-agent--sample-among (logits ids temp)
  "Sample an id from IDS by softmax(LOGITS/TEMP) over those ids (uses `random')."
  (let* ((mx (apply #'max (mapcar (lambda (i) (aref logits i)) ids)))
         (ws (mapcar (lambda (i) (exp (/ (- (aref logits i) mx) (max 1e-6 temp)))) ids))
         (z (apply #'+ ws)) (r (* (/ (float (random 1000000)) 1000000.0) z)) (c 0.0) (pick (car (last ids))))
    (cl-loop for i in ids for w in ws do (setq c (+ c w)) (when (<= r c) (setq pick i) (cl-return)))
    pick))

(defun nl-llm-agent--sample-utf8-character
    (model chars tokens tokenizer temp)
  "Sample one of CHARS by UTF-8 byte prefix from MODEL.
Return (CHAR . TOKENS) after appending the selected character's complete byte
sequence to the existing training TOKENS."
  (let ((tail (append chars nil))
        (candidates nil)
        (depth 0))
    (unless tail
      (error "P5 rollout received an empty allowed character set"))
    (dolist (char tail)
      (unless (integerp char)
        (error "P5 rollout candidate is not a character: %S" char))
      (push (cons char
                  (nl-llm-agent-tokenizer-encode
                   (string char) tokenizer))
            candidates))
    (setq candidates (nreverse candidates))
    (catch 'selected
      (while candidates
        (let ((ids nil))
          (dolist (candidate candidates)
            (let ((encoded (cdr candidate)))
              (unless (< depth (length encoded))
                (error "P5 rollout encountered an empty candidate"))
              (let ((id (nth depth encoded)))
                (unless (memq id ids)
                  (push id ids)))))
          (let* ((logits (nl-llm-agent--p5-last-logits model tokens))
                 (id
                  (nl-llm-agent--sample-among
                   logits (nreverse ids) temp)))
            (setq tokens (append tokens (list id))
                  candidates
                  (delq nil
                        (mapcar
                         (lambda (candidate)
                           (and (= (nth depth (cdr candidate)) id)
                                candidate))
                         candidates))
                  depth (1+ depth))
            (let ((complete
                   (cl-find-if
                    (lambda (candidate)
                      (= (length (cdr candidate)) depth))
                    candidates)))
              (when complete
                (throw 'selected (cons (car complete) tokens)))))))
      (error "P5 rollout could not select an allowed character"))))

(defun nl-llm-agent-p5-rollout (m grammar temp &optional prompt)
  "Generate one action under GRAMMAR by sampling M's free positions.
TEMP controls sampling.  PROMPT, when non-nil, is fed as context first, so
free-slot choices are conditioned on it.  Return (EMITTED . TOKS), where TOKS
is the prompt plus action training sequence."
  (let* ((emitted "")
         (tokenizer
          (nl-llm-agent-improve--model-tokenizer m "P5 rollout model"))
         (toks
          (when prompt
            (nl-llm-agent-tokenizer-encode prompt tokenizer))))
    (catch 'done
      (while t
        (let ((g (funcall grammar emitted)))
          (pcase g
            (:stop (throw 'done nil))
            (`(:force ,ch)
             (setq emitted (concat emitted (string ch))
                   toks
                   (append
                    toks
                    (nl-llm-agent-tokenizer-encode
                     (string ch) tokenizer))))
            (`(:allow ,chars)
             (if (equal tokenizer nl-llm-agent-tokenizer-ascii)
                 ;; Keep the legacy candidate order and random draw exactly.
                 (let* ((logits (nl-llm-agent--p5-last-logits m toks))
                        (ids
                         (mapcar
                          #'nl-llm-agent--char->id (append chars nil)))
                        (id (nl-llm-agent--sample-among logits ids temp)))
                   (setq emitted
                         (concat emitted
                                 (string (nl-llm-agent--id->char id)))
                         toks (append toks (list id))))
               (let ((choice
                      (nl-llm-agent--sample-utf8-character
                       m chars toks tokenizer temp)))
                 (setq emitted (concat emitted (string (car choice)))
                       toks (cdr choice)))))))))
    (cons emitted toks)))

(defun nl-llm-agent-p5-finetune (m examples lr epochs)
  "Fine-tune M on token-id EXAMPLES using next-token cross-entropy."
  (let ((params (nl-llm-agent--p5-params m)))
    (dotimes (_ epochs)
      (dolist (toks examples)
        (when (> (length toks) 1)
          (let ((loss (nl-llm-agent--p5-forward m (butlast toks) (apply #'vector (cdr toks)))))
            (photon-autograd-zero-grad params)
            (photon-autograd-backward loss)
            (photon-autograd-sgd params lr)))))))

(defun nl-llm-agent-p5-success-rate (m grammar reward-fn n temp)
  "Fraction of N sampled rollouts whose reward (REWARD-FN EMITTED) is >= 1."
  (let ((ok 0)) (dotimes (_ n) (when (>= (funcall reward-fn (car (nl-llm-agent-p5-rollout m grammar temp))) 1.0) (setq ok (1+ ok))))
       (/ (float ok) n)))

;;;###autoload
(cl-defun nl-llm-agent-improve (m grammar reward-fn &key (rounds 3) (rollouts 24) (lr 0.3) (epochs 2) (temp 1.0) (eval-n 30) trace)
  "Run the self-improvement loop on M.
Each round samples ROLLOUTS actions under GRAMMAR, keeps those whose REWARD-FN
score is at least one, fine-tunes M on them, and evaluates EVAL-N samples.
Return the initial success rate followed by one rate per round."
  (let ((rates (list (nl-llm-agent-p5-success-rate m grammar reward-fn eval-n temp))))
    (dotimes (r rounds)
      (let ((succ nil))
        (dotimes (_ rollouts)
          (let ((roll (nl-llm-agent-p5-rollout m grammar temp)))
            (when (>= (funcall reward-fn (car roll)) 1.0) (push (cdr roll) succ))))
        (when succ (nl-llm-agent-p5-finetune m succ lr epochs))
        (let ((rate (nl-llm-agent-p5-success-rate m grammar reward-fn eval-n temp)))
          (push rate rates)
          (when trace (funcall trace (1+ r) (length succ) rate)))))
    (nreverse rates)))

;; ---- richer reward: a multi-task curriculum, spec-conditioned, multi-case ----

(cl-defun nl-llm-agent-improve-tasks (m grammar tasks &key (rounds 6) (rollouts 32) (lr 0.4) (epochs 2) (temp 1.0) (eval-n 12) trace)
  "Self-improve M over TASKS of (PROMPT . REWARD-FN) pairs.
Each round synthesizes prompt-conditioned actions and fine-tunes on successful
prompt plus action trajectories.  Shared skills can transfer across tasks.
Return the initial mean success rate followed by one rate per round."
  (cl-flet ((avg-rate ()
              (/ (apply #'+ (mapcar (lambda (tk)
                                      (let ((ok 0)) (dotimes (_ eval-n)
                                        (when (>= (funcall (cdr tk) (car (nl-llm-agent-p5-rollout m grammar temp (car tk)))) 1.0) (setq ok (1+ ok))))
                                        (/ (float ok) eval-n)))
                                    tasks))
                 (length tasks))))
    ;; A replay buffer of all solutions seen so far (capped, recent-first) keeps the
    ;; fine-tune BALANCED across tasks -- without it the model overfits whichever
    ;; task it solved most this round and never learns the conditional mapping.
    (let ((rates (list (avg-rate))) (replay nil) (cap 64))
      (dotimes (r rounds)
        (let ((ti 0))
          (dotimes (_ rollouts)
            (let* ((tk (nth (mod ti (length tasks)) tasks)) (roll (nl-llm-agent-p5-rollout m grammar temp (car tk))))
              (when (>= (funcall (cdr tk) (car roll)) 1.0) (push (cdr roll) replay))
              (setq ti (1+ ti))))
          (when (> (length replay) cap) (setq replay (cl-subseq replay 0 cap)))
          (when replay (nl-llm-agent-p5-finetune m replay lr epochs))
          (let ((rate (avg-rate))) (push rate rates) (when trace (funcall trace (1+ r) (length replay) rate)))))
      (nreverse rates))))

(provide 'nl-llm-agent-improve)
;;; nl-llm-agent-improve.el ends here
