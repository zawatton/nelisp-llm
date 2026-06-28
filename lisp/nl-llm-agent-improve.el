;;; nl-llm-agent-improve.el --- the self-improvement loop (STaR)  -*- lexical-binding: t; -*-

;; Phase 5 of the agent harness (docs/design/05-agent-harness.org): close the loop
;; so the model gets better at acting by training on its OWN successful actions.
;;
;;   model --(constrained sampling)--> actions --(reward)--> keep successes
;;     --(next-char CE fine-tune via the CPU autograd)--> better model --> repeat
;;
;; This is expert-iteration / STaR in miniature: each round the model samples K
;; trajectories under the action grammar, a reward function scores them, the
;; successful ones become supervised next-char training data, and the model is
;; fine-tuned (real backprop, photon-autograd + nl-llm-autograd) to make those
;; actions more likely.  The measured success rate climbs round over round --
;; self-improvement with no external labels, only the model's own wins.
;;
;; Same modern block (RMSNorm + RoPE attention + SwiGLU + tied-ish head) as
;; examples/train-modern.el, so the trained weights are a real nelisp-llm model.
;; CPU; deterministic under a seeded RNG.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)

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

;;;###autoload
(defun nl-llm-agent-improve-model (&optional dim ff vocab nblocks heads)
  "Build a small trainable char-level model (pav params) for the self-improvement
loop: NBLOCKS stacked GQA/SwiGLU blocks (default 2 -- enough depth to LEARN to copy
from the spec prompt into the action, which a single layer cannot) with HEADS heads
(default 2).  Weights mutate in place, so one plist serves both rollout and
training.  Returns a plist of params + dims."
  (let* ((dim (or dim 24)) (ff (or ff dim)) (vocab (or vocab nl-llm-agent-char-vocab))
         (nblocks (or nblocks 1)) (heads (or heads 1)) (sc (/ 1.0 (sqrt (float dim)))))
    (list :wte (nl-llm-agent--p5-p (list vocab dim) 1 sc)
          :blocks (cl-loop for i below nblocks collect (nl-llm-agent--p5-block dim ff (* 20 (1+ i))))
          :lnfg (nl-llm-agent--p5-c dim 1.0) :wh (nl-llm-agent--p5-p (list vocab dim) 9 sc) :bh (nl-llm-agent--p5-c vocab 0.0)
          :dim dim :ff ff :vocab vocab :heads heads :nblocks nblocks)))

(defun nl-llm-agent--p5-params (m)
  (append (list (plist-get m :wte))
          (cl-loop for blk in (plist-get m :blocks) append
                   (mapcar (lambda (k) (plist-get blk k))
                           '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd)))
          (list (plist-get m :lnfg) (plist-get m :wh) (plist-get m :bh))))

(defun nl-llm-agent--p5-forward (m toks &optional targets)
  "Stacked-block forward over TOKS (a list of ids).  Returns the softmax-CE loss
against TARGETS (a vector) if given, else the logits pav."
  (photon-autograd-reset-tape)
  (let* ((dim (plist-get m :dim)) (heads (plist-get m :heads))
         (x (photon-autograd-embedding (plist-get m :wte) toks dim)))
    (dolist (blk (plist-get m :blocks)) (setq x (nl-llm-ag-block x blk heads heads)))
    (let* ((xf (nl-llm-ag-rmsnorm x (plist-get m :lnfg)))
           (logits (photon-autograd-linear xf (plist-get m :wh) (plist-get m :bh))))
      (if targets (photon-autograd-softmax-ce logits targets) logits))))

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

(defun nl-llm-agent-p5-rollout (m grammar temp &optional prompt)
  "Generate one action under GRAMMAR by sampling the model M's free positions at
temperature TEMP.  If PROMPT is given it is fed as context first, so the free-slot
choices are CONDITIONED on it (synthesis from a spec).  Returns (EMITTED . TOKS):
EMITTED is the action text; TOKS is the PROMPT+action char-id list (the training
sequence, so fine-tuning learns P(action | prompt))."
  (let ((emitted "") (toks (when prompt (mapcar #'nl-llm-agent--char->id (append prompt nil)))))
    (catch 'done
      (while t
        (let ((g (funcall grammar emitted)))
          (pcase g
            (:stop (throw 'done nil))
            (`(:force ,ch) (setq emitted (concat emitted (string ch)) toks (append toks (list (nl-llm-agent--char->id ch)))))
            (`(:allow ,chars)
             (let* ((logits (nl-llm-agent--p5-last-logits m toks))
                    (ids (mapcar #'nl-llm-agent--char->id (append chars nil)))
                    (id (nl-llm-agent--sample-among logits ids temp)))
               (setq emitted (concat emitted (string (nl-llm-agent--id->char id))) toks (append toks (list id)))))))))
    (cons emitted toks)))

(defun nl-llm-agent-p5-finetune (m examples lr epochs)
  "Fine-tune M on EXAMPLES (each a char-id list of a successful action) by
next-char cross-entropy -- raising the likelihood of those actions."
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
  "Run the self-improvement loop on model M: each round, sample ROLLOUTS actions
under GRAMMAR, keep those REWARD-FN scores >= 1, fine-tune M on them, and measure
the success rate over EVAL-N samples.  Returns the list of success rates (initial
+ one per round) -- they should climb as the model learns from its own wins."
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
  "Self-improve M over a bank of TASKS (each a cons (PROMPT . REWARD-FN)): every
round the model synthesises an action for each task CONDITIONED on the task's
prompt, the reward (e.g. a multi-test-case grader) scores it, and the model is
fine-tuned on all the successful prompt+action trajectories at once.  Because the
useful skill (read the spec, fill the template) is shared, learning transfers
across tasks.  Returns the list of MEAN success rates (initial + one per round)."
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
