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

;;;###autoload
(defun nl-llm-agent-improve-model (&optional dim ff vocab)
  "Build a small trainable char-level model (pav params) for the self-improvement
loop.  Returns a plist with the params + dims; weights mutate in place during
fine-tuning, so the same plist is used for both rollout and training."
  (let* ((dim (or dim 16)) (ff (or ff 32)) (vocab (or vocab nl-llm-agent-char-vocab))
         (sc (/ 1.0 (sqrt (float dim)))))
    (list :wte (nl-llm-agent--p5-p (list vocab dim) 1 sc) :ln1g (nl-llm-agent--p5-c dim 1.0)
          :wq (nl-llm-agent--p5-p (list dim dim) 2 sc) :bq (nl-llm-agent--p5-c dim 0.0)
          :wk (nl-llm-agent--p5-p (list dim dim) 3 sc) :bk (nl-llm-agent--p5-c dim 0.0)
          :wv (nl-llm-agent--p5-p (list dim dim) 4 sc) :bv (nl-llm-agent--p5-c dim 0.0)
          :wo (nl-llm-agent--p5-p (list dim dim) 5 sc) :bo (nl-llm-agent--p5-c dim 0.0) :ln2g (nl-llm-agent--p5-c dim 1.0)
          :wg (nl-llm-agent--p5-p (list ff dim) 6 sc) :bg (nl-llm-agent--p5-c ff 0.0)
          :wu (nl-llm-agent--p5-p (list ff dim) 7 sc) :bu (nl-llm-agent--p5-c ff 0.0)
          :wd (nl-llm-agent--p5-p (list dim ff) 8 sc) :bd (nl-llm-agent--p5-c dim 0.0)
          :lnfg (nl-llm-agent--p5-c dim 1.0) :wh (nl-llm-agent--p5-p (list vocab dim) 9 sc) :bh (nl-llm-agent--p5-c vocab 0.0)
          :dim dim :ff ff :vocab vocab)))

(defun nl-llm-agent--p5-params (m)
  (mapcar (lambda (k) (plist-get m k)) '(:wte :ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd :lnfg :wh :bh)))

(defun nl-llm-agent--p5-forward (m toks &optional targets)
  "Modern-block forward over TOKS (a list of ids).  Returns the softmax-CE loss
against TARGETS (a vector) if given, else the logits pav."
  (photon-autograd-reset-tape)
  (let* ((dim (plist-get m :dim)) (sc (/ 1.0 (sqrt (float dim)))) (seq (length toks)) (heads 1)
         (mask (let ((md (make-vector (* seq seq) 0.0)) (i 0))
                 (while (< i seq) (let ((j (1+ i))) (while (< j seq) (aset md (+ (* i seq) j) -1.0e30) (setq j (1+ j)))) (setq i (1+ i)))
                 (photon-autograd-const (photon-tensor (list seq seq) md))))
         (x (photon-autograd-embedding (plist-get m :wte) toks dim))
         (a (nl-llm-ag-rmsnorm x (plist-get m :ln1g)))
         (q (nl-llm-ag-rope (photon-autograd-linear a (plist-get m :wq) (plist-get m :bq)) heads))
         (k (nl-llm-ag-rope (photon-autograd-linear a (plist-get m :wk) (plist-get m :bk)) heads))
         (v (photon-autograd-linear a (plist-get m :wv) (plist-get m :bv)))
         (s (photon-autograd-scale (photon-autograd-matmul q (photon-autograd-transpose k)) sc))
         (p (photon-autograd-softmax-rows (photon-autograd-add s mask)))
         (ctx (photon-autograd-matmul p v))
         (x1 (photon-autograd-add x (photon-autograd-linear ctx (plist-get m :wo) (plist-get m :bo))))
         (bb (nl-llm-ag-rmsnorm x1 (plist-get m :ln2g)))
         (mout (nl-llm-ag-swiglu bb (plist-get m :wg) (plist-get m :bg) (plist-get m :wu) (plist-get m :bu) (plist-get m :wd) (plist-get m :bd)))
         (x2 (photon-autograd-add x1 mout))
         (xf (nl-llm-ag-rmsnorm x2 (plist-get m :lnfg)))
         (logits (photon-autograd-linear xf (plist-get m :wh) (plist-get m :bh))))
    (if targets (photon-autograd-softmax-ce logits targets) logits)))

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

(defun nl-llm-agent-p5-rollout (m grammar temp)
  "Generate one action under GRAMMAR by sampling the model M's free positions at
temperature TEMP.  Returns (EMITTED . TOKS) where TOKS is the char-id list."
  (let ((emitted "") (toks nil))
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

(provide 'nl-llm-agent-improve)
;;; nl-llm-agent-improve.el ends here
