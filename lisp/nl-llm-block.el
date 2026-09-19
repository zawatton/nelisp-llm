;;; nl-llm-block.el --- modern (Llama/Qwen-style) transformer block + model  -*- lexical-binding: t; -*-

;; Composes the nelisp-llm primitives into a pre-norm transformer:
;;   x1 = x  + GQA(RMSNorm(x))
;;   x2 = x1 + FFN(RMSNorm(x1))          ; FFN = MoE if the block has a router,
;;                                         else a single SwiGLU
;; and stacks blocks into a model: embed -> blocks -> RMSNorm -> head.
;; Position information comes from RoPE inside attention (no learned wpe).

;;; Code:

(require 'nl-llm-compat)
(require 'photon-tensor)
(require 'nl-llm-arch)
(require 'nl-llm-attn)
(require 'nl-llm-moe)

;;;###autoload
(defun nl-llm-block (x block heads kv-heads &optional rope-base head-dim)
  "Run one pre-norm transformer BLOCK on X (seq x dim).
BLOCK holds :ln1g :ln2g (RMSNorm gains), attention :wq :wk :wv :wo, and a
feed-forward: either (:router :experts :top-k) for MoE, or (:wg :wu :wd) for
a single SwiGLU.  HEAD-DIM is the per-head width to use when BLOCK carries no
:head-dim of its own; both absent means (/ dim heads)."
  (let* ((a (nl-llm-rmsnorm x (plist-get block :ln1g)))
         (x1 (photon-tensor-add
              x (nl-llm-gqa a block heads kv-heads rope-base head-dim)))
         (b (nl-llm-rmsnorm x1 (plist-get block :ln2g)))
         (ffn (if (plist-get block :router)
                  (nl-llm-moe b (plist-get block :router) (plist-get block :experts)
                              (or (plist-get block :top-k) 1))
                (nl-llm-swiglu b (plist-get block :wg) (plist-get block :wu)
                               (plist-get block :wd)))))
    (photon-tensor-add x1 ffn)))

;;;###autoload
(defun nl-llm-model-forward (model tokens)
  "Run MODEL over TOKENS (list of ids); return (seq x vocab) logits.
MODEL holds :wte (vocab x dim), :blocks (list), :lnf (final RMSNorm gain),
:head (vocab x dim), :dim, :heads and optional :kv-heads, :rope-base,
:head-dim.  :head-dim is the per-head width for blocks that do not state their
own, and defaults to (/ dim heads) -- the Qwen3 family sets it independently
of dim and heads."
  (let* ((dim (plist-get model :dim)) (heads (plist-get model :heads))
         (kvh (or (plist-get model :kv-heads) heads))
         (rb (plist-get model :rope-base))
         (hd (plist-get model :head-dim))
         (x (photon-tensor-embedding (plist-get model :wte) tokens dim)))
    (dolist (blk (plist-get model :blocks))
      (setq x (nl-llm-block x blk heads kvh rb hd)))
    (photon-tensor-linear (nl-llm-rmsnorm x (plist-get model :lnf))
                          (plist-get model :head))))

;;;###autoload
(defun nl-llm-sample (logits base vocab &optional temp topk)
  "Sample a token id from LOGITS[BASE .. BASE+VOCAB) with TEMP (default 1.0) and
TOP-K (default 0 = keep all).  Temperature scales the logits, top-k keeps only
the K largest before softmax, then a categorical draw is taken.  TEMP at/near 0
or TOP-K = 1 reduces to greedy argmax.  Uses the global RNG -- seed with
`(random \"...\")' for reproducibility."
  (let* ((tp (max 1.0e-6 (or temp 1.0))) (k (or topk 0))
         (lg (make-vector vocab 0.0))
         (keep (make-vector vocab t))
         (i 0))
    (while (< i vocab) (aset lg i (/ (aref logits (+ base i)) tp)) (setq i (1+ i)))
    (when (and (> k 0) (< k vocab))
      (let* ((thresh (nth (1- k) (sort (append lg nil) #'>))) (j 0))
        (while (< j vocab)
          (aset keep j (>= (aref lg j) thresh))
          (setq j (1+ j)))))
    (let ((mx nil) (j 0))
      (while (< j vocab)
        (when (and (aref keep j)
                   (or (null mx) (> (aref lg j) mx)))
          (setq mx (aref lg j)))
        (setq j (1+ j)))
      (let ((s 0.0) (j2 0))
        (while (< j2 vocab)
          (if (aref keep j2)
              (progn
                (aset lg j2 (exp (- (aref lg j2) mx)))
                (setq s (+ s (aref lg j2))))
            (aset lg j2 0.0))
          (setq j2 (1+ j2)))
        (let ((r (* s (/ (float (random 1000000)) 1000000.0)))
              (acc 0.0) (j3 0) (pick nil) (done nil))
          (while (and (< j3 vocab) (not done))
            (when (aref keep j3)
              (setq pick j3)
              (setq acc (+ acc (aref lg j3)))
              (when (> acc r) (setq done t)))
            (setq j3 (1+ j3)))
          pick)))))

;;;###autoload
(defun nl-llm-dropout-mask (shape p)
  "Inverted-dropout mask tensor of SHAPE: each element is 1/(1-P) with probability
1-P, else 0, so E[mask]=1 and no train/eval scale mismatch.  Uses the global RNG
\(seed with `(random \"...\")' for reproducibility).  Refresh per training step and
feed it to the dropout mask rt with `nlga-update'; for eval use an all-ones mask
\(p=0) or skip dropout."
  (let* ((keep (- 1.0 p)) (inv (/ 1.0 keep)) (n 1))
    (dolist (d shape) (setq n (* n d)))
    (photon-tensor shape
                   (let ((v (make-vector n 0.0)) (i 0))
                     (while (< i n)
                       (when (< (/ (float (random 1000000)) 1000000.0) keep) (aset v i inv))
                       (setq i (1+ i)))
                     v))))

(provide 'nl-llm-block)
;;; nl-llm-block.el ends here
