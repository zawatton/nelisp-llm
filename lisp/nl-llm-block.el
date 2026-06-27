;;; nl-llm-block.el --- modern (Llama/Qwen-style) transformer block + model  -*- lexical-binding: t; -*-

;; Composes the nelisp-llm primitives into a pre-norm transformer:
;;   x1 = x  + GQA(RMSNorm(x))
;;   x2 = x1 + FFN(RMSNorm(x1))          ; FFN = MoE if the block has a router,
;;                                         else a single SwiGLU
;; and stacks blocks into a model: embed -> blocks -> RMSNorm -> head.
;; Position information comes from RoPE inside attention (no learned wpe).

;;; Code:

(require 'photon-tensor)
(require 'nl-llm-arch)
(require 'nl-llm-attn)
(require 'nl-llm-moe)

;;;###autoload
(defun nl-llm-block (x block heads kv-heads &optional rope-base)
  "Run one pre-norm transformer BLOCK on X (seq x dim).
BLOCK holds :ln1g :ln2g (RMSNorm gains), attention :wq :wk :wv :wo, and a
feed-forward: either (:router :experts :top-k) for MoE, or (:wg :wu :wd) for
a single SwiGLU."
  (let* ((a (nl-llm-rmsnorm x (plist-get block :ln1g)))
         (x1 (photon-tensor-add x (nl-llm-gqa a block heads kv-heads rope-base)))
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
:head (vocab x dim), :dim, :heads and optional :kv-heads, :rope-base."
  (let* ((dim (plist-get model :dim)) (heads (plist-get model :heads))
         (kvh (or (plist-get model :kv-heads) heads))
         (rb (plist-get model :rope-base))
         (x (photon-tensor-embedding (plist-get model :wte) tokens dim)))
    (dolist (blk (plist-get model :blocks))
      (setq x (nl-llm-block x blk heads kvh rb)))
    (photon-tensor-linear (nl-llm-rmsnorm x (plist-get model :lnf))
                          (plist-get model :head))))

(provide 'nl-llm-block)
;;; nl-llm-block.el ends here
