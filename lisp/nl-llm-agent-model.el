;;; nl-llm-agent-model.el --- wire an nelisp-llm model in as the agent policy  -*- lexical-binding: t; -*-

;; Phase 4 of the agent harness (docs/design/05-agent-harness.org): drive the
;; agent loop with a REAL nelisp-llm model instead of a scripted policy.
;;
;; A small model cannot free-form a well-formed action block reliably, so the
;; bridge generates under a GRAMMAR with constrained decoding: at every position
;; the grammar says which characters are legal, the model's logits are masked to
;; that set, and the argmax among the legal characters is taken.  Fixed scaffold
;; characters are forced; only the variable slots are chosen by the model.  The
;; result is ALWAYS parseable by `nl-llm-agent--parse', however weak the model --
;; the model picks the content, the grammar guarantees the structure.
;;
;; The generation engine is just `nl-llm-decode-step' (CPU, the same transformer
;; decode the project trains); swap in the GPU/integrated decode by passing a
;; different STEP-FN -- the interface is (token-id -> next-token-logit-vector).

;;; Code:

(require 'cl-lib)
(require 'subr-x)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-agent)

;; ---- character tokenizer (printable ASCII + newline) -----------------------
;; A self-contained char vocab (vocab=96): id 0..94 = chars 32..126, id 95 = \n.
;; (Production would use the BPE tokenizer; char level keeps the agent PoC
;; deterministic and dependency-free.)

(defconst nl-llm-agent-char-vocab 96 "Size of the char tokenizer vocabulary.")

(defun nl-llm-agent--char->id (c)
  "Map character C to a token id in [0, 96)."
  (cond ((= c ?\n) 95)
        ((and (>= c 32) (<= c 126)) (- c 32))
        (t 0)))                                 ; everything else -> space

(defun nl-llm-agent--id->char (id)
  "Map token ID back to a character."
  (if (= id 95) ?\n (+ id 32)))

;; ---- constrained decoding --------------------------------------------------

(defun nl-llm-agent--argmax-among (logits ids)
  "Index in IDS with the largest LOGITS value (IDS already filtered in range)."
  (let ((best (car ids)) (bv (aref logits (car ids))))
    (dolist (i (cdr ids)) (when (> (aref logits i) bv) (setq bv (aref logits i) best i)))
    best))

(defun nl-llm-agent-constrained-generate (init-logits step-fn grammar)
  "Generate a string under GRAMMAR using INIT-LOGITS as the first next-token
distribution and STEP-FN (a function ID -> next-token-logit-vector) to advance.
GRAMMAR is a function EMITTED -> :stop | (:force CHAR) | (:allow CHARS); forced
characters are emitted as-is, allowed positions take the argmax among the legal
characters' ids.  Returns the generated string (always grammar-valid)."
  (let ((logits init-logits) (emitted ""))
    (catch 'done
      (while t
        (let ((g (funcall grammar emitted)))
          (pcase g
            (:stop (throw 'done emitted))
            (`(:force ,ch)
             (setq emitted (concat emitted (string ch)))
             (setq logits (funcall step-fn (nl-llm-agent--char->id ch))))
            (`(:allow ,chars)
             (let* ((ids (delq nil (mapcar (lambda (c)
                                             (let ((i (nl-llm-agent--char->id c)))
                                               (and (< i (length logits)) i)))
                                           (append chars nil))))
                    (id (nl-llm-agent--argmax-among logits ids))
                    (ch (nl-llm-agent--id->char id)))
               (setq emitted (concat emitted (string ch)))
               (setq logits (funcall step-fn id))))))))))

;; ---- action grammars -------------------------------------------------------

(defun nl-llm-agent-grammar-message (n &optional allow)
  "Grammar forcing a valid Elisp action: ```elisp (message \"<N model chars>\")```.
The N free characters are chosen by the model from ALLOW (default a safe set with
no quote/backslash/newline, so the emitted Elisp always parses)."
  (let ((pre "```elisp\n(message \"") (post "\")\n```")
        (safe (or allow "abcdefghijklmnopqrstuvwxyz0123456789 ")))
    (lambda (emitted)
      (let ((p (length emitted)) (lp (length "```elisp\n(message \"")))
        (cond
         ((< p lp) (list :force (aref pre p)))
         ((< p (+ lp n)) (list :allow safe))
         ((< p (+ lp n (length post))) (list :force (aref post (- p lp n))))
         (t :stop))))))

;; ---- the model policy ------------------------------------------------------

(defun nl-llm-agent--render (messages)
  "Render MESSAGES (list of (ROLE . CONTENT)) into a single prompt string."
  (concat (mapconcat (lambda (m) (format "%s: %s" (car m) (cdr m))) messages "\n") "\nassistant:\n"))

;;;###autoload
(defun nl-llm-agent-model-step-fn (model caches)
  "Return a STEP-FN (ID -> logit-vector) over MODEL (a plist with :blocks :wte
:lnfg :bh :dim) using mutable per-block CACHES (CPU `nl-llm-decode-step')."
  (let ((blocks (plist-get model :blocks)) (wte (plist-get model :wte))
        (lnfg (plist-get model :lnfg)) (bh (plist-get model :bh)) (dim (plist-get model :dim)))
    (lambda (id) (nl-llm-decode-step id blocks caches wte lnfg bh dim))))

;;;###autoload
(defun nl-llm-agent-model-policy (model grammar &optional maxseq)
  "Return an agent policy (MESSAGES -> text) that drives MODEL's generation under
GRAMMAR with constrained decoding.  MODEL is a plist (:blocks :wte :lnfg :bh :dim
:heads :kvh).  Each call feeds the rendered history through a fresh KV cache, then
constrained-generates the action -- guaranteed parseable however weak the model."
  (let ((dim (plist-get model :dim)) (heads (plist-get model :heads)) (kvh (plist-get model :kvh)))
    (lambda (messages)
      (let* ((caches (mapcar (lambda (_) (nl-llm-dcache-new (or maxseq 1024) dim heads kvh)) (plist-get model :blocks)))
             (step (nl-llm-agent-model-step-fn model caches))
             (prompt (nl-llm-agent--render messages))
             (logits nil))
        (dolist (ch (string-to-list prompt)) (setq logits (funcall step (nl-llm-agent--char->id ch))))
        (unless logits (setq logits (funcall step 0)))
        (nl-llm-agent-constrained-generate logits step grammar)))))

(provide 'nl-llm-agent-model)
;;; nl-llm-agent-model.el ends here
