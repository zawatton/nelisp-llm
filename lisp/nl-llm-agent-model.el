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

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'subr-x)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-inference-runtime)
(require 'nl-llm-agent)
(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-tokenizer)

(defcustom nl-llm-agent-model-inference-mode 'auto
  "Inference implementation prepared at the start of each policy invocation.
`auto' selects the best available implementation, `source' keeps source
definitions, and `byte-code' requests compiled numeric kernels.  Preparation
updates process-global shared numeric function bindings between agent turns;
the implementation is not hot-upgraded in the middle of token generation."
  :type '(choice (const :tag "Automatic" auto)
                 (const :tag "Source" source)
                 (const :tag "Byte code" byte-code))
  :group 'nl-llm-inference-runtime)

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

(defun nl-llm-agent--token-logits (logits tokenizer)
  "Validate LOGITS against TOKENIZER and return LOGITS."
  (let ((vocab (nl-llm-agent-tokenizer-vocab tokenizer)))
    (unless (and (vectorp logits) (= (length logits) vocab))
      (error "Native model logits length must be %d for %s, got %S"
             vocab tokenizer
             (and (vectorp logits) (length logits))))
    logits))

(defun nl-llm-agent--feed-character (char logits step-fn tokenizer)
  "Feed CHAR's complete token sequence and return the resulting logits."
  (let ((ids (nl-llm-agent-tokenizer-encode (string char) tokenizer)))
    (dolist (id ids logits)
      (setq logits
            (nl-llm-agent--token-logits
             (funcall step-fn id) tokenizer)))))

(defun nl-llm-agent--allowed-character
    (chars logits step-fn tokenizer)
  "Greedily select one of CHARS by byte-token prefix.
Return (CHAR . NEXT-LOGITS) after consuming the selected character's complete
token sequence."
  (let ((tail (append chars nil))
        (candidates nil)
        (depth 0))
    (unless tail
      (error "Constrained decoder received an empty allowed character set"))
    (dolist (char tail)
      (unless (integerp char)
        (error "Constrained decoder candidate is not a character: %S" char))
      (push (cons char
                  (nl-llm-agent-tokenizer-encode (string char) tokenizer))
            candidates))
    (setq candidates (nreverse candidates))
    (catch 'selected
      (while candidates
        (nl-llm-agent--token-logits logits tokenizer)
        (let ((ids nil))
          (dolist (candidate candidates)
            (let ((tokens (cdr candidate)))
              (unless (< depth (length tokens))
                (error "Constrained decoder encountered an empty candidate"))
              (let ((id (nth depth tokens)))
                (unless (memq id ids) (push id ids)))))
          (let ((id (nl-llm-agent--argmax-among logits (nreverse ids))))
            (setq logits
                  (nl-llm-agent--token-logits
                   (funcall step-fn id) tokenizer)
                  candidates
                  (delq nil
                        (mapcar
                         (lambda (candidate)
                           (and (= (nth depth (cdr candidate)) id) candidate))
                         candidates)))
            (setq depth (1+ depth))
            (let ((complete
                   (cl-find-if
                    (lambda (candidate) (= (length (cdr candidate)) depth))
                    candidates)))
              (when complete
                (throw 'selected (cons (car complete) logits)))))))
      (error "Constrained decoder could not select an allowed character"))))

(defun nl-llm-agent-constrained-generate
    (init-logits step-fn grammar &optional tokenizer)
  "Generate a string under GRAMMAR using INIT-LOGITS as the first next-token
distribution and STEP-FN (a function ID -> next-token-logit-vector) to advance.
GRAMMAR is a function EMITTED -> :stop | (:force CHAR) | (:allow CHARS); forced
characters are emitted as-is, allowed positions take the argmax among the legal
characters' encoded token prefixes.  TOKENIZER defaults to `ascii-char-v1'.
Each grammar callback observes only completed Unicode characters.  Returns the
generated string (always grammar-valid)."
  (let ((logits init-logits)
        (emitted "")
        (tokenizer (nl-llm-agent-tokenizer-id tokenizer)))
    (catch 'done
      (while t
        (let ((g (funcall grammar emitted)))
          (pcase g
            (:stop (throw 'done emitted))
            (`(:force ,ch)
             (setq emitted (concat emitted (string ch)))
             (nl-llm-agent--token-logits logits tokenizer)
             (setq logits
                   (nl-llm-agent--feed-character
                    ch logits step-fn tokenizer)))
            (`(:allow ,chars)
             (let* ((choice
                     (nl-llm-agent--allowed-character
                      chars logits step-fn tokenizer))
                    (ch (car choice)))
               (setq emitted (concat emitted (string ch)))
               (setq logits (cdr choice))))))))))

;; ---- action grammars -------------------------------------------------------

(defun nl-llm-agent-grammar-message (n &optional allow)
  "Return a grammar forcing a valid Elisp action.
The action has the form ```elisp (message \"<N model chars>\")```.
The N free characters are chosen by the model from ALLOW, which defaults to a
safe set without quote, backslash, or newline, so the result always parses."
  (let ((pre "```elisp\n(message \"") (post "\")\n```")
        (safe (or allow "abcdefghijklmnopqrstuvwxyz0123456789 ")))
    (lambda (emitted)
      (let ((p (length emitted)) (lp (length "```elisp\n(message \"")))
        (cond
         ((< p lp) (list :force (aref pre p)))
         ((< p (+ lp n)) (list :allow safe))
         ((< p (+ lp n (length post))) (list :force (aref post (- p lp n))))
         (t :stop))))))

(defun nl-llm-agent-grammar-template (segments)
  "Build a grammar from SEGMENTS, a list of either a forced string or a model
slot (:slot CHARS) (one model-chosen character from CHARS).  Generalises
`nl-llm-agent-grammar-message' to several variable slots, e.g. synthesising a
function with a chosen operator and constant."
  (lambda (emitted)
    (let ((p (length emitted)) (pos 0) (result :stop))
      (catch 'done
        (dolist (seg segments)
          (if (stringp seg)
              (let ((len (length seg)))
                (when (< p (+ pos len)) (throw 'done (setq result (list :force (aref seg (- p pos))))))
                (setq pos (+ pos len)))
            (when (= p pos) (throw 'done (setq result (list :allow (cadr seg)))))
            (setq pos (1+ pos)))))
      result)))

;; ---- the model policy ------------------------------------------------------

(defun nl-llm-agent--render (messages)
  "Render MESSAGES (list of (ROLE . CONTENT)) into a single prompt string."
  (concat (mapconcat (lambda (m) (format "%s: %s" (car m) (cdr m))) messages "\n") "\nassistant:\n"))

;;;###autoload
(defun nl-llm-agent-model-step-fn (model caches)
  "Return a STEP-FN (ID -> logit-vector) over MODEL (a plist with :blocks :wte
:lnfg :bh :dim and optional untied :wh) using mutable per-block CACHES (CPU
`nl-llm-decode-step')."
  (let ((blocks (plist-get model :blocks)) (wte (plist-get model :wte))
        (lnfg (plist-get model :lnfg)) (bh (plist-get model :bh))
        (dim (plist-get model :dim)) (wh (plist-get model :wh)))
    (lambda (id)
      (nl-llm-decode-step id blocks caches wte lnfg bh dim nil wh))))

(defun nl-llm-agent-model--validate-tokenizer (model)
  "Return MODEL's canonical tokenizer after checking known vocabulary shapes."
  (let* ((declared (plist-get model :tokenizer))
         (tokenizer (nl-llm-agent-tokenizer-id declared))
         (expected (nl-llm-agent-tokenizer-vocab tokenizer))
         (configured-vocab (plist-get model :vocab))
         (wte (plist-get model :wte))
         (bh (plist-get model :bh))
         (wh (plist-get model :wh))
         (wte-vocab (and (vectorp wte) (car (photon-tensor-shape wte))))
         (bh-vocab (and (vectorp bh) (car (photon-tensor-shape bh))))
         (wh-vocab (and (vectorp wh) (car (photon-tensor-shape wh)))))
    (when (and (null declared) wte-vocab (/= wte-vocab 96))
      (error "A model without :tokenizer is only compatible with vocabulary 96"))
    (when (and configured-vocab
               (not (and (integerp configured-vocab)
                         (> configured-vocab 0))))
      (error "Native model :vocab must be a positive integer"))
    (dolist (entry (list (cons :vocab configured-vocab)
                         (cons :wte wte-vocab)
                         (cons :bh bh-vocab)
                         (cons :wh wh-vocab)))
      (when (and (cdr entry) (/= (cdr entry) expected))
        (error "Native model %s vocabulary %d does not match tokenizer %s (%d)"
               (car entry) (cdr entry) tokenizer expected)))
    tokenizer))

;;;###autoload
(defun nl-llm-agent-model-policy (model grammar &optional maxseq)
  "Return an agent policy that drives MODEL under GRAMMAR.
The policy maps MESSAGES to text using constrained decoding.  MODEL is a plist
with :blocks, :wte, :lnfg, :bh, :dim, :heads, and :kvh.  Optional :tokenizer
is `ascii-char-v1' or `utf8-byte-v1'; omission retains the 96-token legacy
ASCII vocabulary.  Each call feeds the rendered history through a fresh KV
cache, then generates an action that is parseable regardless of model quality.
An optional :wh is used as an untied output head; otherwise :wte remains the
tied embedding/head matrix."
  (let ((dim (plist-get model :dim))
        (heads (plist-get model :heads))
        (kvh (plist-get model :kvh))
        (tokenizer (nl-llm-agent-model--validate-tokenizer model)))
    (lambda (messages)
      (let* ((capacity (or maxseq 1024))
             (prompt (nl-llm-agent--render messages)))
        (unless (and (integerp capacity) (> capacity 0))
          (error "Native model context capacity must be a positive integer, got %S"
                 capacity))
        ;; Every supported tokenizer emits at least one token per character.
        ;; This rejects obviously oversized input without first allocating a
        ;; potentially much larger UTF-8 id list.
        (when (> (length prompt) capacity)
          (error "Native model prompt length %d exceeds context capacity %d"
                 (length prompt) capacity))
        (let ((prompt-ids
               (nl-llm-agent-tokenizer-encode prompt tokenizer)))
          (when (> (length prompt-ids) capacity)
            (error
             "Native model prompt token length %d exceeds context capacity %d"
             (length prompt-ids) capacity))
          (nl-llm-inference-runtime-prepare
           nl-llm-agent-model-inference-mode)
          (let* ((caches
                  (mapcar
                   (lambda (_)
                     (nl-llm-dcache-new capacity dim heads kvh))
                   (plist-get model :blocks)))
                 (step (nl-llm-agent-model-step-fn model caches))
                 (logits nil))
            (dolist (id prompt-ids)
              (setq logits (funcall step id)))
            (unless logits (setq logits (funcall step 0)))
            (nl-llm-agent-constrained-generate
             logits step grammar tokenizer)))))))

;; ---- native provider adapter ----------------------------------------------

(defun nl-llm-agent-model--public-spec (spec)
  "Return SPEC without private model and grammar objects."
  (let ((rest spec)
        (result nil))
    (while rest
      (let ((key (car rest))
            (value (cadr rest)))
        (unless (memq key '(:model :grammar))
          (setq result (append result (list key value))))
        (setq rest (cddr rest))))
    result))

(defun nl-llm-agent-model--find-spec (models model-id)
  "Return MODEL-ID's private descriptor from MODELS, or nil."
  (cl-find-if
   (lambda (spec)
     (equal (nl-llm-agent-provider--id
             (plist-get spec :id) "native model" t)
            model-id))
   models))

;;;###autoload
(cl-defun nl-llm-agent-model-provider
    (id models &key name
        (capabilities '(generate local constrained-decoding)))
  "Expose MODELS through an `nl-llm-agent-provider'.

Each MODELS entry is a plist containing :id, :model, and :grammar, with optional
:name, :maxseq, and public :capabilities.  Private model and grammar objects are
not returned by the provider catalog.  A session :maxseq option overrides the
model descriptor.  Generation delegates to `nl-llm-agent-model-policy'."
  (unless (listp models)
    (error "nl-llm-agent-model-provider: MODELS must be a list"))
  (dolist (spec models)
    (unless (and (listp spec)
                 (plist-member spec :id)
                 (plist-member spec :model)
                 (functionp (plist-get spec :grammar)))
      (error "nl-llm-agent-model-provider: invalid model spec %S" spec)))
  (let ((private-models (copy-tree models)))
    (nl-llm-agent-provider-new
     id
     :name name
     :capabilities capabilities
     :models
     (lambda ()
       (mapcar #'nl-llm-agent-model--public-spec private-models))
     :open
     (lambda (model-id options)
       (let ((spec (nl-llm-agent-model--find-spec
                    private-models model-id)))
         (unless spec
           (error "native provider %s: unknown model %s" id model-id))
         (nl-llm-agent-model-policy
          (plist-get spec :model)
          (plist-get spec :grammar)
          (if (plist-member options :maxseq)
              (plist-get options :maxseq)
            (plist-get spec :maxseq)))))
     :complete
     (lambda (policy messages)
       (funcall policy messages)))))

(provide 'nl-llm-agent-model)
;;; nl-llm-agent-model.el ends here
