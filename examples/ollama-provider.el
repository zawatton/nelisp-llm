;;; ollama-provider.el --- Phase 0 of Doc 08: a local open-weight teacher  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 0: register a locally served
;; Apache-2.0 open-weight model as an OpenAI-compatible provider.  No weight
;; conversion and no training -- this is both an immediately usable remote model
;; for the agent harness and the data generator for the distillation phase.
;;
;; Licensing note: the donor must be locally-run open weights (Apache-2.0 or
;; MIT).  Outputs taken from a hosted API may not be used to train a model, so
;; this path deliberately talks to a local server only.
;;
;; A reasoning model (Qwen3) returns its chain of thought in a separate
;; `reasoning' field, which `nl-llm-agent-openai--response-text' already
;; ignores; only the final `content' is kept.  The consequence worth guarding
;; is that a completion truncated while still thinking arrives as an EMPTY
;; string rather than as an error, so an empty completion is rejected here --
;; silently training on blank targets is the failure mode this avoids.
;;
;; Run:
;;   ollama serve &
;;   ollama pull qwen3:4b
;;   make ollama-provider      (or the emacs -Q --batch line in the Makefile)

;;; Code:

(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-openai)

(defvar ollama-provider-base-url
  (or (getenv "NL_LLM_OLLAMA_BASE_URL") "http://127.0.0.1:11434/v1")
  "Base URL of the local OpenAI-compatible server.")

(defvar ollama-provider-model
  (or (getenv "NL_LLM_OLLAMA_MODEL") "qwen3:4b")
  "Model tag to open a session on.")

(defun ollama-provider-registry (&rest models)
  "Return a registry holding the local Ollama provider serving MODELS."
  (let ((registry (nl-llm-agent-provider-registry-new)))
    (nl-llm-agent-provider-register
     registry
     (nl-llm-agent-openai-provider
      "ollama"
      :name "local Ollama (open weights)"
      :base-url ollama-provider-base-url
      :models (or models (list ollama-provider-model))))
    registry))

(defun ollama-provider-ask (prompt &rest options)
  "Send PROMPT to the local model and return its final answer text.
OPTIONS are session options (:max_tokens, :temperature, :seed, ...).  Signals
when the model returns no text, which is what a mid-thought truncation looks
like over the OpenAI-compatible contract."
  (let* ((registry (ollama-provider-registry))
         (session (nl-llm-agent-session-open
                   registry ollama-provider-model
                   :options (or options '(:max_tokens 2048))))
         (text (nl-llm-agent-session-complete
                session (list (cons 'user prompt)))))
    (nl-llm-agent-session-close session)
    (when (string-empty-p (string-trim text))
      (error "empty completion: the model was probably truncated while \
reasoning -- raise :max_tokens (currently %S)"
             (plist-get options :max_tokens)))
    text))

;; --- demo ---------------------------------------------------------------

(let ((cases '(("Reply with exactly: OK" . "OK")
               ("What is 17*23? Answer with the number only." . "391"))))
  (message "provider: %s  model: %s"
           ollama-provider-base-url ollama-provider-model)
  (dolist (c cases)
    (let* ((start (float-time))
           (answer (string-trim (ollama-provider-ask (car c))))
           (secs (- (float-time) start)))
      (message "  %-45s -> %-8s (%.1fs) %s"
               (car c) answer secs
               (if (string= answer (cdr c)) "ok" (format "EXPECTED %s" (cdr c))))))
  (message "Phase 0 reachable: a teacher is available with zero conversion."))

;;; ollama-provider.el ends here
