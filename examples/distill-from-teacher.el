;;; distill-from-teacher.el --- a dataset from the local open-weight teacher  -*- lexical-binding: t; -*-

;; Doc 08 Phase 3.  Joins the two ends of this work: the Phase 0 provider serves
;; a locally-run Apache-2.0 teacher, and `nl-llm-distill' turns its answers into
;; the completion-only dataset `nl-llm-agent-supervised' takes.  The agent's own
;; provider layer generating the agent's own training data.
;;
;; Licensing, which is why the teacher is local rather than an API: Qwen's core
;; models are Apache-2.0, which places no restriction on the use of outputs, so
;; distillation from locally-run weights is permitted.  Outputs from a hosted
;; API are a different matter -- OpenAI, Anthropic and Gemini all forbid using
;; them to train a model -- and no gate in this file can tell which endpoint was
;; called.  That choice is the caller's, and this file only makes the permitted
;; one convenient.
;;
;; Run:
;;   ollama serve &                       (once)
;;   ollama pull qwen3:4b                 (once)
;;   make distill                         (uses examples/distill-prompts.txt)
;;   make distill PROMPTS=my-prompts.txt OUT=my-data.eld
;;
;; Personal prompts belong in a file outside this repository.  What is shipped
;; is a shape to copy.

;;; Code:

(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-openai)
(require 'nl-llm-distill)
(require 'nl-llm-teacher-cache)

(defvar distill--base-url
  (or (getenv "NL_LLM_OLLAMA_BASE_URL") "http://127.0.0.1:11434/v1"))
(defvar distill--model (or (getenv "NL_LLM_OLLAMA_MODEL") "qwen3:4b"))
(defvar distill--prompts
  (or (getenv "NL_LLM_DISTILL_PROMPTS") "examples/distill-prompts.txt"))
(defvar distill--out
  (or (getenv "NL_LLM_DISTILL_OUT") "build/distilled.eld"))
(defvar distill--timeout
  (string-to-number (or (getenv "NL_LLM_TEACHER_TIMEOUT") "900"))
  "Seconds to wait for one answer.  A 4B reasoning model on a loaded machine
takes around two minutes per prompt here, well past the provider's 60-second
default -- which is how a whole run came back as nothing but timeouts.")
(defvar distill--options
  (list :max_tokens (string-to-number
                     (or (getenv "NL_LLM_TEACHER_MAX_TOKENS") "2048"))
        :temperature 0.0
        :timeout-sec distill--timeout))
(defvar distill--stats (nl-llm-teacher-cache-stats-new))

(defun distill--read-prompts (path)
  "Return the prompt lines of PATH, skipping blanks and # comments."
  (unless (file-readable-p path)
    (error "cannot read prompts from %s" path))
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8-unix)) (insert-file-contents path))
    (let (out)
      (dolist (line (split-string (buffer-string) "\n"))
        (let ((s (string-trim line)))
          (unless (or (string-empty-p s) (string-prefix-p "#" s))
            (push s out))))
      (nreverse out))))

(defun distill--teacher ()
  "Return a one-argument function that asks the local teacher for an answer.
A session per call, so one bad answer cannot poison the next; the provider
reads only the final `content', so a reasoning model's chain of thought is
discarded rather than trained on.

Answers go through `nl-llm-teacher-cache', keyed on everything that can change
one.  At temperature 0 a re-run of the same prompt set is free, which is the
usual shape of this loop: the dataset gets rebuilt because something
downstream of it changed, not because the teacher did."
  (nl-llm-teacher-cache-wrap
   (distill--ask)
   (list :model distill--model :base-url distill--base-url
         :options distill--options)
   nil distill--stats))

(defun distill--ask ()
  "Return the uncached one-argument teacher."
  (lambda (prompt)
    (let* ((registry (nl-llm-agent-provider-registry-new))
           (_ (nl-llm-agent-provider-register
               registry
               (nl-llm-agent-openai-provider
                "ollama" :name "local teacher"
                :base-url distill--base-url
                :models (list distill--model))))
           (session (nl-llm-agent-session-open
                     registry distill--model
                     :options distill--options))
           (text (nl-llm-agent-session-complete
                  session (list (cons 'user prompt)))))
      (nl-llm-agent-session-close session)
      text)))

;; --- run -----------------------------------------------------------------

(let* ((prompts (distill--read-prompts distill--prompts))
       (t0 (float-time)))
  (message "teacher %s at %s; %d prompts from %s"
           distill--model distill--base-url (length prompts) distill--prompts)
  (let* ((res (nl-llm-distill-collect
               prompts (distill--teacher)
               (lambda (i reason)
                 (message "  %2d/%d %-10s %.0fs" (1+ i) (length prompts)
                          (or reason "ok") (- (float-time) t0)))))
         (examples (car res))
         (report (cdr res)))
    (message "\n%s" (nl-llm-distill-report-summary report))
    (dolist (note (reverse (nl-llm-distill-report-notes report)))
      (message "  rejected (%s): %s" (car note) (cdr note)))
    (if (null examples)
        (message "\nnothing accepted -- not writing a file")
      (make-directory (file-name-directory distill--out) t)
      (nl-llm-distill-write
       examples distill--out
       (list :teacher distill--model
             :base-url distill--base-url
             :prompts (length prompts)
             :generated (format-time-string "%Y-%m-%dT%H:%M:%S%z")))
      (message "\nwrote %s" distill--out)
      (let ((first (car examples)))
        (message "first example:\n  prompt     %s\n  completion %s"
                 (plist-get first :prompt)
                 (let ((c (plist-get first :completion)))
                   (if (> (length c) 160) (concat (substring c 0 160) "...") c))))
      (message "%s" (nl-llm-teacher-cache-report distill--stats))
      (message "elapsed %.0fs" (- (float-time) t0)))))

;;; distill-from-teacher.el ends here
