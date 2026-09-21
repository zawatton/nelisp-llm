;;; distill-soft-from-teacher.el --- a soft-target dataset from the local teacher  -*- lexical-binding: t; -*-

;; Doc 08 Phase 3.  The same run as `distill-from-teacher.el', asking the
;; teacher for its top-K per position instead of only the token it sampled.
;; Same gates, same file shape, K times the supervision per teacher call --
;; and the teacher call is what the run costs.
;;
;; Licensing is unchanged and still the caller's: a locally-run Apache-2.0 or
;; MIT teacher may be distilled from, a hosted API may not, and no gate here
;; can tell which endpoint was called.
;;
;; Run:
;;   ollama serve &                       (once)
;;   make distill-soft                    (uses examples/distill-prompts.txt)
;;   make distill-soft PROMPTS=my.txt OUT=my-data.eld TOPK=8
;;
;; The report ends with the vocabulary size, which is the number to size a
;; student from: soft targets are only usable by a model that can represent
;; every token the teacher mentioned.

;;; Code:

(require 'nl-llm-teacher-cache)
(require 'nl-llm-teacher-logprobs)
(require 'nl-llm-distill-soft)

(defvar soft--base-url
  (or (getenv "NL_LLM_OLLAMA_BASE_URL") "http://127.0.0.1:11434/v1"))
(defvar soft--model (or (getenv "NL_LLM_OLLAMA_MODEL") "qwen3:4b"))
(defvar soft--prompts
  (or (getenv "NL_LLM_DISTILL_PROMPTS") "examples/distill-prompts.txt"))
(defvar soft--out
  (or (getenv "NL_LLM_DISTILL_OUT") "build/distilled-soft.eld"))
(defvar soft--top-k
  (string-to-number (or (getenv "NL_LLM_DISTILL_TOPK") "8")))
(defvar soft--timeout
  (string-to-number (or (getenv "NL_LLM_TEACHER_TIMEOUT") "900"))
  "Seconds to wait for one answer.  A 4B reasoning model on a loaded machine
takes around two minutes per prompt here, well past the provider's 60-second
default -- which is how a whole run came back as nothing but timeouts.")
(defvar soft--options
  (list :max_tokens (string-to-number
                     (or (getenv "NL_LLM_TEACHER_MAX_TOKENS") "2048"))
        :temperature 0.0
        :timeout-sec soft--timeout))
(defvar soft--stats (nl-llm-teacher-cache-stats-new))

(defun soft--read-prompts (path)
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

(defun soft--teacher ()
  "Return a cached one-argument teacher that answers with its distribution.
The cache identity carries the top-k as well as the model and options, because
a K of 8 and a K of 4 are different answers to the same prompt."
  (nl-llm-teacher-logprobs-cached-teacher
   :base-url soft--base-url :model soft--model :options soft--options
   :top-k soft--top-k :stats soft--stats))

;; --- run -----------------------------------------------------------------

(let* ((prompts (soft--read-prompts soft--prompts))
       (t0 (float-time)))
  (message "teacher %s at %s; %d prompts, top-%d"
           soft--model soft--base-url (length prompts) soft--top-k)
  (let* ((res (nl-llm-distill-soft-collect
               prompts (soft--teacher)
               (lambda (i reason)
                 (message "  %3d/%d %s" (1+ i) (length prompts)
                          (if reason (format "rejected: %s" reason) "ok")))))
         (examples (car res))
         (report (cdr res)))
    (message "\n%s" (nl-llm-distill-report-summary report))
    (message "%s" (nl-llm-distill-soft-summary examples))
    (if (null examples)
        (message "nothing accepted; not writing %s" soft--out)
      (nl-llm-distill-soft-write
       examples soft--out
       (list :teacher soft--model
             :base-url soft--base-url
             :top-k soft--top-k
             :prompts (length prompts)
             :generated (format-time-string "%Y-%m-%dT%H:%M:%S%z")))
      (message "wrote %s" soft--out)
      ;; What a student would have to represent, and how much of the mass the
      ;; first N of it carry -- the numbers a vocabulary is pruned on.
      (let ((vocab (nl-llm-distill-soft-vocabulary examples)))
        (message "vocabulary %d distinct token(s)" (length vocab))
        (dolist (n '(256 1024 4096 16384))
          (when (< n (length vocab))
            (message "  top %-6d covers %.4f of all soft-target mass"
                     n (nl-llm-distill-soft-coverage vocab n))))))
    (message "%s" (nl-llm-teacher-cache-report soft--stats))
    (message "elapsed %.0fs" (- (float-time) t0))))

;;; distill-soft-from-teacher.el ends here
