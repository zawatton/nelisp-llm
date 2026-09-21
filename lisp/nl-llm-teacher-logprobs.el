;;; nl-llm-teacher-logprobs.el --- soft targets from a local teacher  -*- lexical-binding: t; -*-

;; Doc 08 Phase 3.  A completion-only dataset teaches one token per position:
;; the one the teacher happened to sample.  The teacher knew more than that --
;; it had a distribution -- and an OpenAI-compatible endpoint will hand over
;; the top K of it per position if asked.  Training against those soft targets
;; carries roughly K times the signal per teacher call, which matters because
;; the teacher call is the expensive part.
;;
;; This is deliberately separate from `nl-llm-agent-openai'.  That module's
;; contract is a chat session that returns text, and widening it to return
;; token-level structure would change a shape several other callers depend on.
;; Here the transport is reused and the response is read differently.
;;
;; The guard that matters: an endpoint which does not implement `logprobs'
;; answers perfectly normally, just without them.  A dataset built through such
;; an endpoint would have no soft targets at all and nothing would say so until
;; a training run quietly learned nothing.  `nl-llm-teacher-logprobs-ask'
;; therefore signals when the field is missing rather than returning a
;; plausible-looking answer with an empty :tokens.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-agent-openai)

(defcustom nl-llm-teacher-logprobs-top-k 8
  "How many alternatives per position to ask the teacher for.
The endpoint caps this; OpenAI's documented maximum is 20."
  :type 'integer
  :group 'nl-llm-inference-runtime)

(defun nl-llm-teacher-logprobs--alt (entry)
  "Read one {token, logprob} object into a plist."
  (list :token (plist-get entry :token)
        :logprob (plist-get entry :logprob)))

(defun nl-llm-teacher-logprobs--position (entry)
  "Read one sampled position, with its alternatives, into a plist."
  (append (nl-llm-teacher-logprobs--alt entry)
          (list :top (mapcar #'nl-llm-teacher-logprobs--alt
                             (plist-get entry :top_logprobs)))))

;;;###autoload
(cl-defun nl-llm-teacher-logprobs-ask
    (prompt &key base-url model options top-k transport timeout-sec)
  "Ask MODEL at BASE-URL for PROMPT and return text plus per-token soft targets.

The result is (:text S :tokens LIST :model M), where each element of LIST is
(:token T :logprob L :top ((:token T :logprob L) ...)) in generated order.

TRANSPORT defaults to `nl-llm-agent-openai-default-transport' and exists so a
test can drive this without a server.  Signals when the response carries no
`logprobs', because that is indistinguishable from success at every later
stage."
  (let* ((top-k (or top-k nl-llm-teacher-logprobs-top-k))
         ;; `:timeout-sec' belongs to the transport, not the request body: the
         ;; caller keeps one options plist for both, and sending an unknown
         ;; field to a strict endpoint is a 400 waiting to happen.
         (timeout-sec (or timeout-sec (plist-get options :timeout-sec) 600))
         (body-options (cl-loop for (k v) on options by #'cddr
                                unless (eq k :timeout-sec) append (list k v)))
         (body (append
                (list :model model
                      :messages (vector (list :role "user" :content prompt))
                      :logprobs t
                      :top_logprobs top-k
                      :stream :json-false)
                body-options))
         (response (funcall (or transport
                                #'nl-llm-agent-openai-default-transport)
                            (list :url (concat base-url "/chat/completions")
                                  :headers '(("Content-Type" . "application/json"))
                                  :body body
                                  :timeout-sec timeout-sec)))
         (choice (car (plist-get response :choices)))
         (logprobs (plist-get choice :logprobs))
         (content (plist-get logprobs :content)))
    (unless choice
      (error "teacher-logprobs: no choice in the response"))
    (unless content
      (error "teacher-logprobs: %s returned no logprobs; \
the endpoint may not implement them" model))
    (list :text (plist-get (plist-get choice :message) :content)
          :model (or (plist-get response :model) model)
          :top-k top-k
          :tokens (mapcar #'nl-llm-teacher-logprobs--position content))))

;;;###autoload
(defun nl-llm-teacher-logprobs-text (answer)
  "Return ANSWER's text, whether it is a plist from here or a plain string."
  (if (stringp answer) answer (plist-get answer :text)))

;;;###autoload
(defun nl-llm-teacher-logprobs-joined (answer)
  "Return ANSWER's sampled tokens concatenated.
Compared against `:text' this says whether the token stream really is the
answer, rather than a parallel sequence that merely looks like one."
  (mapconcat (lambda (p) (or (plist-get p :token) ""))
             (plist-get answer :tokens) ""))

;;;###autoload
(defun nl-llm-teacher-logprobs-summary (answer)
  "Return a one-line description of ANSWER's soft targets."
  (let* ((tokens (plist-get answer :tokens))
         (n (length tokens))
         (widths (mapcar (lambda (p) (length (plist-get p :top))) tokens)))
    (format "%d token(s), top-%s per position, %d soft targets"
            n
            (if widths
                (if (apply #'= widths) (format "%d" (car widths))
                  (format "%d-%d" (apply #'min widths) (apply #'max widths)))
              "0")
            (apply #'+ 0 widths))))

(provide 'nl-llm-teacher-logprobs)
;;; nl-llm-teacher-logprobs.el ends here
