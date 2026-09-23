;;; agent-openai-provider-test.el --- OpenAI-compatible adapter  -*- lexical-binding: t; -*-

;; Tests use an injected transport: no network or API key is required.
;;   emacs -Q --batch -l test/agent-openai-provider-test.el

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-openai)

(defvar agent-openai-provider--fail 0)

(defun agent-openai-provider--ck (name ok &optional extra)
  (princ (format "%-58s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-openai-provider--fail
                         (1+ agent-openai-provider--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-openai-provider--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(let ((requests nil)
      (token "first-token"))
  (let* ((transport
          (lambda (request)
            (setq requests (append requests (list request)))
            '(:choices ((:message
                         (:content
                          ((:type "output_text" :text "DONE ")
                           (:type "output_text" :text "ok"))))))))
         (provider
          (nl-llm-agent-openai-provider
           "openrouter"
           :base-url "https://example.invalid/v1/"
           :models '("poolside/laguna-s-2.1:free"
                     (:id "fast" :name "Fast model"))
           :api-key (lambda () token)
           :headers '(("X-App" . "nelisp-agent"))
           :transport transport))
         (registry (nl-llm-agent-provider-registry-new)))
    (nl-llm-agent-provider-register registry provider)
    (let ((catalog (nl-llm-agent-provider-models registry)))
      (agent-openai-provider--ck
       "catalog preserves slash-containing remote model ids"
       (equal (plist-get (car catalog) :qualified-id)
              "openrouter/poolside/laguna-s-2.1:free")))
    (let* ((session
            (nl-llm-agent-session-open
             registry "openrouter/poolside/laguna-s-2.1:free"
             :options '(:temperature 0.25 :max_tokens 128
                        :timeout-sec 9)))
           (out
            (nl-llm-agent-session-complete
             session '((system . "rules") (user . "hello"))))
           (request (car requests))
           (body (plist-get request :body))
           (messages (plist-get body :messages))
           (headers (plist-get request :headers)))
      (agent-openai-provider--ck
       "adapter targets the chat completions endpoint"
       (equal (plist-get request :url)
              "https://example.invalid/v1/chat/completions"))
      (agent-openai-provider--ck
       "Bearer auth and configured headers stay in HTTP headers"
       (and (equal (cdr (assoc "Authorization" headers))
                   "Bearer first-token")
            (equal (cdr (assoc "X-App" headers)) "nelisp-agent")))
      (agent-openai-provider--ck
       "request carries model and provider-neutral messages"
       (and (equal (plist-get body :model)
                   "poolside/laguna-s-2.1:free")
            (= (length messages) 2)
            (equal (plist-get (aref messages 0) :role) "system")
            (equal (plist-get (aref messages 1) :content) "hello")))
      (agent-openai-provider--ck
       "generation options enter the body but timeout stays transport-only"
       (and (= (plist-get body :temperature) 0.25)
            (= (plist-get body :max_tokens) 128)
            (not (plist-member body :timeout-sec))
            (= (plist-get request :timeout-sec) 9)))
      (agent-openai-provider--ck
       "content blocks are combined into agent text"
       (equal out "DONE ok"))

      (setq token "rotated-token")
      (nl-llm-agent-session-complete
       session '((:role user :content "again")))
      (let* ((latest (car (last requests)))
             (latest-messages
              (plist-get (plist-get latest :body) :messages)))
        (agent-openai-provider--ck
         "plist messages normalize to the same wire format"
         (and (equal (plist-get (aref latest-messages 0) :role)
                     "user")
              (equal (plist-get (aref latest-messages 0) :content)
                     "again"))))
      (agent-openai-provider--ck
       "API key functions are resolved for every request"
       (equal (cdr (assoc "Authorization"
                          (plist-get (car (last requests)) :headers)))
              "Bearer rotated-token"))))

  (let* ((provider
          (nl-llm-agent-openai-provider
           "broken"
           :base-url "https://example.invalid/v1"
           :models '("failure")
           :transport
           (lambda (_request) '(:error (:message "quota exceeded")))))
         (registry (nl-llm-agent-provider-registry-new))
         session)
    (nl-llm-agent-provider-register registry provider)
    (setq session (nl-llm-agent-session-open registry "broken/failure"))
    (agent-openai-provider--ck
     "provider error objects become completion errors"
     (agent-openai-provider--error-p
      (lambda ()
        (nl-llm-agent-session-complete session '((user . "hello"))))))))

(let* ((requests nil)
       (provider
        (nl-llm-agent-openai-provider
         "default-timeout"
         :base-url "https://example.invalid/v1"
         :models '("slow")
         :timeout-sec 180
         :transport
         (lambda (request)
           (setq requests (append requests (list request)))
           '(:choices ((:message (:content "slow model")))))))
       (registry (nl-llm-agent-provider-registry-new)))
  (nl-llm-agent-provider-register registry provider)
  (nl-llm-agent-session-complete
   (nl-llm-agent-session-open registry "default-timeout/slow")
   '((user . "hello")))
  (nl-llm-agent-session-complete
   (nl-llm-agent-session-open registry "default-timeout/slow"
                              :options '(:timeout-sec 5))
   '((user . "again")))
  (agent-openai-provider--ck
   "provider timeout defaults the request timeout"
   (= (plist-get (nth 0 requests) :timeout-sec) 180))
  (agent-openai-provider--ck
   "session timeout overrides provider timeout"
   (= (plist-get (nth 1 requests) :timeout-sec) 5)))

(agent-openai-provider--ck
 "non-positive provider timeouts are rejected"
 (agent-openai-provider--error-p
  (lambda ()
    (nl-llm-agent-openai-provider
     "bad-timeout"
     :base-url "https://example.invalid/v1"
     :models '("x")
     :timeout-sec -1))))

(agent-openai-provider--ck
 "non-HTTP base URLs are rejected"
 (agent-openai-provider--error-p
  (lambda ()
    (nl-llm-agent-openai-provider
     "bad-url" :base-url "file:///tmp/socket" :models '("x")))))

(princ (format "NL-LLM-AGENT-OPENAI-PROVIDER %s (%d failures)\n"
               (if (= agent-openai-provider--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-openai-provider--fail))
(kill-emacs (if (= agent-openai-provider--fail 0) 0 1))

;;; agent-openai-provider-test.el ends here
