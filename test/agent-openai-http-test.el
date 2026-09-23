;;; agent-openai-http-test.el --- default Emacs HTTP transport  -*- lexical-binding: t; -*-

;; URL retrieval is stubbed, so this test performs no network access.
;;   emacs -Q --batch -l test/agent-openai-http-test.el

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'url)
(require 'json)
(require 'nl-llm-agent-openai)

(defvar agent-openai-http--fail 0)

(defun agent-openai-http--ck (name ok &optional extra)
  (princ (format "%-58s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-openai-http--fail
                         (1+ agent-openai-http--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-openai-http--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(let ((status 200)
      (captured nil)
      (buffers nil))
  (cl-letf (((symbol-function 'url-retrieve-synchronously)
             (lambda (url silent inhibit-cookies timeout)
               (setq captured
                     (list :url url
                           :silent silent
                           :inhibit-cookies inhibit-cookies
                           :timeout timeout
                           :method url-request-method
                           :headers url-request-extra-headers
                           :data url-request-data))
               (let ((buffer (generate-new-buffer " *openai-http-test*")))
                 (push buffer buffers)
                 (with-current-buffer buffer
                   (setq-local url-http-response-status status)
                   (insert
                    (if (= status 200)
                        (concat
                         "HTTP/1.1 200 OK\r\nContent-Type: application/json"
                         "\r\n\r\n{\"choices\":[{\"message\":{\"content\":"
                         "\"DONE local\"}}]}")
                      (concat
                       "HTTP/1.1 401 Unauthorized\r\n"
                       "Content-Type: application/json\r\n\r\n"
                       "{\"error\":{\"message\":\"bad token\"}}"))))
                 buffer))))
    (let* ((response
            (nl-llm-agent-openai-default-transport
             '(:url "https://example.invalid/v1/chat/completions"
               :headers (("Authorization" . "Bearer secret"))
               :body (:model "m" :messages [])
               :timeout-sec 7)))
           (sent-body
            (json-parse-string (plist-get captured :data)
                               :object-type 'plist
                               :array-type 'list)))
      (agent-openai-http--ck
       "default transport sends POST JSON with headers and timeout"
       (and (equal (plist-get captured :method) "POST")
            (= (plist-get captured :timeout) 7)
            (equal (cdr (assoc "Authorization"
                               (plist-get captured :headers)))
                   "Bearer secret")
            (equal (plist-get sent-body :model) "m")))
      (agent-openai-http--ck
       "default transport parses a successful JSON response"
       (equal (nl-llm-agent-openai--response-text response)
              "DONE local")))
    (setq status 401)
    (agent-openai-http--ck
     "default transport rejects non-success HTTP status"
     (agent-openai-http--error-p
      (lambda ()
        (nl-llm-agent-openai-default-transport
         '(:url "https://example.invalid/v1/chat/completions"
           :headers nil :body (:model "m" :messages []))))))
    (agent-openai-http--ck "default transport always releases response buffers"
                           (not (cl-some #'buffer-live-p buffers)))))

(let ((captured nil))
  (cl-letf (((symbol-function 'url-retrieve-synchronously)
             (lambda (&rest _)
               (setq captured
                     (list :method url-request-method
                           :headers url-request-extra-headers
                           :data url-request-data))
               (let ((buffer (generate-new-buffer " *openai-http-utf8-test*")))
                 (with-current-buffer buffer
                   (setq-local url-http-response-status 200)
                   (insert
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json"
                    "\r\n\r\n{\"choices\":[{\"message\":{\"content\":"
                    "\"DONE local\"}}]}"))
                 buffer))))
    (nl-llm-agent-openai-default-transport
     (list :url "https://example.invalid/v1/chat/completions"
           :headers (list (cons "Authorization"
                                (string-to-multibyte "Bearer sk-test")))
           :body '(:model "m"
                    :messages
                    [(:role "assistant" :content "I’m")])
           :timeout-sec 7))
    (let ((request
           (concat
            "POST /v1/chat/completions HTTP/1.1\r\n"
            (mapconcat
             (lambda (h) (concat (car h) ": " (cdr h) "\r\n"))
             (plist-get captured :headers) "")
            "\r\n" (plist-get captured :data))))
      (agent-openai-http--ck
       "default transport request passes the url-http multibyte guard"
       (= (string-bytes request) (length request))))))

(let ((captured-message nil))
  (cl-letf (((symbol-function 'url-retrieve-synchronously)
             (lambda (&rest _)
               (error
                "Multibyte text in HTTP request: POST /x HTTP/1.1\r\nAuthorization: Bearer sk-or-v1-SECRET\r\n"))))
    (condition-case err
        (progn
          (nl-llm-agent-openai-default-transport
           '(:url "https://example.invalid/v1/chat/completions"
             :headers nil :body (:model "m" :messages [])))
          (setq captured-message ""))
      (error
       (setq captured-message (error-message-string err)))))
  (agent-openai-http--ck
   "default transport redacts credentials from request errors"
   (and (string-match-p "OpenAI provider: request to" captured-message)
        (not (string-match-p "SECRET" captured-message)))))

(agent-openai-http--ck
 "openai redactor removes Bearer token punctuation runs"
 (not (string-match-p
       "abc\\.def"
       (nl-llm-agent-openai--redact "x Bearer abc.def y"))))

(princ (format "NL-LLM-AGENT-OPENAI-HTTP %s (%d failures)\n"
               (if (= agent-openai-http--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-openai-http--fail))
(kill-emacs (if (= agent-openai-http--fail 0) 0 1))

;;; agent-openai-http-test.el ends here
