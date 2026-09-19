;;; agent-openai-http-test.el --- default Emacs HTTP transport  -*- lexical-binding: t; -*-

;; URL retrieval is stubbed, so this test performs no network access.
;;   emacs -Q --batch -L lisp -l test/agent-openai-http-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
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

(princ (format "NL-LLM-AGENT-OPENAI-HTTP %s (%d failures)\n"
               (if (= agent-openai-http--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-openai-http--fail))
(kill-emacs (if (= agent-openai-http--fail 0) 0 1))

;;; agent-openai-http-test.el ends here
