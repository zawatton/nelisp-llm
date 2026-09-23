;;; nl-llm-agent-openai.el --- OpenAI-compatible provider  -*- lexical-binding: t; -*-

;; Translate provider-neutral agent messages to the widely implemented
;; /chat/completions contract.  HTTP is behind one injectable transport so tests
;; and future standalone NeLisp networking use the same request/response logic.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'subr-x)
(require 'nl-llm-agent-provider)

(declare-function json-parse-string "json" (string &rest args))
(declare-function json-serialize "json" (object &rest args))
(declare-function url-retrieve-synchronously
                  "url" (url &optional silent inhibit-cookies timeout))
(defvar url-http-response-status)
(defvar url-request-data)
(defvar url-request-extra-headers)
(defvar url-request-method)

(defconst nl-llm-agent-openai--body-option-keys
  '(:temperature :top_p :max_tokens :seed :stop
    :tools :tool_choice :response_format)
  "Session option keys copied into a chat completion request body.")

(defun nl-llm-agent-openai--header-set (headers name value)
  "Return HEADERS with case-insensitive NAME set to VALUE."
  (let ((result nil))
    (dolist (header headers)
      (unless (string-equal (downcase (car header)) (downcase name))
        (push header result)))
    (nreverse (cons (cons name value) result))))

(defun nl-llm-agent-openai--headers (configured api-key)
  "Build request headers from CONFIGURED and API-KEY."
  (let ((headers nil))
    (setq headers
          (nl-llm-agent-openai--header-set
           headers "Content-Type" "application/json"))
    (setq headers
          (nl-llm-agent-openai--header-set
           headers "Accept" "application/json"))
    (dolist (header configured)
      (unless (and (consp header)
                   (stringp (car header))
                   (stringp (cdr header)))
        (error "OpenAI provider: invalid header %S" header))
      (setq headers
            (nl-llm-agent-openai--header-set
             headers (car header) (cdr header))))
    ;; Explicit API credentials win over a configured Authorization header.
    (when (and api-key (not (string-empty-p api-key)))
      (setq headers
            (nl-llm-agent-openai--header-set
             headers "Authorization" (concat "Bearer " api-key))))
    headers))

(defun nl-llm-agent-openai--resolve-api-key (source)
  "Resolve API key SOURCE, which may be nil, a string, or a function."
  (let ((value (cond ((null source) nil)
                     ((stringp source) source)
                     ((functionp source) (funcall source))
                     (t (error "OpenAI provider: invalid API key source")))))
    (unless (or (null value) (stringp value))
      (error "OpenAI provider: API key function returned %S" value))
    value))

(defun nl-llm-agent-openai--role (value)
  "Return provider-neutral role VALUE as an API string."
  (cond ((stringp value) value)
        ((keywordp value) (substring (symbol-name value) 1))
        ((symbolp value) (symbol-name value))
        (t (error "OpenAI provider: invalid message role %S" value))))

(defun nl-llm-agent-openai--message (message)
  "Return one OpenAI request object for provider-neutral MESSAGE."
  (let (role content)
    (cond
     ((and (consp message)
           (keywordp (car message))
           (listp (cdr message))
           (plist-member message :role))
      (setq role (plist-get message :role)
            content (plist-get message :content)))
     ((and (consp message)
           (or (stringp (car message)) (symbolp (car message))))
      (setq role (car message)
            content (cdr message)))
     (t (error "OpenAI provider: invalid message %S" message)))
    (unless (stringp content)
      (error "OpenAI provider: message content must be text, got %S"
             content))
    (list :role (nl-llm-agent-openai--role role)
          :content content)))

(defun nl-llm-agent-openai--body (model messages options)
  "Build a chat completion body for MODEL, MESSAGES, and OPTIONS."
  (let ((body
         (list :model model
               :messages
               (vconcat (mapcar #'nl-llm-agent-openai--message messages)))))
    (dolist (key nl-llm-agent-openai--body-option-keys)
      (when (plist-member options key)
        (setq body (plist-put body key (plist-get options key)))))
    body))

(defun nl-llm-agent-openai--get (object key)
  "Read string KEY from JSON-like OBJECT.
OBJECT may be a hash table, keyword plist, or alist."
  (let ((keyword (intern (concat ":" key)))
        (symbol (intern key)))
    (cond
     ((hash-table-p object)
      (or (gethash key object)
          (gethash keyword object)
          (gethash symbol object)))
     ((and (listp object) (keywordp (car object)))
      (plist-get object keyword))
     ((listp object)
      (let ((cell (or (assoc key object)
                      (assq keyword object)
                      (assq symbol object))))
        (and cell (cdr cell))))
     (t nil))))

(defun nl-llm-agent-openai--first (sequence)
  "Return the first element of list or vector SEQUENCE."
  (cond ((vectorp sequence)
         (and (> (length sequence) 0) (aref sequence 0)))
        ((consp sequence) (car sequence))
        (t nil)))

(defun nl-llm-agent-openai--content-text (content)
  "Normalize string or content-block CONTENT into text."
  (cond
   ((stringp content) content)
   ((or (listp content) (vectorp content))
    (let ((blocks (if (vectorp content) (append content nil) content))
          (pieces nil))
      (dolist (block blocks)
        (let ((text (if (stringp block)
                        block
                      (or (nl-llm-agent-openai--get block "text")
                          (nl-llm-agent-openai--get block "content")))))
          (when (stringp text)
            (push text pieces))))
      (mapconcat #'identity (nreverse pieces) "")))
   (t nil)))

(defun nl-llm-agent-openai--response-text (response)
  "Extract assistant text from parsed OpenAI-compatible RESPONSE."
  (let ((remote-error (nl-llm-agent-openai--get response "error")))
    (when remote-error
      (error "OpenAI provider error: %s"
             (or (nl-llm-agent-openai--get remote-error "message")
                 remote-error)))
    (let* ((choice
            (nl-llm-agent-openai--first
             (nl-llm-agent-openai--get response "choices")))
           (message (and choice
                         (nl-llm-agent-openai--get choice "message")))
           (content (or (and message
                             (nl-llm-agent-openai--get message "content"))
                        (and choice
                             (nl-llm-agent-openai--get choice "text"))))
           (text (nl-llm-agent-openai--content-text content)))
      (unless (stringp text)
        (error "OpenAI provider: response has no assistant text"))
      text)))

(defun nl-llm-agent-openai--redact (text)
  "Return TEXT with Authorization headers and Bearer tokens redacted."
  (let ((case-fold-search t))
    (replace-regexp-in-string
     "Authorization:[^\r\n]*" "Authorization: [redacted]"
     (replace-regexp-in-string
      "Bearer [^[:space:]\"\\]+" "Bearer [redacted]"
      text))))

(defun nl-llm-agent-openai-default-transport (request)
  "Perform OpenAI-compatible REQUEST with Emacs URL and JSON libraries.
REQUEST is a plist containing :url, :headers, :body, and :timeout-sec."
  (require 'url)
  (require 'url-http)
  (require 'json)
  (let* ((url (plist-get request :url))
         (url-request-method "POST")
         (url-request-extra-headers
          ;; getenv returns multibyte strings; one multibyte header makes
          ;; url-http reject a request whose body has non-ASCII bytes.
          (mapcar (lambda (header)
                    ;; Without NOCOPY: an ASCII-only multibyte string would
                    ;; otherwise come back unchanged, still multibyte.
                    (cons (encode-coding-string (car header) 'utf-8)
                          (encode-coding-string (cdr header) 'utf-8)))
                  (plist-get request :headers)))
         (url-request-data
          (encode-coding-string
           (json-serialize (plist-get request :body)
                           :null-object nil
                           :false-object :json-false)
           'utf-8 t))
         (timeout (or (plist-get request :timeout-sec) 60))
         (buffer
          (condition-case err
              (url-retrieve-synchronously url t t timeout)
            (error
             (signal 'error
                     (list (format
                            "OpenAI provider: request to %s failed: %s"
                            url
                            (nl-llm-agent-openai--redact
                             (error-message-string err)))))))))
    ;; `url-retrieve-synchronously' returns nil when it gave up waiting, which
    ;; is what a slow local teacher looks like from here.  The old wording --
    ;; "failed without a response" -- reads as a server fault and cost a whole
    ;; distillation run before the 60-second default was the suspect.
    (unless buffer
      (error "OpenAI provider: no response within %ss; raise :timeout-sec if the model is simply slow" timeout))
    (unwind-protect
        (with-current-buffer buffer
          ;; No status at all means nothing spoke HTTP: the server is down, the
          ;; port is refusing, or the name did not resolve.  That used to
          ;; surface as "malformed HTTP response", which reads as a protocol
          ;; fault at the far end and sends the reader looking at their request
          ;; -- twice here, once for 25 minutes.
          (unless url-http-response-status
            (error "OpenAI provider: no HTTP response from %s; the server may be down or refusing connections" url))
          (let ((status url-http-response-status)
                (body-start
                 (progn
                   (goto-char (point-min))
                   (unless (re-search-forward "\r?\n\r?\n" nil t)
                     (error "OpenAI provider: HTTP %S with no header terminator"
                            url-http-response-status))
                   (point))))
            (let ((body (buffer-substring-no-properties
                         body-start (point-max))))
              (unless (and (integerp status)
                           (>= status 200)
                           (< status 300))
                (error "OpenAI provider: HTTP %S: %s"
                       status
                       (substring body 0 (min 1000 (length body)))))
              (json-parse-string body
                                 :object-type 'plist
                                 :array-type 'list
                                 :null-object nil
                                 :false-object nil))))
      (kill-buffer buffer))))

;;;###autoload
(cl-defun nl-llm-agent-openai-provider
    (id &key base-url models api-key headers transport name
        (chat-path "/chat/completions") timeout-sec)
  "Create an OpenAI-compatible provider named ID.

BASE-URL normally ends in /v1.  MODELS is the static or callable catalog
accepted by `nl-llm-agent-provider-new'.  API-KEY may be a string or a
zero-argument function, allowing credential rotation without rebuilding a
session.  HEADERS is an alist.  TRANSPORT receives a request plist and returns
parsed JSON.  The default uses Emacs URL without exposing credentials in process
arguments.  TIMEOUT-SEC is the default per-request timeout in seconds, used
when the session options do not set :timeout-sec."
  (unless (and (stringp base-url)
               (string-match-p "\\`https?://" base-url))
    (error "OpenAI provider: BASE-URL must use http or https"))
  (unless (and (stringp chat-path) (string-prefix-p "/" chat-path))
    (error "OpenAI provider: CHAT-PATH must start with slash"))
  (unless (or (null api-key) (stringp api-key) (functionp api-key))
    (error "OpenAI provider: API-KEY must be nil, string, or function"))
  (when (and transport (not (functionp transport)))
    (error "OpenAI provider: TRANSPORT must be nil or a function"))
  (unless (or (null timeout-sec)
              (and (numberp timeout-sec) (> timeout-sec 0)))
    (error "OpenAI provider: TIMEOUT-SEC must be nil or a positive number"))
  (let ((endpoint
         (concat (replace-regexp-in-string "/+\\'" "" base-url)
                 chat-path))
        (configured-headers (copy-tree headers))
        (request-transport
         (or transport #'nl-llm-agent-openai-default-transport)))
    (nl-llm-agent-provider-new
     id
     :name name
     :models models
     :capabilities '(generate remote openai-compatible)
     :open
     (lambda (model options)
       (list :model model :options (copy-tree options)))
     :complete
     (lambda (state messages)
       (let* ((options (plist-get state :options))
              (request
               (list :url endpoint
                     :headers
                     (nl-llm-agent-openai--headers
                      configured-headers
                      (nl-llm-agent-openai--resolve-api-key api-key))
                     :body
                     (nl-llm-agent-openai--body
                      (plist-get state :model) messages options)
                     :timeout-sec (or (plist-get options :timeout-sec)
                                      timeout-sec)))
              (response (funcall request-transport request)))
         (nl-llm-agent-openai--response-text response))))))

(provide 'nl-llm-agent-openai)
;;; nl-llm-agent-openai.el ends here
