;;; teacher-logprobs-test.el --- soft targets, and the silence that hides them  -*- lexical-binding: t; -*-

;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/teacher-logprobs-test.el
;;
;; A stub transport carries most of this, because the failure worth testing is
;; a server that answers normally *without* logprobs -- which a live server
;; that implements them cannot demonstrate.  The last check does go to a real
;; teacher, and skips rather than fails when one is not running.

(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-teacher-logprobs)
(require 'nl-llm-teacher-cache)

(defvar tl--fail 0)
(defvar tl--pass 0)

(defun tl--ck (name ok &optional detail)
  (if ok (setq tl--pass (1+ tl--pass)) (setq tl--fail (1+ tl--fail)))
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(defun tl--alt (tok lp &optional bytes)
  (list :token tok :logprob lp :bytes bytes))

(defun tl--response (&optional with-logprobs)
  "A minimal chat completion for \"Hi\" + \"!\", with or without logprobs."
  (let ((choice (list :message (list :role "assistant" :content "Hi!"))))
    (when with-logprobs
      (setq choice
            (plist-put choice :logprobs
                       (list :content
                             (list
                              (append (tl--alt "Hi" -0.1 '(72 105))
                                      (list :top_logprobs
                                            (list (tl--alt "Hi" -0.1 '(72 105))
                                                  (tl--alt "Hey" -2.0 '(72 101 121)))))
                              (append (tl--alt "!" -0.4 '(33))
                                      (list :top_logprobs
                                            (list (tl--alt "!" -0.4 '(33))
                                                  (tl--alt "." -1.2 '(46))))))))))
    (list :model "stub:1b" :choices (list choice))))

(defun tl--transport (response &optional seen whole)
  (lambda (request)
    (when seen (setcar seen (plist-get request :body)))
    (when whole (setcar whole request))
    response))

;; The happy path: text and per-token alternatives both come back.
(let* ((seen (list nil))
       (a (nl-llm-teacher-logprobs-ask
           "say hi" :base-url "http://stub/v1" :model "stub:1b" :top-k 2
           :transport (tl--transport (tl--response t) seen))))
  (tl--ck "text comes back" (equal (plist-get a :text) "Hi!"))
  (tl--ck "every position carries its alternatives"
          (equal (mapcar (lambda (p) (mapcar (lambda (x) (plist-get x :token))
                                             (plist-get p :top)))
                         (plist-get a :tokens))
                 '(("Hi" "Hey") ("!" ".")))
          (nl-llm-teacher-logprobs-summary a))
  (tl--ck "the sampled token stream is the answer"
          (equal (nl-llm-teacher-logprobs-joined a) (plist-get a :text))
          "joined tokens == :text")
  (tl--ck "bytes survive on sampled and alternative tokens"
          (and (equal (plist-get (car (plist-get a :tokens)) :bytes) '(72 105))
               (equal (plist-get (car (plist-get (car (plist-get a :tokens)) :top))
                                :bytes)
                      '(72 105))
               (cl-every (lambda (p)
                           (cl-every (lambda (x) (plist-member x :bytes))
                                     (plist-get p :top)))
                         (plist-get a :tokens))))
  ;; The request has to actually ask, or a compliant server returns nothing and
  ;; the failure looks like the server's.
  (tl--ck "the request asks for logprobs"
          (and (eq (plist-get (car seen) :logprobs) t)
               (= (plist-get (car seen) :top_logprobs) 2))
          (format "top_logprobs=%S" (plist-get (car seen) :top_logprobs))))

;; The failure this module exists for: a normal-looking answer with no soft
;; targets at all.  Returning it would poison a dataset silently.
(tl--ck "a response without logprobs is refused"
        (condition-case err
            (progn (nl-llm-teacher-logprobs-ask
                    "say hi" :base-url "http://stub/v1" :model "stub:1b"
                    :transport (tl--transport (tl--response nil)))
                   nil)
          (error (and (string-match-p "no logprobs" (error-message-string err))
                      t)))
        "and names the model")

(let* ((response (tl--response nil))
       (fetched (nl-llm-teacher-logprobs-fetch
                 "say hi" :base-url "http://stub/v1" :model "stub:1b"
                 :transport (tl--transport response))))
  (tl--ck "fetch returns the raw response unchanged"
          (equal fetched response))
  (tl--ck "fetch does not refuse missing logprobs"
          (equal fetched response))
  (tl--ck "parse refuses that response"
          (condition-case nil
              (progn (nl-llm-teacher-logprobs-parse fetched :model "stub:1b") nil)
            (error t))))

;; The cache must preserve the raw response.  Parsing after the cache boundary
;; keeps future parser fields available without paying for the teacher again.
(let* ((dir (make-temp-file "tl-cache-" t))
       (calls (list 0))
       (response (tl--response t))
       (transport (lambda (_request)
                    (setcar calls (1+ (car calls)))
                    response))
       (teacher (nl-llm-teacher-logprobs-cached-teacher
                 :base-url "http://stub/v1" :model "stub:1b" :top-k 2
                 :dir dir :transport transport)))
  (unwind-protect
      (progn
        (let ((first (funcall teacher "cached hi")))
          (tl--ck "cached teacher returns parsed shape on a miss"
                  (plist-get first :tokens)))
        (let* ((path (car (directory-files-recursively dir "\\.eld\\'")))
               (entry (with-temp-buffer
                        (insert-file-contents path)
                        (goto-char (point-min))
                        (read (current-buffer))))
               (stored (plist-get entry :value)))
          (tl--ck "cache stores the raw response with choices"
                  (plist-get stored :choices))
          (tl--ck "cached value is not the parsed token shape"
                  (not (plist-member stored :tokens))))
        (let ((second (funcall teacher "cached hi")))
          (tl--ck "cache hit avoids transport and still parses"
                  (and (= (car calls) 1) (plist-get second :tokens)))))
    (delete-directory dir t)))

(let* ((dir (make-temp-file "tl-cache-missing-" t))
       (teacher (nl-llm-teacher-logprobs-cached-teacher
                 :base-url "http://stub/v1" :model "stub:1b" :top-k 2
                 :dir dir :transport (tl--transport (tl--response nil)))))
  (unwind-protect
      (tl--ck "cached teacher refuses and does not write missing logprobs"
              (condition-case nil
                  (progn (funcall teacher "uncached refusal") nil)
                (error (null (directory-files-recursively dir "\\.eld\\'")))))
    (delete-directory dir t)))

(let* ((response (tl--response t))
       (content (plist-get (plist-get (car (plist-get response :choices))
                                       :logprobs)
                           :content)))
  (dolist (entry content)
    (dolist (alt (cons entry (plist-get entry :top_logprobs)))
      (plist-put alt :bytes nil)))
  (let ((parsed (nl-llm-teacher-logprobs-parse response :model "stub:1b")))
    (tl--ck "missing bytes remain nil rather than refusing"
            (cl-every (lambda (p)
                        (and (null (plist-get p :bytes))
                             (cl-every (lambda (x) (null (plist-get x :bytes)))
                                       (plist-get p :top))))
                      (plist-get parsed :tokens)))))

(let* ((response (tl--response t))
       (entry (car (plist-get (plist-get (car (plist-get response :choices))
                                         :logprobs)
                             :content)))
       (alt (car (plist-get entry :top_logprobs))))
  (plist-put entry :token "�")
  (plist-put entry :bytes '(227 130 170))
  (plist-put alt :token "�")
  (plist-put alt :bytes '(227 130 170))
  (let ((parsed (nl-llm-teacher-logprobs-parse response :model "stub:1b")))
    (tl--ck "multi-byte token bytes survive exactly"
            (and (equal (plist-get (car (plist-get parsed :tokens)) :bytes)
                        '(227 130 170))
                 (equal (plist-get (car (plist-get (car (plist-get parsed :tokens))
                                                   :top))
                                    :bytes)
                        '(227 130 170))))))

(tl--ck "an empty response is refused too"
        (condition-case nil
            (progn (nl-llm-teacher-logprobs-ask
                    "say hi" :base-url "http://stub/v1" :model "stub:1b"
                    :transport (tl--transport '(:choices nil)))
                   nil)
          (error t)))

(tl--ck "text accessor takes a plain string as well"
        (equal (nl-llm-teacher-logprobs-text "plain") "plain"))

;; One options plist serves the body and the transport, so the transport's own
;; key must be taken out of the body: a strict endpoint answers an unknown
;; field with a 400, and a lenient one hides the mistake until it meets a
;; strict one.
(let* ((whole (list nil))
       (_ (nl-llm-teacher-logprobs-ask
           "say hi" :base-url "http://stub/v1" :model "stub:1b"
           :options '(:temperature 0.0 :timeout-sec 900)
           :transport (tl--transport (tl--response t) nil whole)))
       (req (car whole)))
  (tl--ck ":timeout-sec reaches the transport"
          (= (plist-get req :timeout-sec) 900))
  (tl--ck "and is kept out of the request body"
          (and (not (plist-member (plist-get req :body) :timeout-sec))
               (equal (plist-get (plist-get req :body) :temperature) 0.0))
          "other options still pass through"))

;; Live, when a teacher is running.  A stub cannot show that a real endpoint
;; honours `top_logprobs', which is the one thing a stub is free to fake.
(let* ((base (or (getenv "NL_LLM_OLLAMA_BASE_URL") "http://127.0.0.1:11434/v1"))
       (model (or (getenv "NL_LLM_OLLAMA_MODEL") "qwen3:4b"))
       (up (ignore-errors
             (= 0 (call-process "curl" nil nil nil "-sf" "-m" "3"
                                (concat (string-remove-suffix "/v1" base)
                                        "/api/version"))))))
  (if (not up)
      (princ (format "%-52s %s  %s\n" "live teacher" "skip"
                     (format "no server at %s" base)))
    (let ((a (nl-llm-teacher-logprobs-ask
              "Reply with the single word: ready"
              :base-url base :model model :top-k 5
              :options '(:temperature 0.0 :max_tokens 8))))
      (tl--ck "a live teacher returns per-token alternatives"
              (and (plist-get a :tokens)
                   (cl-every (lambda (p) (= (length (plist-get p :top)) 5))
                             (plist-get a :tokens)))
              (nl-llm-teacher-logprobs-summary a))
      (tl--ck "and its token stream reconstructs its text"
              (equal (nl-llm-teacher-logprobs-joined a) (plist-get a :text))))))

(princ (format "\nteacher-logprobs: %d passed, %d failed\n" tl--pass tl--fail))
(kill-emacs (if (= tl--fail 0) 0 1))

;;; teacher-logprobs-test.el ends here
