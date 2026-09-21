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

(defvar tl--fail 0)
(defvar tl--pass 0)

(defun tl--ck (name ok &optional detail)
  (if ok (setq tl--pass (1+ tl--pass)) (setq tl--fail (1+ tl--fail)))
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(defun tl--alt (tok lp) (list :token tok :logprob lp))

(defun tl--response (&optional with-logprobs)
  "A minimal chat completion for \"Hi\" + \"!\", with or without logprobs."
  (list :model "stub:1b"
        :choices
        (list (append
               (list :message (list :role "assistant" :content "Hi!"))
               (when with-logprobs
                 (list :logprobs
                       (list :content
                             (list (append (tl--alt "Hi" -0.1)
                                           (list :top_logprobs
                                                 (list (tl--alt "Hi" -0.1)
                                                       (tl--alt "Hey" -2.0))))
                                   (append (tl--alt "!" -0.4)
                                           (list :top_logprobs
                                                 (list (tl--alt "!" -0.4)
                                                       (tl--alt "." -1.2))))))))))))

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
