;;; agent-provider-test.el --- provider registry and live model switching  -*- lexical-binding: t; -*-

;; The agent runtime must be able to switch between local and remote models
;; without coupling its message loop to either backend.  A failed switch must
;; leave the current backend usable.
;;   emacs -Q --batch -L lisp -l test/agent-provider-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-provider)

(defvar agent-provider--fail 0)

(defun agent-provider--ck (name ok &optional extra)
  (princ (format "%-58s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-provider--fail
                         (1+ agent-provider--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-provider--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(let* ((events nil)
       (make-mock
        (lambda (id models)
          (nl-llm-agent-provider-new
           id
           :models models
           :capabilities '(generate stream)
           :open
           (lambda (model options)
             (setq events
                   (append events (list (list 'open id model options))))
             (when (equal model "broken")
               (error "mock open failure"))
             (list :provider id :model model :options options))
           :complete
           (lambda (state messages)
             (let ((model (plist-get state :model)))
               (setq events
                     (append events
                             (list (list 'complete id model
                                         (length messages)))))
               (format "%s/%s:%d" id model (length messages))))
           :close
           (lambda (state)
             (setq events
                   (append events
                           (list (list 'close id
                                       (plist-get state :model)))))))))
       (local (funcall make-mock
                       "local"
                       '((:id "small" :name "Small" :capabilities (generate train))
                         (:id "shared" :name "Local shared"))))
       (remote (funcall make-mock
                        "remote"
                        '((:id "large" :name "Large")
                          (:id "shared" :name "Remote shared")
                          (:id "poolside/laguna:free" :name "Slash id")
                          (:id "broken" :name "Broken"))))
       (registry (nl-llm-agent-provider-registry-new)))
  (nl-llm-agent-provider-register registry local)
  (nl-llm-agent-provider-register registry remote)

  (let ((models (nl-llm-agent-provider-models registry)))
    (agent-provider--ck "registry lists every provider model"
                        (= (length models) 6))
    (agent-provider--ck "model descriptors carry a qualified id"
                        (equal (mapcar (lambda (m)
                                         (plist-get m :qualified-id))
                                       models)
                               '("local/small" "local/shared"
                                 "remote/large" "remote/shared"
                                 "remote/poolside/laguna:free"
                                 "remote/broken"))))

  (agent-provider--ck
   "duplicate provider ids are rejected"
   (agent-provider--error-p
    (lambda () (nl-llm-agent-provider-register registry local))))
  (agent-provider--ck
   "an ambiguous unqualified model is rejected"
   (agent-provider--error-p
    (lambda () (nl-llm-agent-session-open registry "shared"))))
  (let ((slash-session
         (nl-llm-agent-session-open registry "poolside/laguna:free")))
    (agent-provider--ck "unqualified remote ids may contain slash"
                        (and (equal
                              (nl-llm-agent-session-provider-id slash-session)
                              "remote")
                             (equal
                              (nl-llm-agent-session-model-id slash-session)
                              "poolside/laguna:free")))
    (nl-llm-agent-session-close slash-session))

  (let* ((session
          (nl-llm-agent-session-open
           registry "local/small" :options '(:temperature 0.2)))
         (policy (nl-llm-agent-session-policy session))
         (first-messages '((system . "rules") (user . "hello")))
         (first (funcall policy first-messages)))
    (agent-provider--ck "session opens the selected provider and model"
                        (and (equal (nl-llm-agent-session-provider-id session)
                                    "local")
                             (equal (nl-llm-agent-session-model-id session)
                                    "small")
                             (equal first "local/small:2")))
    (agent-provider--ck "session records provider-neutral message history"
                        (= (length (nl-llm-agent-session-messages session)) 3))

    (let ((before (length events)))
      (nl-llm-agent-session-switch session "remote/large")
      (agent-provider--ck
       "a successful switch opens new before closing old"
       (equal (cl-subseq events before)
              '((open "remote" "large" (:temperature 0.2))
                (close "local" "small")))))
    (agent-provider--ck "model switching preserves conversation history"
                        (= (length
                            (nl-llm-agent-session-messages session))
                           3))

    (let ((after (funcall policy first-messages)))
      (agent-provider--ck "an existing policy sees the switched model"
                          (equal after "remote/large:2")))
    (agent-provider--ck "successful switching advances the generation"
                        (= (nl-llm-agent-session-generation session) 1))

    (condition-case nil
        (nl-llm-agent-session-switch session "remote/broken")
      (error nil))
    (agent-provider--ck
     "a failed switch keeps the previous backend active"
     (and (equal (nl-llm-agent-session-provider-id session) "remote")
          (equal (nl-llm-agent-session-model-id session) "large")
          (equal (funcall policy first-messages) "remote/large:2")))
    (agent-provider--ck "a failed switch is present in the audit history"
                        (eq (plist-get
                             (car (nl-llm-agent-session-history session))
                             :status)
                            'failed))
    (condition-case nil
        (nl-llm-agent-session-switch session "missing")
      (error nil))
    (agent-provider--ck
     "an unresolved switch is audited without changing the model"
     (let ((event (car (nl-llm-agent-session-history session))))
       (and (eq (plist-get event :status) 'failed)
            (equal (plist-get event :to) "missing")
            (equal (nl-llm-agent-session-model-id session) "large"))))

    ;; An unqualified selector first resolves within the current provider.
    (nl-llm-agent-session-switch session "shared")
    (agent-provider--ck "unqualified switching prefers the current provider"
                        (equal (nl-llm-agent-session-model-id session)
                               "shared"))
    (nl-llm-agent-session-switch session "shared" :options nil)
    (agent-provider--ck "explicit nil resets inherited provider options"
                        (null (nl-llm-agent-session-options session)))

    (nl-llm-agent-session-close session)
    (agent-provider--ck "closing a session is idempotent"
                        (progn (nl-llm-agent-session-close session) t))
    (agent-provider--ck
     "a closed session refuses completion"
     (agent-provider--error-p
      (lambda () (funcall policy first-messages)))))

  (let* ((bad
          (nl-llm-agent-provider-new
           "bad"
           :models '("invalid-output")
           :open (lambda (_model _options) 'state)
           :complete (lambda (_state _messages) 42)))
         (bad-registry (nl-llm-agent-provider-registry-new)))
    (nl-llm-agent-provider-register bad-registry bad)
    (let ((session
           (nl-llm-agent-session-open bad-registry "bad/invalid-output")))
      (agent-provider--ck
       "providers must return text completions"
       (agent-provider--error-p
       (lambda ()
          (nl-llm-agent-session-complete session '((user . "hello")))))))))

(let* ((provider
        (nl-llm-agent-provider-new
         "scripted"
         :models '("finish")
         :open (lambda (_model _options) nil)
         :complete (lambda (_state _messages) "DONE provider-backed")))
       (registry (nl-llm-agent-provider-registry-new)))
  (nl-llm-agent-provider-register registry provider)
  (let* ((session
          (nl-llm-agent-session-open registry "scripted/finish"))
         (result
          (nl-llm-agent-run
           "finish" (nl-llm-agent-session-policy session) :max-steps 1)))
    (agent-provider--ck
     "provider session policy plugs into the existing agent loop"
     (and (eq (plist-get result :status) 'done)
          (equal (plist-get result :result) "provider-backed")))))

(princ (format "NL-LLM-AGENT-PROVIDER %s (%d failures)\n"
               (if (= agent-provider--fail 0) "ALL-PASS" "HAS-FAILURES")
               agent-provider--fail))
(kill-emacs (if (= agent-provider--fail 0) 0 1))

;;; agent-provider-test.el ends here
