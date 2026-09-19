;;; nl-llm-agent-provider.el --- model providers and sessions  -*- lexical-binding: t; -*-

;; Keep the agent loop independent from any inference implementation.  A
;; provider owns backend-specific state; a session owns provider-neutral message
;; history and can transactionally switch models while an existing policy
;; closure keeps working.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'subr-x)

(cl-defstruct (nl-llm-agent-provider
               (:constructor nl-llm-agent-provider--make))
  id
  name
  catalog-fn
  open-fn
  complete-fn
  close-fn
  capabilities)

(cl-defstruct (nl-llm-agent-provider-registry
               (:constructor nl-llm-agent-provider-registry-new))
  (providers nil))

(cl-defstruct (nl-llm-agent-session
               (:constructor nl-llm-agent-session--make))
  registry
  provider-id
  model-id
  provider
  backend-state
  options
  messages
  generation
  status
  history
  last-error)

(defun nl-llm-agent-provider--id (value where &optional provider-id)
  "Return VALUE as a validated identifier for WHERE.
When PROVIDER-ID is non-nil, slash is accepted because remote model ids often
contain an organisation prefix."
  (let ((id (cond ((stringp value) value)
                  ((symbolp value) (symbol-name value))
                  (t nil))))
    (unless (and id (not (string-empty-p id)))
      (error "%s: id must be a non-empty string or symbol, got %S"
             where value))
    (when (and (not provider-id) (string-match-p "/" id))
      (error "%s: provider id must not contain slash, got %S" where id))
    id))

(defun nl-llm-agent-provider--model (value provider-id)
  "Normalize model descriptor VALUE for PROVIDER-ID."
  (let* ((descriptor
          (cond ((or (stringp value) (symbolp value))
                 (list :id (nl-llm-agent-provider--id
                            value "model" provider-id)))
                ((listp value) (copy-tree value))
                (t (error "provider %s: invalid model descriptor %S"
                          provider-id value))))
         (id (nl-llm-agent-provider--id
              (plist-get descriptor :id) "model" provider-id)))
    (append (list :provider provider-id
                  :qualified-id (concat provider-id "/" id)
                  :id id)
            descriptor)))

(defun nl-llm-agent-provider--catalog (provider)
  "Return PROVIDER's validated public model descriptors."
  (let ((raw (funcall (nl-llm-agent-provider-catalog-fn provider)))
        (provider-id (nl-llm-agent-provider-id provider))
        (seen nil)
        (result nil))
    (unless (listp raw)
      (error "provider %s: model catalog must be a list" provider-id))
    (dolist (value raw)
      (let* ((model (nl-llm-agent-provider--model value provider-id))
             (id (plist-get model :id)))
        (when (member id seen)
          (error "provider %s: duplicate model id %S" provider-id id))
        (push id seen)
        (push model result)))
    (nreverse result)))

;;;###autoload
(cl-defun nl-llm-agent-provider-new
    (id &key name models open complete close capabilities)
  "Create a model provider named ID.

MODELS is a list, or a zero-argument function returning model descriptors.
Each descriptor is a model id string or a plist containing :id.  OPEN is called
with (MODEL-ID OPTIONS) and returns private backend state.  COMPLETE is called
with (STATE MESSAGES) and must return text.  CLOSE, when non-nil, is called with
STATE.  CAPABILITIES describes provider-wide optional features."
  (let ((id (nl-llm-agent-provider--id id
                                       "nl-llm-agent-provider-new")))
    (unless (or (listp models) (functionp models))
      (error "provider %s: MODELS must be a list or function" id))
    (unless (functionp open)
      (error "provider %s: OPEN must be a function" id))
    (unless (functionp complete)
      (error "provider %s: COMPLETE must be a function" id))
    (when (and close (not (functionp close)))
      (error "provider %s: CLOSE must be nil or a function" id))
    (let ((catalog (if (functionp models)
                       models
                     (let ((fixed models))
                       (lambda () fixed)))))
      (nl-llm-agent-provider--make
       :id id
       :name (or name id)
       :catalog-fn catalog
       :open-fn open
       :complete-fn complete
       :close-fn (or close (lambda (_state) nil))
       :capabilities (copy-sequence capabilities)))))

(defun nl-llm-agent-provider--get (registry provider-id)
  "Return PROVIDER-ID from REGISTRY, or nil."
  (let ((id (nl-llm-agent-provider--id provider-id "provider lookup")))
    (cl-find-if
     (lambda (provider)
       (equal (nl-llm-agent-provider-id provider) id))
     (nl-llm-agent-provider-registry-providers registry))))

;;;###autoload
(defun nl-llm-agent-provider-register (registry provider)
  "Register PROVIDER in REGISTRY and return PROVIDER.
Provider order is stable so catalog presentation is deterministic."
  (unless (nl-llm-agent-provider-registry-p registry)
    (error "nl-llm-agent-provider-register: invalid registry"))
  (unless (nl-llm-agent-provider-p provider)
    (error "nl-llm-agent-provider-register: invalid provider"))
  (when (nl-llm-agent-provider--get
         registry (nl-llm-agent-provider-id provider))
    (error "provider already registered: %s"
           (nl-llm-agent-provider-id provider)))
  ;; Validate the catalog at registration so bad providers fail before use.
  (nl-llm-agent-provider--catalog provider)
  (setf (nl-llm-agent-provider-registry-providers registry)
        (append (nl-llm-agent-provider-registry-providers registry)
                (list provider)))
  provider)

;;;###autoload
(defun nl-llm-agent-provider-models (registry &optional provider-id)
  "Return public model descriptors from REGISTRY.
When PROVIDER-ID is non-nil, return only that provider's catalog."
  (unless (nl-llm-agent-provider-registry-p registry)
    (error "nl-llm-agent-provider-models: invalid registry"))
  (if provider-id
      (let ((provider (nl-llm-agent-provider--get registry provider-id)))
        (unless provider
          (error "unknown provider: %s" provider-id))
        (nl-llm-agent-provider--catalog provider))
    (apply #'append
           (mapcar #'nl-llm-agent-provider--catalog
                   (nl-llm-agent-provider-registry-providers registry)))))

(defun nl-llm-agent-provider--find-model (registry provider-id model-id)
  "Return MODEL-ID's descriptor under PROVIDER-ID in REGISTRY, or nil."
  (cl-find-if
   (lambda (model) (equal (plist-get model :id) model-id))
   (nl-llm-agent-provider-models registry provider-id)))

;;;###autoload
(defun nl-llm-agent-provider-resolve
    (registry selector &optional preferred-provider)
  "Resolve model SELECTOR in REGISTRY and return its descriptor.

SELECTOR may be PROVIDER/MODEL or an unqualified model id.  Model ids may
themselves contain slash.  A prefix is treated as a provider only when that
provider is registered.  An unqualified id first tries PREFERRED-PROVIDER, then
requires exactly one match across the registry."
  (let* ((selector
          (nl-llm-agent-provider--id selector "model selector" t))
         (slash (string-match "/" selector))
         (prefix (and slash (substring selector 0 slash)))
         (explicit (and prefix
                        (nl-llm-agent-provider--get registry prefix))))
    (cond
     (explicit
      (let* ((model-id (substring selector (1+ slash)))
             (model (nl-llm-agent-provider--find-model
                     registry prefix model-id)))
        (or model
            (error "unknown model: %s" selector))))
     ((and preferred-provider
           (nl-llm-agent-provider--get registry preferred-provider)
           (nl-llm-agent-provider--find-model
            registry preferred-provider selector))
      (nl-llm-agent-provider--find-model
       registry preferred-provider selector))
     (t
      (let ((matches
             (cl-remove-if-not
              (lambda (model)
                (equal (plist-get model :id) selector))
              (nl-llm-agent-provider-models registry))))
        (cond ((null matches) (error "unknown model: %s" selector))
              ((cdr matches)
               (error "ambiguous model %s; use PROVIDER/MODEL" selector))
              (t (car matches))))))))

(defun nl-llm-agent-session--record (session event)
  "Add EVENT to SESSION's newest-first audit history."
  (setf (nl-llm-agent-session-history session)
        (cons event (nl-llm-agent-session-history session)))
  event)

;;;###autoload
(cl-defun nl-llm-agent-session-open (registry selector &key options)
  "Open a provider session for model SELECTOR in REGISTRY.
OPTIONS is opaque provider configuration retained across later model switches."
  (let* ((model (nl-llm-agent-provider-resolve registry selector))
         (provider-id (plist-get model :provider))
         (model-id (plist-get model :id))
         (provider (nl-llm-agent-provider--get registry provider-id))
         (state (funcall (nl-llm-agent-provider-open-fn provider)
                         model-id options)))
    (nl-llm-agent-session--make
     :registry registry
     :provider-id provider-id
     :model-id model-id
     :provider provider
     :backend-state state
     :options options
     :messages nil
     :generation 0
     :status 'open
     :history nil
     :last-error nil)))

(defun nl-llm-agent-session--ensure-open (session where)
  "Signal when SESSION is not open, identifying WHERE."
  (unless (and (nl-llm-agent-session-p session)
               (eq (nl-llm-agent-session-status session) 'open))
    (error "%s: session is not open" where)))

;;;###autoload
(defun nl-llm-agent-session-complete (session messages)
  "Generate one completion from SESSION for provider-neutral MESSAGES.
The successful request and result become SESSION's message snapshot."
  (nl-llm-agent-session--ensure-open
   session "nl-llm-agent-session-complete")
  (unless (listp messages)
    (error "nl-llm-agent-session-complete: MESSAGES must be a list"))
  (let* ((provider (nl-llm-agent-session-provider session))
         (result
          (funcall (nl-llm-agent-provider-complete-fn provider)
                   (nl-llm-agent-session-backend-state session)
                   messages)))
    (unless (stringp result)
      (error "provider %s returned a non-string completion: %S"
             (nl-llm-agent-provider-id provider) result))
    (setf (nl-llm-agent-session-messages session)
          (copy-tree
           (append messages (list (cons 'assistant result)))))
    result))

;;;###autoload
(defun nl-llm-agent-session-policy (session)
  "Return a live agent policy backed by SESSION.
The closure dereferences SESSION on every turn, so model switching takes effect
without rebuilding `nl-llm-agent-run' or losing provider-neutral history."
  (lambda (messages)
    (nl-llm-agent-session-complete session messages)))

;;;###autoload
(defun nl-llm-agent-session-switch (session selector &rest keys)
  "Transactionally switch SESSION to model SELECTOR.

The new backend is opened before the old one is closed.  If opening fails, the
old backend remains active and the failed attempt is audited.  OPTIONS defaults
to the current options when omitted.  Provider-neutral messages are preserved."
  (nl-llm-agent-session--ensure-open session
                                     "nl-llm-agent-session-switch")
  ;; NeLisp's cl-defun currently cannot distinguish an omitted &key argument
  ;; from an explicitly supplied nil.  Parse this one key directly so provider
  ;; options have identical semantics on both substrates.
  (unless (or (null keys)
              (and (= (length keys) 2) (eq (car keys) :options)))
    (error "nl-llm-agent-session-switch: expected optional :options VALUE"))
  (let* ((registry (nl-llm-agent-session-registry session))
         (old-provider (nl-llm-agent-session-provider session))
         (old-provider-id (nl-llm-agent-session-provider-id session))
         (old-model-id (nl-llm-agent-session-model-id session))
         (old-state (nl-llm-agent-session-backend-state session))
         (target selector))
    (condition-case err
        (let* ((model (nl-llm-agent-provider-resolve
                       registry selector old-provider-id))
               (new-provider-id (plist-get model :provider))
               (new-model-id (plist-get model :id))
               (new-provider
                (nl-llm-agent-provider--get registry new-provider-id))
               (new-options
                (if keys
                    (cadr keys)
                  (nl-llm-agent-session-options session))))
          (setq target (plist-get model :qualified-id))
          (if (and (equal old-provider-id new-provider-id)
                   (equal old-model-id new-model-id)
                   (equal (nl-llm-agent-session-options session)
                          new-options))
              session
            (let ((new-state
                   (funcall (nl-llm-agent-provider-open-fn new-provider)
                            new-model-id new-options))
                  (close-error nil)
                  (generation
                   (1+ (nl-llm-agent-session-generation session))))
              ;; Commit only after the new backend opened successfully.
              (setf (nl-llm-agent-session-provider-id session)
                    new-provider-id)
              (setf (nl-llm-agent-session-model-id session) new-model-id)
              (setf (nl-llm-agent-session-provider session) new-provider)
              (setf (nl-llm-agent-session-backend-state session) new-state)
              (setf (nl-llm-agent-session-options session) new-options)
              (setf (nl-llm-agent-session-generation session) generation)
              (setf (nl-llm-agent-session-last-error session) nil)
              (condition-case close-err
                  (funcall (nl-llm-agent-provider-close-fn old-provider)
                           old-state)
                (error (setq close-error (format "%S" close-err))))
              (nl-llm-agent-session--record
               session
               (list :status 'switched
                     :generation generation
                     :from (concat old-provider-id "/" old-model-id)
                     :to target
                     :close-error close-error))
              session)))
      (error
       (let ((message (format "%S" err)))
         (setf (nl-llm-agent-session-last-error session) message)
         (nl-llm-agent-session--record
          session
          (list :status 'failed
                :generation (nl-llm-agent-session-generation session)
                :from (concat old-provider-id "/" old-model-id)
                :to target
                :error message))
         (signal (car err) (cdr err)))))))

;;;###autoload
(defun nl-llm-agent-session-close (session)
  "Close SESSION once and release its current backend state."
  (unless (nl-llm-agent-session-p session)
    (error "nl-llm-agent-session-close: invalid session"))
  (when (eq (nl-llm-agent-session-status session) 'open)
    (let ((provider (nl-llm-agent-session-provider session))
          (state (nl-llm-agent-session-backend-state session))
          (close-error nil))
      (setf (nl-llm-agent-session-status session) 'closed)
      (setf (nl-llm-agent-session-backend-state session) nil)
      (condition-case err
          (funcall (nl-llm-agent-provider-close-fn provider) state)
        (error (setq close-error (format "%S" err))))
      (setf (nl-llm-agent-session-last-error session) close-error)
      (nl-llm-agent-session--record
       session
       (list :status 'closed
             :generation (nl-llm-agent-session-generation session)
             :model (concat (nl-llm-agent-session-provider-id session)
                            "/"
                            (nl-llm-agent-session-model-id session))
             :close-error close-error))))
  session)

(provide 'nl-llm-agent-provider)
;;; nl-llm-agent-provider.el ends here
