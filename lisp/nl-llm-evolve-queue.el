;;; nl-llm-evolve-queue.el --- evaluated model improvement queue  -*- lexical-binding: t; -*-

;; Model output may request an experiment, but it must never select executable
;; code or replace the fixed evaluator.  This controller accepts bounded data
;; proposals, resolves them through trusted allowlisted handlers, and delegates
;; the actual champion/challenger transaction to `nl-llm-evolve'.

;;; Code:

(require 'nl-llm-compat)
(require 'cl-lib)
(require 'nl-llm-evolve)

(defvar read-eval)

(cl-defstruct (nl-llm-evolve-queue-handler
               (:constructor nl-llm-evolve-queue-handler--make))
  kind
  description
  validate-fn
  propose-fn
  train-fn
  resume-fn
  finish-fn)

(cl-defstruct (nl-llm-evolve-queue-job
               (:constructor nl-llm-evolve-queue-job--make))
  id
  kind
  payload
  metadata
  priority
  sequence
  status
  result)

(cl-defstruct (nl-llm-evolve-queue
               (:constructor nl-llm-evolve-queue--make))
  evolution
  handlers
  jobs
  history
  max-pending
  max-history
  checkpoint-file
  next-sequence
  active-claim)

(defconst nl-llm-evolve-queue-checkpoint-format
  "nl-llm-evolve-queue-v1"
  "Format tag for private durable proposal queue state.")

(defconst nl-llm-evolve-queue-max-checkpoint-bytes (* 16 1024 1024)
  "Maximum accepted durable proposal queue checkpoint size.")

(defconst nl-llm-evolve-queue-max-data-nodes 4096
  "Maximum nodes accepted in one model-supplied proposal tree.")

(defconst nl-llm-evolve-queue-max-data-depth 32
  "Maximum nesting depth accepted in one proposal tree.")

(defconst nl-llm-evolve-queue-max-string-length 65536
  "Maximum length of one string in a proposal tree.")

(defvar nl-llm-evolve-queue-execution-context nil
  "Dynamically bound private context for a currently executing queue job.")

(defun nl-llm-evolve-queue-current-execution-context ()
  "Return detached private context for the current trusted handler."
  (unless nl-llm-evolve-queue-execution-context
    (error "no model improvement job is currently executing"))
  (copy-tree nl-llm-evolve-queue-execution-context))

(defun nl-llm-evolve-queue--identifier (value where)
  "Return VALUE as a bounded identifier for WHERE."
  (let ((text (cond ((stringp value) value)
                    ((symbolp value) (symbol-name value))
                    (t nil))))
    (unless (and text
                 (<= 1 (length text))
                 (<= (length text) 128)
                 (string-match-p "\\`[A-Za-z0-9_.-]+\\'" text))
      (error "%s has invalid identifier %S" where value))
    text))

(defun nl-llm-evolve-queue--data-copy (value where)
  "Validate and detach data-only VALUE supplied for WHERE."
  (let ((nodes 0)
        (seen nil))
    (cl-labels
        ((copy-one
          (object depth)
          (setq nodes (1+ nodes))
          (when (> nodes nl-llm-evolve-queue-max-data-nodes)
            (error "%s exceeds %d data nodes"
                   where nl-llm-evolve-queue-max-data-nodes))
          (when (> depth nl-llm-evolve-queue-max-data-depth)
            (error "%s exceeds data depth %d"
                   where nl-llm-evolve-queue-max-data-depth))
          (cond
           ((or (null object) (eq object t) (symbolp object)) object)
           ((numberp object)
            (unless (= object object)
              (error "%s contains a non-finite number" where))
            object)
           ((stringp object)
            (when (> (length object)
                     nl-llm-evolve-queue-max-string-length)
              (error "%s contains an oversized string" where))
            (copy-sequence object))
           ((and (functionp object) (not (symbolp object)))
            (error "%s contains an executable function" where))
           ((consp object)
            (when (memq object seen)
              (error "%s contains shared or cyclic list data" where))
            (push object seen)
            (cons (copy-one (car object) (1+ depth))
                  (copy-one (cdr object) (1+ depth))))
           ((vectorp object)
            (when (memq object seen)
              (error "%s contains shared or cyclic vector data" where))
            (push object seen)
            (let* ((length (length object))
                   (result (make-vector length nil)))
              (dotimes (index length)
                (aset result index
                      (copy-one (aref object index) (1+ depth))))
              result))
           (t (error "%s contains unsupported data %S" where object)))))
      (copy-one value 0))))

(defun nl-llm-evolve-queue--keys (value allowed where)
  "Validate plist VALUE keys against ALLOWED for WHERE."
  (unless (and (listp value) (= (% (length value) 2) 0))
    (error "%s must be a plist" where))
  (let ((tail value))
    (while tail
      (unless (memq (car tail) allowed)
        (error "%s contains unknown key %S" where (car tail)))
      (setq tail (cddr tail))))
  value)

(defun nl-llm-evolve-queue--handler (queue kind)
  "Return QUEUE handler KIND, or nil."
  (let ((kind (nl-llm-evolve-queue--identifier kind "proposal kind")))
    (cl-find-if
     (lambda (handler)
       (equal (nl-llm-evolve-queue-handler-kind handler) kind))
     (nl-llm-evolve-queue-handlers queue))))

(defun nl-llm-evolve-queue--job (queue id)
  "Return QUEUE job ID, or nil."
  (let ((id (nl-llm-evolve-queue--identifier id "proposal id")))
    (cl-find-if
     (lambda (job) (equal (nl-llm-evolve-queue-job-id job) id))
     (nl-llm-evolve-queue-jobs queue))))

(defun nl-llm-evolve-queue--public-job (job)
  "Return model-visible state for JOB without its executable payload."
  (append
   (list :id (nl-llm-evolve-queue-job-id job)
         :kind (nl-llm-evolve-queue-job-kind job)
         :priority (nl-llm-evolve-queue-job-priority job)
         :sequence (nl-llm-evolve-queue-job-sequence job)
         :status (nl-llm-evolve-queue-job-status job)
         :metadata (copy-tree (nl-llm-evolve-queue-job-metadata job)))
   (when (nl-llm-evolve-queue-job-result job)
     (list :result (copy-tree (nl-llm-evolve-queue-job-result job))))))

(defun nl-llm-evolve-queue--terminal-status-p (status)
  "Return non-nil when STATUS cannot execute again."
  (memq status '(promoted rejected error cancelled)))

(defun nl-llm-evolve-queue--outstanding-status-p (status)
  "Return non-nil when STATUS consumes admission capacity."
  (memq status '(pending running interrupted)))

(defun nl-llm-evolve-queue--outstanding-count (queue)
  "Return number of outstanding jobs in QUEUE."
  (cl-count-if
   (lambda (job)
     (nl-llm-evolve-queue--outstanding-status-p
      (nl-llm-evolve-queue-job-status job)))
   (nl-llm-evolve-queue-jobs queue)))

(defun nl-llm-evolve-queue--validate-job (queue job)
  "Validate JOB's payload against its currently registered handler."
  (let ((handler (nl-llm-evolve-queue--handler
                  queue (nl-llm-evolve-queue-job-kind job))))
    (unless handler
      (error "improvement proposal %s has no registered handler"
             (nl-llm-evolve-queue-job-id job)))
    (when (nl-llm-evolve-queue-handler-validate-fn handler)
      (funcall (nl-llm-evolve-queue-handler-validate-fn handler)
               (nl-llm-evolve-queue--data-copy
                (nl-llm-evolve-queue-job-payload job)
                "proposal validation")))
    handler))

(defun nl-llm-evolve-queue--private-job (job)
  "Return a data-only private checkpoint form for JOB."
  (list :id (nl-llm-evolve-queue-job-id job)
        :kind (nl-llm-evolve-queue-job-kind job)
        :payload (nl-llm-evolve-queue--data-copy
                  (nl-llm-evolve-queue-job-payload job)
                  "checkpoint proposal payload")
        :metadata (nl-llm-evolve-queue--data-copy
                   (nl-llm-evolve-queue-job-metadata job)
                   "checkpoint proposal metadata")
        :priority (nl-llm-evolve-queue-job-priority job)
        :sequence (nl-llm-evolve-queue-job-sequence job)
        :status (nl-llm-evolve-queue-job-status job)
        :result (nl-llm-evolve-queue--data-copy
                 (nl-llm-evolve-queue-job-result job)
                 "checkpoint proposal result")))

(defun nl-llm-evolve-queue--rebuild-history (queue)
  "Rebuild QUEUE's newest-first public terminal history."
  (setf (nl-llm-evolve-queue-history queue)
        (mapcar
         #'nl-llm-evolve-queue--public-job
         (reverse
          (cl-remove-if-not
           (lambda (job)
             (nl-llm-evolve-queue--terminal-status-p
              (nl-llm-evolve-queue-job-status job)))
           (nl-llm-evolve-queue-jobs queue))))))

(defun nl-llm-evolve-queue--trim-history (queue)
  "Bound terminal jobs retained by QUEUE and rebuild public history."
  (let ((terminal 0)
        (kept nil)
        (maximum (nl-llm-evolve-queue-max-history queue)))
    (dolist (job (reverse (nl-llm-evolve-queue-jobs queue)))
      (if (nl-llm-evolve-queue--terminal-status-p
           (nl-llm-evolve-queue-job-status job))
          (when (< terminal maximum)
            (setq terminal (1+ terminal))
            (push job kept))
        (push job kept)))
    (setf (nl-llm-evolve-queue-jobs queue) kept)
    (nl-llm-evolve-queue--rebuild-history queue)))

;;;###autoload
(defun nl-llm-evolve-queue-snapshot (queue)
  "Return private, data-only durable state for QUEUE."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-snapshot: invalid queue"))
  (list
   :format nl-llm-evolve-queue-checkpoint-format
   :next-sequence (nl-llm-evolve-queue-next-sequence queue)
   :evolution-generation
   (nl-llm-evolution-generation (nl-llm-evolve-queue-evolution queue))
   :jobs
   (apply #'vector
          (mapcar #'nl-llm-evolve-queue--private-job
                  (nl-llm-evolve-queue-jobs queue)))))

;;;###autoload
(defun nl-llm-evolve-queue-save (queue &optional file)
  "Atomically save private QUEUE state to mode-0600 FILE.

FILE defaults to the queue's configured checkpoint path."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-save: invalid queue"))
  (let* ((selected (or file (nl-llm-evolve-queue-checkpoint-file queue)))
         (path (and selected (expand-file-name selected))))
    (unless path
      (error "nl-llm-evolve-queue-save: no checkpoint file configured"))
    (let* ((directory (file-name-directory path))
           (temporary nil)
           (text
            (let ((print-length nil) (print-level nil))
              (prin1-to-string (nl-llm-evolve-queue-snapshot queue)))))
      (make-directory directory t)
      (setq temporary
            (make-temp-file
             (expand-file-name ".evolve-queue-checkpoint-" directory)))
      (unwind-protect
          (progn
            (let ((coding-system-for-write 'utf-8))
              (write-region text nil temporary nil 'silent))
            (set-file-modes temporary #o600)
            (rename-file temporary path t)
            (setq temporary nil)
            (setf (nl-llm-evolve-queue-checkpoint-file queue) path)
            path)
        (when (and temporary (file-exists-p temporary))
          (delete-file temporary))))))

(defun nl-llm-evolve-queue--save-if-configured (queue)
  "Persist QUEUE when it has a checkpoint path."
  (when (nl-llm-evolve-queue-checkpoint-file queue)
    (nl-llm-evolve-queue-save queue)))

(defun nl-llm-evolve-queue--checkpoint-read (path)
  "Read and return a bounded data checkpoint from PATH."
  (unless (file-regular-p path)
    (error "improvement queue checkpoint does not exist: %s" path))
  (when (> (file-attribute-size (file-attributes path))
           nl-llm-evolve-queue-max-checkpoint-bytes)
    (error "improvement queue checkpoint exceeds %d bytes"
           nl-llm-evolve-queue-max-checkpoint-bytes))
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8))
      (insert-file-contents path))
    (let* ((text (buffer-string))
           (read-eval nil)
           (parsed (read-from-string text))
           (value (car parsed))
           (trailing (substring text (cdr parsed))))
      (unless (string-match-p "\\`[ \\t\\r\\n]*\\'" trailing)
        (error "improvement queue checkpoint contains trailing data"))
      value)))

;;;###autoload
(defun nl-llm-evolve-queue-restore (queue &optional file missing-ok)
  "Restore private jobs into empty QUEUE from FILE.

FILE defaults to the configured checkpoint path.  When MISSING-OK is non-nil,
a missing file leaves QUEUE empty.  A job marked running is restored as
interrupted and is never executed automatically."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-restore: invalid queue"))
  (when (nl-llm-evolve-queue-jobs queue)
    (error "improvement queue restore requires an empty queue"))
  (let* ((selected (or file (nl-llm-evolve-queue-checkpoint-file queue)))
         (path (and selected (expand-file-name selected))))
    (unless path
      (error "nl-llm-evolve-queue-restore: no checkpoint file configured"))
    (setf (nl-llm-evolve-queue-checkpoint-file queue) path)
    (if (not (file-exists-p path))
        (if missing-ok queue
          (error "improvement queue checkpoint does not exist: %s" path))
      (let* ((data (nl-llm-evolve-queue--checkpoint-read path))
             (jobs nil)
             (seen nil)
             (outstanding 0)
             (maximum-sequence 0)
             (interrupted nil))
        (nl-llm-evolve-queue--keys
         data '(:format :next-sequence :evolution-generation :jobs)
         "improvement queue checkpoint")
        (unless (equal (plist-get data :format)
                       nl-llm-evolve-queue-checkpoint-format)
          (error "unsupported improvement queue checkpoint format %S"
                 (plist-get data :format)))
        (let ((saved-next (plist-get data :next-sequence))
              (saved-generation (plist-get data :evolution-generation))
              (forms (plist-get data :jobs)))
          (unless (and (integerp saved-next) (>= saved-next 0))
            (error "improvement queue checkpoint has invalid sequence"))
          (unless (vectorp forms)
            (error "improvement queue checkpoint requires a jobs vector"))
          (unless (and (integerp saved-generation)
                       (>= saved-generation 0))
            (error "improvement queue checkpoint has invalid generation"))
          (when (> saved-generation
                   (nl-llm-evolution-generation
                    (nl-llm-evolve-queue-evolution queue)))
            (error "improvement queue checkpoint is ahead of its champion"))
          (when (> (length forms)
                   (+ (nl-llm-evolve-queue-max-pending queue)
                      (nl-llm-evolve-queue-max-history queue) 1))
            (error "improvement queue checkpoint contains too many jobs"))
          (dolist (form (append forms nil))
            (nl-llm-evolve-queue--keys
             form '(:id :kind :payload :metadata :priority :sequence
                        :status :result)
             "improvement queue job")
            (let* ((id
                    (nl-llm-evolve-queue--identifier
                     (plist-get form :id) "checkpoint proposal id"))
                   (kind
                    (nl-llm-evolve-queue--identifier
                     (plist-get form :kind) "checkpoint proposal kind"))
                   (payload
                    (nl-llm-evolve-queue--data-copy
                     (plist-get form :payload) "checkpoint proposal payload"))
                   (metadata
                    (nl-llm-evolve-queue--data-copy
                     (plist-get form :metadata) "checkpoint proposal metadata"))
                   (priority (plist-get form :priority))
                   (sequence (plist-get form :sequence))
                   (status (plist-get form :status))
                   (result
                    (nl-llm-evolve-queue--data-copy
                     (plist-get form :result) "checkpoint proposal result")))
              (when (member id seen)
                (error "duplicate checkpoint proposal id: %s" id))
              (unless (and (integerp priority)
                           (<= -1000 priority) (<= priority 1000))
                (error "checkpoint proposal %s has invalid priority" id))
              (unless (and (integerp sequence) (> sequence 0))
                (error "checkpoint proposal %s has invalid sequence" id))
              (unless (memq status
                            '(pending running promoted rejected error
                                      cancelled interrupted))
                (error "checkpoint proposal %s has invalid status" id))
              (when (nl-llm-evolve-queue--outstanding-status-p status)
                (setq outstanding (1+ outstanding)))
              (when (memq status '(pending running interrupted))
                (let ((handler (nl-llm-evolve-queue--handler queue kind)))
                  (unless handler
                    (error "checkpoint proposal %s has unknown handler %s"
                           id kind))
                  (when (nl-llm-evolve-queue-handler-validate-fn handler)
                    (funcall
                     (nl-llm-evolve-queue-handler-validate-fn handler)
                     (nl-llm-evolve-queue--data-copy
                      payload "checkpoint proposal validation"))))
                (if (eq status 'running)
                    (setq status 'interrupted
                          result
                          '(:status interrupted :reason host-restart)
                          interrupted t)
                  nil))
              (setq seen (cons id seen))
              (setq maximum-sequence (max maximum-sequence sequence))
              (setq jobs
                    (append
                     jobs
                     (list
                      (nl-llm-evolve-queue-job--make
                       :id id :kind kind :payload payload :metadata metadata
                       :priority priority :sequence sequence :status status
                       :result result))))))
          ;; Check all outstanding states before committing any restored jobs;
          ;; RUNNING is counted here before its local restart conversion.
          (when (> outstanding (nl-llm-evolve-queue-max-pending queue))
            (error "checkpoint exceeds outstanding queue capacity"))
          (when (< saved-next maximum-sequence)
            (error "checkpoint next sequence precedes a stored job"))
          (setf (nl-llm-evolve-queue-jobs queue) jobs)
          (setf (nl-llm-evolve-queue-next-sequence queue) saved-next)
          (nl-llm-evolve-queue--trim-history queue)
          (when interrupted
            (nl-llm-evolve-queue-save queue))
          queue)))))

;;;###autoload
(cl-defun nl-llm-evolve-queue-new
    (evolution &key (max-pending 64) (max-history 256) checkpoint-file)
  "Create an evaluated proposal queue around EVOLUTION.

MAX-PENDING bounds pending, running, and interrupted experiments;
MAX-HISTORY bounds terminal jobs retained for audit.  CHECKPOINT-FILE enables
private atomic persistence.
The evolution state's fixed evaluator and publisher remain authoritative;
proposals cannot replace them."
  (unless (nl-llm-evolution-p evolution)
    (error "nl-llm-evolve-queue-new: invalid evolution state"))
  (unless (and (integerp max-pending) (> max-pending 0))
    (error "nl-llm-evolve-queue-new: MAX-PENDING must be positive"))
  (unless (and (integerp max-history) (>= max-history 0))
    (error "nl-llm-evolve-queue-new: MAX-HISTORY must be non-negative"))
  (when (and checkpoint-file (not (stringp checkpoint-file)))
    (error "nl-llm-evolve-queue-new: CHECKPOINT-FILE must be text"))
  (nl-llm-evolve-queue--make
   :evolution evolution :handlers nil :jobs nil :history nil
   :max-pending max-pending :max-history max-history
   :checkpoint-file (and checkpoint-file (expand-file-name checkpoint-file))
   :next-sequence 0
   :active-claim nil))

;;;###autoload
(cl-defun nl-llm-evolve-queue-register
    (queue kind propose &key train resume finish validate description)
  "Register trusted proposal handler KIND in QUEUE.

PROPOSE receives (CANDIDATE PAYLOAD EVOLUTION) and may mutate the isolated
candidate.  TRAIN has the same arguments and runs after PROPOSE.  VALIDATE,
when non-nil, receives a detached payload and must signal on invalid data; its
return value is ignored.  RESUME is a separate trusted training callback used
only for an explicitly resumed interrupted job.  FINISH receives private
execution context and the terminal result after durable queue persistence, for
bounded checkpoint cleanup.  Model-supplied data never selects these functions."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-register: invalid queue"))
  (let ((kind (nl-llm-evolve-queue--identifier kind "handler kind")))
    (when (nl-llm-evolve-queue--handler queue kind)
      (error "improvement handler already registered: %s" kind))
    (unless (functionp propose)
      (error "improvement handler %s requires a proposal function" kind))
    (when (and train (not (functionp train)))
      (error "improvement handler %s has invalid training function" kind))
    (when (and resume (not (functionp resume)))
      (error "improvement handler %s has invalid resume function" kind))
    (when (and finish (not (functionp finish)))
      (error "improvement handler %s has invalid finish function" kind))
    (when (and validate (not (functionp validate)))
      (error "improvement handler %s has invalid validator" kind))
    (unless (or (null description) (stringp description))
      (error "improvement handler %s description must be text" kind))
    (let ((handler
           (nl-llm-evolve-queue-handler--make
            :kind kind :description (or description "")
            :validate-fn validate :propose-fn propose :train-fn train
            :resume-fn resume :finish-fn finish)))
      (setf (nl-llm-evolve-queue-handlers queue)
            (append (nl-llm-evolve-queue-handlers queue) (list handler)))
      handler)))

;;;###autoload
(defun nl-llm-evolve-queue-catalog (queue)
  "Return QUEUE's public allowlisted proposal kinds without callbacks."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-catalog: invalid queue"))
  (mapcar
   (lambda (handler)
     (append
      (list :kind (nl-llm-evolve-queue-handler-kind handler)
            :description
            (nl-llm-evolve-queue-handler-description handler))
      (when (nl-llm-evolve-queue-handler-resume-fn handler)
        (list :resumable t))))
   (nl-llm-evolve-queue-handlers queue)))

;;;###autoload
(cl-defun nl-llm-evolve-queue-submit
    (queue kind payload &key id (priority 0) metadata)
  "Submit a bounded data proposal to QUEUE and return its public job state."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-submit: invalid queue"))
  (let* ((kind (nl-llm-evolve-queue--identifier kind "proposal kind"))
         (handler (nl-llm-evolve-queue--handler queue kind))
         (outstanding (nl-llm-evolve-queue--outstanding-count queue)))
    (unless handler
      (error "unknown improvement proposal kind: %s" kind))
    (when (>= outstanding (nl-llm-evolve-queue-max-pending queue))
      (error "improvement proposal queue is full"))
    (unless (and (integerp priority) (<= -1000 priority) (<= priority 1000))
      (error "proposal priority must be an integer in [-1000, 1000]"))
    (let* ((sequence (1+ (nl-llm-evolve-queue-next-sequence queue)))
           (id (nl-llm-evolve-queue--identifier
                (or id (format "proposal-%d" sequence)) "proposal id"))
           (payload
            (nl-llm-evolve-queue--data-copy payload "proposal payload"))
           (metadata
            (nl-llm-evolve-queue--data-copy metadata "proposal metadata")))
      (when (nl-llm-evolve-queue--job queue id)
        (error "improvement proposal id already exists: %s" id))
      (when (nl-llm-evolve-queue-handler-validate-fn handler)
        (funcall
         (nl-llm-evolve-queue-handler-validate-fn handler)
         (nl-llm-evolve-queue--data-copy payload "proposal validation")))
      (let ((job
             (nl-llm-evolve-queue-job--make
              :id id :kind kind :payload payload :metadata metadata
              :priority priority :sequence sequence :status 'pending
              :result nil)))
        (let ((old-jobs (nl-llm-evolve-queue-jobs queue))
              (old-sequence (nl-llm-evolve-queue-next-sequence queue)))
          (setf (nl-llm-evolve-queue-next-sequence queue) sequence)
          (setf (nl-llm-evolve-queue-jobs queue)
                (append old-jobs (list job)))
          (condition-case err
              (nl-llm-evolve-queue--save-if-configured queue)
            (error
             (setf (nl-llm-evolve-queue-jobs queue) old-jobs)
             (setf (nl-llm-evolve-queue-next-sequence queue) old-sequence)
             (signal (car err) (cdr err))))
          (nl-llm-evolve-queue--public-job job))))))

(defun nl-llm-evolve-queue--next-pending (queue)
  "Return QUEUE's highest-priority stable pending job, or nil."
  (let ((best nil))
    (dolist (job (nl-llm-evolve-queue-jobs queue))
      (when (and (eq (nl-llm-evolve-queue-job-status job) 'pending)
                 (or (null best)
                     (> (nl-llm-evolve-queue-job-priority job)
                        (nl-llm-evolve-queue-job-priority best))))
        (setq best job)))
    best))

(defun nl-llm-evolve-queue--running-p (queue)
  "Return non-nil when QUEUE contains a running job."
  (cl-some (lambda (job) (eq (nl-llm-evolve-queue-job-status job) 'running))
           (nl-llm-evolve-queue-jobs queue)))

(defun nl-llm-evolve-queue--attempt-id (queue job evolution)
  "Return a process-bound, restart-resistant opaque attempt identifier."
  (let ((pid (if (fboundp 'emacs-pid) (emacs-pid) 0))
        (seed (format "%s-%s-%s-%s-%s"
                      (nl-llm-evolve-queue-job-id job)
                      (nl-llm-evolution-generation evolution)
                      (nl-llm-evolve-queue-next-sequence queue)
                      (float-time)
                      (if (fboundp 'emacs-pid) (emacs-pid) 0))))
    (if (fboundp 'secure-hash)
        (secure-hash 'sha256 seed)
      (format "%x-%x-%x" (truncate (* 1000000 (float-time)))
              (random most-positive-fixnum) pid))))

(defun nl-llm-evolve-queue--claim-valid (queue claim)
  "Return the job for CLAIM, requiring the exact active claim object."
  (unless (and (nl-llm-evolve-queue-p queue)
               (eq claim (nl-llm-evolve-queue-active-claim queue)))
    (error "invalid or inactive improvement queue claim"))
  (let ((job (nl-llm-evolve-queue--job queue (plist-get claim :job-id))))
    (unless (and job (eq (nl-llm-evolve-queue-job-status job) 'running))
      (error "improvement queue claim is no longer running"))
    job))

;;;###autoload
(defun nl-llm-evolve-queue-claim (queue &optional id resuming)
  "Claim one pending job, or explicitly resume interrupted ID.

The returned opaque host token contains detached payload and parent binding;
callbacks and the token are never exposed by status or checkpoint APIs."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-claim: invalid queue"))
  (when (nl-llm-evolve-queue-active-claim queue)
    (error "an asynchronous improvement claim is already active"))
  (when (nl-llm-evolve-queue--running-p queue)
    (error "a synchronous improvement job is already running"))
  (let ((job (if id
                 (nl-llm-evolve-queue--job queue id)
               (nl-llm-evolve-queue--next-pending queue))))
    (unless job (error "no claimable improvement proposal"))
    (let ((status (nl-llm-evolve-queue-job-status job)))
      (unless (if resuming (eq status 'interrupted) (eq status 'pending))
        (error "improvement proposal %s is not claimable as requested"
               (nl-llm-evolve-queue-job-id job))))
    (let* ((handler (nl-llm-evolve-queue--validate-job queue job))
           (resume-fn (nl-llm-evolve-queue-handler-resume-fn handler)))
      (when (and resuming (not resume-fn))
        (error "improvement proposal %s is not resumable"
               (nl-llm-evolve-queue-job-id job)))
      (let* ((evolution (nl-llm-evolve-queue-evolution queue))
             (claim (list :job-id (nl-llm-evolve-queue-job-id job)
                          :payload (nl-llm-evolve-queue--data-copy
                                    (nl-llm-evolve-queue-job-payload job)
                                    "claim payload")
                          :kind (nl-llm-evolve-queue-job-kind job)
                          :resuming (and resuming t)
                          :parent-generation
                          (nl-llm-evolution-generation evolution)
                          :parent-score
                          (nl-llm-evolution-champion-score evolution)
                          :attempt (nl-llm-evolve-queue--attempt-id
                                    queue job evolution)))
             (old-status (nl-llm-evolve-queue-job-status job)))
        (setf (nl-llm-evolve-queue-job-status job) 'running
              (nl-llm-evolve-queue-active-claim queue) claim)
        (condition-case err
            (nl-llm-evolve-queue--save-if-configured queue)
          (error
           (setf (nl-llm-evolve-queue-job-status job) old-status
                 (nl-llm-evolve-queue-active-claim queue) nil)
           (signal (car err) (cdr err))))
        claim))))

(defun nl-llm-evolve-queue--finish (queue job claim result)
  "Persisted terminal RESULT cleanup for JOB, if its trusted handler has one."
  (let* ((handler (nl-llm-evolve-queue--handler
                   queue (nl-llm-evolve-queue-job-kind job)))
         (finish (and handler (nl-llm-evolve-queue-handler-finish-fn handler))))
    (when finish
      (condition-case err
          (funcall finish
                   (list :job-id (nl-llm-evolve-queue-job-id job)
                         :kind (nl-llm-evolve-queue-job-kind job)
                         :resuming (plist-get claim :resuming)
                         :attempt (plist-get claim :attempt))
                   (copy-tree result))
        (error
         (setf (nl-llm-evolve-queue-job-result job)
               (append (nl-llm-evolve-queue-job-result job)
                       (list :cleanup-error (format "%S" err))))
         (nl-llm-evolve-queue--rebuild-history queue)
         (ignore-errors (nl-llm-evolve-queue--save-if-configured queue)))))))

;;;###autoload
(defun nl-llm-evolve-queue-complete (queue claim candidate score)
  "Complete active CLAIM with trusted CANDIDATE and SCORE."
  (let* ((job (nl-llm-evolve-queue--claim-valid queue claim))
         (evolution (nl-llm-evolve-queue-evolution queue))
         (result
          (nl-llm-evolution-accept-scored
           evolution candidate score (plist-get claim :parent-generation)
           (plist-get claim :parent-score)
           (list :proposal-id (nl-llm-evolve-queue-job-id job)
                 :proposal-kind (nl-llm-evolve-queue-job-kind job)
                 :resumed (plist-get claim :resuming)
                 :proposal-metadata
                 (copy-tree (nl-llm-evolve-queue-job-metadata job))))))
    (setf (nl-llm-evolve-queue-job-status job) (plist-get result :status)
          (nl-llm-evolve-queue-job-result job) (copy-tree result))
    (nl-llm-evolve-queue--trim-history queue)
    (let ((persisted nil))
      (condition-case err
          (progn (nl-llm-evolve-queue--save-if-configured queue)
                 (setq persisted t))
        (error
         (setf (nl-llm-evolve-queue-job-result job)
               (append (nl-llm-evolve-queue-job-result job)
                       (list :persistence-error (format "%S" err))))
         (nl-llm-evolve-queue--rebuild-history queue)))
      (when persisted (nl-llm-evolve-queue--finish queue job claim result)))
    (setf (nl-llm-evolve-queue-active-claim queue) nil)
    (copy-tree (nl-llm-evolve-queue--public-job job))))

;;;###autoload
(defun nl-llm-evolve-queue-fail (queue claim message)
  "Record trusted trainer failure for active CLAIM without changing champion."
  (let* ((job (nl-llm-evolve-queue--claim-valid queue claim))
         (evolution (nl-llm-evolve-queue-evolution queue))
         (generation (nl-llm-evolution-generation evolution))
         (score (nl-llm-evolution-champion-score evolution))
         (attempt (1+ (nl-llm-evolution-attempts evolution)))
         (result (list :attempt attempt :status 'error
                       :generation-before generation :generation-after generation
                       :score-before score :score-after nil :delta nil
                       :stage 'train :error (format "%s" message)
                       :metadata nil)))
    (setf (nl-llm-evolution-attempts evolution) attempt)
    (nl-llm-evolution--record evolution result)
    (setf (nl-llm-evolve-queue-job-status job) 'error
          (nl-llm-evolve-queue-job-result job) (copy-tree result))
    (nl-llm-evolve-queue--trim-history queue)
    (let ((persisted nil))
      (condition-case err
          (progn (nl-llm-evolve-queue--save-if-configured queue)
                 (setq persisted t))
        (error
         (setf (nl-llm-evolve-queue-job-result job)
               (append result (list :persistence-error (format "%S" err))))
         (nl-llm-evolve-queue--rebuild-history queue)))
      (when persisted (nl-llm-evolve-queue--finish queue job claim result)))
    (setf (nl-llm-evolve-queue-active-claim queue) nil)
    (copy-tree (nl-llm-evolve-queue--public-job job))))

;;;###autoload
(defun nl-llm-evolve-queue-interrupt (queue claim)
  "Mark active CLAIM interrupted after its subprocess has stopped."
  (let* ((job (nl-llm-evolve-queue--claim-valid queue claim))
         (old-status (nl-llm-evolve-queue-job-status job))
         (old-result (nl-llm-evolve-queue-job-result job)))
    (setf (nl-llm-evolve-queue-job-status job) 'interrupted
          (nl-llm-evolve-queue-job-result job)
          '(:status interrupted :reason host-interrupt))
    (condition-case err
        (nl-llm-evolve-queue--save-if-configured queue)
      (error
       (setf (nl-llm-evolve-queue-job-status job) old-status
             (nl-llm-evolve-queue-job-result job) old-result)
       (signal (car err) (cdr err))))
    (setf (nl-llm-evolve-queue-active-claim queue) nil)
    (copy-tree (nl-llm-evolve-queue--public-job job))))

(defun nl-llm-evolve-queue--execute (queue job resuming)
  "Execute QUEUE JOB, using its trusted resume callback when RESUMING."
  (let* ((handler (nl-llm-evolve-queue--validate-job queue job))
         (payload (nl-llm-evolve-queue-job-payload job))
         (propose (and handler
                       (nl-llm-evolve-queue-handler-propose-fn handler)))
         (train
          (and handler
               (if resuming
                   (nl-llm-evolve-queue-handler-resume-fn handler)
                 (nl-llm-evolve-queue-handler-train-fn handler))))
         (finish (and handler
                      (nl-llm-evolve-queue-handler-finish-fn handler)))
         (context
          (list :job-id (nl-llm-evolve-queue-job-id job)
                :kind (nl-llm-evolve-queue-job-kind job)
                :resuming (and resuming t)))
         result
         (persisted nil))
    (when (and resuming (not train))
      (error "improvement proposal %s is not resumable"
             (nl-llm-evolve-queue-job-id job)))
    (setf (nl-llm-evolve-queue-job-status job) 'running)
    (condition-case err
        (nl-llm-evolve-queue--save-if-configured queue)
      (error
       (setf (nl-llm-evolve-queue-job-status job)
             (if resuming 'interrupted 'pending))
       (signal (car err) (cdr err))))
    (let ((nl-llm-evolve-queue-execution-context context))
      (setq result
            (nl-llm-evolution-step
             (nl-llm-evolve-queue-evolution queue)
             (lambda (candidate evolution)
               (funcall
                propose candidate
                (nl-llm-evolve-queue--data-copy payload "proposal execution")
                evolution))
             :train
             (when train
               (lambda (candidate evolution)
                 (funcall
                  train candidate
                  (nl-llm-evolve-queue--data-copy payload
                                                   "proposal training")
                  evolution)))
             :metadata
             (list :proposal-id (nl-llm-evolve-queue-job-id job)
                   :proposal-kind (nl-llm-evolve-queue-job-kind job)
                   :resumed (and resuming t)
                   :proposal-metadata
                   (copy-tree (nl-llm-evolve-queue-job-metadata job)))))
      (setf (nl-llm-evolve-queue-job-status job)
            (plist-get result :status))
      (setf (nl-llm-evolve-queue-job-result job) (copy-tree result))
      (nl-llm-evolve-queue--trim-history queue)
      (condition-case err
          (progn
            (nl-llm-evolve-queue--save-if-configured queue)
            (setq persisted t))
        (error
         ;; Publication and champion commit may already have succeeded, so do
         ;; not pretend the experiment rolled back.  Surface the durability
         ;; failure alongside the authoritative evaluation result.
         (setf (nl-llm-evolve-queue-job-result job)
               (append
                (nl-llm-evolve-queue-job-result job)
                (list :persistence-error (format "%S" err))))
         (nl-llm-evolve-queue--rebuild-history queue)))
      (when (and persisted finish)
        (condition-case err
            (funcall finish (copy-tree context) (copy-tree result))
          (error
           (setf (nl-llm-evolve-queue-job-result job)
                 (append
                  (nl-llm-evolve-queue-job-result job)
                  (list :cleanup-error (format "%S" err))))
           (nl-llm-evolve-queue--rebuild-history queue)
           (ignore-errors
             (nl-llm-evolve-queue--save-if-configured queue))))))
    (nl-llm-evolve-queue--public-job job)))

;;;###autoload
(defun nl-llm-evolve-queue-run (queue &optional id)
  "Run pending proposal ID, or QUEUE's next proposal, through the eval gate."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-run: invalid queue"))
  (when (nl-llm-evolve-queue-active-claim queue)
    (error "cannot run synchronously while an asynchronous claim is active"))
  (let* ((job (if id
                  (nl-llm-evolve-queue--job queue id)
                (nl-llm-evolve-queue--next-pending queue))))
    (unless job
      (error "no pending improvement proposal"))
    (unless (eq (nl-llm-evolve-queue-job-status job) 'pending)
      (error "improvement proposal %s is not pending"
             (nl-llm-evolve-queue-job-id job)))
    (nl-llm-evolve-queue--execute queue job nil)))

;;;###autoload
(defun nl-llm-evolve-queue-resume (queue id)
  "Explicitly resume interrupted proposal ID through its trusted callback."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-resume: invalid queue"))
  (when (nl-llm-evolve-queue-active-claim queue)
    (error "cannot resume synchronously while an asynchronous claim is active"))
  (let ((job (nl-llm-evolve-queue--job queue id)))
    (unless job
      (error "unknown improvement proposal: %s" id))
    (unless (eq (nl-llm-evolve-queue-job-status job) 'interrupted)
      (error "improvement proposal %s is not interrupted" id))
    (nl-llm-evolve-queue--execute queue job t)))

;;;###autoload
(defun nl-llm-evolve-queue-cancel (queue id)
  "Cancel pending or interrupted proposal ID in QUEUE and return its state."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-cancel: invalid queue"))
  (let ((job (nl-llm-evolve-queue--job queue id)))
    (unless job
      (error "unknown improvement proposal: %s" id))
    (unless (memq (nl-llm-evolve-queue-job-status job) '(pending interrupted))
      (error "improvement proposal %s is not pending or interrupted" id))
    (let ((old-status (nl-llm-evolve-queue-job-status job))
          (old-result (nl-llm-evolve-queue-job-result job))
          (old-jobs (nl-llm-evolve-queue-jobs queue))
          (interrupted (eq (nl-llm-evolve-queue-job-status job) 'interrupted))
          (finish nil))
      (when interrupted
        (let ((handler (nl-llm-evolve-queue--handler
                        queue (nl-llm-evolve-queue-job-kind job))))
          (setq finish (and handler
                            (nl-llm-evolve-queue-handler-finish-fn handler)))))
      (setf (nl-llm-evolve-queue-job-status job) 'cancelled)
      (setf (nl-llm-evolve-queue-job-result job)
            '(:status cancelled))
      (nl-llm-evolve-queue--trim-history queue)
      (condition-case err
          (nl-llm-evolve-queue--save-if-configured queue)
        (error
         (setf (nl-llm-evolve-queue-jobs queue) old-jobs)
         (setf (nl-llm-evolve-queue-job-status job) old-status)
         (setf (nl-llm-evolve-queue-job-result job) old-result)
         (nl-llm-evolve-queue--rebuild-history queue)
         (signal (car err) (cdr err))))
      (when (and interrupted finish)
        (condition-case err
            (funcall finish
                     (list :job-id (nl-llm-evolve-queue-job-id job)
                           :kind (nl-llm-evolve-queue-job-kind job)
                           :resuming t :cancelled t)
                     (list :status 'cancelled))
          (error
           (setf (nl-llm-evolve-queue-job-result job)
                 (append (nl-llm-evolve-queue-job-result job)
                         (list :cleanup-error (format "%S" err))))
           (nl-llm-evolve-queue--rebuild-history queue)
           (ignore-errors
             (nl-llm-evolve-queue--save-if-configured queue)))))
      (copy-tree (nl-llm-evolve-queue--public-job job)))))

;;;###autoload
(defun nl-llm-evolve-queue-status (queue)
  "Return detached public status for QUEUE and its fixed evolution state."
  (unless (nl-llm-evolve-queue-p queue)
    (error "nl-llm-evolve-queue-status: invalid queue"))
  (let ((jobs (nl-llm-evolve-queue-jobs queue)))
    (list
     :generation
     (nl-llm-evolution-generation (nl-llm-evolve-queue-evolution queue))
     :champion-score
     (nl-llm-evolution-champion-score
      (nl-llm-evolve-queue-evolution queue))
     :pending
     (cl-count-if
      (lambda (job) (eq (nl-llm-evolve-queue-job-status job) 'pending)) jobs)
     :completed
     (cl-count-if
      (lambda (job)
        (memq (nl-llm-evolve-queue-job-status job)
              '(promoted rejected error cancelled)))
      jobs)
     :interrupted
     (cl-count-if
      (lambda (job)
        (eq (nl-llm-evolve-queue-job-status job) 'interrupted))
      jobs)
     :handlers (nl-llm-evolve-queue-catalog queue)
     :jobs (mapcar #'nl-llm-evolve-queue--public-job jobs))))

(provide 'nl-llm-evolve-queue)
;;; nl-llm-evolve-queue.el ends here
