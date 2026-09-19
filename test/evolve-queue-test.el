;;; evolve-queue-test.el --- evaluated improvement proposal queue  -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-evolve-queue)

(defvar evolve-queue--fail 0)

(defun evolve-queue--ck (name ok)
  (princ (format "%-67s %s\n" name
                 (if ok "PASS"
                   (setq evolve-queue--fail (1+ evolve-queue--fail))
                   "FAIL"))))

(defun evolve-queue--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(defun evolve-queue--fitness (model)
  (aref (plist-get model :fitness) 0))

(let* ((state
        (nl-llm-evolution-new '(:fitness [1.0])
                              #'evolve-queue--fitness))
       (queue (nl-llm-evolve-queue-new state :max-pending 4))
       (caller-payload '(:fitness 2.0 :notes ["detached"]))
       first)
  (nl-llm-evolve-queue-register
   queue "set-fitness"
   (lambda (candidate payload _state)
     (aset (plist-get candidate :fitness) 0
           (plist-get payload :fitness)))
   :validate
   (lambda (payload)
     (unless (numberp (plist-get payload :fitness))
       (error "fitness is required")))
   :description "Set candidate fitness before fixed evaluation")
  (nl-llm-evolve-queue-register
   queue "explode"
   (lambda (_candidate _payload _state)
     (error "proposal failed"))
   :description "Exercise error containment")
  (evolve-queue--ck
   "handler catalog exposes descriptions but no trusted callbacks"
   (equal
    (nl-llm-evolve-queue-catalog queue)
    '((:kind "set-fitness"
	     :description "Set candidate fitness before fixed evaluation")
      (:kind "explode" :description "Exercise error containment"))))
  (nl-llm-evolve-queue-submit
   queue "set-fitness" caller-payload :id "low" :priority 0)
  (aset (plist-get caller-payload :notes) 0 "mutated")
  (nl-llm-evolve-queue-submit
   queue "set-fitness" '(:fitness 3.0) :id "high" :priority 10)
  (setq first (nl-llm-evolve-queue-run queue))
  (evolve-queue--ck
   "highest-priority proposal is evaluated first and promoted"
   (and (equal (plist-get first :id) "high")
        (eq (plist-get first :status) 'promoted)
        (= (nl-llm-evolution-generation state) 1)
        (= (evolve-queue--fitness (nl-llm-evolution-champion state)) 3.0)))
  (let ((low (nl-llm-evolve-queue-run queue "low")))
    (evolve-queue--ck
     "a queued regression is rejected without replacing the champion"
     (and (eq (plist-get low :status) 'rejected)
          (= (evolve-queue--fitness
              (nl-llm-evolution-champion state))
             3.0))))
  (evolve-queue--ck
   "submitted payload is detached and omitted from public job state"
   (let ((low
          (car
           (cl-remove-if-not
            (lambda (job) (equal (plist-get job :id) "low"))
            (plist-get (nl-llm-evolve-queue-status queue) :jobs)))))
     (and low (not (plist-member low :payload)))))
  (nl-llm-evolve-queue-submit queue "explode" nil :id "broken")
  (let ((broken (nl-llm-evolve-queue-run queue "broken")))
    (evolve-queue--ck
     "handler errors become audited failed experiments"
     (and (eq (plist-get broken :status) 'error)
          (eq (plist-get (plist-get broken :result) :stage) 'propose)
          (= (nl-llm-evolution-generation state) 1))))
  (evolve-queue--ck
   "unregistered experiment kinds cannot select executable behavior"
   (evolve-queue--error-p
    (lambda ()
      (nl-llm-evolve-queue-submit queue "unknown" nil))))
  (evolve-queue--ck
   "executable function objects are rejected from proposal payloads"
   (evolve-queue--error-p
    (lambda ()
      (nl-llm-evolve-queue-submit
       queue "set-fitness" (lambda () t)))))
  (nl-llm-evolve-queue-submit
   queue "set-fitness" '(:fitness 4.0) :id "cancel-me")
  (let ((cancelled (nl-llm-evolve-queue-cancel queue "cancel-me")))
    (evolve-queue--ck
     "pending proposals can be cancelled before execution"
     (and (eq (plist-get cancelled :status) 'cancelled)
          (= (plist-get (nl-llm-evolve-queue-status queue) :pending) 0))))
  (evolve-queue--ck
   "queue status reports the fixed evaluator's generation and score"
   (let ((status (nl-llm-evolve-queue-status queue)))
     (and (= (plist-get status :generation) 1)
          (= (plist-get status :champion-score) 3.0)
          (= (plist-get status :completed) 4)))))

(let* ((state
        (nl-llm-evolution-new '(:fitness [0.0])
                              #'evolve-queue--fitness))
       (queue (nl-llm-evolve-queue-new state :max-pending 1)))
  (nl-llm-evolve-queue-register
   queue "set"
   (lambda (candidate payload _state)
     (aset (plist-get candidate :fitness) 0 payload)))
  (nl-llm-evolve-queue-submit queue "set" 1.0)
  (evolve-queue--ck
   "pending queue capacity is bounded before accepting more work"
   (evolve-queue--error-p
    (lambda () (nl-llm-evolve-queue-submit queue "set" 2.0)))))

(let* ((directory (make-temp-file "nl-llm-evolve-queue-state-" t))
       (checkpoint (expand-file-name "queue.sexp" directory))
       (new-queue
        (lambda (state file)
          (let ((queue
                 (nl-llm-evolve-queue-new
                  state :max-pending 4 :max-history 2
                  :checkpoint-file file)))
            (nl-llm-evolve-queue-register
             queue "set"
             (lambda (candidate payload _state)
               (aset (plist-get candidate :fitness) 0 payload)))
            queue)))
       (first-state
        (nl-llm-evolution-new '(:fitness [0.0])
                              #'evolve-queue--fitness))
       (first (funcall new-queue first-state checkpoint)))
  (unwind-protect
      (progn
        (nl-llm-evolve-queue-submit first "set" 2.0 :id "durable")
        (evolve-queue--ck
         "accepted proposal is atomically checkpointed as private mode-0600 data"
         (and (file-regular-p checkpoint)
              (or (not (fboundp 'file-modes))
                  (= (logand (file-modes checkpoint) #o777) #o600))
              (not (plist-member
                    (car (plist-get
                          (nl-llm-evolve-queue-status first) :jobs))
                    :payload))))
        (let* ((restored-state
                (nl-llm-evolution-new '(:fitness [0.0])
                                      #'evolve-queue--fitness))
               (restored (funcall new-queue restored-state checkpoint)))
          (nl-llm-evolve-queue-restore restored nil nil)
          (evolve-queue--ck
           "pending proposal restores with stable identity and executes normally"
           (let ((job (nl-llm-evolve-queue-run restored "durable")))
             (and (eq (plist-get job :status) 'promoted)
                  (= (evolve-queue--fitness
                      (nl-llm-evolution-champion restored-state))
                     2.0))))
          (nl-llm-evolve-queue-submit restored "set" 3.0 :id "second")
          (nl-llm-evolve-queue-run restored "second")
          (nl-llm-evolve-queue-submit restored "set" 4.0 :id "third")
          (nl-llm-evolve-queue-run restored "third")
          (evolve-queue--ck
           "terminal audit retention is bounded without losing sequence state"
           (let ((status (nl-llm-evolve-queue-status restored)))
             (and (= (length (plist-get status :jobs)) 2)
                  (= (plist-get status :completed) 2)
                  (= (nl-llm-evolve-queue-next-sequence restored) 3)))))
        (let* ((interrupted-file
                (expand-file-name "interrupted.sexp" directory))
               (running-state
                (nl-llm-evolution-new '(:fitness [0.0])
                                      #'evolve-queue--fitness))
               (running (funcall new-queue running-state interrupted-file)))
          (nl-llm-evolve-queue-submit running "set" 9.0 :id "in-flight")
          (setf
           (nl-llm-evolve-queue-job-status
            (car (nl-llm-evolve-queue-jobs running)))
           'running)
          (nl-llm-evolve-queue-save running)
          (let* ((recovery-state
                  (nl-llm-evolution-new '(:fitness [0.0])
                                        #'evolve-queue--fitness))
                 (recovery
                  (funcall new-queue recovery-state interrupted-file)))
            (nl-llm-evolve-queue-restore recovery)
            (evolve-queue--ck
             "in-flight work becomes interrupted and is never replayed after restart"
             (let* ((status (nl-llm-evolve-queue-status recovery))
                    (job (car (plist-get status :jobs))))
               (and (= (plist-get status :pending) 0)
                    (= (plist-get status :interrupted) 1)
                    (eq (plist-get job :status) 'interrupted)
                    (= (evolve-queue--fitness
                        (nl-llm-evolution-champion recovery-state))
                       0.0))))
            (evolve-queue--ck
             "an interrupted job without a trusted resume callback stays inert"
             (evolve-queue--error-p
              (lambda ()
                (nl-llm-evolve-queue-resume recovery "in-flight")))))))
    (delete-directory directory t)))

(let* ((directory (make-temp-file "nl-llm-evolve-resume-" t))
       (checkpoint (expand-file-name "queue.sexp" directory))
       (finish-saw-durable nil)
       (new-queue
        (lambda ()
          (let* ((state
                  (nl-llm-evolution-new
                   '(:fitness [0.0]) #'evolve-queue--fitness))
                 (queue
                  (nl-llm-evolve-queue-new
                   state :checkpoint-file checkpoint)))
	    (nl-llm-evolve-queue-register
	     queue "resumable-set"
	     (lambda (_candidate _payload _state) nil)
	     :resume
	     (lambda (candidate payload _state)
	       (aset (plist-get candidate :fitness) 0 payload))
	     :finish
	     (lambda (context _result)
	       (let* ((saved
		       (nl-llm-evolve-queue--checkpoint-read checkpoint))
		      (job (aref (plist-get saved :jobs) 0)))
                 (setq finish-saw-durable
		       (and (plist-get context :resuming)
			    (eq (plist-get job :status) 'promoted)))))
	     :description "Resume only through trusted state restoration")
	    queue)))
       (first (funcall new-queue)))
  (unwind-protect
      (progn
        (nl-llm-evolve-queue-submit
         first "resumable-set" 7.0 :id "resume-me")
        (setf (nl-llm-evolve-queue-job-status
	       (car (nl-llm-evolve-queue-jobs first)))
	      'running)
        (nl-llm-evolve-queue-save first)
        (let ((restored (funcall new-queue)))
          (nl-llm-evolve-queue-restore restored)
          (evolve-queue--ck
           "resumable capability is public but its trusted callback remains private"
           (equal
	    (nl-llm-evolve-queue-catalog restored)
	    '((:kind "resumable-set"
		     :description "Resume only through trusted state restoration"
		     :resumable t))))
          (let ((result
                 (nl-llm-evolve-queue-resume restored "resume-me")))
	    (evolve-queue--ck
	     "explicit resume persists terminal state before checkpoint cleanup"
	     (and (eq (plist-get result :status) 'promoted)
                  finish-saw-durable
                  (= (evolve-queue--fitness
		      (nl-llm-evolution-champion
		       (nl-llm-evolve-queue-evolution restored)))
		     7.0))))))
    (delete-directory directory t)))

(let* ((state (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))
       (queue (nl-llm-evolve-queue-new state :max-pending 1 :max-history 0))
       (allow t)
       (trained nil))
  (nl-llm-evolve-queue-register
   queue "resume"
   (lambda (_candidate _payload _state) nil)
   :resume (lambda (_candidate _payload _state) (setq trained t))
   :validate (lambda (_payload) (unless allow (error "payload revoked"))))
  (nl-llm-evolve-queue-submit queue "resume" 1 :id "interrupted")
  (setf (nl-llm-evolve-queue-job-status
         (car (nl-llm-evolve-queue-jobs queue))) 'interrupted)
  (evolve-queue--ck
   "interrupted jobs consume admission capacity"
   (and (= (plist-get (nl-llm-evolve-queue-status queue) :interrupted) 1)
        (evolve-queue--error-p
         (lambda () (nl-llm-evolve-queue-submit queue "resume" 2)))))
  (setq allow nil)
  (evolve-queue--ck
   "cancel can discard an interrupted job after its payload is revoked"
   (and (eq (plist-get (nl-llm-evolve-queue-cancel queue "interrupted") :status)
            'cancelled)
        (= (plist-get (nl-llm-evolve-queue-status queue) :interrupted) 0)
        (not trained)))
  (evolve-queue--ck
   "cancelling interrupted work frees admission capacity"
   (progn (setq allow t)
          (nl-llm-evolve-queue-submit queue "resume" 2)
          t)))

(let* ((directory (make-temp-file "nl-llm-evolve-lifecycle-" t))
       (checkpoint (expand-file-name "queue.sexp" directory))
       (make-q
        (lambda (state &optional max-pending max-history)
          (let ((q (nl-llm-evolve-queue-new
                    state :max-pending (or max-pending 4)
                    :max-history (if max-history max-history 2)
                    :checkpoint-file checkpoint)))
            (nl-llm-evolve-queue-register
             q "set" (lambda (candidate payload _state)
                       (aset (plist-get candidate :fitness) 0 payload))
             :resume (lambda (candidate payload _state)
		       (aset (plist-get candidate :fitness) 0 payload)))
            q))))
  (unwind-protect
      (progn
        ;; 1: max-history zero still restores and resumes interrupted work.
        (let* ((q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness) 2 0))
	       (s (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))
	       (r (funcall make-q s 2 0)))
          (nl-llm-evolve-queue-submit q "set" 5 :id "zero-history")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'running)
          (nl-llm-evolve-queue-save q)
          (nl-llm-evolve-queue-restore r)
          (let ((job (nl-llm-evolve-queue-resume r "zero-history")))
            (evolve-queue--ck "max-history zero preserves resumable interrupted jobs"
			      (and (eq (plist-get job :status) 'promoted)
                                   (= (evolve-queue--fitness (nl-llm-evolution-champion s)) 5)))))
        ;; 2: terminal trimming never evicts interrupted work.
        (let ((q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness) 4 1)))
          (nl-llm-evolve-queue-submit q "set" 1 :id "keep-interrupted")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
          (dotimes (i 3)
            (nl-llm-evolve-queue-submit q "set" (1+ i) :id (format "terminal-%d" i))
            (nl-llm-evolve-queue-run q (format "terminal-%d" i)))
          (evolve-queue--ck "interrupted survives newer terminal trimming"
                            (and (= (plist-get (nl-llm-evolve-queue-status q) :interrupted) 1)
                                 (nl-llm-evolve-queue--job q "keep-interrupted"))))
        ;; 3: restore rejects excess outstanding before committing any jobs.
        (let* ((source (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness) 4 2))
	       (target (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness) 1 2)))
          (nl-llm-evolve-queue-submit source "set" 1 :id "running")
          (nl-llm-evolve-queue-submit source "set" 2 :id "interrupted")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs source))) 'running)
          (setf (nl-llm-evolve-queue-job-status (cadr (nl-llm-evolve-queue-jobs source))) 'interrupted)
          (nl-llm-evolve-queue-save source)
          (evolve-queue--ck "restore rejects excess outstanding without committing"
                            (and (evolve-queue--error-p (lambda () (nl-llm-evolve-queue-restore target)))
                                 (null (nl-llm-evolve-queue-jobs target)))))
        ;; 4: changed validator blocks resume before status/training mutation.
        (let* ((q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness)))
	       (allow t) (trained 0))
          (setf (nl-llm-evolve-queue-handler-validate-fn (car (nl-llm-evolve-queue-handlers q)))
                (lambda (_payload) (unless allow (error "revoked"))))
          (setf (nl-llm-evolve-queue-handler-resume-fn (car (nl-llm-evolve-queue-handlers q)))
                (lambda (&rest _) (setq trained (1+ trained))))
          (nl-llm-evolve-queue-submit q "set" 1 :id "stale")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
          (setq allow nil)
          (evolve-queue--ck "resume revalidation leaves interrupted job unchanged"
                            (and (evolve-queue--error-p (lambda () (nl-llm-evolve-queue-resume q "stale")))
                                 (eq (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
                                 (= trained 0))))
        ;; 5: rejecting validator also blocks interrupted restore.
        (let* ((source (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness)))
	       (target (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))))
          (nl-llm-evolve-queue-submit source "set" 1 :id "bad-restore")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs source))) 'interrupted)
          (nl-llm-evolve-queue-save source)
          (setf (nl-llm-evolve-queue-handler-validate-fn (car (nl-llm-evolve-queue-handlers target)))
                (lambda (_payload) (error "rejected")))
          (evolve-queue--ck "restore validates interrupted payloads"
                            (and (evolve-queue--error-p (lambda () (nl-llm-evolve-queue-restore target)))
                                 (null (nl-llm-evolve-queue-jobs target)))))
        ;; 6: cancellation cleanup observes durable cancelled state.
        (let* ((marker (expand-file-name "cleanup-marker" directory))
	       (recovery (expand-file-name "cleanup-recovery" directory))
	       (q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))))
          (nl-llm-evolve-queue-register q "cleanup" (lambda (&rest _) nil)
					:finish (lambda (_context _result)
						  (let* ((saved (nl-llm-evolve-queue--checkpoint-read checkpoint))
							 (job (aref (plist-get saved :jobs) 0)))
						    (when (eq (plist-get job :status) 'cancelled)
						      (write-region "ok" nil marker nil 'silent))
						    (delete-file recovery))))
          (nl-llm-evolve-queue-submit q "cleanup" nil :id "cleanup")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
          (nl-llm-evolve-queue-save q)
          (write-region "checkpoint" nil recovery nil 'silent)
          (nl-llm-evolve-queue-cancel q "cleanup")
          (evolve-queue--ck "interrupted cancellation cleanup sees durable state"
                            (and (file-exists-p marker) (file-exists-p checkpoint)
                                 (not (file-exists-p recovery)))))
        ;; 7: save failure rolls back cancellation and suppresses cleanup.
        (let ((called nil)
	      (recovery (expand-file-name "rollback-recovery" directory))
	      (q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))))
          (nl-llm-evolve-queue-register q "rollback" (lambda (&rest _) nil)
					:finish (lambda (&rest _) (setq called t)))
          (nl-llm-evolve-queue-submit q "rollback" nil :id "rollback")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
          (nl-llm-evolve-queue-save q)
          (write-region "checkpoint" nil recovery nil 'silent)
          (cl-letf (((symbol-function 'nl-llm-evolve-queue-save)
                     (lambda (&rest _) (error "injected save failure"))))
		   (evolve-queue--ck "cancel save failure rolls back and skips cleanup"
				     (and (evolve-queue--error-p (lambda () (nl-llm-evolve-queue-cancel q "rollback")))
					  (eq (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
					  (not called) (file-exists-p checkpoint)
					  (file-exists-p recovery)))))
        ;; 8: cleanup failures are audited after durable cancellation.
        (let ((q (funcall make-q (nl-llm-evolution-new '(:fitness [0.0]) #'evolve-queue--fitness))))
          (nl-llm-evolve-queue-register q "cleanup-error" (lambda (&rest _) nil)
					:finish (lambda (&rest _) (error "cleanup failed")))
          (nl-llm-evolve-queue-submit q "cleanup-error" nil :id "cleanup-error")
          (setf (nl-llm-evolve-queue-job-status (car (nl-llm-evolve-queue-jobs q))) 'interrupted)
          (let ((job (nl-llm-evolve-queue-cancel q "cleanup-error")))
            (evolve-queue--ck "cleanup failure is audited"
			      (plist-member (plist-get job :result) :cleanup-error)))))
    (delete-directory directory t)))

(princ (format "NL-LLM-EVOLVE-QUEUE %s (%d failures)\n"
               (if (= evolve-queue--fail 0) "ALL-PASS" "HAS-FAILURES")
               evolve-queue--fail))
(kill-emacs (if (= evolve-queue--fail 0) 0 1))

;;; evolve-queue-test.el ends here
