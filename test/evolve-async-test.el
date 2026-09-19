;;; evolve-async-test.el --- async evolution completion boundary -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-evolve-queue)

(defvar evolve-async--fail 0)
(defun evolve-async--ck (name value)
  (princ (format "%-58s %s\n" name (if value "PASS"
                                        (setq evolve-async--fail
                                              (1+ evolve-async--fail))
                                        "FAIL"))))

(let* ((state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (nl-llm-evolve-queue-new state))
       (finished nil))
  (nl-llm-evolve-queue-register
   queue "async" (lambda (&rest _) nil)
   :finish (lambda (_context _result) (setq finished t)))
  (nl-llm-evolve-queue-submit queue "async" 'payload :id "job")
  (let ((claim (nl-llm-evolve-queue-claim queue)))
    (evolve-async--ck "claim is the active opaque identity"
                      (eq claim (nl-llm-evolve-queue-active-claim queue)))
    (evolve-async--ck "public status omits claim and payload"
                      (not (or (plist-member (nl-llm-evolve-queue-status queue)
                                             :active-claim)
                               (plist-member (car (plist-get
                                                   (nl-llm-evolve-queue-status queue)
                                                   :jobs)) :payload))))
    (evolve-async--ck "synchronous run is blocked by active claim"
                      (condition-case nil
                          (progn (nl-llm-evolve-queue-run queue "job") nil)
                        (error t)))
    (let ((result (nl-llm-evolve-queue-complete queue claim [2.0] 2.0)))
      (evolve-async--ck "completion promotes and persists terminal state"
                        (and (eq (plist-get result :status) 'promoted)
                             (= (aref (nl-llm-evolution-champion state) 0) 2.0)
                             finished
                             (null (nl-llm-evolve-queue-active-claim queue)))))))

(let* ((state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (nl-llm-evolve-queue-new state)))
  (nl-llm-evolve-queue-register queue "async" (lambda (&rest _) nil))
  (nl-llm-evolve-queue-submit queue "async" nil :id "stale")
  (let ((claim (nl-llm-evolve-queue-claim queue)))
    (nl-llm-evolve-queue-submit queue "async" nil :id "blocked")
    (nl-llm-evolve-queue-complete queue claim [3.0] 3.0)
    (evolve-async--ck "new submissions survive async completion"
                      (eq (nl-llm-evolve-queue-job-status
                           (nl-llm-evolve-queue--job queue "blocked")) 'pending))))

(let* ((state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (nl-llm-evolve-queue-new state)))
  (nl-llm-evolve-queue-register queue "async" (lambda (&rest _) nil))
  (nl-llm-evolve-queue-submit queue "async" nil :id "interrupt")
  (let ((claim (nl-llm-evolve-queue-claim queue)))
    (nl-llm-evolve-queue-interrupt queue claim)
    (evolve-async--ck "interrupt clears claim without replay"
                      (and (null (nl-llm-evolve-queue-active-claim queue))
                           (eq (nl-llm-evolve-queue-job-status
                                (nl-llm-evolve-queue--job queue "interrupt"))
                               'interrupted)))))

;; Durable lifecycle: a pending concurrent submission survives restoration,
;; while a running claim is recovered as interrupted and never replayed.
(let* ((directory (make-temp-file "evolve-async-" t))
       (checkpoint (expand-file-name "queue.sexp" directory))
       (make-queue (lambda (state)
                     (let ((q (nl-llm-evolve-queue-new
                               state :checkpoint-file checkpoint)))
                       (nl-llm-evolve-queue-register q "async"
                                                     (lambda (&rest _) nil))
                       q)))
       (state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (funcall make-queue state)))
  (unwind-protect
      (progn
        (nl-llm-evolve-queue-submit queue "async" nil :id "first")
        (let ((claim (nl-llm-evolve-queue-claim queue)))
          (nl-llm-evolve-queue-submit queue "async" nil :id "concurrent")
          (evolve-async--ck "claim checkpoint is private and durable"
                            (and (file-regular-p checkpoint)
                                 (not (plist-member
                                       (car (plist-get
                                             (nl-llm-evolve-queue-status queue)
                                             :jobs)) :payload))))
          (let* ((restored-state
                  (nl-llm-evolution-new [0.0]
                                        (lambda (model) (aref model 0))))
                 (restored (funcall make-queue restored-state)))
            (nl-llm-evolve-queue-restore restored)
            (let ((jobs (plist-get (nl-llm-evolve-queue-status restored) :jobs)))
              (evolve-async--ck "restore converts claim to interrupted"
                                (and (= (plist-get
                                         (nl-llm-evolve-queue-status restored)
                                         :interrupted) 1)
                                     (eq (plist-get (car jobs) :status)
                                         'interrupted)))
              (evolve-async--ck "concurrent pending submission survives restore"
                                (eq (plist-get (cadr jobs) :status) 'pending)))))
        (ignore-errors (nl-llm-evolve-queue-interrupt
                        queue (nl-llm-evolve-queue-active-claim queue))))
    (delete-directory directory t)))

;; Save failures roll back claim/interruption and suppress cleanup callbacks.
(let* ((state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (nl-llm-evolve-queue-new state))
       (finished nil))
  (nl-llm-evolve-queue-register
   queue "async" (lambda (&rest _) nil)
   :finish (lambda (&rest _) (setq finished t)))
  (nl-llm-evolve-queue-submit queue "async" nil :id "rollback")
  (cl-letf (((symbol-function 'nl-llm-evolve-queue--save-if-configured)
             (lambda (_queue) (error "disk full"))))
    (evolve-async--ck "failed claim save restores pending state"
                      (condition-case nil
                          (progn (nl-llm-evolve-queue-claim queue "rollback") nil)
                        (error
                         (and (null (nl-llm-evolve-queue-active-claim queue))
                              (eq (nl-llm-evolve-queue-job-status
                                   (nl-llm-evolve-queue--job queue "rollback"))
                                  'pending))))))
  (let ((claim (nl-llm-evolve-queue-claim queue "rollback")))
    (cl-letf (((symbol-function 'nl-llm-evolve-queue--save-if-configured)
               (lambda (_queue) (error "disk full"))))
      (evolve-async--ck "failed interrupt save restores running state"
                        (condition-case nil
                            (progn (nl-llm-evolve-queue-interrupt queue claim) nil)
                          (error
                           (and (eq (nl-llm-evolve-queue-job-status
                                     (nl-llm-evolve-queue--job queue "rollback"))
                                    'running)
                                (eq (nl-llm-evolve-queue-active-claim queue) claim)))))
      (evolve-async--ck "failed completion save skips finish cleanup"
                        (progn
                          (nl-llm-evolve-queue-complete queue claim [1.0] 1.0)
                          (not finished)))))

;; Forged/duplicate and stale completions never mutate the champion.
(let* ((state (nl-llm-evolution-new [0.0] (lambda (model) (aref model 0))))
       (queue (nl-llm-evolve-queue-new state)))
  (nl-llm-evolve-queue-register queue "async" (lambda (&rest _) nil))
  (nl-llm-evolve-queue-submit queue "async" nil :id "guard")
  (let ((claim (nl-llm-evolve-queue-claim queue "guard")))
    (evolve-async--ck "copied claim is rejected"
                      (condition-case nil
                          (progn (nl-llm-evolve-queue-complete
                                  queue (copy-tree claim) [4.0] 4.0) nil)
                        (error t)))
    (setf (nl-llm-evolution-generation state) 1)
    (evolve-async--ck "stale generation completion is rejected"
                      (condition-case nil
                          (progn (nl-llm-evolve-queue-complete
                                  queue claim [4.0] 4.0) nil)
                        (error t)))
    (setf (nl-llm-evolution-generation state) 0
          (nl-llm-evolution-champion-score state) 9.0)
    (evolve-async--ck "stale score completion is rejected"
                      (condition-case nil
                          (progn (nl-llm-evolve-queue-complete
                                  queue claim [4.0] 4.0) nil)
                        (error t)))
    (setf (nl-llm-evolution-champion-score state) 0.0)
    (evolve-async--ck "NaN and infinity scores are rejected"
                      (and (condition-case nil
                               (progn (nl-llm-evolve-queue-complete
                                       queue claim [4.0] 0.0e+NaN) nil)
                             (error t))
                           (condition-case nil
                               (progn (nl-llm-evolve-queue-complete
                                       queue claim [4.0] 1.0e+INF) nil)
                             (error t))))
    (nl-llm-evolve-queue-interrupt queue claim)))

;; Publication failure leaves champion untouched; accepted candidate is detached.
(let* ((state (nl-llm-evolution-new
               [0.0] (lambda (model) (aref model 0))
               :publish (lambda (&rest _) (error "publication failed"))))
       (queue (nl-llm-evolve-queue-new state)))
  (nl-llm-evolve-queue-register queue "async" (lambda (&rest _) nil))
  (nl-llm-evolve-queue-submit queue "async" nil :id "publish")
  (let ((claim (nl-llm-evolve-queue-claim queue "publish"))
        (candidate [3.0]))
    (let ((result (nl-llm-evolve-queue-complete queue claim candidate 3.0)))
      (aset candidate 0 99.0)
      (evolve-async--ck "publication failure rolls back and clone detaches"
                        (and (eq (plist-get result :status) 'error)
                             (= (aref (nl-llm-evolution-champion state) 0) 0.0)))))

  ))

(princ (format "NL-LLM-EVOLVE-ASYNC %s (%d failures)\n"
               (if (= evolve-async--fail 0) "ALL-PASS" "HAS-FAILURES")
               evolve-async--fail))
(kill-emacs (if (= evolve-async--fail 0) 0 1))

;;; evolve-async-test.el ends here
