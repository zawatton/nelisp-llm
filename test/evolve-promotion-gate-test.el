;;; evolve-promotion-gate-test.el --- generic promotion gates -*- lexical-binding: t; -*-

(require 'ert)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-evolve)
(require 'nl-llm-evolve-queue)

(defun nl-llm-evolve-promotion-gate-test--clone (model)
  "Clone the small plist MODEL used by these tests."
  (copy-tree model t))

(defun nl-llm-evolve-promotion-gate-test--score (model)
  "Read the numeric score from MODEL."
  (plist-get model :score))

(defun nl-llm-evolve-promotion-gate-test--state
    (&optional gate publish-called)
  "Create a tiny evolution STATE with optional GATE and publish flag."
  (nl-llm-evolution-new
   '(:score 1 :label parent)
   #'nl-llm-evolve-promotion-gate-test--score
   :clone #'nl-llm-evolve-promotion-gate-test--clone
   :promotion-gate gate
   :publish (when publish-called
              (lambda (_candidate _entry _state)
                (setcar publish-called t)))))

(defun nl-llm-evolve-promotion-gate-test--promote (state)
  "Run one score-improving synchronous proposal on STATE."
  (nl-llm-evolution-step
   state
   (lambda (candidate _state)
     (setf (plist-get candidate :score) 2))))

(ert-deftest nl-llm-evolve-promotion-gate-legacy-shape-and-noop ()
  (let ((state (nl-llm-evolve-promotion-gate-test--state)))
    (should (null (nl-llm-evolution-promotion-gate-fn state)))
    (let ((entry (nl-llm-evolve-promotion-gate-test--promote state)))
      (should (eq (plist-get entry :status) 'promoted))
      (should-not (plist-member entry :promotion-gate))
      (should (= (nl-llm-evolution-generation state) 1)))))

(ert-deftest nl-llm-evolve-promotion-gate-pass-and-isolation ()
  (let* ((seen nil)
         (published (list nil))
         (gate
          (lambda (parent candidate)
            (setq seen (list (plist-get parent :score)
                             (plist-get candidate :score)))
            (setf (plist-get parent :score) 99
                  (plist-get candidate :score) 98)
            t))
         (state (nl-llm-evolve-promotion-gate-test--state gate published))
         (entry (nl-llm-evolve-promotion-gate-test--promote state)))
    (should (equal seen '(1 2)))
    (should (eq (plist-get entry :status) 'promoted))
    (should (eq (plist-get entry :promotion-gate) 'passed))
    (should (equal (plist-get (nl-llm-evolution-champion state) :score) 2))
    (should (equal (plist-get (nl-llm-evolution-champion state) :label)
                   'parent))
    (should (car published))))

(ert-deftest nl-llm-evolve-promotion-gate-vetoes-after-score-gain ()
  (let* ((published (list nil))
        (state
         (nl-llm-evolve-promotion-gate-test--state
          (lambda (_parent _candidate) nil) published)))
    (let ((entry (nl-llm-evolve-promotion-gate-test--promote state)))
      (should (eq (plist-get entry :status) 'rejected))
      (should (eq (plist-get entry :promotion-gate) 'rejected))
      (should (= (nl-llm-evolution-generation state) 0))
      (should (= (plist-get (nl-llm-evolution-champion state) :score) 1))
      (should-not (car published)))))

(ert-deftest nl-llm-evolve-promotion-gate-rejects-malformed-and-errors ()
  (dolist (gate (list (lambda (_parent _candidate) 'yes)
                      (lambda (_parent _candidate) (error "gate boom"))))
    (let* ((published (list nil))
          (state (nl-llm-evolve-promotion-gate-test--state gate published)))
      (let ((entry (nl-llm-evolve-promotion-gate-test--promote state)))
        (should (eq (plist-get entry :status) 'error))
        (should (eq (plist-get entry :stage) 'promotion-gate))
        (should-not (car published))
        (should (= (nl-llm-evolution-generation state) 0))))))

(ert-deftest nl-llm-evolve-promotion-gate-skips-numeric-rejection ()
  (let* ((calls 0)
        (state
         (nl-llm-evolve-promotion-gate-test--state
          (lambda (_parent _candidate) (setq calls (1+ calls)) t))))
    (let ((entry
           (nl-llm-evolution-step
            state
            (lambda (candidate _state)
              (setf (plist-get candidate :score) 1)))))
      (should (eq (plist-get entry :status) 'rejected))
      (should (= calls 0))
      (should-not (plist-member entry :promotion-gate)))))

(ert-deftest nl-llm-evolve-promotion-gate-rechecks-nested-state ()
  (let ((state nil)
        (inside nil))
    (setq state
          (nl-llm-evolve-promotion-gate-test--state
           (lambda (_parent _candidate)
             (unless inside
               (setq inside t)
               (nl-llm-evolution-step
                state
                (lambda (candidate _state)
                  (setf (plist-get candidate :score) 3))))
             t)))
    (let ((entry (nl-llm-evolve-promotion-gate-test--promote state)))
      (should (eq (plist-get entry :status) 'error))
      (should (eq (plist-get entry :stage) 'promotion-gate))
      (should (= (plist-get entry :generation-after) 1))
      (should (= (nl-llm-evolution-generation state) 1))
      (should (= (plist-get (nl-llm-evolution-champion state) :score) 3)))))

(ert-deftest nl-llm-evolve-promotion-gate-async-veto ()
  (let* ((published (list nil))
        (state
         (nl-llm-evolve-promotion-gate-test--state
          (lambda (_parent _candidate) nil) published)))
    (let ((entry
           (nl-llm-evolution-accept-scored
            state '(:score 2 :label async) 2 0 1
                   '(:source async))))
      (should (eq (plist-get entry :status) 'rejected))
      (should (eq (plist-get entry :promotion-gate) 'rejected))
      (should-not (car published))
      (should (= (nl-llm-evolution-generation state) 0)))))

(ert-deftest nl-llm-evolve-promotion-gate-queue-sync-and-async-veto ()
  (dolist (async '(nil t))
    (let* ((published (list nil))
          (state
           (nl-llm-evolve-promotion-gate-test--state
            (lambda (_parent _candidate) nil) published))
          (queue nil))
      (setq queue (nl-llm-evolve-queue-new state :max-pending 2))
      (nl-llm-evolve-queue-register
       queue "set-score"
       (lambda (candidate payload _state)
         (setf (plist-get candidate :score) payload)))
      (nl-llm-evolve-queue-submit queue "set-score" 2 :id "gate-job")
      (if async
          (let ((claim (nl-llm-evolve-queue-claim queue "gate-job")))
            (nl-llm-evolve-queue-complete queue claim '(:score 2) 2))
        (nl-llm-evolve-queue-run queue "gate-job"))
      (let ((job (nl-llm-evolve-queue--job queue "gate-job")))
        (should (eq (nl-llm-evolve-queue-job-status job) 'rejected))
        (should-not (car published))
        (should (= (nl-llm-evolution-generation state) 0))))))

(ert-deftest nl-llm-evolve-promotion-gate-validates-constructor ()
  (should-error
   (nl-llm-evolution-new '(:score 1)
                         #'nl-llm-evolve-promotion-gate-test--score
                         :promotion-gate 1)))

(provide 'evolve-promotion-gate-test)

(ert-run-tests-batch-and-exit)

;;; evolve-promotion-gate-test.el ends here
