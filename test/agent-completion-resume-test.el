;;; agent-completion-resume-test.el --- bound completion GPU resume tests -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)
(require 'photon-tensor)

(defconst nl-llm-agent-completion-resume-test--here
  (file-name-directory (or load-file-name buffer-file-name)))
(add-to-list 'load-path
             (expand-file-name "../lisp"
                               nl-llm-agent-completion-resume-test--here))
(add-to-list 'load-path
             (expand-file-name "../../nelisp-photon/lisp"
                               nl-llm-agent-completion-resume-test--here))
(require 'nl-llm-agent-ondevice)

(defun nl-llm-agent-completion-resume-test--model ()
  "Return the small CPU-view metadata used by fake GPU contexts."
  '(:tokenizer "ascii-char-v1" :vocab 96))

(defun nl-llm-agent-completion-resume-test--plan
    (&optional optimizer epochs learning-rate shuffle-seed)
  "Return a small canonical completion plan."
  (nl-llm-agent-completion-plan-make
   '((1 2 3) (4 5 6)) [1 1]
   :tokenizer "ascii-char-v1" :sequence 4
   :learning-rate (or learning-rate 0.1)
   :epochs (or epochs 2) :optimizer (or optimizer 'sgd)
   :shuffle-seed shuffle-seed))

(defun nl-llm-agent-completion-resume-test--ctx
    (&optional plan step optimizer)
  "Return a fake resident context bound to PLAN."
  (setq plan (or plan (nl-llm-agent-completion-resume-test--plan)))
  (list :cpu (nl-llm-agent-completion-resume-test--model)
        :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
        :seq (plist-get plan :sequence) :vocab (plist-get plan :vocab)
        :pad-token (plist-get plan :pad-token)
        :tokenizer (plist-get plan :tokenizer)
        :learning-rate (plist-get plan :learning-rate)
        :optimizer (or optimizer (plist-get plan :optimizer))
        :loss-mode 'completion :transfer-mode (plist-get plan :transfer-mode)
        :completion-plan plan :step (or step 0)))

(defun nl-llm-agent-completion-resume-test--run
    (ctx trajs starts epochs &optional start-step callback)
  "Run fake resident training and return seen inputs and callback events."
  (let ((seen nil) (steps 0) (events nil))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (resident value)
                 (when (eq resident 'oh)
                   (push (copy-sequence value) seen))))
              ((symbol-function 'nlga-step)
               (lambda (_builder) (setq steps (1+ steps))))
              ((symbol-function 'nl-llm-agent--onehot-pad)
               (lambda (tokens _seq _vocab _pad) (copy-sequence tokens)))
              ((symbol-function 'nl-llm-agent--shift-pad)
               (lambda (tokens _seq &optional _pad) (copy-sequence tokens))))
      (let ((total
             (nl-llm-agent-ondevice-train
              ctx trajs epochs :start-step (or start-step 0)
              :loss-starts starts
              :shuffle-seed (plist-get (plist-get ctx :completion-plan)
                                       :shuffle-seed)
              :after-step
              (lambda (_ctx completed planned)
                (push (list completed planned) events)
                (when callback (funcall callback))))))
        (list total steps (nreverse seen) (nreverse events))))))

(ert-deftest nl-llm-agent-ondevice-completion-plan-rejects-before-gpu-allocation ()
  (let* ((model (nl-llm-agent-improve-model 2 2 96 1 1 "ascii-char-v1"))
         (utf8-model (nl-llm-agent-improve-model 2 2 256 1 1 "utf8-byte-v1"))
         (plan (nl-llm-agent-completion-resume-test--plan))
         (allocated 0))
    (cl-letf (((symbol-function 'nlga-new)
               (lambda () (setq allocated (1+ allocated)))))
      (dolist (call
               (list
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   model 8 0.1 :optimizer 'sgd :loss-mode 'completion
                   :completion-plan plan))
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   model 4 0.2 :optimizer 'sgd :loss-mode 'completion
                   :completion-plan plan))
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   model 4 0.1 :optimizer 'adam :loss-mode 'completion
                   :completion-plan plan))
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   model 4 0.1 :optimizer 'sgd :loss-mode 'completion
                   :transfer-mode 'compact :completion-plan plan))
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   utf8-model 4 0.1 :optimizer 'sgd :loss-mode 'completion
                   :completion-plan plan))
                (lambda ()
                  (nl-llm-agent-ondevice-from-model
                   model 4 0.1 :optimizer 'sgd :completion-plan plan))))
        (should-error (funcall call))))
    (should (= allocated 0))))

(ert-deftest nl-llm-agent-ondevice-bound-train-validates-plan-before-updates ()
  (let* ((plan (nl-llm-agent-completion-resume-test--plan))
         (ctx (nl-llm-agent-completion-resume-test--ctx plan))
         (updates 0) (steps 0))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (&rest _args) (setq updates (1+ updates))))
              ((symbol-function 'nlga-step)
               (lambda (&rest _args) (setq steps (1+ steps)))))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2 3) (4 5 6)) 1 :loss-starts [1 1]))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2 3) (4 5 6)) 2 :start-step 1 :loss-starts [1 1]))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2 3) (4 5 6)) 2 :loss-starts [1 2]))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2 3) (4 5 6)) 2 :loss-starts [1 1]
        :shuffle-seed 2)))
    (should (= updates 0))
    (should (= steps 0))
    (should (= (plist-get ctx :step) 0))
    (let ((bad-ctx (copy-tree ctx t)))
      (setf (plist-get bad-ctx :learning-rate) 0.2)
      (should-error
       (nl-llm-agent-ondevice-train
        bad-ctx '((1 2 3) (4 5 6)) 2 :loss-starts [1 1])))))

(ert-deftest nl-llm-agent-ondevice-bound-train-detaches-callback-inputs ()
  (let* ((plan (nl-llm-agent-completion-resume-test--plan))
         (ctx (nl-llm-agent-completion-resume-test--ctx plan))
         (trajs (list (list 1 2 3) (list 4 5 6)))
         (result
          (nl-llm-agent-completion-resume-test--run
           ctx trajs [1 1] 2
           nil
           (lambda ()
             (setcar (car trajs) 95)
             (aset (aref (plist-get (plist-get ctx :completion-plan)
                                   :trajectories)
                         0)
                   0 95)))))
    (should (= (car result) 4))
    (should (= (cadr result) 4))
    (should (= (plist-get ctx :step) 4))
    (should (equal (nth 2 result)
                   '((1 2 3) (4 5 6) (1 2 3) (4 5 6))))
    (should-error (nl-llm-agent-ondevice-optimizer-state ctx))))

(ert-deftest nl-llm-agent-ondevice-bound-resume-replays-shuffle-callbacks ()
  (let* ((plan (nl-llm-agent-completion-resume-test--plan 'sgd 2 0.1 104729))
         (one (nl-llm-agent-completion-resume-test--run
               (nl-llm-agent-completion-resume-test--ctx plan 1)
               '((1 2 3) (4 5 6)) [1 1] 2 1))
         (two (nl-llm-agent-completion-resume-test--run
               (nl-llm-agent-completion-resume-test--ctx plan 1)
               '((1 2 3) (4 5 6)) [1 1] 2 1)))
    (should (equal one two))
    (should (equal (nth 3 one) '((2 4) (3 4) (4 4))))
    (should (= (car one) 4))
    (should (= (cadr one) 3))))

(ert-deftest nl-llm-agent-ondevice-bound-snapshot-detaches-plan-and-legacy-keys ()
  (let* ((plan (nl-llm-agent-completion-resume-test--plan))
         (ctx (nl-llm-agent-completion-resume-test--ctx plan))
         (syncs 0)
         (snapshot
          (cl-letf (((symbol-function 'nl-llm-agent-ondevice-sync)
                     (lambda (_ctx) (setq syncs (1+ syncs)))))
            (nl-llm-agent-ondevice-snapshot ctx)))
         (saved (plist-get snapshot :completion-plan)))
    (should-not (nl-llm-agent-ondevice-optimizer-state ctx))
    (let ((bad (copy-tree plan t)))
      (setf (plist-get bad :sequence) 99
            (plist-get ctx :completion-plan) bad)
      (should-error (nl-llm-agent-ondevice-optimizer-state ctx))
      (setf (plist-get ctx :completion-plan) plan))
    (should (= syncs 1))
    (should (equal saved plan))
    (should-not (eq saved plan))
    (should (plist-member snapshot :completion-plan))
    (setf (plist-get saved :sequence) 99)
    (should (= (plist-get plan :sequence) 4))
    (let ((legacy (list :cpu 'model :b 'builder :step 0
                        :optimizer 'sgd :loss-mode nil)))
      (cl-letf (((symbol-function 'nl-llm-agent-ondevice-sync) #'ignore))
        (should-not (plist-member
                     (nl-llm-agent-ondevice-snapshot legacy)
                     :completion-plan))))))

(ert-deftest nl-llm-agent-ondevice-bound-restore-requires-plan-and-validates-state ()
  (let* ((plan (nl-llm-agent-completion-resume-test--plan))
         (ctx (nl-llm-agent-completion-resume-test--ctx plan))
         (restores 0)
         (state nil))
    (cl-letf (((symbol-function 'nlga-adam-restore)
               (lambda (&rest _args) (setq restores (1+ restores)))))
      (should-error
       (nl-llm-agent-ondevice-restore-training-state ctx 1 nil))
      (should-error
       (nl-llm-agent-ondevice-restore-training-state
        ctx 1 nil (nl-llm-agent-completion-resume-test--plan 'sgd 1)))
      (should-error
       (nl-llm-agent-ondevice-restore-training-state
        ctx 5 nil plan))
      (should (= restores 0)))
    (let ((legacy (list :b 'builder :step 0 :optimizer 'sgd :loss-mode nil)))
      (should-error
       (nl-llm-agent-ondevice-restore-training-state legacy 0 nil plan)))
    (let ((unbound (list :b 'builder :step 0 :optimizer 'sgd
                         :loss-mode 'completion)))
      (should-error (nl-llm-agent-ondevice-optimizer-state unbound))
      (should-error (nl-llm-agent-ondevice-snapshot unbound))
      (should-error
       (nl-llm-agent-ondevice-restore-training-state unbound 0 nil)))
    (let* ((adam-plan (nl-llm-agent-completion-resume-test--plan 'adam))
           (adam-ctx (nl-llm-agent-completion-resume-test--ctx
                      adam-plan 0 'adam))
           (parameter (photon-tensor '(2) [0.0 0.0]))
           (moment (photon-tensor '(2) [0.0 0.0]))
           (pair (cons moment (photon-tensor '(2) [0.0 0.0])))
           (builder (nlga--make
                     :params (list (list :tensor parameter)))))
      (setf (plist-get adam-ctx :b) builder)
      (setq state (list pair))
      (cl-letf (((symbol-function 'nlga-params)
                 (lambda (_builder) (list (list :tensor parameter))))
                ((symbol-function 'nlga-adam-restore)
                 (lambda (&rest _args) (setq restores (1+ restores)))))
        (nl-llm-agent-ondevice-restore-training-state
         adam-ctx 1 state adam-plan)
        (should (= restores 1))
        (should (= (plist-get adam-ctx :step) 1))
        (setf (plist-get adam-ctx :step) 0)
        (setq state (list pair pair))
        (should-error
         (nl-llm-agent-ondevice-restore-training-state
          adam-ctx 1 state adam-plan))
        (should (= restores 1))
        (should (= (plist-get adam-ctx :step) 0))
        (setq state (list (cons (photon-tensor '(2) [0.0])
                                (photon-tensor '(2) [0.0 0.0]))))
        (should-error
         (nl-llm-agent-ondevice-restore-training-state
          adam-ctx 1 state adam-plan))
        (should (= restores 1))
        (should (= (plist-get adam-ctx :step) 0))))))

(provide 'agent-completion-resume-test)

(ert-run-tests-batch-and-exit)

;;; agent-completion-resume-test.el ends here
