;;; agent-loss-masks-test.el --- sparse completion target-mask plans -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(setq load-prefer-newer t)
(require 'ert)
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent-ondevice)

(defun nl-llm-agent-loss-masks-test--ctx (&optional compact)
  (list :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
        :seq 8 :vocab 5 :pad-token 255 :optimizer 'sgd
        :loss-mode 'completion :transfer-mode (and compact 'compact)
        :step 0))

(defun nl-llm-agent-loss-masks-test--data (value)
  (condition-case nil
      (photon-tensor-data value)
    (error value)))

(defun nl-llm-agent-loss-masks-test--run
    (ctx trajs starts &optional masks shuffle-seed callback)
  (let ((updates nil)
        (steps 0))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (resident value)
                 (push (list resident
                             (nl-llm-agent-loss-masks-test--data value))
                       updates)))
              ((symbol-function 'nlga-step)
               (lambda (_builder) (setq steps (1+ steps))))
              ((symbol-function 'nl-llm-agent--onehot-pad)
               (lambda (tokens _seq _vocab _pad) (copy-sequence tokens)))
              ((symbol-function 'nl-llm-agent--shift-pad)
               (lambda (tokens _seq &optional _pad) (copy-sequence tokens)))
              ((symbol-function 'nl-llm-agent-ondevice--index-pad)
               (lambda (tokens _seq _pad) (copy-sequence tokens))))
      (nl-llm-agent-ondevice-train
       ctx trajs 1 :loss-starts starts :loss-masks masks
       :shuffle-seed shuffle-seed :after-step callback))
    (list updates steps)))

(defun nl-llm-agent-loss-masks-test--scale (updates)
  (cadr (assq 'scale updates)))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-exact-dense-and-compact-rows ()
  (let* ((trajectory '(1 2 3 4 0 1))
         (mask [0 0 1 0 1 1])
         (dense (nl-llm-agent-loss-masks-test--run
                 (nl-llm-agent-loss-masks-test--ctx)
                 (list trajectory) [2] (vector mask)))
         (compact (nl-llm-agent-loss-masks-test--run
                   (nl-llm-agent-loss-masks-test--ctx t)
                   (list trajectory) [2] (vector mask)))
         (factor (/ 8.0 3.0))
         (rows (vector 0.0 factor 0.0 factor factor 0.0 0.0 0.0))
         (dense-rows (make-vector (* 8 5) 0.0)))
    (dotimes (row 8)
      (dotimes (column 5)
        (aset dense-rows (+ (* row 5) column) (aref rows row))))
    (should (equal (nl-llm-agent-loss-masks-test--scale (car dense))
                   dense-rows))
    (should (equal (nl-llm-agent-loss-masks-test--scale (car compact))
                   rows))
    (should (= (cadr dense) 1))
    (should (= (cadr compact) 1))))

(ert-deftest nl-llm-agent-ondevice-nil-and-all-completion-ones-match ()
  (let* ((trajectory '(1 2 3 4 0 1))
         (starts [2])
         (nil-run (nl-llm-agent-loss-masks-test--run
                   (nl-llm-agent-loss-masks-test--ctx)
                   (list trajectory) starts))
         (ones-run (nl-llm-agent-loss-masks-test--run
                    (nl-llm-agent-loss-masks-test--ctx)
                    (list trajectory) starts (vector [0 0 1 1 1 1]))))
    (should (equal (car nil-run) (car ones-run)))
    (should (= (cadr nil-run) (cadr ones-run)))))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-validation-before-gpu-calls ()
  (let* ((trajectory '(1 2 3 4 0 1))
        (valid [0 0 1 0 1 1])
        (cases
         (list
          (list (list trajectory) [2] '((0 0 1 0 1 1)))
          (list (list trajectory) [2] [[0 0 1 0 1]])
          (list (list trajectory) [2] (vector '(0 0 1 0 1 1)))
          (list (list trajectory) [2]
                (vector valid [0 0 1 0 1]))
          (list (list trajectory) [2] (vector [0 0 1 0 1.0 1]))
          (list (list trajectory) [2] (vector [0 0 1 0 -2 1]))
          (list (list trajectory) [2] (vector [0 0 1 0 2 1]))
          (list (list trajectory) [2] (vector [1 0 1 0 1 1]))
          (list (list trajectory) [2] (vector [0 0 0 0 0 0]))
          ;; The first example is valid; the later malformed example must
          ;; still be rejected before the first resident update.
          (list (list trajectory trajectory) [2 2]
                (vector valid [0 0 0 0 0 0])))))
    (dolist (case cases)
      (let ((ctx (nl-llm-agent-loss-masks-test--ctx))
            (updates 0)
            (steps 0))
        (cl-letf (((symbol-function 'nlga-update)
                   (lambda (&rest _args) (setq updates (1+ updates))))
                  ((symbol-function 'nlga-step)
                   (lambda (&rest _args) (setq steps (1+ steps)))))
          (should-error
           (nl-llm-agent-ondevice-train
            ctx (nth 0 case) 1 :loss-starts (nth 1 case)
            :loss-masks (nth 2 case)))
          (should (= updates 0))
          (should (= steps 0))
          (should (= (plist-get ctx :step) 0)))))))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-preserves-mode-and-resume-validation ()
  (let ((updates 0)
        (ctx (nl-llm-agent-loss-masks-test--ctx)))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (&rest _args) (setq updates (1+ updates))))
              ((symbol-function 'nlga-step) #'ignore))
      (setf (plist-get ctx :loss-mode) nil)
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2)) 1 :loss-masks (vector [0 1])))
      (setf (plist-get ctx :loss-mode) 'completion)
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2)) 1 :start-step 1 :loss-starts [1]
        :loss-masks (vector [0 1])))
      (should (= updates 0))
      (should (= (plist-get ctx :step) 0)))))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-callback-mutation-is-detached ()
  (let* ((first [0 1 0 0])
         (second [0 0 1 0])
         (run (nl-llm-agent-loss-masks-test--run
               (nl-llm-agent-loss-masks-test--ctx t)
               '((1 2 3 4) (2 3 4 0)) [1 1]
               (vector first second)
               nil
               (lambda (_ctx completed _total)
                 (when (= completed 1)
                   (dotimes (index 4)
                     (aset first index 0)
                     (aset second index 0))))))
         (updates (car run))
         (scales nil))
    (dolist (update (reverse updates))
      (when (eq (car update) 'scale)
        (push (cadr update) scales)))
    (should (equal (nreverse scales)
                   (list [8.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                         [0.0 8.0 0.0 0.0 0.0 0.0 0.0 0.0])))))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-follows-shuffle-plan ()
  (let* ((trajs '((1 2 3 4) (2 3 4 0) (3 4 0 1)))
         (masks [[0 1 0 0] [0 0 1 0] [0 0 0 1]])
         (run (nl-llm-agent-loss-masks-test--run
               (nl-llm-agent-loss-masks-test--ctx t)
               trajs [1 1 1] masks 104729))
         (updates (car run))
         (pairs nil)
         (pending nil))
    (dolist (update (reverse updates))
      (pcase update
        (`(scale ,rows) (setq pending rows))
        (`(oh ,tokens) (push (list tokens pending) pairs))))
    (setq pairs (nreverse pairs))
    (let* ((permutation (cdr (nl-llm-agent-ondevice--shuffle-epoch 3 104729)))
           (expected
            (mapcar (lambda (index) (nth index trajs))
                    (append permutation nil))))
      (should (equal (mapcar #'car pairs) expected))
      (dolist (pair pairs)
        (let* ((token (car (car pair)))
               (rows (cadr pair))
               (selected-row (cond ((= token 1) 0)
                                   ((= token 2) 1)
                                   ((= token 3) 2))))
          (should (= (aref rows selected-row) 8.0))
          (dotimes (row 8)
            (unless (= row selected-row)
              (should (= (aref rows row) 0.0)))))))))

(provide 'agent-loss-masks-test)

(ert-run-tests-batch-and-exit)

;;; agent-loss-masks-test.el ends here
