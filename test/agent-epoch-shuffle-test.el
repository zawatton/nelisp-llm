;;; agent-epoch-shuffle-test.el --- deterministic on-device epoch order -*- lexical-binding: t; -*-

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(setq load-prefer-newer t)
(require 'ert)
(require 'nl-llm-agent-ondevice)

(ert-deftest nl-llm-agent-ondevice-shuffle-known-permutations ()
  (let ((state 104729)
        (permutations nil))
    (dotimes (_ 3)
      (let ((epoch (nl-llm-agent-ondevice--shuffle-epoch 4 state)))
        (setq state (car epoch))
        (push (append (cdr epoch) nil) permutations)))
    (should
     (equal (nreverse permutations)
            '((3 1 2 0) (1 0 2 3) (3 2 1 0))))
    (should (= state 40761457))
    (let* ((one (nl-llm-agent-ondevice--shuffle-epoch 4 1))
           (one-again (nl-llm-agent-ondevice--shuffle-epoch 4 1))
           (other (nl-llm-agent-ondevice--shuffle-epoch 4 2)))
      (should (equal one one-again))
      (should-not (equal (cdr one) (cdr other)))
      (should (equal (sort (append (cdr one) nil) #'<) '(0 1 2 3)))
      (should (equal (sort (append (cdr other) nil) #'<) '(0 1 2 3))))))

(ert-deftest nl-llm-agent-ondevice-shuffle-pairs-plan-and-progress ()
  (let* ((trajs '((10 11 12) (20 21 22 23) (30 31) (40 41 42 43 44)))
         (original-trajs (copy-tree trajs))
         (starts [1 2 1 3])
         (original-starts (copy-sequence starts))
         (ctx (list :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
                    :seq 8 :vocab 100 :pad-token 0 :optimizer 'adam
                    :loss-mode 'completion :step 7))
         (updates nil)
         (adam-times nil)
         (steps 0)
         (progress nil))
    (cl-letf (((symbol-function 'nl-llm-agent--onehot-pad)
               (lambda (tokens _seq _vocab _pad) (copy-sequence tokens)))
              ((symbol-function 'nl-llm-agent--shift-pad)
               (lambda (tokens _seq &optional _pad) tokens))
              ((symbol-function 'nl-llm-agent-ondevice--expand-row-scales)
               (lambda (rows _vocab) rows))
              ((symbol-function 'nlga-update)
               (lambda (resident value)
                 (push (list resident value) updates)))
              ((symbol-function 'nlga-adam-update-t)
               (lambda (_builder time) (push time adam-times)))
              ((symbol-function 'nlga-step)
               (lambda (_builder) (setq steps (1+ steps))))
              ((symbol-function 'nl-llm-agent-ondevice--index-pad)
               (lambda (tokens _seq _pad) (copy-sequence tokens))))
      (should (= 12
                 (nl-llm-agent-ondevice-train
                  ctx trajs 3 :loss-starts starts :shuffle-seed 104729
                  :after-step
                  (lambda (_ctx completed total)
                    (push (list completed total) progress)))))
      (should (= (plist-get ctx :step) 19))
      (should (= steps 12))
      (should (equal (nreverse adam-times)
                     '(8 9 10 11 12 13 14 15 16 17 18 19)))
      (should (equal (nreverse progress)
                     '((1 12) (2 12) (3 12) (4 12) (5 12) (6 12)
                       (7 12) (8 12) (9 12) (10 12) (11 12) (12 12))))
      (let ((seen-tokens nil)
            (seen-scales nil))
        (dolist (update (nreverse updates))
          (pcase update
            (`(oh ,tokens) (push tokens seen-tokens))
            (`(scale ,rows) (push rows seen-scales))))
        (should (equal (nreverse seen-tokens)
                       '((40 41 42 43 44) (20 21 22 23) (30 31) (10 11 12)
                         (20 21 22 23) (10 11 12) (30 31) (40 41 42 43 44)
                         (40 41 42 43 44) (30 31) (20 21 22 23) (10 11 12))))
        ;; Each completion mask must stay with the trajectory it was built for.
        (should
         (equal (nreverse seen-scales)
                '([0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0]
                  [0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0]
                  [8.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0]
                  [4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [8.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0]
                  [0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0]
                  [8.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0]
                  [4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0]))))
      ;; The caller's list and its nested token lists are never shuffled.
      (should (equal trajs original-trajs))
      (should (equal starts original-starts)))))

(ert-deftest nl-llm-agent-ondevice-shuffle-nil-keeps-traversal ()
  (let ((ctx (list :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
                   :seq 4 :vocab 100 :pad-token 0 :optimizer 'sgd
                   :loss-mode 'completion :step 0))
        (seen nil))
    (cl-letf (((symbol-function 'nl-llm-agent--onehot-pad)
               (lambda (tokens _seq _vocab _pad) (copy-sequence tokens)))
              ((symbol-function 'nl-llm-agent--shift-pad)
               (lambda (tokens _seq &optional _pad) tokens))
              ((symbol-function 'nl-llm-agent-ondevice--expand-row-scales)
               (lambda (rows _vocab) rows))
              ((symbol-function 'nl-llm-agent-ondevice--index-pad)
               (lambda (tokens _seq _pad) (copy-sequence tokens)))
              ((symbol-function 'nlga-update)
               (lambda (resident value)
                 (when (eq resident 'oh) (push value seen))))
              ((symbol-function 'nlga-step) #'ignore))
      (nl-llm-agent-ondevice-train
       ctx '((10 11) (20 21) (30 31) (40 41)) 1 :loss-starts [1 1 1 1])
      (should (equal (nreverse seen)
                     '((10 11) (20 21) (30 31) (40 41)))))))

(ert-deftest nl-llm-agent-ondevice-shuffle-validation-before-updates ()
  (let ((updates 0)
        (ctx (list :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
                   :seq 4 :vocab 100 :pad-token 0 :optimizer 'sgd
                   :loss-mode 'completion :step 0)))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (&rest _args) (setq updates (1+ updates))))
              ((symbol-function 'nlga-step) #'ignore))
      (dolist (seed (list 0 -1 1.0
                          (1+ nl-llm-agent-ondevice--uint32-mask)
                          'not-an-integer))
        (should-error
         (nl-llm-agent-ondevice-train
          ctx '((1 2) (3 4)) 1 :loss-starts [1 1] :shuffle-seed seed)))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2) (3 4)) 1 :loss-starts [1 1]
        :shuffle-seed 1 :start-step 1))
      (setf (plist-get ctx :loss-mode) nil)
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2) (3 4)) 1 :shuffle-seed 1))
      (should (= updates 0))
      (should (= (plist-get ctx :step) 0)))))

(ert-deftest nl-llm-agent-ondevice-shuffle-invalid-plan-before-updates ()
  (let ((updates 0)
        (ctx (list :b 'builder :oh 'oh :ohtgt 'ohtgt :loss-scale 'scale
                   :seq 4 :vocab 100 :pad-token 0 :optimizer 'sgd
                   :loss-mode 'completion :step 0)))
    (cl-letf (((symbol-function 'nlga-update)
               (lambda (&rest _args) (setq updates (1+ updates))))
              ((symbol-function 'nlga-step) #'ignore))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2) (3 999)) 1 :loss-starts [1 1] :shuffle-seed 1))
      (should-error
       (nl-llm-agent-ondevice-train
        ctx '((1 2) (3 4)) 1 :loss-starts [1 4] :shuffle-seed 1))
      (should (= updates 0))
      (should (= (plist-get ctx :step) 0)))))

(provide 'agent-epoch-shuffle-test)

(ert-run-tests-batch-and-exit)

;;; agent-epoch-shuffle-test.el ends here
