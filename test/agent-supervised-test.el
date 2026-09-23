;;; agent-supervised-test.el --- completion-only supervision tests -*- lexical-binding: t; -*-

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(setq load-prefer-newer t)
(require 'ert)
(require 'cl-lib)
(require 'nl-llm-agent-supervised)

(declare-function nl-llm-gpu-enable "nl-llm-gpu" ())
(declare-function nl-llm-gpu-disable "nl-llm-gpu" ())

(defun nl-llm-agent-supervised-test--weights (model)
  "Return a detached flat vector of MODEL parameter values."
  (apply
   #'vconcat
   (mapcar
    (lambda (parameter)
      (copy-sequence (photon-tensor-data (pav-value parameter))))
    (nl-llm-agent--p5-params model))))

(ert-deftest nl-llm-agent-supervised-encode-binds-boundaries-and-unicode ()
  (let* ((ascii
          (nl-llm-agent-supervised-encode
           [(:prompt "Q:" :completion "A!")]))
         (utf8
          (nl-llm-agent-supervised-encode
           [(:prompt "問:" :completion "答")]
           nl-llm-agent-tokenizer-utf8))
         (left
          (nl-llm-agent-supervised-encode
           [(:prompt "a" :completion "bc")]))
         (right
          (nl-llm-agent-supervised-encode
           [(:prompt "ab" :completion "c")])))
    (should (equal (plist-get ascii :tokenizer)
                   nl-llm-agent-tokenizer-ascii))
    (should (equal (plist-get ascii :trajectories)
                   (list (nl-llm-agent-tokenizer-encode "Q:A!"))))
    (should (equal (plist-get ascii :loss-starts) [2]))
    (should (= (plist-get ascii :completion-tokens) 2))
    (should (equal (plist-get utf8 :tokenizer)
                   nl-llm-agent-tokenizer-utf8))
    (should (equal (plist-get utf8 :loss-starts) [4]))
    (should (= (plist-get utf8 :completion-tokens) 3))
    (should (= (length (car (plist-get utf8 :trajectories))) 7))
    ;; Equal concatenated text has different supervision boundaries and digest.
    (should (equal (plist-get left :trajectories)
                   (plist-get right :trajectories)))
    (should-not (equal (plist-get left :loss-starts)
                       (plist-get right :loss-starts)))
    (should-not (equal (plist-get left :dataset-sha256)
                       (plist-get right :dataset-sha256)))))

(ert-deftest nl-llm-agent-supervised-encode-is-detached-and-strictly-bounded ()
  (let* ((prompt (propertize "P:" 'face 'bold))
         (completion (propertize "ok" 'secret t))
         (examples (vector (list :prompt prompt :completion completion)))
         (encoded (nl-llm-agent-supervised-encode examples)))
    (aset prompt 0 ?X)
    (aset completion 0 ?Y)
    (setf (plist-get (aref examples 0) :prompt) "changed")
    (should (equal (plist-get encoded :trajectories)
                   (list (nl-llm-agent-tokenizer-encode "P:ok"))))
    (should-not
     (text-properties-at 0 (plist-get encoded :dataset-sha256))))
  (dolist (bad
           (list []
                 [(:prompt "" :completion "x")]
                 [(:prompt "x" :completion "")]
                 [(:prompt "x" :completion "y" :extra t)]
                 [(:prompt "x" :prompt "z" :completion "y")]
                 (make-vector 129 '(:prompt "x" :completion "y"))
                 (vector (list :prompt (make-string 4096 ?a)
                               :completion "b"))
                 (vector (list :prompt (make-string 1400 ?日)
                               :completion "本"))))
    (should-error
     (nl-llm-agent-supervised-encode
      bad (if (and (vectorp bad) (> (length bad) 0)
                   (stringp (plist-get (aref bad 0) :prompt))
                   (string-match-p "日" (plist-get (aref bad 0) :prompt)))
              nl-llm-agent-tokenizer-utf8
            nil))))
  (let ((too-large (make-vector 17 nil)))
    (dotimes (index 17)
      (aset too-large index
            (list :prompt (make-string 4095 ?a) :completion "b")))
    (should-error (nl-llm-agent-supervised-encode too-large))))

(ert-deftest nl-llm-agent-supervised-cpu-train-lowers-completion-loss ()
  (let* ((model (nl-llm-agent-improve-model 4 4 nil 1 1))
         (examples [(:prompt "Q:" :completion "A")])
         (before-weights (nl-llm-agent-supervised-test--weights model))
         (before (nl-llm-agent-supervised-loss model examples))
         (result
          (nl-llm-agent-supervised-train
           model examples :backend 'cpu :lr 0.2 :epochs 16))
         (after (nl-llm-agent-supervised-loss model examples)))
    (should (eq (plist-get result :backend) 'cpu))
    (should (= (plist-get result :steps) 16))
    (should (= (plist-get result :examples) 1))
    (should (= (plist-get result :completion-tokens) 1))
    (should (= (plist-get result :loss-before) before))
    (should (= (plist-get result :loss-after) after))
    (should (< after before))
    (should-not
     (equal before-weights
            (nl-llm-agent-supervised-test--weights model)))))

(ert-deftest nl-llm-agent-supervised-loss-is-completion-token-weighted ()
  (let* ((model (nl-llm-agent-improve-model 4 4 nil 1 1))
         (short [(:prompt "Q:" :completion "A")])
         (long [(:prompt "R:" :completion "BC")])
         (short-loss (nl-llm-agent-supervised-loss model short))
         (long-loss (nl-llm-agent-supervised-loss model long))
         (combined
          (nl-llm-agent-supervised-loss
           model [(:prompt "Q:" :completion "A")
                  (:prompt "R:" :completion "BC")])))
    (should
     (< (abs (- combined (/ (+ short-loss (* 2.0 long-loss)) 3.0)))
        1.0e-12))))

(ert-deftest nl-llm-agent-supervised-invalid-plan-never-mutates-or-allocates ()
  (let* ((model (nl-llm-agent-improve-model 4 4 nil 1 1))
         (examples [(:prompt "Q:" :completion "A")])
         (before (nl-llm-agent-supervised-test--weights model))
         (allocations 0))
    (cl-letf (((symbol-function 'nl-llm-agent-ondevice-from-model)
               (lambda (&rest _arguments)
                 (setq allocations (1+ allocations))
                 (error "unexpected GPU allocation"))))
      (dolist (keys
               (list '(:backend cpu :optimizer adam)
                     '(:backend cpu :sequence 8)
                     '(:backend cpu :lr 0.0)
                     '(:backend cpu :epochs 33)
                     '(:backend gpu :sequence 2)
                     '(:backend gpu :sequence 8 :unknown t)))
        (should-error
         (apply #'nl-llm-agent-supervised-train model examples keys)))
      (should-error
       (nl-llm-agent-supervised-train
        model [(:prompt "Q:" :completion "")]
        :backend 'gpu :sequence 8))
      (should (= allocations 0))
      (should (equal before
                     (nl-llm-agent-supervised-test--weights model))))))

(ert-deftest nl-llm-agent-supervised-rejects-model-tokenizer-mismatch ()
  (let* ((model (nl-llm-agent-improve-model 4 4 nil 1 1))
         (before (nl-llm-agent-supervised-test--weights model)))
    (setf (plist-get model :tokenizer) nl-llm-agent-tokenizer-utf8)
    (should-error
     (nl-llm-agent-supervised-loss
      model [(:prompt "問" :completion "答")]))
    (should-error
     (nl-llm-agent-supervised-train
      model [(:prompt "問" :completion "答")] :backend 'cpu))
    (should (equal before
                   (nl-llm-agent-supervised-test--weights model)))))

(ert-deftest nl-llm-agent-supervised-gpu-wrapper-trains-and-cleans-failure ()
  (unless (and (require 'nl-llm-gpu nil t)
               (require 'nl-llm-agent-ondevice nil t)
               (nl-llm-gpu-enable))
    (ert-skip "no supported Vulkan device"))
  (unwind-protect
      (let* ((examples [(:prompt "Q" :completion "A")])
             (model (nl-llm-agent-improve-model 2 2 nil 1 1))
             (before (nl-llm-agent-supervised-loss model examples))
             (before-weights
              (nl-llm-agent-supervised-test--weights model))
             (real-from
              (symbol-function 'nl-llm-agent-ondevice-from-model))
             (selected-transfer nil)
             (result
              (cl-letf
                  (((symbol-function 'nl-llm-agent-ondevice-from-model)
                    (lambda (child sequence learning-rate &rest keys)
                      (setq selected-transfer
                            (plist-get keys :transfer-mode))
                      (apply real-from child sequence learning-rate keys))))
                (nl-llm-agent-supervised-train
                 model examples :backend 'gpu :lr 0.1 :epochs 4)))
             (after (nl-llm-agent-supervised-loss model examples)))
        (should (eq (plist-get result :backend) 'gpu))
        (should (eq selected-transfer 'compact))
        (should (= (plist-get result :steps) 4))
        (should (< after before))
        (should-not
         (equal before-weights
                (nl-llm-agent-supervised-test--weights model)))
        (let* ((failed (nl-llm-agent-improve-model 2 2 nil 1 1))
               (failed-before
                (nl-llm-agent-supervised-test--weights failed))
               (real-from
                (symbol-function 'nl-llm-agent-ondevice-from-model))
               (real-free
                (symbol-function 'nl-llm-agent-ondevice-free))
               (allocations 0)
               (frees 0))
          (cl-letf
              (((symbol-function 'nl-llm-agent-ondevice-from-model)
                (lambda (&rest arguments)
                  (setq allocations (1+ allocations))
                  (apply real-from arguments)))
               ((symbol-function 'nl-llm-agent-ondevice-train)
                (lambda (&rest _arguments)
                  (error "injected supervised GPU step failure")))
               ((symbol-function 'nl-llm-agent-ondevice-free)
                (lambda (context)
                  (setq frees (1+ frees))
                  (funcall real-free context))))
            (should-error
             (nl-llm-agent-supervised-train
              failed examples :backend 'gpu :lr 0.1 :epochs 1)))
          (should (= allocations 1))
          (should (= frees 1))
          (should
           (equal failed-before
                  (nl-llm-agent-supervised-test--weights failed)))))
    (ignore-errors (nl-llm-gpu-disable))))

(provide 'agent-supervised-test)
(ert-run-tests-batch-and-exit)

;;; agent-supervised-test.el ends here
