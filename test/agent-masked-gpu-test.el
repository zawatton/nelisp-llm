;;; agent-masked-gpu-test.el --- completion-only GPU loss tests -*- lexical-binding: t; -*-

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(setq load-prefer-newer t)
(require 'ert)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-gpu)

(defun nl-llm-agent-masked-gpu-test--parameters (model)
  "Return detached flattened parameter values from MODEL."
  (apply
   #'vconcat
   (mapcar
    (lambda (parameter)
      (copy-sequence (photon-tensor-data (pav-value parameter))))
    (nl-llm-agent--p5-params model))))

(defun nl-llm-agent-masked-gpu-test--maxdiff (left right)
  "Return maximum absolute difference between LEFT and RIGHT."
  (let ((maximum 0.0))
    (dotimes (index (length left))
      (setq maximum
            (max maximum
                 (abs (- (aref left index) (aref right index))))))
    maximum))

(defun nl-llm-agent-masked-gpu-test--cpu-step
    (model tokens start learning-rate)
  "Take one completion-masked CPU SGD step on TOKENS from START."
  (let* ((inputs (butlast tokens))
         (targets (vconcat (cdr tokens)))
         (rows (length inputs))
         (mask (make-vector rows 0))
         (row (1- start))
         (parameters (nl-llm-agent--p5-params model)))
    (while (< row rows)
      (aset mask row 1)
      (setq row (1+ row)))
    (let ((loss (nl-llm-agent--p5-forward model inputs targets mask)))
      (photon-autograd-zero-grad parameters)
      (photon-autograd-backward loss)
      (photon-autograd-sgd parameters learning-rate))))

(defun nl-llm-agent-masked-gpu-test--simple-step (masked)
  "Return logits after one GPU CE step; use all-active mask when MASKED."
  (let* ((builder (nlga-new))
         (tensor (photon-tensor '(3 2) (make-vector 6 0.0)))
         (logits (nlga-param builder tensor))
         (targets
          (nlga-const
           builder
           (photon-tensor '(3 2) (vector 1.0 0.0 0.0 1.0 1.0 0.0))))
         (scales
          (and masked
               (nlga-const
                builder
                (photon-tensor '(3 2) (make-vector 6 1.0))))))
    (unwind-protect
        (progn
          (if masked
              (nlga-seed-ce-masked builder logits targets scales)
            (nlga-seed-ce builder logits targets))
          (nlga-finish builder (nlga-scalar builder 0.1))
          (nlga-compile builder)
          (nlga-step builder)
          (nlga-readback builder)
          (copy-sequence (photon-tensor-data tensor)))
      (nlga-free builder))))

(ert-deftest nl-llm-agent-completion-loss-real-gpu ()
  (unless (nl-llm-gpu-enable)
    (ert-skip "No Vulkan device"))
  (unwind-protect
      (progn
        ;; A malformed scale is rejected before the seed allocates a gradient
        ;; slot or appends any dispatch to the existing graph.
        (let* ((builder (nlga-new))
               (logits
                (nlga-param
                 builder
                 (photon-tensor '(2 2) (make-vector 4 0.0))))
               (targets
                (nlga-const
                 builder
                 (photon-tensor '(2 2) (vector 1.0 0.0 0.0 1.0))))
               (wrong-scale
                (nlga-const
                 builder
                 (photon-tensor '(2 1) (vector 1.0 1.0))))
               (slots (nlga-nslot builder))
               (dispatches (copy-tree (nlga-disp builder))))
          (unwind-protect
              (progn
                (should-error
                 (nlga-seed-ce-masked
                  builder logits targets wrong-scale))
                (should (= (nlga-nslot builder) slots))
                (should (equal (nlga-disp builder) dispatches))
                (should-not (nlga-rt-grad logits)))
            (nlga-free builder)))

        ;; Applying an all-one scale is exactly the legacy CE seed.
        (let ((legacy (nl-llm-agent-masked-gpu-test--simple-step nil))
              (masked (nl-llm-agent-masked-gpu-test--simple-step t)))
          (should (< (nl-llm-agent-masked-gpu-test--maxdiff legacy masked)
                     1.0e-7)))

        ;; A single active middle row changes only that row of a logits param.
        (let* ((builder (nlga-new))
               (tensor (photon-tensor '(3 2) (make-vector 6 0.0)))
               (logits (nlga-param builder tensor))
               (targets
                (nlga-const
                 builder
                 (photon-tensor
                  '(3 2) (vector 1.0 0.0 0.0 1.0 1.0 0.0))))
               (scales
                (nlga-const
                 builder
                 (photon-tensor
                  '(3 2) (vector 0.0 0.0 3.0 3.0 0.0 0.0)))))
          (unwind-protect
              (progn
                (nlga-seed-ce-masked builder logits targets scales)
                (nlga-finish builder (nlga-scalar builder 0.1))
                (nlga-compile builder)
                (nlga-step builder)
                (nlga-readback builder)
                (let ((values (photon-tensor-data tensor)))
                  (should (= (aref values 0) 0.0))
                  (should (= (aref values 1) 0.0))
                  (should (< (abs (- (aref values 2) -0.05)) 1.0e-6))
                  (should (< (abs (- (aref values 3) 0.05)) 1.0e-6))
                  (should (= (aref values 4) 0.0))
                  (should (= (aref values 5) 0.0))))
            (nlga-free builder)))

        ;; The full P5 GPU update agrees with completion-masked CPU autograd.
        (let* ((tokens '(0 1 2 3 4))
               (start 2)
               (learning-rate 0.01)
               (cpu (nl-llm-agent-improve-model 2 2 96 1 1))
               (gpu (nl-llm-agent-improve-model 2 2 96 1 1))
               (context nil))
          (nl-llm-agent-masked-gpu-test--cpu-step
           cpu tokens start learning-rate)
          (unwind-protect
              (progn
                (setq context
                      (nl-llm-agent-ondevice-from-model
                       gpu 7 learning-rate :loss-mode 'completion))
                (nl-llm-agent-ondevice-train
                 context (list tokens) 1 :loss-starts (vector start))
                ;; Ephemeral masked contexts cannot leak into old checkpoint
                ;; semantics, and rejection happens before host readback.
                (let ((host-before
                       (nl-llm-agent-masked-gpu-test--parameters gpu)))
                  (should-error (nl-llm-agent-ondevice-snapshot context))
                  (should
                   (equal host-before
                          (nl-llm-agent-masked-gpu-test--parameters gpu))))
                (should-error
                 (nl-llm-agent-ondevice-optimizer-state context))
                (should-error
                 (nl-llm-agent-ondevice-restore-training-state
                  context 0 nil))
                (nl-llm-agent-ondevice-sync context)
                (should
                 (< (nl-llm-agent-masked-gpu-test--maxdiff
                     (nl-llm-agent-masked-gpu-test--parameters cpu)
                     (nl-llm-agent-masked-gpu-test--parameters gpu))
                    5.0e-4)))
            (when context
              (nl-llm-agent-ondevice-free context))))

        ;; Causal completion loss is invariant to the identity of later
        ;; padding tokens because every padded target row has zero weight.
        (let* ((tokens '(0 1 2 3))
               (left (nl-llm-agent-improve-model 2 2 96 1 1))
               (right (nl-llm-agent-improve-model 2 2 96 1 1))
               (left-context
                (nl-llm-agent-ondevice-from-model
                 left 7 0.01 :loss-mode 'completion))
               (right-context
                (nl-llm-agent-ondevice-from-model
                 right 7 0.01 :loss-mode 'completion)))
          (unwind-protect
              (progn
                (setf (plist-get left-context :pad-token) 0
                      (plist-get right-context :pad-token) 5)
                (nl-llm-agent-ondevice-train
                 left-context (list tokens) 1 :loss-starts [2])
                (nl-llm-agent-ondevice-train
                 right-context (list tokens) 1 :loss-starts [2])
                (nl-llm-agent-ondevice-sync left-context)
                (nl-llm-agent-ondevice-sync right-context)
                (should
                 (< (nl-llm-agent-masked-gpu-test--maxdiff
                     (nl-llm-agent-masked-gpu-test--parameters left)
                     (nl-llm-agent-masked-gpu-test--parameters right))
                    1.0e-7)))
            (nl-llm-agent-ondevice-free left-context)
            (nl-llm-agent-ondevice-free right-context)))

        ;; A bad later example is rejected before an earlier example uploads or
        ;; advances either the optimizer counter or resident parameters.
        (let* ((model (nl-llm-agent-improve-model 2 2 96 1 1))
               (context
                (nl-llm-agent-ondevice-from-model
                 model 6 0.01 :loss-mode 'completion))
               before)
          (unwind-protect
              (progn
                ;; Establish the resident float32 representation before the
                ;; failed call, so this compares exact pre/post GPU state.
                (nl-llm-agent-ondevice-sync context)
                (setq before
                      (nl-llm-agent-masked-gpu-test--parameters model))
                (should-error
                 (nl-llm-agent-ondevice-train
                  context (list '(0 1 2) '(0 1 999)) 1
                  :loss-starts [1 1]))
                (should (= (plist-get context :step) 0))
                (nl-llm-agent-ondevice-sync context)
                (should
                 (equal before
                        (nl-llm-agent-masked-gpu-test--parameters model)))
                (should-error
                 (nl-llm-agent-ondevice-train
                  context (list '(0 1 2)) 1))
                (should-error
                 (nl-llm-agent-ondevice-train
                  context (list '(0 1 2)) 1
                  :start-step 1 :loss-starts [1]))
                (should (= (plist-get context :step) 0))
                (should-error
                 (let ((legacy
                        (nl-llm-agent-ondevice-from-model
                         (nl-llm-agent-improve-model 2 2 96 1 1) 6 0.01)))
                   (unwind-protect
                       (nl-llm-agent-ondevice-train
                        legacy (list '(0 1 2)) 1 :loss-starts [1])
                     (nl-llm-agent-ondevice-free legacy)))))
            (nl-llm-agent-ondevice-free context))))
    (nl-llm-gpu-disable)))

(provide 'agent-masked-gpu-test)

(ert-run-tests-batch-and-exit)

;;; agent-masked-gpu-test.el ends here
