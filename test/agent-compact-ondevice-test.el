;;; agent-compact-ondevice-test.el --- compact completion transfers -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(setq load-prefer-newer t)
(require 'ert)
(require 'cl-lib)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-evolve)
(require 'nl-llm-gpu)

(defun nl-llm-agent-compact-test--flat-parameters (model)
  "Return a detached flat vector of MODEL parameters."
  (apply
   #'vconcat
   (mapcar
    (lambda (parameter)
      (copy-sequence (photon-tensor-data (pav-value parameter))))
    (nl-llm-agent--p5-params model))))

(defun nl-llm-agent-compact-test--flat-optimizer (builder)
  "Return detached Adam moments from BUILDER in parameter order."
  (apply
   #'vconcat
   (mapcar
    (lambda (pair)
      (vconcat (photon-tensor-data (car pair))
               (photon-tensor-data (cdr pair))))
    (nlga-adam-state builder))))

(defun nl-llm-agent-compact-test--maxdiff (left right)
  "Return maximum absolute difference between vectors LEFT and RIGHT."
  (let ((maximum 0.0))
    (should (= (length left) (length right)))
    (dotimes (index (length left))
      (let ((left-value (aref left index))
            (right-value (aref right index)))
        (should (and (= left-value left-value)
                     (< (abs left-value) 1.0e100)))
        (should (and (= right-value right-value)
                     (< (abs right-value) 1.0e100)))
        (setq maximum
              (max maximum (abs (- left-value right-value))))))
    maximum))

(defun nl-llm-agent-compact-test--model ()
  "Return a deterministic tiny UTF-8 P5 model."
  (nl-llm-agent-improve-model 2 4 256 1 1 "utf8-byte-v1"))

(defun nl-llm-agent-compact-test--paired (optimizer)
  "Compare dense and compact multi-step training under OPTIMIZER."
  (let* ((initial (nl-llm-agent-compact-test--model))
         (before (nl-llm-agent-compact-test--flat-parameters initial))
         (dense-model (nl-llm-evolve-copy-model initial))
         (compact-model (nl-llm-evolve-copy-model initial))
         (tokens-a
          (nl-llm-agent-tokenizer-encode "日日a" "utf8-byte-v1"))
         (tokens-b
          (nl-llm-agent-tokenizer-encode "日日b" "utf8-byte-v1"))
         (trajs (list tokens-a tokens-b))
         ;; Include repeated multibyte input, a nonempty masked prompt, and
         ;; padding rows after each seven-token trajectory in SEQ 10.
         (starts [3 4])
         (dense nil)
         (compact nil))
    (unwind-protect
        (progn
          (setq dense
                (nl-llm-agent-ondevice-from-model
                 dense-model 10 0.003 :optimizer optimizer
                 :loss-mode 'completion))
          (setq compact
                (nl-llm-agent-ondevice-from-model
                 compact-model 10 0.003 :optimizer optimizer
                 :loss-mode 'completion :transfer-mode 'compact))
          (should-not (plist-get dense :transfer-mode))
          (should (= (nlga-nout (plist-get dense :b)) 1))
          (should (integerp (plist-get dense :lout)))
          (should (eq (plist-get compact :transfer-mode) 'compact))
          (should (= (nlga-nout (plist-get compact :b)) 0))
          (should-not (plist-get compact :lout))
          (dolist (key '(:oh :ohtgt :loss-scale))
            (let ((resident (plist-get compact key)))
              (should (= (nlga-rt-rows resident) 10))
              (should (= (nlga-rt-cols resident) 1))))
          ;; Compact construction has no dense causal-mask resident or output
          ;; slot.  Attention applies its causal mask in the fused kernel.
          (should-not
           (cl-find-if
            (lambda (slot)
              (and (eq (car-safe slot) 'res)
                   (= (or (nth 2 slot) -1) 100)))
            (nlga-slots (plist-get compact :b))))
          (should-not
           (cl-find-if
            (lambda (slot) (eq (car-safe slot) 'out))
            (nlga-slots (plist-get compact :b))))
          (nl-llm-agent-ondevice-train dense trajs 2 :loss-starts starts)
          (nl-llm-agent-ondevice-train compact trajs 2 :loss-starts starts)
          (should (= (plist-get dense :step) 4))
          (should (= (plist-get compact :step) 4))
          (when (eq optimizer 'adam)
            (should
             (< (nl-llm-agent-compact-test--maxdiff
                 (nl-llm-agent-compact-test--flat-optimizer
                  (plist-get dense :b))
                 (nl-llm-agent-compact-test--flat-optimizer
                  (plist-get compact :b)))
                2.0e-5)))
          (nl-llm-agent-ondevice-sync dense)
          (nl-llm-agent-ondevice-sync compact)
          (let ((dense-values
                 (nl-llm-agent-compact-test--flat-parameters dense-model))
                (compact-values
                 (nl-llm-agent-compact-test--flat-parameters compact-model)))
            (should
             (> (nl-llm-agent-compact-test--maxdiff before dense-values)
                1.0e-8))
            (should
             (< (nl-llm-agent-compact-test--maxdiff
                 dense-values compact-values)
                2.0e-5))))
      (when dense
        (nl-llm-agent-ondevice-free dense))
      (when compact
        (nl-llm-agent-ondevice-free compact)))))

(ert-deftest nl-llm-agent-compact-transfer-validation-before-allocation ()
  (let ((model (nl-llm-agent-compact-test--model))
        (allocations 0))
    (cl-letf (((symbol-function 'nlga-new)
               (lambda ()
                 (setq allocations (1+ allocations))
                 (error "unexpected allocation"))))
      (should-error
       (nl-llm-agent-ondevice-from-model
        model 8 0.01 :transfer-mode 'compact))
      (should-error
       (nl-llm-agent-ondevice-from-model
        model 8 0.01 :loss-mode 'completion :transfer-mode 'dense))
      (should (= allocations 0)))))

(ert-deftest nl-llm-agent-compact-completion-real-gpu ()
  (unless (nl-llm-gpu-enable)
    (ert-skip "No Vulkan device"))
  (unwind-protect
      (progn
        (nl-llm-agent-compact-test--paired 'sgd)
        (nl-llm-agent-compact-test--paired 'adam)

        ;; Per SGD step the compact path writes exactly token indices, target
        ;; indices, and row scales: three SEQ-length vectors and no output.
        (let* ((model (nl-llm-agent-compact-test--model))
               (context
                (nl-llm-agent-ondevice-from-model
                 model 10 0.01 :loss-mode 'completion
                 :transfer-mode 'compact))
               (original-write
                (symbol-function 'nelisp-gpu-server-write-resident))
               (writes nil))
          (unwind-protect
              (cl-letf
                  (((symbol-function 'nelisp-gpu-server-write-resident)
                    (lambda (handle values)
                      (push (length values) writes)
                      (funcall original-write handle values))))
                (nl-llm-agent-ondevice-train
                 context
                 (list
                  (nl-llm-agent-tokenizer-encode
                   "日日a" "utf8-byte-v1"))
                 1 :loss-starts [3])
                (should (equal (nreverse writes) '(10 10 10))))
            (nl-llm-agent-ondevice-free context)))

        ;; A malformed later trajectory is rejected before any resident write,
        ;; optimizer step, or parameter mutation from the valid first item.
        (let* ((model (nl-llm-agent-compact-test--model))
               (context
                (nl-llm-agent-ondevice-from-model
                 model 10 0.01 :loss-mode 'completion
                 :transfer-mode 'compact))
               (before nil)
               (writes 0))
          (unwind-protect
              (progn
                (nl-llm-agent-ondevice-sync context)
                (setq before
                      (nl-llm-agent-compact-test--flat-parameters model))
                (cl-letf
                    (((symbol-function 'nelisp-gpu-server-write-resident)
                      (lambda (&rest _)
                        (setq writes (1+ writes))
                        (error "unexpected write"))))
                  (should-error
                   (nl-llm-agent-ondevice-train
                    context (list '(1 2 3) '(1 2 999)) 1
                    :loss-starts [1 1])))
                (should (= writes 0))
                (should (= (plist-get context :step) 0))
                (nl-llm-agent-ondevice-sync context)
                (should
                 (equal before
                        (nl-llm-agent-compact-test--flat-parameters model))))
            (nl-llm-agent-ondevice-free context))))
    (nl-llm-gpu-disable)))

(provide 'agent-compact-ondevice-test)

(ert-run-tests-batch-and-exit)

;;; agent-compact-ondevice-test.el ends here
