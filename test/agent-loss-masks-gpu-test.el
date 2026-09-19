;;; agent-loss-masks-gpu-test.el --- sparse completion GPU parity -*- lexical-binding: t; -*-

;; Run manually on a Vulkan device:
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
;;     -l test/agent-loss-masks-gpu-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(setq load-prefer-newer t)
(require 'ert)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-agent-improve)
(require 'nl-llm-evolve)
(require 'nl-llm-gpu)

(defconst nl-llm-agent-loss-masks-gpu-test--tokens
  [65 66 65 67 68 65])

(defun nl-llm-agent-loss-masks-gpu-test--flat-parameters (model)
  "Return detached MODEL parameters in the library's canonical order."
  (apply #'vconcat
         (mapcar
          (lambda (parameter)
            (copy-sequence (photon-tensor-data (pav-value parameter))))
          (nl-llm-agent--p5-params model))))

(defun nl-llm-agent-loss-masks-gpu-test--max-diff (left right)
  "Return maximum absolute difference between equally sized vectors."
  (unless (= (length left) (length right))
    (error "loss-mask GPU parity parameter lengths differ: %d and %d"
           (length left) (length right)))
  (let ((maximum 0.0))
    (dotimes (index (length left))
      (let ((a (aref left index))
            (b (aref right index)))
        (unless (and (= a a) (= b b)
                     (< (abs a) 1.0e100) (< (abs b) 1.0e100))
          (error "loss-mask GPU parity produced a non-finite parameter"))
        (setq maximum (max maximum (abs (- a b))))))
    maximum))

(defun nl-llm-agent-loss-masks-gpu-test--assert-cpu-dispatch ()
  "Assert that all saved GPU tensor function cells currently point to CPU."
  (unless (and (boundp 'photon-tensor-gpu--saved)
               photon-tensor-gpu--saved)
    (error "loss-mask GPU parity has no saved CPU tensor dispatch"))
  (dolist (pair photon-tensor-gpu--saved)
    (unless (eq (symbol-function (car pair)) (cdr pair))
      (error "loss-mask GPU parity tensor op is not on CPU dispatch: %S"
             (car pair))))
  t)

(defun nl-llm-agent-loss-masks-gpu-test--cpu-step
    (model learning-rate)
  "Run the CPU reference update and return its detached post-step parameters."
  (photon-tensor-use-cpu-backend)
  (nl-llm-agent-loss-masks-gpu-test--assert-cpu-dispatch)
  (let* ((tokens (append nl-llm-agent-loss-masks-gpu-test--tokens nil))
         ;; Keep the CPU reference unpadded.  Causal rows before the end of the
         ;; trajectory are identical to the padded GPU graphs, while this makes
         ;; pad-id handling an independent part of the GPU check.
         (inputs (butlast tokens))
         (targets (apply #'vector (cdr tokens)))
         ;; Target token J is predicted at row J-1.  The sparse target mask is
         ;; [0 0 1 0 1 1], so the unpadded CPU row mask is [0 1 0 1 1].
         (row-mask [0 1 0 1 1])
         (parameters (nl-llm-agent--p5-params model)))
    (let ((loss
           (nl-llm-agent--p5-forward
            model inputs targets row-mask)))
      (photon-autograd-zero-grad parameters)
      (photon-autograd-backward loss)
      (photon-autograd-sgd parameters learning-rate))
    (nl-llm-agent-loss-masks-gpu-test--flat-parameters model)))

(defun nl-llm-agent-loss-masks-gpu-test--gpu-pair
    (initial sequence-length pad-id learning-rate cpu-values)
  "Run one sparse masked step through dense and compact resident graphs."
  (let* ((dense-model (nl-llm-evolve-copy-model initial))
         (compact-model (nl-llm-evolve-copy-model initial))
         (dense nil)
         (compact nil)
         (tokens (append nl-llm-agent-loss-masks-gpu-test--tokens nil))
         (starts [2])
         (masks (vector [0 0 1 0 1 1]))
         (initial-values
          (nl-llm-agent-loss-masks-gpu-test--flat-parameters initial)))
    (unwind-protect
        (progn
          (setq dense
                (nl-llm-agent-ondevice-from-model
                 dense-model sequence-length learning-rate
                 :loss-mode 'completion))
          (setq compact
                (nl-llm-agent-ondevice-from-model
                 compact-model sequence-length learning-rate
                 :loss-mode 'completion :transfer-mode 'compact))
          ;; The production tokenizer-derived default is not part of this
          ;; numerical contract; explicitly exercise both requested padding ids.
          (setf (plist-get dense :pad-token) pad-id
                (plist-get compact :pad-token) pad-id)
          (nl-llm-agent-ondevice-train
           dense (list tokens) 1 :loss-starts starts :loss-masks masks)
          (nl-llm-agent-ondevice-train
           compact (list tokens) 1 :loss-starts starts :loss-masks masks)
          (nl-llm-agent-ondevice-sync dense)
          (nl-llm-agent-ondevice-sync compact)
          (let ((dense-values
                 (nl-llm-agent-loss-masks-gpu-test--flat-parameters dense-model))
                (compact-values
                 (nl-llm-agent-loss-masks-gpu-test--flat-parameters compact-model)))
            (should (= (length cpu-values) (length initial-values)))
            (should (= (length dense-values) (length initial-values)))
            (should (= (length compact-values) (length initial-values)))
            (let ((cpu-change
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    initial-values cpu-values))
                  (dense-change
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    initial-values dense-values))
                  (compact-change
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    initial-values compact-values))
                  (cpu-dense
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    cpu-values dense-values))
                  (cpu-compact
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    cpu-values compact-values))
                  (dense-compact
                   (nl-llm-agent-loss-masks-gpu-test--max-diff
                    dense-values compact-values)))
              (princ (format "loss-mask GPU parity seq=%d pad=%d changes cpu=%.3g dense=%.3g compact=%.3g diffs cpu/dense=%.3g cpu/compact=%.3g dense/compact=%.3g\n"
                             sequence-length pad-id cpu-change dense-change compact-change
                             cpu-dense cpu-compact dense-compact))
              (should (> cpu-change 1.0e-10))
              (should (> dense-change 1.0e-10))
              (should (> compact-change 1.0e-10))
            ;; Keep this tolerance fixed: a failure is a parity regression, not
            ;; a reason to weaken the numerical contract.
              (should (< cpu-dense 1.0e-5))
              (should (< cpu-compact 1.0e-5))
              (should (< dense-compact 1.0e-5)))))
      (when dense (nl-llm-agent-ondevice-free dense))
      (when compact (nl-llm-agent-ondevice-free compact)))))

(ert-deftest nl-llm-agent-ondevice-sparse-mask-real-gpu-parity ()
  (unless (nl-llm-gpu-enable)
    (ert-skip "No Vulkan device"))
  (unwind-protect
      (dolist (case '((8 0) (8 255) (16 0) (16 255)))
        (let* ((sequence-length (nth 0 case))
               (pad-id (nth 1 case))
               (learning-rate 0.01)
               ;; UTF-8 byte vocabulary admits both requested pad ids.
               (initial (nl-llm-agent-improve-model 8 12 256 1 2
                                                     "utf8-byte-v1"))
               (cpu-model (nl-llm-evolve-copy-model initial))
               ;; CPU dispatch is explicitly selected before resident GPU
               ;; construction; this remains true even if a prior test changed
               ;; the process-wide backend selection.
               (cpu-values
                (progn
                  (photon-tensor-use-cpu-backend)
                   (nl-llm-agent-loss-masks-gpu-test--cpu-step
                   cpu-model learning-rate))))
          (nl-llm-agent-loss-masks-gpu-test--gpu-pair
           initial sequence-length pad-id learning-rate cpu-values)))
    (nl-llm-gpu-disable)))

(provide 'agent-loss-masks-gpu-test)

(ert-run-tests-batch-and-exit)

;;; agent-loss-masks-gpu-test.el ends here
