;;; bench-completion-transfers.el --- dense vs compact completion transfer  -*- lexical-binding: t; -*-

;; Synthetic transport/correctness benchmark for the completion-loss GPU path.
;; This does not use a held-out corpus and does not measure model quality.
;;
;;   emacs -Q --batch -l examples/bench-completion-transfers.el

;;; Code:

(let* ((here (file-name-directory
              (or load-file-name buffer-file-name default-directory)))
       (root (expand-file-name ".." here)))
  (add-to-list 'load-path (expand-file-name "lisp" root))
  (load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                          (file-name-directory (or load-file-name buffer-file-name
                                                   default-directory))) nil t)
  (add-to-list 'load-path (expand-file-name "../nelisp-gpu/lisp" root)))

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-ondevice)

(defvar nl-llm-bench-completion-transfers-suppress-autorun nil
  "When non-nil, loading this benchmark only defines its functions.")

(defconst nl-llm-bench-completion-transfers--epochs 2)
(defconst nl-llm-bench-completion-transfers--optimizer 'sgd)
(defconst nl-llm-bench-completion-transfers--tolerance 1.0e-4)

(defun nl-llm-bench-completion-transfers--model ()
  "Return the deterministic tiny UTF-8 model used by the benchmark."
  (nl-llm-agent-improve-model 8 16 nil 1 1 "utf8-byte-v1"))

(defun nl-llm-bench-completion-transfers--trajectories ()
  "Return a fresh fixed synthetic token batch."
  (list '(65 66 65 66 65 66 240 159 152 128 10 33)
        '(120 121 120 121 120 230 151 165 230 156 172 10)
        '(49 50 49 50 49 50 51 52 53 54 55 10)))

(defun nl-llm-bench-completion-transfers--loss-starts ()
  "Return completion boundaries aligned with the synthetic trajectories."
  (vector 6 5 7))

(defun nl-llm-bench-completion-transfers--parameter-data (model)
  "Return detached MODEL parameter data vectors in training order."
  (mapcar (lambda (parameter)
            (copy-sequence (photon-tensor-data (pav-value parameter))))
          (nl-llm-agent--p5-params model)))

(defun nl-llm-bench-completion-transfers--finite-data-p (vectors)
  "Return non-nil when every value in VECTORS is finite and bounded."
  (cl-every
   (lambda (vector)
     (let ((valid t)
           (index 0))
       (while (and valid (< index (length vector)))
         (let ((value (aref vector index)))
           (setq valid
                 (and (numberp value)
                      (= value value)
                      (< (abs value) 1.0e+300))))
         (setq index (1+ index)))
       valid))
   vectors))

(defun nl-llm-bench-completion-transfers--float-count (vectors)
  "Return the total number of floats in VECTORS."
  (let ((count 0))
    (dolist (vector vectors count)
      (setq count (+ count (length vector))))))

(defun nl-llm-bench-completion-transfers--max-difference (left right)
  "Return the maximum absolute difference across LEFT and RIGHT vectors."
  (unless (= (length left) (length right))
    (error "parameter list lengths differ"))
  (let ((maximum 0.0))
    (cl-loop for a in left for b in right do
             (unless (= (length a) (length b))
               (error "parameter vector lengths differ"))
             (dotimes (index (length a))
               (setq maximum
                     (max maximum
                          (abs (- (aref a index) (aref b index)))))))
    maximum))

(defun nl-llm-bench-completion-transfers--run-mode
    (model sequence transfer-mode)
  "Train MODEL at SEQUENCE using TRANSFER-MODE and return measurements."
  (let ((original-write (symbol-function 'nelisp-gpu-server-write-resident))
        (original-compiled (symbol-function 'nelisp-gpu-server-run-compiled))
        (original-run2 (symbol-function 'nelisp-gpu-server-run2))
        (phase nil)
        (upload-counts nil)
        (compiled-output-counts nil)
        (sync-output-floats 0)
        (context nil)
        (build-seconds 0.0)
        (train-seconds 0.0)
        (sync-seconds 0.0)
        (trajectories
         (nl-llm-bench-completion-transfers--trajectories))
        (loss-starts
         (nl-llm-bench-completion-transfers--loss-starts)))
    (cl-letf (((symbol-function 'nelisp-gpu-server-write-resident)
               (lambda (handle vector)
                 (when (eq phase 'train)
                   (push (length vector) upload-counts))
                 (funcall original-write handle vector)))
              ((symbol-function 'nelisp-gpu-server-run-compiled)
               (lambda (handle output-sizes)
                 (let ((outputs (funcall original-compiled handle output-sizes)))
                   (when (eq phase 'train)
                     (push (nl-llm-bench-completion-transfers--float-count
                            outputs)
                           compiled-output-counts))
                   outputs)))
              ((symbol-function 'nelisp-gpu-server-run2)
               (lambda (name descriptors push-constants groups)
                 (let ((outputs
                        (funcall original-run2 name descriptors
                                 push-constants groups)))
                   (when (eq phase 'sync)
                     (setq sync-output-floats
                           (+ sync-output-floats
                              (nl-llm-bench-completion-transfers--float-count
                               outputs))))
                   outputs))))
      (unwind-protect
          (progn
            (let ((started (float-time)))
              (setq context
                    (nl-llm-agent-ondevice-from-model
                     model sequence 0.05
                     :optimizer nl-llm-bench-completion-transfers--optimizer
                     :loss-mode 'completion
                     :transfer-mode transfer-mode)
                    build-seconds (- (float-time) started)))
            (let ((graph-outputs (nlga-nout (plist-get context :b))))
              (unless (= graph-outputs
                         (if (eq transfer-mode 'compact) 0 1))
                (error "%S graph retained %d outputs"
                       transfer-mode graph-outputs)))
            (let ((started (float-time)))
              (setq phase 'train)
              (nl-llm-agent-ondevice-train
               context trajectories
               nl-llm-bench-completion-transfers--epochs
               :loss-starts loss-starts)
              (setq train-seconds (- (float-time) started)
                    phase nil))
            (let ((started (float-time)))
              (setq phase 'sync)
              (nl-llm-agent-ondevice-sync context)
              (setq sync-seconds (- (float-time) started)
                    phase nil)))
        (setq phase nil)
        (when context
          (nl-llm-agent-ondevice-free context))))
    (let* ((steps (* nl-llm-bench-completion-transfers--epochs
                     (length trajectories)))
           (uploads (nreverse upload-counts))
           (outputs (nreverse compiled-output-counts))
           (expected-upload-per-step
            (* 3 sequence
               (if (eq transfer-mode 'compact) 1 256)))
           (expected-output-per-step
            (if (eq transfer-mode 'compact) 0 (* sequence 256))))
      (unless (= (length uploads) (* steps 3))
        (error "%S recorded %d input transfers, expected %d"
               transfer-mode (length uploads) (* steps 3)))
      (unless (cl-every
               (lambda (count)
                 (= count (if (eq transfer-mode 'compact)
                              sequence
                            (* sequence 256))))
               uploads)
        (error "%S input transfer dimensions differ from the graph mode"
               transfer-mode))
      (unless (and (= (length outputs) steps)
                   (cl-every (lambda (count)
                               (= count expected-output-per-step))
                             outputs))
        (error "%S compiled output dimensions differ from the graph mode"
               transfer-mode))
      (list :status 'done
            :transfer-mode (or transfer-mode 'dense)
            :optimizer nl-llm-bench-completion-transfers--optimizer
            :sequence sequence
            :epochs nl-llm-bench-completion-transfers--epochs
            :steps steps
            :build-seconds build-seconds
            :train-seconds train-seconds
            :sync-seconds sync-seconds
            :uploaded-floats (apply #'+ uploads)
            :uploaded-floats-per-step expected-upload-per-step
            :compiled-output-floats (apply #'+ outputs)
            :compiled-output-floats-per-step expected-output-per-step
            :training-transport-bytes
            (* 4 (+ (apply #'+ uploads) (apply #'+ outputs)))
            :training-transport-byte-scope
            'float-payload-excluding-protocol-headers
            :sync-output-floats sync-output-floats
            :sync-output-bytes (* 4 sync-output-floats)))))

;;;###autoload
(defun nl-llm-bench-completion-transfers (&optional sequence)
  "Compare dense and compact completion transfers at fixed SEQUENCE.

SEQUENCE defaults to 64 and must be between 64 and 2048.  The returned report
contains measured graph-build, train, and sync times plus actual float counts
observed at the GPU transport functions.  The data are synthetic and the
timings characterize only this paired run; they are not a model-quality or
general performance claim."
  (setq sequence (or sequence 64))
  (unless (and (integerp sequence) (<= 64 sequence) (<= sequence 2048))
    (error "benchmark sequence must be an integer in [64, 2048]"))
  (if (not (nl-llm-gpu-enable))
      (list :status 'unavailable
            :reason "Vulkan GPU server unavailable"
            :sequence sequence)
    (unwind-protect
        (let* ((dense-model (nl-llm-bench-completion-transfers--model))
             (compact-model (nl-llm-bench-completion-transfers--model))
             (dense-initial
              (nl-llm-bench-completion-transfers--parameter-data dense-model))
             (compact-initial
              (nl-llm-bench-completion-transfers--parameter-data compact-model))
             dense compact dense-final compact-final
             dense-update compact-update difference)
        (unless (= 0.0
                   (nl-llm-bench-completion-transfers--max-difference
                    dense-initial compact-initial))
          (error "paired models do not have identical initial weights"))
        (setq dense
              (nl-llm-bench-completion-transfers--run-mode
               dense-model sequence nil)
              compact
              (nl-llm-bench-completion-transfers--run-mode
               compact-model sequence 'compact)
              dense-final
              (nl-llm-bench-completion-transfers--parameter-data dense-model)
              compact-final
              (nl-llm-bench-completion-transfers--parameter-data compact-model)
              dense-update
              (nl-llm-bench-completion-transfers--max-difference
               dense-initial dense-final)
              compact-update
              (nl-llm-bench-completion-transfers--max-difference
               compact-initial compact-final)
              difference
              (nl-llm-bench-completion-transfers--max-difference
               dense-final compact-final))
        (unless (and (nl-llm-bench-completion-transfers--finite-data-p
                      dense-final)
                     (nl-llm-bench-completion-transfers--finite-data-p
                      compact-final))
          (error "dense or compact training produced non-finite weights"))
        (unless (and (> dense-update 0.0) (> compact-update 0.0))
          (error "dense or compact training did not update model weights"))
        (unless (<= difference
                    nl-llm-bench-completion-transfers--tolerance)
          (error "dense/compact trained weights differ by %g (tolerance %g)"
                 difference nl-llm-bench-completion-transfers--tolerance))
        (list :status 'done
              :workload 'synthetic-completion-transfer
              :tokenizer "utf8-byte-v1"
              :optimizer nl-llm-bench-completion-transfers--optimizer
              :dense dense
              :compact compact
              :dense-max-update dense-update
              :compact-max-update compact-update
              :max-weight-difference difference
              :weight-tolerance
              nl-llm-bench-completion-transfers--tolerance))
      (nl-llm-gpu-disable))))

(unless nl-llm-bench-completion-transfers-suppress-autorun
  (prin1 (nl-llm-bench-completion-transfers 64))
  (terpri))

(provide 'bench-completion-transfers)
;;; bench-completion-transfers.el ends here
