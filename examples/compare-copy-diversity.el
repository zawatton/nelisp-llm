;;; compare-copy-diversity.el --- COPY data-diversity ablation harness -*- lexical-binding: t; -*-

;;; Commentary:
;; This is an opt-in experiment harness.  It keeps the frozen COPY
;; curriculum's development set and model/evaluation code, while comparing a
;; repeated 128-example arm with a fresh 128-example arm at each epoch.

;;; Code:

(require 'cl-lib)

(defconst nl-llm-compare-copy-diversity-format
  "nl-llm-copy-diversity-report-v1")
(defconst nl-llm-compare-copy-diversity-data-format
  "nl-copy-diversity-data-v1")
(defconst nl-llm-compare-copy-diversity-seed 2654435769)
(defconst nl-llm-compare-copy-diversity-epochs 32)
(defconst nl-llm-compare-copy-diversity-examples-per-length 16)
(defconst nl-llm-compare-copy-diversity-batch-size 128)
(defconst nl-llm-compare-copy-diversity-steps 4096)
(defconst nl-llm-compare-copy-diversity-uint32-mask #xffffffff)
(defconst nl-llm-compare-copy-diversity-source-sha256
  "ac0742137067ad93eccdafd9e4182ffa56744480f4e2fe9e75edc4770994d459")
(defvar nl-llm-learn-copy-curriculum-auto-run nil)
(defvar nl-llm-learn-copy-curriculum-initializer-seed)
(defvar nl-llm-learn-copy-curriculum-alphabet)
(defvar nl-llm-learn-copy-curriculum-lengths)
(defvar nl-llm-learn-copy-curriculum-tokenizer)
(defvar nl-llm-learn-copy-curriculum-dim)
(defvar nl-llm-learn-copy-curriculum-ff)
(defvar nl-llm-learn-copy-curriculum-vocab)
(defvar nl-llm-learn-copy-curriculum-blocks)
(defvar nl-llm-learn-copy-curriculum-heads)
(defvar nl-llm-learn-copy-curriculum-sequence)
(defvar nl-llm-learn-copy-curriculum-learning-rate)
(defvar nl-llm-compare-copy-diversity-auto-run nil
  "When non-nil, loading this example runs the GPU experiment.")

(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (curriculum (expand-file-name "learn-copy-curriculum.el" here))
       (nl-llm-learn-copy-curriculum-auto-run nil))
  (load curriculum nil nil t))

(require 'nl-llm-agent-initialization)
(require 'nl-llm-agent-supervised)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-gpu)
(require 'nl-llm-inference-runtime)

(declare-function nl-llm-agent-initialization-create
                  "nl-llm-agent-initialization" (&rest keys))
(declare-function nl-llm-agent-supervised-encode
                  "nl-llm-agent-supervised" (examples &optional tokenizer))
(declare-function nl-llm-agent-ondevice-from-model
                  "nl-llm-agent-ondevice" (model seq lr &rest keys))
(declare-function nl-llm-agent-ondevice-train
                  "nl-llm-agent-ondevice" (ctx trajs epochs &rest keys))
(declare-function nl-llm-agent-ondevice-sync
                  "nl-llm-agent-ondevice" (ctx))
(declare-function nl-llm-agent-ondevice-free
                  "nl-llm-agent-ondevice" (ctx))
(declare-function nl-llm-gpu-available-p "nl-llm-gpu" ())
(declare-function nl-llm-gpu-enable "nl-llm-gpu" ())
(declare-function nl-llm-gpu-disable "nl-llm-gpu" ())
(declare-function nl-llm-learn-copy-curriculum-dataset
                  "learn-copy-curriculum" ())
(declare-function nl-llm-learn-copy-curriculum--scores
                  "learn-copy-curriculum" (model dataset))
(declare-function nl-llm-learn-copy-curriculum--data-digest
                  "learn-copy-curriculum" (dataset key))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))

(defun nl-llm-compare-copy-diversity--xorshift32 (state)
  "Return the next local uint32 value from STATE."
  (let ((mask nl-llm-compare-copy-diversity-uint32-mask))
    (setq state (logand mask (logxor state (ash state 13))))
    (setq state (logand mask (logxor state (ash state -17))))
    (logand mask (logxor state (ash state 5)))))

(defun nl-llm-compare-copy-diversity--sha256 (value)
  "Hash semantic VALUE with stable, full-precision printing."
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t) (float-output-format nil))
    (secure-hash 'sha256 (prin1-to-string value))))

(defun nl-llm-compare-copy-diversity--literal-sha256 (examples)
  "Hash EXAMPLES' literals as UTF-8 lines, without a trailing newline."
  (secure-hash
   'sha256
   (mapconcat (lambda (example) (plist-get example :literal))
              (append examples nil) "\n")))

(defun nl-llm-compare-copy-diversity--example (index epoch length literal)
  (list :index index :epoch epoch :length length :literal literal
        :prompt (format "COPY: %s\nOUTPUT:\n" literal)
        :completion (concat literal "\n")))

(defun nl-llm-compare-copy-diversity--examples (examples)
  "Convert EXAMPLES to detached supervised example records."
  (vconcat
   (mapcar (lambda (example)
             (list :prompt (plist-get example :prompt)
                   :completion (plist-get example :completion)))
           (append examples nil))))

(defun nl-llm-compare-copy-diversity-dataset ()
  "Return the original splits and deterministic fresh-exposure dataset.

Each epoch contains sixteen examples of each configured length.  A literal
equal to any frozen development literal is rejected and regenerated as a
whole; duplicate literals elsewhere are intentionally allowed."
  (let* ((original (nl-llm-learn-copy-curriculum-dataset))
         (alphabet nl-llm-learn-copy-curriculum-alphabet)
         (lengths nl-llm-learn-copy-curriculum-lengths)
         (dev-literals (make-hash-table :test #'equal))
         (state nl-llm-compare-copy-diversity-seed)
         (candidate nil) (index 0) (rejected-dev 0)
         (unique-tables
          (mapcar (lambda (length)
                    (cons length (make-hash-table :test #'equal)))
                  lengths)))
    (dolist (example (append (plist-get original :dev) nil))
      (puthash (plist-get example :literal) t dev-literals))
    ;; The epoch is the outer loop: every 128-example batch has the same
    ;; length schedule, which makes the arm's exposure order explicit.
    (dotimes (epoch nl-llm-compare-copy-diversity-epochs)
      (dolist (length lengths)
        (dotimes (_ nl-llm-compare-copy-diversity-examples-per-length)
          (let ((literal nil) (attempts 0))
            (while (or (null literal) (gethash literal dev-literals))
              (when (and literal (gethash literal dev-literals))
                (setq rejected-dev (1+ rejected-dev)))
              (setq attempts (1+ attempts))
              (when (> attempts 10000)
                (error "copy diversity rejection exceeded bound at length %d"
                       length))
              (let ((chars nil))
                (dotimes (_ length)
                  (setq state
                        (nl-llm-compare-copy-diversity--xorshift32 state))
                  (push (aref alphabet (mod state (length alphabet))) chars))
                (setq literal (apply #'string (nreverse chars)))))
            (puthash literal t (cdr (assq length unique-tables)))
            (push (nl-llm-compare-copy-diversity--example
                   index epoch length literal)
                  candidate)
            (setq index (1+ index))))))
    (setq candidate (vconcat (nreverse candidate)))
    (list :format nl-llm-compare-copy-diversity-data-format
          :seed nl-llm-compare-copy-diversity-seed
          :alphabet (copy-sequence alphabet)
          :lengths (copy-sequence lengths)
          :train (plist-get original :train)
          :dev (plist-get original :dev)
          :candidate candidate
          :rejected-dev-count rejected-dev
          :unique-counts
          (vconcat
           (mapcar (lambda (length)
                     (cons length
                           (hash-table-count (cdr (assq length unique-tables)))))
                   lengths))
          :exposures-by-length
          (vconcat
           (mapcar (lambda (length)
                     (cons length
                           (* nl-llm-compare-copy-diversity-epochs
                              nl-llm-compare-copy-diversity-examples-per-length)))
                   lengths))
          :candidate-sha256
          (nl-llm-compare-copy-diversity--sha256
           (list :format nl-llm-compare-copy-diversity-data-format
                 :seed nl-llm-compare-copy-diversity-seed
                 :alphabet alphabet :lengths lengths
                 :examples (append candidate nil)))
          :literal-sha256
          (nl-llm-compare-copy-diversity--literal-sha256 candidate)
          :train-sha256
          (nl-llm-learn-copy-curriculum--data-digest original :train)
          :dev-sha256
          (nl-llm-learn-copy-curriculum--data-digest original :dev)
          :final-state state)))

(defun nl-llm-compare-copy-diversity--encoded (examples)
  "Encode EXAMPLES for one bounded on-device batch."
  (nl-llm-agent-supervised-encode
   (nl-llm-compare-copy-diversity--examples examples)
   nl-llm-learn-copy-curriculum-tokenizer))

;;;###autoload
(defun nl-llm-compare-copy-diversity-train-arm (model examples &optional epochs)
  "Train MODEL for EPOCHS fixed 128-example calls and return metadata.

When EXAMPLES has 128 entries, that same batch is repeated each epoch.  When
it has 4096 entries, consecutive 128-entry batches provide fresh data for the
32 epochs.  The model and one Adam context are retained across all calls."
  (let* ((epochs (or epochs nl-llm-compare-copy-diversity-epochs))
         (count (length examples)))
    (unless (and (integerp epochs) (= epochs 32)
                 (or (= count nl-llm-compare-copy-diversity-batch-size)
                     (= count nl-llm-compare-copy-diversity-steps)))
      (error "copy diversity arm requires 128 or 4096 examples and 32 epochs"))
    (when (nl-llm-gpu-available-p)
      (error "copy diversity training refuses an already-active GPU"))
    (let ((gpu-enabled nil) (context nil) (batch-digests nil)
          (encoded-digests nil))
      (unwind-protect
          (progn
            (setq gpu-enabled (nl-llm-gpu-enable))
            (unless gpu-enabled
              (error "copy diversity training requires a Vulkan GPU"))
            (setq context
                  (nl-llm-agent-ondevice-from-model
                   model nl-llm-learn-copy-curriculum-sequence
                   nl-llm-learn-copy-curriculum-learning-rate
                   :optimizer 'adam :loss-mode 'completion
                   :transfer-mode 'compact))
            (dotimes (epoch epochs)
              (let* ((start (if (= count nl-llm-compare-copy-diversity-steps)
                                (* epoch nl-llm-compare-copy-diversity-batch-size)
                              0))
                     (batch (cl-subseq examples start
                                       (+ start nl-llm-compare-copy-diversity-batch-size)))
                     (encoded (nl-llm-compare-copy-diversity--encoded batch))
                     (steps
                      (nl-llm-agent-ondevice-train
                       context (plist-get encoded :trajectories) 1
                       :loss-starts (plist-get encoded :loss-starts)
                       :after-step
                       (lambda (_ctx completed _total)
                         (when (= completed nl-llm-compare-copy-diversity-batch-size)
                           (message "COPY diversity training: epoch %d/%d"
                                    (1+ epoch) epochs))))))
                (unless (= steps nl-llm-compare-copy-diversity-batch-size)
                  (error "copy diversity batch returned %S steps" steps))
                (unless (= (plist-get context :step)
                           (* (1+ epoch)
                              nl-llm-compare-copy-diversity-batch-size))
                  (error "copy diversity context step is %S after batch %d"
                         (plist-get context :step) (1+ epoch)))
                (push (copy-sequence (plist-get encoded :dataset-sha256))
                      encoded-digests)
                (push (nl-llm-compare-copy-diversity--sha256
                       (append batch nil)) batch-digests)))
            (nl-llm-agent-ondevice-sync context)
            (list :epochs epochs :batch-count epochs
                  :batch-size nl-llm-compare-copy-diversity-batch-size
                  :exposures (* epochs nl-llm-compare-copy-diversity-batch-size)
                  :steps (* epochs nl-llm-compare-copy-diversity-batch-size)
                  :batch-sha256 (vconcat (nreverse batch-digests))
                  :dataset-sha256 (vconcat (nreverse encoded-digests))))
        (when context
          (unwind-protect
              (nl-llm-agent-ondevice-free context)
            (when gpu-enabled (nl-llm-gpu-disable))))
        (when (and gpu-enabled (not context))
          (nl-llm-gpu-disable))))))

(defun nl-llm-compare-copy-diversity--model ()
  "Create one fresh production-initialized COPY model."
  (nl-llm-agent-initialization-create
   :initializer 'xorshift32 :seed nl-llm-learn-copy-curriculum-initializer-seed
   :dim nl-llm-learn-copy-curriculum-dim :ff nl-llm-learn-copy-curriculum-ff
   :vocab nl-llm-learn-copy-curriculum-vocab
   :nblocks nl-llm-learn-copy-curriculum-blocks
   :heads nl-llm-learn-copy-curriculum-heads
   :tokenizer nl-llm-learn-copy-curriculum-tokenizer))

;;;###autoload
(cl-defun nl-llm-compare-copy-diversity-run (&optional (mode 'paired))
  "Run the opt-in COPY data-diversity comparison in MODE.
MODE is `paired' (baseline and challenger) or `candidate' (challenger only).
The report contains hashes, bounded scores, and data digests, never model
objects or saved files."
  (unless (memq mode '(paired candidate))
    (error "copy diversity mode must be `paired' or `candidate'"))
  (when (nl-llm-gpu-available-p)
    (error "copy diversity run refuses an already-active GPU"))
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (baseline (and (eq mode 'paired)
                        (nl-llm-compare-copy-diversity--model)))
         (candidate (nl-llm-compare-copy-diversity--model))
         (baseline-before (and baseline
                               (nl-llm-learn-literal-copy--model-hash baseline)))
         (candidate-before (nl-llm-learn-literal-copy--model-hash candidate)))
    (unless (and (or (null baseline)
                     (equal baseline-before candidate-before)))
      (error "copy diversity arms do not share their initial seed"))
    (let* ((before-baseline (and baseline
                                (nl-llm-learn-copy-curriculum--scores
                                 baseline dataset)))
           (baseline-before-eval-hash
            (and baseline
                 (nl-llm-learn-literal-copy--model-hash baseline)))
           (before-candidate
            (nl-llm-learn-copy-curriculum--scores candidate dataset))
           (candidate-before-eval-hash
            (nl-llm-learn-literal-copy--model-hash candidate)))
      (unless (and (equal baseline-before baseline-before-eval-hash)
                   (equal candidate-before candidate-before-eval-hash))
        (error "COPY evaluation mutated model weights"))
      (let* ((baseline-training
              (and baseline
                   (nl-llm-compare-copy-diversity-train-arm
                    baseline (plist-get dataset :train))))
             (candidate-training
              (nl-llm-compare-copy-diversity-train-arm
               candidate (plist-get dataset :candidate)))
             (baseline-after-hash
              (and baseline (nl-llm-learn-literal-copy--model-hash baseline)))
             (candidate-after-hash
              (nl-llm-learn-literal-copy--model-hash candidate))
             (after-baseline
              (and baseline (nl-llm-learn-copy-curriculum--scores
                             baseline dataset)))
             (after-candidate
              (nl-llm-learn-copy-curriculum--scores candidate dataset))
             (baseline-after-eval-hash
              (and baseline
                   (nl-llm-learn-literal-copy--model-hash baseline)))
             (candidate-after-eval-hash
              (nl-llm-learn-literal-copy--model-hash candidate)))
        (unless (and (equal baseline-after-hash baseline-after-eval-hash)
                     (equal candidate-after-hash candidate-after-eval-hash))
          (error "COPY evaluation mutated model weights"))
        (list :format nl-llm-compare-copy-diversity-format
              :source-sha256 nl-llm-compare-copy-diversity-source-sha256
              :settings (list :mode mode :seed nl-llm-compare-copy-diversity-seed
                              :epochs nl-llm-compare-copy-diversity-epochs
                              :steps nl-llm-compare-copy-diversity-steps
                              :batch-size nl-llm-compare-copy-diversity-batch-size
                              :initializer-seed
                              nl-llm-learn-copy-curriculum-initializer-seed
                              :dim nl-llm-learn-copy-curriculum-dim
                              :ff nl-llm-learn-copy-curriculum-ff
                              :vocab nl-llm-learn-copy-curriculum-vocab
                              :blocks nl-llm-learn-copy-curriculum-blocks
                              :heads nl-llm-learn-copy-curriculum-heads
                              :tokenizer nl-llm-learn-copy-curriculum-tokenizer
                              :sequence nl-llm-learn-copy-curriculum-sequence
                              :learning-rate nl-llm-learn-copy-curriculum-learning-rate
                              :optimizer 'adam :transfer-mode 'compact)
              :data (list :format nl-llm-compare-copy-diversity-data-format
                          :train-count (length (plist-get dataset :train))
                          :dev-count (length (plist-get dataset :dev))
                          :candidate-count (length (plist-get dataset :candidate))
                          :candidate-sha256 (plist-get dataset :candidate-sha256)
                          :literal-sha256 (plist-get dataset :literal-sha256)
                          :train-sha256 (plist-get dataset :train-sha256)
                          :dev-sha256 (plist-get dataset :dev-sha256)
                          :rejected-dev-count (plist-get dataset :rejected-dev-count)
                          :unique-counts (plist-get dataset :unique-counts)
                          :exposures-by-length
                          (plist-get dataset :exposures-by-length)
                          :final-state (plist-get dataset :final-state))
              :baseline (and baseline
                             (list :model-before-sha256 baseline-before
                                   :model-after-sha256 baseline-after-hash
                                   :training baseline-training
                                   :before before-baseline
                                   :after after-baseline
                                   :weights-changed
                                   (not (equal baseline-before baseline-after-hash))))
              :candidate (list :model-before-sha256 candidate-before
                               :model-after-sha256 candidate-after-hash
                               :training candidate-training
                               :before before-candidate
                               :after after-candidate
                               :weights-changed
                               (not (equal candidate-before candidate-after-hash))))))))

(when (and noninteractive nl-llm-compare-copy-diversity-auto-run)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t))
    (prin1 (nl-llm-compare-copy-diversity-run))
    (terpri)))

(provide 'compare-copy-diversity)
;;; compare-copy-diversity.el ends here
