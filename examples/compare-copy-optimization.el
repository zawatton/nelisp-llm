;;; compare-copy-optimization.el --- COPY optimization factorial harness -*- lexical-binding: t; -*-

;;; Commentary:
;; Opt-in comparison of learning-rate and exposure-order choices.  This file
;; keeps model construction, scoring, and bounded training in the frozen
;; architecture/diversity examples; it only supplies the four schedules.

;;; Code:

(require 'cl-lib)
(require 'seq)

(defconst nl-llm-compare-copy-optimization-format
  "nl-llm-copy-optimization-report-v1")
(defconst nl-llm-compare-copy-optimization-seed 104729)
(defconst nl-llm-compare-copy-optimization-batch-size 128)
(defconst nl-llm-compare-copy-optimization-batches 32)
(defconst nl-llm-compare-copy-optimization-steps 4096)
(defconst nl-llm-compare-copy-optimization-specs
  (vector
   (list :id 'sorted-high :learning-rate 0.003 :order 'sorted)
   (list :id 'sorted-low :learning-rate 0.0003 :order 'sorted)
   (list :id 'shuffled-high :learning-rate 0.003 :order 'shuffled)
   (list :id 'shuffled-low :learning-rate 0.0003 :order 'shuffled)))
(defvar nl-llm-compare-copy-optimization-auto-run nil
  "When non-nil, loading this example runs the GPU comparison.")

;; Loading the reused examples must never inherit an ambient experiment flag.
(defvar nl-llm-compare-copy-architecture-auto-run nil)
(defvar nl-llm-compare-copy-diversity-auto-run nil)
(defvar nl-llm-learn-copy-curriculum-auto-run nil)
(defvar nl-llm-learn-literal-copy-no-run nil)
(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (architecture (expand-file-name "compare-copy-architecture.el" here))
       (nl-llm-compare-copy-architecture-auto-run nil)
       (nl-llm-compare-copy-diversity-auto-run nil)
       (nl-llm-learn-copy-curriculum-auto-run nil)
       (nl-llm-learn-literal-copy-no-run t))
  (load architecture nil nil t))

(require 'nl-llm-agent-ondevice)

(defvar nl-llm-learn-copy-curriculum-tokenizer)
(defvar nl-llm-learn-copy-curriculum-vocab)
(defvar nl-llm-learn-copy-curriculum-sequence)
(defvar nl-llm-learn-copy-curriculum-learning-rate)
(defvar nl-llm-learn-copy-curriculum-initializer-seed)
(defvar nl-llm-compare-copy-diversity-data-format)
(defvar nl-llm-compare-copy-diversity-steps)

(declare-function nl-llm-compare-copy-diversity-dataset
                  "compare-copy-diversity" ())
(declare-function nl-llm-compare-copy-diversity-train-arm
                  "compare-copy-diversity" (model examples &optional epochs))
(declare-function nl-llm-compare-copy-diversity--literal-sha256
                  "compare-copy-diversity" (examples))
(declare-function nl-llm-compare-copy-diversity--sha256
                  "compare-copy-diversity" (value))
(declare-function nl-llm-compare-copy-architecture--geometry
                  "compare-copy-architecture" (dim ff blocks heads))
(declare-function nl-llm-compare-copy-architecture--model
                  "compare-copy-architecture" (geometry))
(declare-function nl-llm-compare-copy-architecture--parameter-count
                  "compare-copy-architecture" (model))
(declare-function nl-llm-compare-copy-architecture--scores
                  "compare-copy-architecture" (model dataset))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))
(declare-function nl-llm-agent-ondevice--shuffle-epoch
                  "nl-llm-agent-ondevice" (count state))
(declare-function nl-llm-gpu-available-p "nl-llm-gpu" ())

(defun nl-llm-compare-copy-optimization--copy-value (value)
  "Return a bounded detached copy of VALUE's lists, vectors, and strings."
  (cond ((stringp value) (copy-sequence value))
        ((vectorp value)
         (let ((copy (make-vector (length value) nil)))
           (let ((index 0))
             (while (< index (length value))
               (aset copy index
                     (nl-llm-compare-copy-optimization--copy-value
                      (aref value index)))
               (setq index (1+ index))))
           copy))
        ((consp value)
         (cons (nl-llm-compare-copy-optimization--copy-value (car value))
               (nl-llm-compare-copy-optimization--copy-value (cdr value))))
        (t value)))

(defun nl-llm-compare-copy-optimization--identity-permutation ()
  "Return the identity permutation for one training batch."
  (vconcat (number-sequence 0 (1- nl-llm-compare-copy-optimization-batch-size))))

(defun nl-llm-compare-copy-optimization--model ()
  "Create the shared candidate architecture for one optimization arm."
  (nl-llm-compare-copy-architecture--model
   (nl-llm-compare-copy-architecture--geometry 24 48 2 1)))

(defun nl-llm-compare-copy-optimization--without-examples (schedule)
  "Return SCHEDULE with its private, potentially large :examples omitted."
  (let ((result nil)
        (rest schedule))
    (while rest
      (let ((key (pop rest))
            (value (pop rest)))
        (unless (eq key :examples)
          (setq result (append result (list key value))))))
    result))

(defun nl-llm-compare-copy-optimization--schedule (examples order)
  "Return an ORDERED schedule for the 4,096-entry EXAMPLES vector.

Each consecutive 128-entry batch is permuted independently.  SHUFFLED uses
the existing local Fisher--Yates implementation with one continuous state;
SORTED preserves each batch's order and has no PRNG state."
  (unless (memq order '(sorted shuffled))
    (error "copy optimization order must be `sorted' or `shuffled'"))
  (unless (and (vectorp examples)
               (= (length examples) nl-llm-compare-copy-optimization-steps))
    (error "copy optimization schedule requires a 4096-entry vector"))
  (let ((scheduled (make-vector (length examples) nil))
        (permutations (make-vector nl-llm-compare-copy-optimization-batches nil))
        (state (and (eq order 'shuffled)
                    nl-llm-compare-copy-optimization-seed))
        (output-index 0))
    (dotimes (batch nl-llm-compare-copy-optimization-batches)
      (let* ((result (if (eq order 'shuffled)
                         (nl-llm-agent-ondevice--shuffle-epoch
                          nl-llm-compare-copy-optimization-batch-size state)
                       (cons nil
                             (nl-llm-compare-copy-optimization--identity-permutation))))
             (next-state (car result))
             (permutation (cdr result))
             (base (* batch nl-llm-compare-copy-optimization-batch-size)))
        (setq state next-state)
        (aset permutations batch (copy-sequence permutation))
        (dotimes (offset nl-llm-compare-copy-optimization-batch-size)
          (aset scheduled output-index
                (nl-llm-compare-copy-optimization--copy-value
                 (aref examples (+ base (aref permutation offset)))))
          (setq output-index (1+ output-index)))))
    (list :examples scheduled :order order
          :seed (and (eq order 'shuffled)
                     nl-llm-compare-copy-optimization-seed)
          :final-state (and (eq order 'shuffled) state)
          :permutations permutations
          :literal-sha256
          (nl-llm-compare-copy-diversity--literal-sha256 scheduled)
          :schedule-sha256
          (nl-llm-compare-copy-diversity--sha256
           (list :order order
                 :seed (and (eq order 'shuffled)
                            nl-llm-compare-copy-optimization-seed)
                 :final-state (and (eq order 'shuffled) state)
                 :permutations permutations)))))

(defun nl-llm-compare-copy-optimization--run-arm (spec dataset)
  "Run one SPEC against DATASET and return its detached data report."
  (message "COPY optimization arm %S starting" (plist-get spec :id))
  (let* ((geometry
          (nl-llm-compare-copy-architecture--geometry 24 48 2 1))
         (model (nl-llm-compare-copy-optimization--model))
         (before-hash (nl-llm-learn-literal-copy--model-hash model))
         (before (nl-llm-compare-copy-architecture--scores model dataset))
         (before-eval-hash (nl-llm-learn-literal-copy--model-hash model)))
    (unless (equal before-hash before-eval-hash)
      (error "COPY optimization evaluation mutated model weights"))
    (let* ((schedule
            (nl-llm-compare-copy-optimization--schedule
             (plist-get dataset :candidate) (plist-get spec :order)))
           (training
            (let ((nl-llm-learn-copy-curriculum-learning-rate
                   (plist-get spec :learning-rate)))
              (nl-llm-compare-copy-diversity-train-arm
               model (plist-get schedule :examples))))
           (after-hash (nl-llm-learn-literal-copy--model-hash model))
           (after (nl-llm-compare-copy-architecture--scores model dataset))
           (after-eval-hash (nl-llm-learn-literal-copy--model-hash model)))
      (unless (equal after-hash after-eval-hash)
        (error "COPY optimization evaluation mutated model weights"))
      (let ((report
             (list :id (plist-get spec :id) :geometry geometry
            :parameter-count
            (nl-llm-compare-copy-architecture--parameter-count model)
            :learning-rate (plist-get spec :learning-rate)
            :order (plist-get spec :order)
            :initializer 'xorshift32
            :initializer-seed
            nl-llm-learn-copy-curriculum-initializer-seed
            :model-before-sha256 before-hash
            :model-after-sha256 after-hash
            :before before :after after
            :training training
            :schedule
            (nl-llm-compare-copy-optimization--without-examples schedule)
            :weights-changed (not (equal before-hash after-hash)))))
        (message "COPY optimization arm %S finished" (plist-get spec :id))
        report))))

;;;###autoload
(cl-defun nl-llm-compare-copy-optimization-run (&optional arm-id)
  "Run one or all four opt-in COPY optimization arms.

With ARM-ID nil, run all specifications; otherwise ARM-ID must be one of the
four specification identifiers.  Loading this file never trains; explicitly
calling this function starts the configured GPU training arms."
  (when (and arm-id
             (not (seq-contains-p
                   (mapcar (lambda (spec) (plist-get spec :id))
                           (append nl-llm-compare-copy-optimization-specs nil))
                   arm-id)))
    (error "unknown copy optimization arm %S" arm-id))
  (when (nl-llm-gpu-available-p)
    (error "copy optimization run refuses an already-active GPU"))
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (specs (if arm-id
                    (list (seq-find
                           (lambda (spec) (eq arm-id (plist-get spec :id)))
                           (append nl-llm-compare-copy-optimization-specs nil)))
                  (append nl-llm-compare-copy-optimization-specs nil)))
         (reports (mapcar (lambda (spec)
                            (nl-llm-compare-copy-optimization--run-arm
                             spec dataset))
                          specs))
         (initial-hashes
          (delete-dups
           (mapcar (lambda (report)
                     (plist-get report :model-before-sha256)) reports)))
         (data (list :format nl-llm-compare-copy-diversity-data-format
                     :candidate-count (length (plist-get dataset :candidate))
                     :candidate-sha256 (plist-get dataset :candidate-sha256)
                     :literal-sha256 (plist-get dataset :literal-sha256)
                     :train-sha256 (plist-get dataset :train-sha256)
                     :dev-sha256 (plist-get dataset :dev-sha256)
                     :final-state (plist-get dataset :final-state))))
    (unless (= (length initial-hashes) 1)
      (error "COPY optimization arms do not share their initial hash"))
    (list :format nl-llm-compare-copy-optimization-format
          :data data :arms (vconcat reports))))

(when (and noninteractive nl-llm-compare-copy-optimization-auto-run)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t))
    (prin1 (nl-llm-compare-copy-optimization-run))
    (terpri)))

(provide 'compare-copy-optimization)
;;; compare-copy-optimization.el ends here
