;;; compare-copy-architecture.el --- COPY architecture comparison -*- lexical-binding: t; -*-

;;; Commentary:
;; Opt-in architecture comparison for the bounded COPY experiment.  This file
;; reuses the frozen diversity dataset and bounded training arm; it changes
;; only the model geometry for the challenger.

;;; Code:

(require 'cl-lib)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)

(defconst nl-llm-compare-copy-architecture-format
  "nl-llm-copy-architecture-report-v1")
(defconst nl-llm-compare-copy-architecture-seed 439041101)
(defvar nl-llm-compare-copy-architecture-auto-run nil
  "When non-nil, loading this example runs the GPU comparison.")

;; The reused example is deliberately loaded with its opt-in runner disabled.
(defvar nl-llm-compare-copy-diversity-auto-run nil)
(defvar nl-llm-learn-copy-curriculum-auto-run nil)
(defvar nl-llm-learn-literal-copy-no-run nil)
(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (diversity (expand-file-name "compare-copy-diversity.el" here))
       (nl-llm-compare-copy-diversity-auto-run nil)
       (nl-llm-learn-copy-curriculum-auto-run nil)
       (nl-llm-learn-literal-copy-no-run t))
  (load diversity nil nil t))

;; These variables are defined by the frozen curriculum/diversity sources.
(defvar nl-llm-learn-copy-curriculum-tokenizer)
(defvar nl-llm-learn-copy-curriculum-vocab)
(defvar nl-llm-learn-copy-curriculum-sequence)
(defvar nl-llm-learn-copy-curriculum-learning-rate)
(defvar nl-llm-learn-copy-curriculum-initializer-seed)
(defvar nl-llm-compare-copy-diversity-data-format)
(defvar nl-llm-compare-copy-diversity-epochs)
(defvar nl-llm-compare-copy-diversity-steps)
(defvar nl-llm-compare-copy-diversity-batch-size)

(require 'nl-llm-agent-initialization)
(require 'nl-llm-gpu)

(declare-function nl-llm-compare-copy-diversity-dataset
                  "compare-copy-diversity" ())
(declare-function nl-llm-compare-copy-diversity-train-arm
                  "compare-copy-diversity" (model examples &optional epochs))
(declare-function nl-llm-learn-copy-curriculum--scores
                  "learn-copy-curriculum" (model dataset))
(declare-function nl-llm-learn-copy-curriculum--examples
                  "learn-copy-curriculum" (dataset key))
(declare-function nl-llm-copy-teacher-forcing-score
                  "copy-teacher-forcing" (model examples))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))
(declare-function nl-llm-agent-initialization-create
                  "nl-llm-agent-initialization" (&rest keys))
(declare-function nl-llm-agent--p5-params
                  "nl-llm-agent-improve" (model))
(declare-function nl-llm-gpu-available-p "nl-llm-gpu" ())

(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (teacher (expand-file-name "copy-teacher-forcing.el" here))
       (nl-llm-learn-copy-curriculum-auto-run nil)
       (nl-llm-learn-literal-copy-no-run t))
  (load teacher nil nil t))

(defun nl-llm-compare-copy-architecture--geometry
    (dim ff blocks heads)
  "Return detached model geometry DIM FF BLOCKS HEADS."
  (list :dim dim :ff ff :vocab nl-llm-learn-copy-curriculum-vocab
        :blocks blocks :heads heads
        :tokenizer (copy-sequence nl-llm-learn-copy-curriculum-tokenizer)))

(defun nl-llm-compare-copy-architecture--parameter-count (model)
  "Return the number of scalar values in MODEL's canonical P5 parameters."
  (apply #'+
         (mapcar (lambda (parameter)
                   (photon-tensor-size (pav-value parameter)))
                 (nl-llm-agent--p5-params model))))

(defun nl-llm-compare-copy-architecture--model (geometry)
  "Create a fresh production model matching GEOMETRY."
  (nl-llm-agent-initialization-create
   :initializer 'xorshift32 :seed nl-llm-compare-copy-architecture-seed
   :dim (plist-get geometry :dim) :ff (plist-get geometry :ff)
   :vocab (plist-get geometry :vocab)
   :nblocks (plist-get geometry :blocks) :heads (plist-get geometry :heads)
   :tokenizer (plist-get geometry :tokenizer)))

(defun nl-llm-compare-copy-architecture--arm-report
    (model geometry before before-hash training after after-hash)
  "Return data-only report for MODEL and its evaluation/training metadata."
  (list :geometry geometry
        :parameter-count
        (nl-llm-compare-copy-architecture--parameter-count model)
        :initializer 'xorshift32
        :initializer-seed nl-llm-compare-copy-architecture-seed
        :model-before-sha256 before-hash
        :model-after-sha256 after-hash
        :training training
        :before before
        :after after
        :weights-changed (not (equal before-hash after-hash))))

(defun nl-llm-compare-copy-architecture--scores (model dataset)
  "Return greedy and teacher-forced scores for MODEL on DATASET."
  (list :greedy (nl-llm-learn-copy-curriculum--scores model dataset)
        :teacher-forcing
        (list :train
              (nl-llm-copy-teacher-forcing-score
               model (plist-get dataset :train))
              :dev
              (nl-llm-copy-teacher-forcing-score
               model (plist-get dataset :dev)))))

;;;###autoload
(cl-defun nl-llm-compare-copy-architecture-run (&optional (mode 'paired))
  "Run the opt-in COPY architecture comparison in MODE.

MODE is `paired' (both geometries) or `candidate' (challenger only).  Both
arms use the exact 4,096-exposure diversity sequence, while only the model
geometry differs.  The returned report contains no model objects or files;
load this example and call this function explicitly to run training."
  (unless (memq mode '(paired candidate))
    (error "copy architecture mode must be `paired' or `candidate'"))
  (when (nl-llm-gpu-available-p)
    (error "copy architecture run refuses an already-active GPU"))
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (baseline-geometry
          (nl-llm-compare-copy-architecture--geometry 32 64 1 1))
         (candidate-geometry
          (nl-llm-compare-copy-architecture--geometry 24 48 2 1))
         (baseline (and (eq mode 'paired)
                        (nl-llm-compare-copy-architecture--model
                         baseline-geometry)))
         (candidate
          (nl-llm-compare-copy-architecture--model candidate-geometry))
         (baseline-before-hash
          (and baseline (nl-llm-learn-literal-copy--model-hash baseline)))
         (candidate-before-hash
          (nl-llm-learn-literal-copy--model-hash candidate))
         (before-baseline
          (and baseline
               (nl-llm-compare-copy-architecture--scores baseline dataset)))
         (baseline-before-eval-hash
          (and baseline (nl-llm-learn-literal-copy--model-hash baseline)))
         (before-candidate
          (nl-llm-compare-copy-architecture--scores candidate dataset))
         (candidate-before-eval-hash
          (nl-llm-learn-literal-copy--model-hash candidate)))
    (unless (and (equal baseline-before-hash baseline-before-eval-hash)
                 (equal candidate-before-hash candidate-before-eval-hash))
      (error "COPY architecture evaluation mutated model weights"))
    (let* ((baseline-training
            (and baseline
                 (nl-llm-compare-copy-diversity-train-arm
                  baseline (plist-get dataset :candidate))))
           (candidate-training
            (nl-llm-compare-copy-diversity-train-arm
             candidate (plist-get dataset :candidate)))
           (baseline-after-hash
            (and baseline (nl-llm-learn-literal-copy--model-hash baseline)))
           (candidate-after-hash
            (nl-llm-learn-literal-copy--model-hash candidate))
           (after-baseline
            (and baseline
                 (nl-llm-compare-copy-architecture--scores baseline dataset)))
           (after-candidate
            (nl-llm-compare-copy-architecture--scores candidate dataset))
           (baseline-after-eval-hash
            (and baseline (nl-llm-learn-literal-copy--model-hash baseline)))
           (candidate-after-eval-hash
            (nl-llm-learn-literal-copy--model-hash candidate)))
      (unless (and (equal baseline-after-hash baseline-after-eval-hash)
                   (equal candidate-after-hash candidate-after-eval-hash))
        (error "COPY architecture evaluation mutated model weights"))
      (list :format nl-llm-compare-copy-architecture-format
            :data
            (list :format nl-llm-compare-copy-diversity-data-format
                  :candidate-count (length (plist-get dataset :candidate))
                  :candidate-sha256 (plist-get dataset :candidate-sha256)
                  :literal-sha256 (plist-get dataset :literal-sha256)
                  :train-sha256 (plist-get dataset :train-sha256)
                  :dev-sha256 (plist-get dataset :dev-sha256)
                  :final-state (plist-get dataset :final-state))
            :settings
            (list :mode mode :initializer 'xorshift32
                  :initializer-seed nl-llm-compare-copy-architecture-seed
                  :tokenizer (copy-sequence nl-llm-learn-copy-curriculum-tokenizer)
                  :vocab nl-llm-learn-copy-curriculum-vocab
                  :sequence nl-llm-learn-copy-curriculum-sequence
                  :loss-mode 'completion
                  :learning-rate nl-llm-learn-copy-curriculum-learning-rate
                  :optimizer 'adam :transfer-mode 'compact
                  :epochs nl-llm-compare-copy-diversity-epochs
                  :steps nl-llm-compare-copy-diversity-steps
                  :batch-size nl-llm-compare-copy-diversity-batch-size)
            :baseline
            (and baseline
                 (nl-llm-compare-copy-architecture--arm-report
                  baseline baseline-geometry before-baseline
                  baseline-before-hash baseline-training after-baseline
                  baseline-after-hash))
            :candidate
            (nl-llm-compare-copy-architecture--arm-report
             candidate candidate-geometry before-candidate
             candidate-before-hash candidate-training after-candidate
             candidate-after-hash)))))

(when (and noninteractive nl-llm-compare-copy-architecture-auto-run)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t))
    (prin1 (nl-llm-compare-copy-architecture-run))
    (terpri)))

(provide 'compare-copy-architecture)
;;; compare-copy-architecture.el ends here
