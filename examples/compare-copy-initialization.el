;;; compare-copy-initialization.el --- paired literal-copy initialization probe -*- lexical-binding: t; -*-

;; This file deliberately keeps the frozen literal-copy example as the runner.
;; The baseline is unchanged; the challenger only replaces values in rank-2
;; parameters immediately after construction.

(require 'cl-lib)

(defvar nl-llm-compare-copy-initialization-no-run nil
  "When non-nil, load this file without running the paired experiment.")

(defconst nl-llm-compare-copy-initialization-seed #x1A2B3C4D
  "Default non-zero uint32 seed for the candidate initializer.")

(defconst nl-llm-compare-copy-initialization-source-sha256
  "ac0742137067ad93eccdafd9e4182ffa56744480f4e2fe9e75edc4770994d459"
  "SHA-256 of the frozen literal-copy runner used by this probe.")

(defconst nl-llm-compare-copy-initialization-epochs 32
  "The immutable epoch setting shared by both paired runs.")

(defconst nl-llm-compare-copy-initialization-parameter-order
  '(:wte
    :ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd
    :lnfg :wh :bh)
  "Canonical one-block P5 parameter order, including all 1D values.")

(defconst nl-llm-compare-copy-initialization-block-parameter-order
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd)
  "Canonical P5 parameter order for one block.")

;; Declare the source guard before the dynamically scoped load binding.  With
;; lexical binding, an undeclared variable would otherwise become lexical and
;; fail to suppress the source file's top-level runner.
(defvar nl-llm-learn-literal-copy-no-run nil)
(defvar nl-llm-learn-literal-copy-epochs)
(declare-function nl-llm-agent--p5-params "nl-llm-agent-improve" (model))
(declare-function pav-value "photon-autograd" (parameter))
(declare-function photon-tensor-shape "photon-tensor" (tensor))
(declare-function photon-tensor-data "photon-tensor" (tensor))

(defun nl-llm-compare-copy-initialization--file-sha256 (path)
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert-file-contents-literally path)
    (secure-hash 'sha256 (current-buffer))))

(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (source (expand-file-name "learn-literal-copy.el" here))
       (actual-sha256
        (nl-llm-compare-copy-initialization--file-sha256 source)))
  (unless (equal actual-sha256
                 nl-llm-compare-copy-initialization-source-sha256)
    (error "frozen literal-copy source SHA mismatch: %s" actual-sha256))
  ;; The source example has its own no-run guard.  Bind it only while loading
  ;; so loading this comparison file does not mutate the caller's flag.
  (let ((nl-llm-learn-literal-copy-no-run t))
    (load source nil nil t)))

(defvar nl-llm-compare-copy-initialization-run-function
  'nl-llm-learn-literal-copy-run
  "Function used for each paired base run.

Tests may bind this to a cheap runner; production use leaves it unchanged.")

(defun nl-llm-compare-copy-initialization--uint32 (value)
  (logand value #xffffffff))

(defun nl-llm-compare-copy-initialization--seed (seed)
  (unless (and (integerp seed)
               (> seed 0)
               (<= seed #xffffffff))
    (error "initializer seed must be a non-zero uint32: %S" seed))
  seed)

(defun nl-llm-compare-copy-initialization-xorshift32 (state)
  "Return the next uint32 from STATE using the specified xorshift stages.

The state is masked after each stage, making this independent of Elisp's
integer width and avoiding the global `random' generator entirely."
  (nl-llm-compare-copy-initialization--seed state)
  (setq state
        (nl-llm-compare-copy-initialization--uint32
         (logxor state (ash state 13))))
  (setq state
        (nl-llm-compare-copy-initialization--uint32
         (logxor state (ash state -17))))
  (nl-llm-compare-copy-initialization--uint32
   (logxor state (ash state 5))))

(defun nl-llm-compare-copy-initialization--matrix-p (tensor)
  (= (length (photon-tensor-shape tensor)) 2))

(defun nl-llm-compare-copy-initialization--parameter-signature (parameters)
  (mapcar
   (lambda (parameter)
     (let ((tensor (pav-value parameter)))
       (list (copy-sequence (photon-tensor-shape tensor))
             (copy-sequence (photon-tensor-data tensor)))))
   parameters))

(defun nl-llm-compare-copy-initialization--parameter-order (model)
  (let ((blocks (plist-get model :blocks)))
    (append '(:wte)
            (cl-loop repeat (length blocks)
                     append (copy-sequence
                             nl-llm-compare-copy-initialization-block-parameter-order))
            '(:lnfg :wh :bh))))

(defun nl-llm-compare-copy-initialization-reinitialize (model &optional seed)
  "Replace only MODEL's rank-2 parameter values with deterministic values.

Return initializer metadata.  Parameters are visited in the canonical order
of `nl-llm-agent--p5-params'; all rank-1 values remain untouched.  SEED must
be a non-zero uint32."
  (setq seed
        (nl-llm-compare-copy-initialization--seed
         (or seed nl-llm-compare-copy-initialization-seed)))
  (let* ((parameters (nl-llm-agent--p5-params model))
         (before (nl-llm-compare-copy-initialization--parameter-signature
                  parameters))
         (dim (plist-get model :dim))
         (state seed)
         (matrix-count 0)
         (matrix-elements 0)
         (one-dimensional-count 0))
    (unless (and (integerp dim) (> dim 0))
      (error "initializer model has invalid dimension: %S" dim))
    (let ((scale (/ 1.0 (sqrt (float dim)))))
    (dolist (parameter parameters)
      (let* ((tensor (pav-value parameter))
             (shape (photon-tensor-shape tensor))
             (data (photon-tensor-data tensor)))
        (if (nl-llm-compare-copy-initialization--matrix-p tensor)
            (progn
              (setq matrix-count (1+ matrix-count))
              (dotimes (index (length data))
                (setq state
                      (nl-llm-compare-copy-initialization-xorshift32 state))
                (let ((u (/ (float state) 4294967296.0)))
                  (aset data index (* scale (- (* 2.0 u) 1.0))))
                (setq matrix-elements (1+ matrix-elements))))
          (setq one-dimensional-count (1+ one-dimensional-count))
          ;; Keep this explicit assertion close to the mutation: an accidental
          ;; rank-0/rank-3 parameter must not silently become a preserved value.
          (unless (= (length shape) 1)
            (error "unexpected non-matrix parameter shape: %S" shape)))))
    (let* ((after (nl-llm-compare-copy-initialization--parameter-signature
                   parameters))
           (before-shapes (mapcar #'car before))
           (after-shapes (mapcar #'car after))
           (one-dimensional-preserved
            (cl-loop for old in before
                     for new in after
                     for tensor-old = (cadr old)
                     for tensor-new = (cadr new)
                     for shape = (car old)
                     always (or (= (length shape) 2)
                                (equal tensor-old tensor-new)))))
      (unless (equal before-shapes after-shapes)
        (error "initializer changed parameter dimensions"))
      (unless one-dimensional-preserved
        (error "initializer changed a one-dimensional parameter"))
      (list :algorithm 'xorshift32
            :seed seed
            :scale scale
            :parameter-order
            (nl-llm-compare-copy-initialization--parameter-order model)
            :parameter-count-before (length before)
            :parameter-count-after (length after)
            :dimensions-before before-shapes
            :dimensions-after after-shapes
            :matrix-count matrix-count
            :matrix-elements matrix-elements
            :one-dimensional-count one-dimensional-count
            :one-dimensional-preserved one-dimensional-preserved)))))

(defun nl-llm-compare-copy-initialization--run-base ()
  (let ((nl-llm-learn-literal-copy-epochs
         nl-llm-compare-copy-initialization-epochs))
    (funcall nl-llm-compare-copy-initialization-run-function)))

(defun nl-llm-compare-copy-initialization--run-candidate (seed)
  (let* ((constructor 'nl-llm-agent-improve-model)
         (original (symbol-function constructor))
         (constructor-count 0)
         (initializer nil)
         (result nil))
    (unwind-protect
        (progn
          ;; The override exists only for the candidate run.  Saving and
          ;; restoring the original function explicitly also covers errors in
          ;; construction, initialization, and the runner.
          (cl-letf (((symbol-function constructor)
                     (lambda (&rest args)
                       (setq constructor-count (1+ constructor-count))
                       (when (> constructor-count 1)
                         (error "candidate constructor called more than once"))
                       (let ((model (apply original args)))
                         (setq initializer
                               (nl-llm-compare-copy-initialization-reinitialize
                                model seed))
                         model))))
            (setq result (nl-llm-compare-copy-initialization--run-base)))
          (unless (= constructor-count 1)
            (error "candidate constructor call count was %d, expected 1"
                   constructor-count))
          (list :report result :constructor-count constructor-count
                :initializer initializer))
      (fset constructor original))))

(defun nl-llm-compare-copy-initialization--paired-settings (baseline candidate)
  (let ((baseline-settings (plist-get baseline :settings))
        (candidate-settings (plist-get candidate :settings))
        (baseline-dataset (plist-get baseline :dataset-sha256))
        (candidate-dataset (plist-get candidate :dataset-sha256)))
    (unless (equal baseline-settings candidate-settings)
      (error "paired runs changed settings"))
    (unless (equal baseline-dataset candidate-dataset)
      (error "paired runs changed dataset"))
    (unless (and (plist-get baseline :model-before-sha256)
                 (plist-get candidate :model-before-sha256)
                 (not (equal (plist-get baseline :model-before-sha256)
                             (plist-get candidate :model-before-sha256))))
      (error "paired model-before hashes did not differ"))
    (list :settings baseline-settings
          :dataset-sha256 baseline-dataset
          :settings-identical t
          :dataset-identical t)))

;;;###autoload
(defun nl-llm-compare-copy-initialization-run (&optional seed)
  "Run the bounded baseline/candidate literal-copy comparison.

The base runner supplies both the immutable data/settings contract and the
actual report.  The candidate constructor is intercepted once, initialized,
and restored before this function returns.  No score threshold is imposed."
  (setq seed
        (nl-llm-compare-copy-initialization--seed
         (or seed nl-llm-compare-copy-initialization-seed)))
  (let* ((baseline (nl-llm-compare-copy-initialization--run-base))
         (candidate-run
          (nl-llm-compare-copy-initialization--run-candidate seed))
         (candidate (plist-get candidate-run :report))
         (paired (nl-llm-compare-copy-initialization--paired-settings
                  baseline candidate)))
    (list :format "nl-llm-copy-initialization-compare-v1"
          :source-sha256 nl-llm-compare-copy-initialization-source-sha256
          :initializer (plist-get candidate-run :initializer)
          :baseline baseline
          :candidate candidate
          :constructor-count (plist-get candidate-run :constructor-count)
          :settings (plist-get paired :settings)
          :dataset-sha256 (plist-get paired :dataset-sha256)
          :settings-identical (plist-get paired :settings-identical)
          :dataset-identical (plist-get paired :dataset-identical))))

(provide 'compare-copy-initialization)

(unless nl-llm-compare-copy-initialization-no-run
  (princ (format "%S\n" (nl-llm-compare-copy-initialization-run))))

;;; compare-copy-initialization.el ends here
