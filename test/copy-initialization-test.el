;;; copy-initialization-test.el --- tests for the paired initialization probe -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)

(defvar nl-llm-compare-copy-initialization-no-run nil)
(declare-function nl-llm-compare-copy-initialization-run
                  "compare-copy-initialization" (&optional seed))
(declare-function nl-llm-compare-copy-initialization--run-candidate
                  "compare-copy-initialization" (seed))
(declare-function nl-llm-compare-copy-initialization-reinitialize
                  "compare-copy-initialization" (model &optional seed))
(declare-function nl-llm-compare-copy-initialization-xorshift32
                  "compare-copy-initialization" (state))
(declare-function nl-llm-agent--p5-params "nl-llm-agent-improve" (model))
(declare-function nl-llm-agent-improve-model
                  "nl-llm-agent-improve"
                  (&optional dim ff vocab nblocks heads tokenizer))
(declare-function pav-value "photon-autograd" (parameter))
(declare-function photon-tensor-shape "photon-tensor" (tensor))
(declare-function photon-tensor-data "photon-tensor" (tensor))

(defconst nl-llm-copy-initialization-test--here
  (file-name-directory (or load-file-name buffer-file-name)))

(let ((here nl-llm-copy-initialization-test--here))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                          (file-name-directory (or load-file-name buffer-file-name
                                                   default-directory))) nil t)
  (setq nl-llm-compare-copy-initialization-no-run t)
  (load (expand-file-name "../examples/compare-copy-initialization.el" here)
        nil nil t))

(defun nl-llm-copy-initialization-test--model ()
  (nl-llm-agent-improve-model 2 2 256 1 1 "utf8-byte-v1"))

(defun nl-llm-copy-initialization-test--matrix-values (model)
  (let ((values nil))
    (dolist (parameter (nl-llm-agent--p5-params model))
      (let ((tensor (pav-value parameter)))
        (when (= (length (photon-tensor-shape tensor)) 2)
          (setq values
                (append values (append (photon-tensor-data tensor) nil))))))
    values))

(ert-deftest nl-llm-copy-initialization-xorshift32-known-vector ()
  (let ((state #x1A2B3C4D)
        (expected '(3388403996 3984204854 2523572680 2838152257 3320909456))
        (actual nil))
    (dotimes (_ 5)
      (setq state
            (nl-llm-compare-copy-initialization-xorshift32 state))
      (push state actual))
    (should (equal (nreverse actual) expected))))

(ert-deftest nl-llm-copy-initialization-rejects-invalid-seeds ()
  (dolist (seed '(0 -1 #x100000000 1.0))
    (should-error
     (nl-llm-compare-copy-initialization-xorshift32 seed))))

(ert-deftest nl-llm-copy-initialization-is-repeatable-and-seed-sensitive ()
  (let* ((left (nl-llm-copy-initialization-test--model))
         (right (nl-llm-copy-initialization-test--model))
         (other (nl-llm-copy-initialization-test--model))
         (left-meta
          (nl-llm-compare-copy-initialization-reinitialize left #x1A2B3C4D)))
    (nl-llm-compare-copy-initialization-reinitialize right #x1A2B3C4D)
    (nl-llm-compare-copy-initialization-reinitialize other #x1A2B3C4E)
    (should (equal (nl-llm-copy-initialization-test--matrix-values left)
                   (nl-llm-copy-initialization-test--matrix-values right)))
    (should-not (equal (nl-llm-copy-initialization-test--matrix-values left)
                       (nl-llm-copy-initialization-test--matrix-values other)))
    (should (= (plist-get left-meta :matrix-count) 9))
    (should (= (plist-get left-meta :matrix-elements) 1052))
    (should (plist-get left-meta :one-dimensional-preserved))))

(ert-deftest nl-llm-copy-initialization-matrix-bounds ()
  (let* ((model (nl-llm-copy-initialization-test--model))
         (meta (nl-llm-compare-copy-initialization-reinitialize
                model #x1A2B3C4D))
         (scale (plist-get meta :scale)))
    (should
     (cl-every (lambda (value)
                 (and (numberp value) (< (abs value) scale)))
               (nl-llm-copy-initialization-test--matrix-values model)))))

(ert-deftest nl-llm-copy-initialization-preserves-one-dimensional-values ()
  (let* ((model (nl-llm-copy-initialization-test--model))
         (parameters (nl-llm-agent--p5-params model))
         (before
          (mapcar
           (lambda (parameter)
             (let ((tensor (pav-value parameter)))
               (when (= (length (photon-tensor-shape tensor)) 1)
                 (copy-sequence (photon-tensor-data tensor)))))
           parameters)))
    (nl-llm-compare-copy-initialization-reinitialize model #x1A2B3C4D)
    (let ((after
           (mapcar
            (lambda (parameter)
              (let ((tensor (pav-value parameter)))
                (when (= (length (photon-tensor-shape tensor)) 1)
                  (photon-tensor-data tensor))))
            parameters)))
      (should (equal before after)))))

(ert-deftest nl-llm-copy-initialization-keeps-dimensions-and-count ()
  (let* ((model (nl-llm-copy-initialization-test--model))
         (before (nl-llm-agent--p5-params model))
         (meta (nl-llm-compare-copy-initialization-reinitialize
                model #x1A2B3C4D)))
    (should (= (plist-get meta :parameter-count-before)
               (plist-get meta :parameter-count-after)))
    (should (equal (plist-get meta :dimensions-before)
                   (plist-get meta :dimensions-after)))
    (should (= (length before) 20))))

(ert-deftest nl-llm-copy-initialization-restores-constructor-on-error ()
  (let* ((constructor 'nl-llm-agent-improve-model)
         (original (symbol-function constructor))
         (calls 0))
    (cl-letf (((symbol-function 'nl-llm-learn-literal-copy-run)
               (lambda ()
                 (setq calls (1+ calls))
                 (funcall constructor 2 2 256 1 1 "utf8-byte-v1")
                 (error "intentional runner failure"))))
      (should-error
       (nl-llm-compare-copy-initialization--run-candidate #x1A2B3C4D)))
    (should (= calls 1))
    (should (eq original (symbol-function constructor)))))

(ert-deftest nl-llm-copy-initialization-paired-run-is-fully-nested ()
  (let ((call-count (list 0)))
    (cl-letf (((symbol-function 'nl-llm-learn-literal-copy-run)
               (lambda ()
                 (setcar call-count (1+ (car call-count)))
                 (funcall 'nl-llm-agent-improve-model 2 2 256 1 1
                          "utf8-byte-v1")
                 (let ((hash (if (= (car call-count) 1)
                                 "baseline-hash" "candidate-hash")))
                   (list :settings '(:dim 2 :ff 2 :epochs 32)
                         :dataset-sha256 "test-dataset"
                         :model-before-sha256 hash :loss-before 1.0)))))
      (let ((report (nl-llm-compare-copy-initialization-run #x1A2B3C4D)))
        (should (equal (plist-get (plist-get report :baseline) :settings)
                       '(:dim 2 :ff 2 :epochs 32)))
        (should (equal (plist-get (plist-get report :candidate) :dataset-sha256)
                       "test-dataset"))
        (should (plist-get report :settings-identical))
        (should (plist-get report :dataset-identical))
        (should (= (plist-get report :constructor-count) 1))
        (should (= (plist-get (plist-get report :initializer) :matrix-count)
                   9))
        (should (= (car call-count) 2))))))

(ert-deftest nl-llm-copy-initialization-load-does-not-train ()
  (let ((called nil)
        (nl-llm-compare-copy-initialization-no-run t))
    (cl-letf (((symbol-function 'nl-llm-learn-literal-copy-run)
               (lambda () (setq called t)))
              ((symbol-function 'nl-llm-gpu-enable)
               (lambda () (setq called t))))
      (load (expand-file-name
             "../examples/compare-copy-initialization.el"
             nl-llm-copy-initialization-test--here)
            nil nil t))
    (should-not called)))

(ert-run-tests-batch-and-exit)

;;; copy-initialization-test.el ends here
