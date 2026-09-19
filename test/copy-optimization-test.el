;;; copy-optimization-test.el --- tests for COPY optimization ablation -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'ert)
(require 'seq)

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here))
  (load (expand-file-name "../examples/compare-copy-optimization.el" here)
        nil nil t))

(declare-function nl-llm-compare-copy-architecture--geometry
                  "compare-copy-architecture" (dim ff blocks heads))
(declare-function nl-llm-compare-copy-architecture--model
                  "compare-copy-architecture" (geometry))
(declare-function nl-llm-compare-copy-architecture--parameter-count
                  "compare-copy-architecture" (model))
(declare-function nl-llm-compare-copy-optimization--model
                  "compare-copy-optimization" ())
(declare-function nl-llm-compare-copy-optimization--schedule
                  "compare-copy-optimization" (examples order))
(declare-function nl-llm-compare-copy-optimization--run-arm
                  "compare-copy-optimization" (spec dataset))
(declare-function nl-llm-compare-copy-optimization-run
                  "compare-copy-optimization" (&optional arm-id))
(declare-function nl-llm-compare-copy-diversity-dataset
                  "compare-copy-diversity" ())
(defvar nl-llm-compare-copy-optimization-specs)
(defvar nl-llm-compare-copy-optimization-auto-run)
(defvar nl-llm-learn-copy-curriculum-learning-rate)

(defun nl-llm-copy-optimization-test--spec (id)
  (seq-find (lambda (spec) (eq (plist-get spec :id) id))
            (append nl-llm-compare-copy-optimization-specs nil)))

(defun nl-llm-copy-optimization-test--literals (examples)
  (mapcar (lambda (example) (plist-get example :literal))
          (append examples nil)))

(ert-deftest nl-llm-copy-optimization-specs-are-four-fixed-arms ()
  (should (vectorp nl-llm-compare-copy-optimization-specs))
  (should (= (length nl-llm-compare-copy-optimization-specs) 4))
  (should (equal (mapcar (lambda (spec) (plist-get spec :id))
                         (append nl-llm-compare-copy-optimization-specs nil))
                 '(sorted-high sorted-low shuffled-high shuffled-low)))
  (dolist (spec (append nl-llm-compare-copy-optimization-specs nil))
    (should (memq (plist-get spec :order) '(sorted shuffled)))
    (should (member (plist-get spec :learning-rate) '(0.003 0.0003)))))

(ert-deftest nl-llm-copy-optimization-schedules-preserve-input ()
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (examples (plist-get dataset :candidate))
         (before (prin1-to-string examples))
         (sorted (nl-llm-compare-copy-optimization--schedule examples 'sorted))
         (shuffled (nl-llm-compare-copy-optimization--schedule
                    examples 'shuffled)))
    (should (vectorp (plist-get sorted :examples)))
    (should (vectorp (plist-get shuffled :examples)))
    (should (= (length (plist-get sorted :examples)) 4096))
    (should (= (length (plist-get shuffled :examples)) 4096))
    (should (equal (nl-llm-copy-optimization-test--literals
                    (plist-get sorted :examples))
                   (nl-llm-copy-optimization-test--literals examples)))
    (dotimes (batch 32)
      (let ((start (* batch 128)))
        (should (equal
                 (sort (copy-sequence
                        (cl-subseq (nl-llm-copy-optimization-test--literals
                                    examples)
                                   start (+ start 128))) #'string<)
                 (sort (copy-sequence
                        (cl-subseq (nl-llm-copy-optimization-test--literals
                                    (plist-get shuffled :examples))
                                   start (+ start 128))) #'string<)))))
    (should (equal (prin1-to-string examples) before))))

(ert-deftest nl-llm-copy-optimization-shuffled-schedule-has-frozen-oracle ()
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (examples (plist-get dataset :candidate))
         (schedule
          (nl-llm-compare-copy-optimization--schedule examples 'shuffled))
         (permutations (plist-get schedule :permutations)))
    (should (= (plist-get schedule :final-state) 4192003634))
    (should (= (length permutations) 32))
    (should (equal (cl-subseq (append (aref permutations 0) nil) 0 12)
                   '(111 55 25 5 115 39 116 73 110 10 68 1)))
    (should (equal (cl-subseq (append (aref permutations 31) nil) 0 12)
                   '(12 99 110 95 55 20 27 3 84 82 96 8)))
    (should (stringp (plist-get schedule :schedule-sha256)))
    (let (indices)
      (dotimes (batch 32)
        (setq indices
              (nconc indices (append (aref permutations batch) nil))))
      (should (equal
               (secure-hash
                'sha256
                (mapconcat #'number-to-string indices ","))
               "ba2fec3518e5aa61eef53e8c69178119e57957041693beafd663d2239f896701")))
    (should (equal (plist-get schedule :literal-sha256)
                   "e41b65b672389d5d168f40eb575f22e4b7118c2c9ba267b36754b6a5d911d193"))))

(ert-deftest nl-llm-copy-optimization-sorted-schedule-has-frozen-oracle ()
  (let* ((dataset (nl-llm-compare-copy-diversity-dataset))
         (examples (plist-get dataset :candidate))
         (schedule
          (nl-llm-compare-copy-optimization--schedule examples 'sorted)))
    (should (eq (plist-get schedule :order) 'sorted))
    (should-not (plist-get schedule :seed))
    (should (equal (plist-get schedule :literal-sha256)
                   "62a0043f734326cda3ec1afc9aad3f8ccefe33447384352325b5eb9e03412036"))))

(ert-deftest nl-llm-copy-optimization-candidate-has-fixed-parameter-count ()
  (let ((model (nl-llm-compare-copy-optimization--model)))
    (should (= (nl-llm-compare-copy-architecture--parameter-count model)
               24616))))

(ert-deftest nl-llm-copy-optimization-run-selects-four-or-one-arm ()
  (let ((calls nil)
        (fake-data '(:candidate [] :candidate-sha256 "candidate"
                     :literal-sha256 "literal" :train-sha256 "train"
                     :dev-sha256 "dev" :final-state 1)))
    (cl-letf (((symbol-function 'nl-llm-gpu-available-p) (lambda () nil))
              ((symbol-function 'nl-llm-compare-copy-diversity-dataset)
               (lambda () fake-data))
              ((symbol-function 'nl-llm-compare-copy-optimization--run-arm)
               (lambda (spec dataset)
                 (push (list spec dataset) calls)
                 (list :id (plist-get spec :id)
                       :model-before-sha256 "same"))))
      (let ((all (nl-llm-compare-copy-optimization-run)))
        (should (equal (mapcar (lambda (report) (plist-get report :id))
                               (append (plist-get all :arms) nil))
                       '(sorted-high sorted-low shuffled-high shuffled-low)))
        (let ((ordered (nreverse calls)))
          (should (equal (mapcar (lambda (call) (plist-get (car call) :id))
                                ordered)
                         '(sorted-high sorted-low shuffled-high shuffled-low)))
          (dolist (call ordered)
            (should (eq (cadr call) fake-data)))
          (should (= (plist-get (car (car ordered)) :learning-rate) 0.003))
          (should (= (plist-get (car (cadr ordered)) :learning-rate) 0.0003))
          (should (eq (plist-get (car (car (cddr ordered))) :order)
                      'shuffled))))
      (setq calls nil)
      (let ((one (nl-llm-compare-copy-optimization-run 'shuffled-low)))
        (should (= (length calls) 1))
        (should (eq (plist-get (caar calls) :id) 'shuffled-low))
        (should (eq (cadar calls) fake-data))
        (should (equal (mapcar (lambda (report) (plist-get report :id))
                               (append (plist-get one :arms) nil))
                       '(shuffled-low)))))))

(ert-deftest nl-llm-copy-optimization-run-rejects-active-gpu ()
  (cl-letf (((symbol-function 'nl-llm-gpu-available-p) (lambda () t)))
    (should-error (nl-llm-compare-copy-optimization-run))))

(ert-deftest nl-llm-copy-optimization-arm-binds-learning-rate-and-restores-it ()
  (let* ((original nl-llm-learn-copy-curriculum-learning-rate)
         (spec (nl-llm-copy-optimization-test--spec 'sorted-low))
         (dataset '(:candidate []))
         (observed nil))
    (cl-letf (((symbol-function 'nl-llm-compare-copy-optimization--model)
               (lambda () '(:weight 0)))
              ((symbol-function 'nl-llm-learn-literal-copy--model-hash)
               (lambda (model) (format "w%d" (plist-get model :weight))))
              ((symbol-function 'nl-llm-compare-copy-architecture--scores)
               (lambda (&rest _args) '(:greedy ok)))
              ((symbol-function 'nl-llm-compare-copy-architecture--parameter-count)
               (lambda (_model) 24616))
              ((symbol-function 'nl-llm-compare-copy-optimization--schedule)
               (lambda (&rest _args) '(:examples [])))
              ((symbol-function 'nl-llm-compare-copy-diversity-train-arm)
               (lambda (&rest _args)
                 (setq observed nl-llm-learn-copy-curriculum-learning-rate)
                 '(:steps 4096))))
      (nl-llm-compare-copy-optimization--run-arm spec dataset))
    (should (= observed 0.0003))
    (should (= nl-llm-learn-copy-curriculum-learning-rate original))))

(ert-deftest nl-llm-copy-optimization-arm-restores-learning-rate-on-error ()
  (let* ((original nl-llm-learn-copy-curriculum-learning-rate)
         (spec (nl-llm-copy-optimization-test--spec 'sorted-low))
         (dataset '(:candidate []))
         (observed nil))
    (cl-letf (((symbol-function 'nl-llm-compare-copy-optimization--model)
               (lambda () '(:weight 0)))
              ((symbol-function 'nl-llm-learn-literal-copy--model-hash)
               (lambda (_model) "same"))
              ((symbol-function 'nl-llm-compare-copy-architecture--scores)
               (lambda (&rest _args) '(:greedy ok)))
              ((symbol-function 'nl-llm-compare-copy-optimization--schedule)
               (lambda (&rest _args) '(:examples [])))
              ((symbol-function 'nl-llm-compare-copy-diversity-train-arm)
               (lambda (&rest _args)
                 (setq observed nl-llm-learn-copy-curriculum-learning-rate)
                 (error "synthetic training failure"))))
      (should-error
       (nl-llm-compare-copy-optimization--run-arm spec dataset)))
    (should (= observed 0.0003))
    (should (= nl-llm-learn-copy-curriculum-learning-rate original))))

(ert-deftest nl-llm-copy-optimization-arm-rejects-evaluation-mutation ()
  (let ((train-calls 0) (score-calls 0))
    (cl-letf (((symbol-function 'nl-llm-compare-copy-optimization--model)
               (lambda () (list :weight 0)))
              ((symbol-function 'nl-llm-learn-literal-copy--model-hash)
               (lambda (model) (format "w%d" (plist-get model :weight))))
              ((symbol-function 'nl-llm-compare-copy-architecture--scores)
               (lambda (model _dataset)
                 (setq score-calls (1+ score-calls))
                 (when (= score-calls 1)
                   (setf (plist-get model :weight) 1))
                 '(:greedy ok)))
              ((symbol-function 'nl-llm-compare-copy-diversity-train-arm)
               (lambda (&rest _args) (setq train-calls (1+ train-calls))))
              ((symbol-function 'nl-llm-compare-copy-optimization--schedule)
               (lambda (&rest _args) '(:examples []))))
      (let ((condition
             (should-error
              (nl-llm-compare-copy-optimization--run-arm
               (nl-llm-copy-optimization-test--spec 'sorted-high)
               '(:candidate [])))))
        (should (string-match-p "evaluation mutated model weights"
                                (error-message-string condition))))
      (should (= train-calls 0)))))

(ert-deftest nl-llm-copy-optimization-arm-rejects-after-evaluation-mutation ()
  (let ((train-calls 0) (score-calls 0))
    (cl-letf (((symbol-function 'nl-llm-compare-copy-optimization--model)
               (lambda () (list :weight 0)))
              ((symbol-function 'nl-llm-learn-literal-copy--model-hash)
               (lambda (model) (format "w%d" (plist-get model :weight))))
              ((symbol-function 'nl-llm-compare-copy-architecture--scores)
               (lambda (model _dataset)
                 (setq score-calls (1+ score-calls))
                 (when (= score-calls 2)
                   (setf (plist-get model :weight) 1))
                 '(:greedy ok)))
              ((symbol-function 'nl-llm-compare-copy-architecture--parameter-count)
               (lambda (_model) 24616))
              ((symbol-function 'nl-llm-compare-copy-optimization--schedule)
               (lambda (&rest _args) '(:examples [])))
              ((symbol-function 'nl-llm-compare-copy-diversity-train-arm)
               (lambda (&rest _args) (setq train-calls (1+ train-calls)))) )
      (let ((condition
             (should-error
              (nl-llm-compare-copy-optimization--run-arm
               (nl-llm-copy-optimization-test--spec 'sorted-high)
               '(:candidate [])))))
        (should (string-match-p "evaluation mutated model weights"
                                (error-message-string condition))))
      (should (= score-calls 2))
      (should (= train-calls 1)))))

(ert-deftest nl-llm-copy-optimization-run-rejects-bad-arm-without-gpu ()
  (should-error (nl-llm-compare-copy-optimization-run 'unknown))
  (should-not nl-llm-compare-copy-optimization-auto-run))

(ert-deftest nl-llm-copy-optimization-load-is-explicit ()
  (let ((gpu-calls 0)
        (nl-llm-compare-copy-optimization-auto-run nil)
        (nl-llm-compare-copy-architecture-auto-run t)
        (nl-llm-compare-copy-diversity-auto-run t)
        (nl-llm-learn-copy-curriculum-auto-run t)
        (here (file-name-directory (or load-file-name buffer-file-name))))
    (cl-letf (((symbol-function 'nl-llm-gpu-available-p)
               (lambda () (setq gpu-calls (1+ gpu-calls)) t))
              ((symbol-function 'nl-llm-gpu-enable)
               (lambda () (error "ambient load must not enable GPU")))
              ((symbol-function 'nl-llm-compare-copy-optimization-run)
               (lambda (&rest _args) (error "ambient load ran optimization"))))
      (load (expand-file-name "../examples/compare-copy-optimization.el" here)
            nil nil t))
    (should (= gpu-calls 0))
    (should nl-llm-compare-copy-architecture-auto-run)
    (should nl-llm-compare-copy-diversity-auto-run)
    (should nl-llm-learn-copy-curriculum-auto-run)))

(ert-run-tests-batch-and-exit)

;;; copy-optimization-test.el ends here
