;;; copy-curriculum-test.el --- tests for COPY curriculum pretraining -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)

(defconst nl-llm-copy-curriculum-test--here
  (file-name-directory (or load-file-name buffer-file-name)))
(defvar nl-llm-learn-copy-curriculum-auto-run nil)
(defvar nl-llm-learn-literal-copy-no-run nil)

(let ((here nl-llm-copy-curriculum-test--here))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here))
  (load (expand-file-name "../examples/learn-copy-curriculum.el" here)
        nil nil t))

(declare-function nl-llm-learn-copy-curriculum-dataset
                  "../examples/learn-copy-curriculum" ())
(declare-function nl-llm-learn-copy-curriculum-train
                  "../examples/learn-copy-curriculum" (model))
(declare-function nl-llm-learn-copy-curriculum-run
                  "../examples/learn-copy-curriculum" ())
(declare-function nl-llm-learn-copy-curriculum--scores
                  "../examples/learn-copy-curriculum" (model dataset))
(declare-function nl-llm-agent-initialization-create
                  "nl-llm-agent-initialization" (&rest keys))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))
(declare-function nl-llm-agent-tokenizer-encode
                  "nl-llm-agent-tokenizer" (text &optional identifier))
(declare-function nl-llm-learn-literal-copy-greedy-decode
                  "learn-literal-copy"
                  (prompt-ids step-function &optional max-generated))

(ert-deftest nl-llm-copy-curriculum-dataset-is-deterministic-and-bounded ()
  (let* ((left (nl-llm-learn-copy-curriculum-dataset))
         (right (nl-llm-learn-copy-curriculum-dataset))
         (all (plist-get left :all))
         (train (plist-get left :train))
         (dev (plist-get left :dev))
         (all-literals (mapcar (lambda (e) (plist-get e :literal))
                               (append all nil)))
         (lengths (mapcar (lambda (e) (plist-get e :length))
                          (append all nil))))
    (should (equal left right))
    (should (= (length all) 160))
    (should (= (length train) 128))
    (should (= (length dev) 32))
    (should (= (length (delete-dups (copy-sequence all-literals))) 160))
    (should (= (plist-get left :final-state) 2412661987))
    (should (equal (cl-count 1 lengths) 20))
    (dolist (length '(2 3 4 8 12 16 24))
      (should (= (cl-count length lengths) 20)))
    (should (equal (plist-get (aref all 0) :literal) "x"))
    (should (equal (plist-get (aref all 159) :literal)
                   "uj_2ytp/nfg6orv:0:xsyo5b"))))

(ert-deftest nl-llm-copy-curriculum-splits-are-disjoint-and-fixed ()
  (let* ((dataset (nl-llm-learn-copy-curriculum-dataset))
         (all (plist-get dataset :all))
         (train (plist-get dataset :train))
         (dev (plist-get dataset :dev))
         (train-literals (mapcar (lambda (e) (plist-get e :literal))
                                 (append train nil)))
         (dev-literals (mapcar (lambda (e) (plist-get e :literal))
                               (append dev nil))))
    (should-not (cl-intersection train-literals dev-literals :test #'equal))
    (should (= (length (cl-remove-if-not
                        (lambda (e) (= (% (plist-get e :index) 5) 0))
                        (append dev nil)))
               32))
    (should (= (+ (length train) (length dev)) (length all)))
    (dolist (example (append all nil))
      (should (<= (+ (length (plist-get example :prompt))
                    (length (plist-get example :completion)))
                  64)))))

(ert-deftest nl-llm-copy-curriculum-initial-model-hash-is-frozen ()
  (let ((model
         (nl-llm-agent-initialization-create
          :initializer 'xorshift32 :seed 439041101
          :dim 32 :ff 64 :vocab 256 :nblocks 1 :heads 1
          :tokenizer "utf8-byte-v1")))
    (should
     (equal
      (nl-llm-learn-literal-copy--model-hash model)
      "bc4ae38649deda046482b1d8eb70b369766cd2a848b905ba44bede75285629c3"))))

(ert-deftest nl-llm-copy-curriculum-training-forwards-fixed-plan ()
  (let* ((dataset (nl-llm-learn-copy-curriculum-dataset))
         (train-examples (append (plist-get dataset :train) nil))
         (expected-trajs
          (mapcar
           (lambda (example)
             (append
              (nl-llm-agent-tokenizer-encode
               (plist-get example :prompt) "utf8-byte-v1")
              (nl-llm-agent-tokenizer-encode
               (plist-get example :completion) "utf8-byte-v1")))
           train-examples))
         (expected-loss-starts
          (vconcat
           (mapcar
            (lambda (example)
              (length
               (nl-llm-agent-tokenizer-encode
                (plist-get example :prompt) "utf8-byte-v1")))
            train-examples)))
         (enable-count 0) (disable-count 0)
        (from-args nil) (train-args nil) (sync-count 0) (free-count 0))
    (cl-letf (((symbol-function 'nl-llm-gpu-available-p)
               (lambda () nil))
              ((symbol-function 'nl-llm-gpu-enable)
               (lambda () (setq enable-count (1+ enable-count)) 'gpu))
              ((symbol-function 'nl-llm-gpu-disable)
               (lambda () (setq disable-count (1+ disable-count)) 'cpu))
              ((symbol-function 'nl-llm-agent-ondevice-from-model)
               (lambda (&rest args) (setq from-args args) 'context))
              ((symbol-function 'nl-llm-agent-ondevice-train)
               (lambda (&rest args) (setq train-args args) 4096))
              ((symbol-function 'nl-llm-agent-ondevice-sync)
               (lambda (_ctx) (setq sync-count (1+ sync-count))))
              ((symbol-function 'nl-llm-agent-ondevice-free)
               (lambda (_ctx) (setq free-count (1+ free-count)))))
      (let ((result (nl-llm-learn-copy-curriculum-train 'model)))
        (should (= (plist-get result :steps) 4096))
        (should (= (plist-get result :train-count) 128))
        (should (= enable-count 1))
        (should (= disable-count 1))
        (should (= sync-count 1))
        (should (= free-count 1))
        (should (equal (car from-args) 'model))
        (should (= (nth 1 from-args) 64))
        (should (= (nth 2 from-args) 0.003))
        (should (equal (nthcdr 3 from-args)
                       '(:optimizer adam :loss-mode completion
                         :transfer-mode compact)))
        (should (eq (car train-args) 'context))
        (should (= (nth 2 train-args) 32))
        (should (equal (nth 1 train-args) expected-trajs))
        (should (eq (nth 3 train-args) :loss-starts))
        (should (equal (nth 4 train-args) expected-loss-starts))
        (should (eq (nth 5 train-args) :after-step))
        (should (functionp (nth 6 train-args)))
        (should (= (length (nth 1 train-args)) 128))))))

(ert-deftest nl-llm-copy-curriculum-training-rejects-active-gpu ()
  (let ((enabled nil))
    (cl-letf (((symbol-function 'nl-llm-gpu-available-p)
               (lambda () t))
              ((symbol-function 'nl-llm-gpu-enable)
               (lambda () (setq enabled t))))
      (should-error (nl-llm-learn-copy-curriculum-train 'model)))
    (should-not enabled)))

(ert-deftest nl-llm-copy-curriculum-training-cleans-up-on-error ()
  (let ((free-count 0) (disable-count 0))
    (cl-letf (((symbol-function 'nl-llm-gpu-available-p)
               (lambda () nil))
              ((symbol-function 'nl-llm-gpu-enable)
               (lambda () 'gpu))
              ((symbol-function 'nl-llm-gpu-disable)
               (lambda () (setq disable-count (1+ disable-count))))
              ((symbol-function 'nl-llm-agent-ondevice-from-model)
               (lambda (&rest _args) 'context))
              ((symbol-function 'nl-llm-agent-ondevice-train)
               (lambda (&rest _args) (error "intentional train failure")))
              ((symbol-function 'nl-llm-agent-ondevice-free)
               (lambda (_ctx)
                 (setq free-count (1+ free-count))
                 (error "intentional free failure")))
              ((symbol-function 'nl-llm-agent-ondevice-sync)
               (lambda (_ctx) nil)))
      (should-error (nl-llm-learn-copy-curriculum-train 'model)))
    (should (= free-count 1))
    (should (= disable-count 1))))

(ert-deftest nl-llm-copy-curriculum-run-and-score-reject-active-gpu ()
  (cl-letf (((symbol-function 'nl-llm-gpu-available-p)
             (lambda () t)))
    (should-error (nl-llm-learn-copy-curriculum-run))
    (should-error
     (nl-llm-learn-copy-curriculum--scores 'model 'dataset))))

(ert-deftest nl-llm-copy-curriculum-decoder-is-unrestricted-and-bounded ()
  (let ((calls nil))
    (let ((result
           (nl-llm-learn-literal-copy-greedy-decode
            '(7 8)
            (lambda (token)
              (push token calls)
              (let ((logits (make-vector 256 0.0)))
                (cond ((= token 8) (aset logits 255 2.0))
                      ((= token 255) (aset logits 10 3.0)))
                logits))
            32)))
      (should (equal (nreverse calls) '(7 8 255)))
      (should (equal (plist-get result :ids) [255 10]))
      (should (plist-get result :terminated))))
  (let ((result
         (nl-llm-learn-literal-copy-greedy-decode
          '(7)
          (lambda (_token)
            (let ((logits (make-vector 256 0.0)))
              (aset logits 255 2.0)
              logits))
          2)))
    (should (equal (plist-get result :ids) [255 255]))
    (should-not (plist-get result :terminated))))

(ert-run-tests-batch-and-exit)

;;; copy-curriculum-test.el ends here
