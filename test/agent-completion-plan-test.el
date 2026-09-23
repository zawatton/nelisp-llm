;;; agent-completion-plan-test.el --- canonical completion plan tests -*- lexical-binding: t; -*-

(require 'ert)
(require 'cl-lib)
(defconst nl-llm-agent-completion-plan-test--here
  (file-name-directory (or load-file-name buffer-file-name)))
(add-to-list 'load-path
             (expand-file-name "../lisp"
                               nl-llm-agent-completion-plan-test--here))
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-agent-completion-plan)

(defun nl-llm-agent-completion-plan-test--base (&optional masks)
  "Return a small canonical plan fixture."
  (nl-llm-agent-completion-plan-make
   '((0 1 2) [3 4 5]) [1 2]
   :sequence 8 :learning-rate 0.1 :epochs 2
   :loss-masks masks :shuffle-seed 12345))

(ert-deftest nl-llm-agent-completion-plan-normalizes-and-detaches-input ()
  (let* ((first (list 0 1 2))
         (second (vector 3 4 5))
         (starts (vector 1 2))
         (masks (vector (vector 0 1 1) (vector 0 0 1)))
         (plan (nl-llm-agent-completion-plan-make
                (list first second) starts
                :sequence 8 :learning-rate (/ 1.0 10.0) :epochs 2
                :optimizer "sgd" :transfer-mode 'dense
                :loss-masks masks :shuffle-seed 1)))
    (setcar first 95)
    (aset second 0 95)
    (aset starts 0 2)
    (aset (aref masks 0) 1 0)
    (should (equal (plist-get plan :trajectories) [[0 1 2] [3 4 5]]))
    (should (equal (plist-get plan :loss-starts) [1 2]))
    (should (equal (plist-get plan :loss-masks)
                   (vector [0 1 1] [0 0 1])))
    (should (= (plist-get plan :vocab) 96))
    (should (= (plist-get plan :pad-token) 0))
    (should (eq (plist-get plan :transfer-mode) nil))
    (should (= (length plan) 28))
    (should (equal (nl-llm-agent-completion-plan-validate plan) plan))
    (let ((tokenizer (plist-get plan :tokenizer)))
      (aset tokenizer 0 ?X)
      (should (equal (plist-get (nl-llm-agent-completion-plan-test--base)
                                :tokenizer)
                     "ascii-char-v1")))
    (let ((format (plist-get plan :format)))
      (aset format 0 ?X)
      (should (equal nl-llm-agent-completion-plan-format
                     "nl-llm-completion-plan-v1")))))

(ert-deftest nl-llm-agent-completion-plan-serializes-and-validates-detached-copy ()
  (let* ((plan (nl-llm-agent-completion-plan-test--base))
         (serialized (prin1-to-string plan))
         (read-back (read serialized))
         (validated (nl-llm-agent-completion-plan-validate read-back)))
    (should (equal validated plan))
    (should-not (eq (plist-get validated :format)
                    (plist-get plan :format)))
    (should-not (eq (plist-get validated :tokenizer)
                    (plist-get plan :tokenizer)))
    (should-not (eq (plist-get validated :digest)
                    (plist-get plan :digest)))
    (should-not (eq (plist-get validated :trajectories)
                    (plist-get plan :trajectories)))
    (should-not (eq (plist-get validated :loss-starts)
                    (plist-get plan :loss-starts)))
    (should (equal (plist-get plan :digest)
                   (nl-llm-agent-completion-plan--digest plan)))))

(ert-deftest nl-llm-agent-completion-plan-digest-binds-semantic-fields ()
  (let ((base (nl-llm-agent-completion-plan-test--base)))
    (dolist (variant
             (list
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 7
               :learning-rate 0.1 :epochs 2 :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 8
               :learning-rate 0.2 :epochs 2 :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 8
               :learning-rate 0.1 :epochs 3 :optimizer 'adam
               :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 8
               :learning-rate 0.1 :epochs 2 :transfer-mode 'compact
               :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 8
               :learning-rate 0.1 :epochs 2 :shuffle-seed 12346)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 6]) [1 2] :sequence 8
               :learning-rate 0.1 :epochs 2 :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [2 2] :sequence 8
               :learning-rate 0.1 :epochs 2 :shuffle-seed 12345)
              (nl-llm-agent-completion-plan-make
               '((0 1 2) [3 4 5]) [1 2] :sequence 8
               :learning-rate 0.1 :epochs 2
               :loss-masks (vector [0 1 1] [0 0 1]))))
      (should-not (equal (plist-get base :digest)
                         (plist-get variant :digest))))))

(ert-deftest nl-llm-agent-completion-plan-validates-masks-and-sparse-bindings ()
  (let ((plan (nl-llm-agent-completion-plan-test--base
               (vector [0 1 0] [0 0 1]))))
    (should (equal (plist-get plan :loss-masks)
                   (vector [0 1 0] [0 0 1])))
    (should-error
     (nl-llm-agent-completion-plan-make
      '((0 1 2) [3 4 5]) [1 2] :sequence 8 :learning-rate 0.1 :epochs 1
      :loss-masks (vector [1 0 0] [0 0 1])))
    (should-error
     (nl-llm-agent-completion-plan-make
      '((0 1 2) [3 4 5]) [1 2] :sequence 8 :learning-rate 0.1 :epochs 1
      :loss-masks (vector [0 0 0] [0 0 1])))
    (should-error
     (nl-llm-agent-completion-plan-make
      '((0 1 2) [3 4 5]) [1 2] :sequence 8 :learning-rate 0.1 :epochs 1
      :loss-masks (vector [0 1 2] [0 0 1])))))

(ert-deftest nl-llm-agent-completion-plan-supports-utf8-and-bounds ()
  (let ((plan
         (nl-llm-agent-completion-plan-make
          '((32 228 184 139) [32 65]) [1 1]
          :tokenizer "utf8-byte-v1" :sequence 8
          :learning-rate 0.01 :epochs 1)))
    (should (= (plist-get plan :vocab) 256))
    (should (= (plist-get plan :pad-token) 32)))
  (dolist (bad
           (list
            (lambda ()
              (nl-llm-agent-completion-plan-make nil [1]
               :sequence 8 :learning-rate 0.1 :epochs 1))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((0 1)) [1]
               :sequence 1 :learning-rate 0.1 :epochs 1))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((96 1)) [1]
               :sequence 8 :learning-rate 0.1 :epochs 1))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((0 1)) [0]
               :sequence 8 :learning-rate 0.1 :epochs 1))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((0 1)) [1]
               :sequence 8 :learning-rate 0.0 :epochs 1))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((0 1)) [1]
               :sequence 8 :learning-rate 0.1 :epochs 0))
            (lambda ()
              (nl-llm-agent-completion-plan-make '((0 1)) [1]
               :sequence 8 :learning-rate 0.1 :epochs 1
               :shuffle-seed 0))))
    (should-error (funcall bad))))

(ert-deftest nl-llm-agent-completion-plan-rejects-malformed-and-tampered-plists ()
  (let ((plan (nl-llm-agent-completion-plan-test--base)))
    (dolist (bad
             (list
              (append plan '(:extra t))
              (let ((copy (copy-sequence plan)))
                (append copy (list :format nl-llm-agent-completion-plan-format)))
              (let ((copy (copy-sequence plan)))
                (plist-put copy :vocab 256))
              (let ((copy (copy-sequence plan)))
                (plist-put copy :digest (make-string 64 ?0)))
              (let ((copy (copy-sequence plan)))
                (plist-put copy :trajectories [[0 1 2] [3 4 6]]))))
      (should-error (nl-llm-agent-completion-plan-validate bad)))
    (let ((cyclic (list :format nl-llm-agent-completion-plan-format)))
      (setcdr (last cyclic) cyclic)
      (should-error (nl-llm-agent-completion-plan-validate cyclic)))
    (should-error
     (nl-llm-agent-completion-plan-validate
      (cons :format '(:not-a-plist))))))

(provide 'agent-completion-plan-test)

(ert-run-tests-batch-and-exit)

;;; agent-completion-plan-test.el ends here
