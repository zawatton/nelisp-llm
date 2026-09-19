;;; literal-copy-test.el --- bounded literal-copy example tests -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'ert)
(require 'cl-lib)

;; Load the example without starting its GPU experiment.  Resolve the source
;; relative to this test so the test remains portable across checkouts.
(defvar nl-llm-learn-literal-copy-no-run nil)
(defvar nl-llm-learn-literal-copy-tokenizer nil)
(defvar nl-llm-learn-literal-copy-max-decode 8)
(declare-function nl-llm-learn-literal-copy-literals "learn-literal-copy" ())
(declare-function nl-llm-learn-literal-copy-dataset "learn-literal-copy" ())
(declare-function nl-llm-learn-literal-copy-greedy-decode
                  "learn-literal-copy" (prompt-ids step-function &optional max-generated))
(declare-function nl-llm-learn-literal-copy-native-decode
                  "learn-literal-copy" (model prompt-ids))
(declare-function nl-llm-agent-supervised-encode
                  "nl-llm-agent-supervised" (examples &optional tokenizer))
(declare-function nl-llm-agent-improve-model
                  "nl-llm-agent-improve" (dim ff vocab blocks heads &optional tokenizer))
(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (setq nl-llm-learn-literal-copy-no-run t)
  (load (expand-file-name "../examples/learn-literal-copy.el" here)
        nil nil t))

(ert-deftest nl-llm-learn-literal-copy-literals-are-length-first-lexical ()
  (should
   (equal
    (nl-llm-learn-literal-copy-literals)
    '("a" "b" "c"
      "aa" "ab" "ac" "ba" "bb" "bc" "ca" "cb" "cc"
      "aaa" "aab" "aac" "aba" "abb" "abc" "aca" "acb" "acc"
      "baa" "bab" "bac" "bba" "bbb" "bbc" "bca" "bcb" "bcc"
      "caa" "cab" "cac" "cba" "cbb" "cbc" "cca" "ccb" "ccc"))))

(ert-deftest nl-llm-learn-literal-copy-dataset-splits-by-index-mod-five ()
  (let* ((dataset (nl-llm-learn-literal-copy-dataset))
         (all (plist-get dataset :all))
         (train (plist-get dataset :train))
         (dev (plist-get dataset :dev)))
    (should (= (length all) 39))
    (should (= (length train) 31))
    (should (= (length dev) 8))
    (should (equal (mapcar (lambda (example) (plist-get example :index))
                           (append dev nil))
                   '(0 5 10 15 20 25 30 35)))
    (should (equal (mapcar (lambda (example) (plist-get example :index))
                           (append train nil))
                   '(1 2 3 4 6 7 8 9 11 12 13 14 16 17 18 19
                     21 22 23 24 26 27 28 29 31 32 33 34 36 37 38)))
    (should (= (length (cl-intersection (mapcar (lambda (x) (plist-get x :index))
                                                 (append train nil))
                                        (mapcar (lambda (x) (plist-get x :index))
                                                (append dev nil))))
               0))))

(ert-deftest nl-llm-learn-literal-copy-examples-encode-bounded-and-deterministic ()
  (let* ((dataset (nl-llm-learn-literal-copy-dataset))
         (all (plist-get dataset :all))
         (first (aref all 0))
         (last (aref all 38))
         (examples
          (vconcat
           (mapcar (lambda (example)
                     (list :prompt (plist-get example :prompt)
                           :completion (plist-get example :completion)))
                   (append all nil))))
         (encoded (nl-llm-agent-supervised-encode
                   examples nl-llm-learn-literal-copy-tokenizer))
         (repeat (nl-llm-agent-supervised-encode
                  examples nl-llm-learn-literal-copy-tokenizer)))
    (should (equal (plist-get first :prompt) "COPY: a\nOUTPUT:\n"))
    (should (equal (plist-get first :completion) "a\n"))
    (should (equal (plist-get last :prompt) "COPY: ccc\nOUTPUT:\n"))
    (should (equal (plist-get last :completion) "ccc\n"))
    (should (= (length (plist-get encoded :trajectories)) 39))
    (should (cl-every (lambda (trajectory) (<= (length trajectory) 32))
                      (plist-get encoded :trajectories)))
    (should (equal (plist-get encoded :dataset-sha256)
                   (plist-get repeat :dataset-sha256)))
    (should (equal (plist-get encoded :trajectories)
                   (plist-get repeat :trajectories)))))

(ert-deftest nl-llm-learn-literal-copy-greedy-decode-is-unrestricted ()
  (let* ((calls nil)
         (step (lambda (token)
                 (push token calls)
                 (let ((logits (make-vector 256 -1.0)))
                   (cond
                    ((= token 65) (aset logits 0 0.0))
                    ((= token 0) (aset logits 255 0.0))
                    ((= token 255) (aset logits 10 0.0)))
                   logits)))
         (decoded
          (nl-llm-learn-literal-copy-greedy-decode (list 65) step)))
    (should (equal (vconcat (plist-get decoded :ids)) [0 255 10]))
    (should (plist-get decoded :terminated))
    (should (equal calls '(255 0 65)))))

(ert-deftest nl-llm-learn-literal-copy-greedy-decode-honors-limit-without-newline ()
  (let ((decoded
         (nl-llm-learn-literal-copy-greedy-decode
          (list 65)
          (lambda (_token)
            (let ((logits (make-vector 256 0.0)))
              (aset logits 0 1.0)
              logits)))))
    (should (= (length (plist-get decoded :ids)) 8))
    (should (equal (vconcat (plist-get decoded :ids))
                   [0 0 0 0 0 0 0 0]))
    (should-not (plist-get decoded :terminated))))

(ert-deftest nl-llm-learn-literal-copy-rejects-nonfinite-logits ()
  (dolist (bad
           (list (string-to-number "1e999")
                 (- (string-to-number "1e999") (string-to-number "1e999"))))
    (should-error
     (nl-llm-learn-literal-copy-greedy-decode
      (list 65)
      (lambda (_token)
        (let ((logits (make-vector 256 0.0)))
          (aset logits 0 bad)
          logits))))))

(ert-deftest nl-llm-learn-literal-copy-native-tiny-model-smoke ()
  (let* ((model
          (nl-llm-agent-improve-model
           2 2 256 1 1 nl-llm-learn-literal-copy-tokenizer))
         (decoded
          (nl-llm-learn-literal-copy-native-decode model (list 65)))
         (ids (plist-get decoded :ids)))
    (should (vectorp ids))
    (should (<= 1 (length ids)))
    (should (<= (length ids) nl-llm-learn-literal-copy-max-decode))
    (should (cl-every (lambda (id) (and (integerp id) (<= 0 id) (< id 256)))
                      (append ids nil)))
    (should (booleanp (plist-get decoded :terminated)))
    (should (eq (plist-get decoded :terminated)
                (= (aref ids (1- (length ids))) 10)))))

(ert-run-tests-batch-and-exit)

;;; literal-copy-test.el ends here
