;;; copy-teacher-forcing-test.el --- tests for teacher-forced COPY scores -*- lexical-binding: t; -*-

(require 'cl-lib)
(require 'ert)

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here))
  (load (expand-file-name "../examples/copy-teacher-forcing.el" here)
        nil nil t))

(declare-function nl-llm-agent-improve-model
                  "nl-llm-agent-improve" (&optional dim ff vocab nblocks heads tokenizer))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))
(declare-function nl-llm-copy-teacher-forcing-score
                  "copy-teacher-forcing" (model examples))

(defun nl-llm-copy-teacher-forcing-test--native ()
  (list :config '(:vocab 256)
        :blocks '(block) :wte 'wte :wh 'wh :lnfg 'lnfg :bh 'bh
        :dim 4 :heads 1 :kvh 1 :tokenizer "utf8-byte-v1"))

(defmacro nl-llm-copy-teacher-forcing-test--with-stubs (&rest body)
  `(cl-letf (((symbol-function 'nl-llm-gpu-available-p) (lambda () nil))
             ((symbol-function 'nl-llm-learn-literal-copy--native-model)
              (lambda (_model)
                (nl-llm-copy-teacher-forcing-test--native)))
             ((symbol-function 'nl-llm-learn-literal-copy--model-hash)
              (lambda (_model) "unchanged"))
             ((symbol-function 'nl-llm-inference-runtime-prepare)
              (lambda (&optional _mode) nil))
             ((symbol-function 'nl-llm-dcache-new)
              (lambda (&rest _args) 'cache)))
     ,@body))

(ert-deftest nl-llm-copy-teacher-forcing-gold-prefix-and-metrics ()
  (let ((calls nil) (decode-count 0))
    (nl-llm-copy-teacher-forcing-test--with-stubs
     (cl-letf (((symbol-function 'nl-llm-decode-step)
                (lambda (token &rest _args)
                  (push token calls)
                  (let ((logits (make-vector 256 0.0))
                        (target (nth decode-count '(97 98 10))))
                    (aset logits target 2.0)
                    (setq decode-count (1+ decode-count))
                    logits))))
       (let* ((result
               (nl-llm-copy-teacher-forcing-score
                'model [(:index 7 :length 2 :prompt "P" :completion "ab\n")]))
              (total (plist-get result :total))
              (case (aref (plist-get result :cases) 0))
              (expected-loss (- (log (+ 255.0 (exp 2.0))) 2.0)))
         ;; P is prefilled, then only gold a and b are consumed.  The future
         ;; newline is scored but never fed, and no model-generated token is
         ;; consulted.
         (should (equal (nreverse calls) '(80 97 98)))
         (should (= (plist-get total :tokens) 3))
         (should (= (plist-get total :correct) 3))
         (should (= (plist-get total :first-byte-correct) 1))
         (should (= (plist-get total :newline-correct) 1))
         (should (< (abs (- (plist-get total :mean-loss) expected-loss))
                    1.0e-12))
         (should (= (plist-get case :length) 2))
         (should (eq (plist-get result :mode) 'teacher-forced)))))))

(ert-deftest nl-llm-copy-teacher-forcing-validates-bounds-and-newline ()
  (nl-llm-copy-teacher-forcing-test--with-stubs
   (dolist (examples
            (list nil
                  '((:prompt "P" :completion "a\n"))
                  [(:prompt "" :completion "a\n")]
                  [(:prompt "P" :completion "a")]
                  [(:prompt "P" :completion "a\n\n")]
                  [(:prompt "P" :completion "a\nb")]
                  (vector (list :prompt (make-string 64 ?p)
                                :completion "a\n"))))
     (should-error
      (nl-llm-copy-teacher-forcing-score 'model examples)))))

(ert-deftest nl-llm-copy-teacher-forcing-rejects-invalid-logits ()
  (nl-llm-copy-teacher-forcing-test--with-stubs
   (cl-letf (((symbol-function 'nl-llm-decode-step)
              (lambda (&rest _args) (make-vector 2 0.0))))
     (should-error
      (nl-llm-copy-teacher-forcing-score
       'model [(:prompt "P" :completion "a\n")])))))

(ert-deftest nl-llm-copy-teacher-forcing-logsumexp-is-shift-invariant ()
  (let ((base nil) (shifted nil))
    (nl-llm-copy-teacher-forcing-test--with-stubs
     (cl-letf (((symbol-function 'nl-llm-decode-step)
                (lambda (&rest _args)
                  (let ((logits (make-vector 256 1.0)))
                    (aset logits 97 3.0)
                    logits))))
       (setq base
             (nl-llm-copy-teacher-forcing-score
              'model [(:prompt "P" :completion "a\n")]))))
    (nl-llm-copy-teacher-forcing-test--with-stubs
     (cl-letf (((symbol-function 'nl-llm-decode-step)
                (lambda (&rest _args)
                  (let ((logits (make-vector 256 1000000000000000.0)))
                    (aset logits 97 1000000000000002.0)
                    logits))))
       (setq shifted
             (nl-llm-copy-teacher-forcing-score
              'model [(:prompt "P" :completion "a\n")]))))
    (should (< (abs (- (plist-get (plist-get base :total) :mean-loss)
                       (plist-get (plist-get shifted :total) :mean-loss)))
               1.0e-12))))

(ert-deftest nl-llm-copy-teacher-forcing-real-cpu-native-model ()
  (let* ((model (nl-llm-agent-improve-model 4 4 256 1 1 "utf8-byte-v1"))
         (before (nl-llm-learn-literal-copy--model-hash model))
         (result
          (nl-llm-copy-teacher-forcing-score
           model [(:prompt "P" :completion "ab\n")]))
         (total (plist-get result :total)))
    (should (= (plist-get total :tokens) 3))
    (should (numberp (plist-get total :mean-loss)))
    (should (equal before (nl-llm-learn-literal-copy--model-hash model)))
    (should (plist-get result :model-unchanged))))

(ert-run-tests-batch-and-exit)

;;; copy-teacher-forcing-test.el ends here
