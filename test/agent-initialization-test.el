;;; agent-initialization-test.el --- tests for opt-in model initialization -*- lexical-binding: t; -*-

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'ert)
(require 'cl-lib)

(defconst nl-llm-agent-initialization-test--here
  (file-name-directory (or load-file-name buffer-file-name)))


(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-initialization)

(declare-function nl-llm-agent--p5-params "nl-llm-agent-improve" (model))
(declare-function nl-llm-agent-improve-model
                  "nl-llm-agent-improve"
                  (&optional dim ff vocab nblocks heads tokenizer))
(declare-function pav-value "photon-autograd" (parameter))
(declare-function photon-tensor-shape "photon-tensor" (tensor))
(declare-function photon-tensor-data "photon-tensor" (tensor))

(defun nl-llm-agent-initialization-test--parameter-signature (model)
  (mapcar
   (lambda (parameter)
     (let ((tensor (pav-value parameter)))
       (list (copy-sequence (photon-tensor-shape tensor))
             (copy-sequence (photon-tensor-data tensor)))))
   (nl-llm-agent--p5-params model)))

(defun nl-llm-agent-initialization-test--model-hash (model)
  "Hash MODEL geometry and weights using the frozen probe's recipe."
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t) (float-output-format nil))
    (secure-hash
     'sha256
     (prin1-to-string
      (list :geometry
            (list :dim (plist-get model :dim)
                  :ff (plist-get model :ff)
                  :vocab (plist-get model :vocab)
                  :heads (plist-get model :heads)
                  :nblocks (plist-get model :nblocks)
                  :tokenizer (plist-get model :tokenizer))
            :weights
            (apply #'append
                   (mapcar
                    (lambda (parameter)
                      (append
                       (photon-tensor-data (pav-value parameter)) nil))
                    (nl-llm-agent--p5-params model))))))))

(ert-deftest nl-llm-agent-initialization-xorshift32-known-vector ()
  (let ((state #x1A2B3C4D)
        (expected '(3388403996 3984204854 2523572680 2838152257
                    3320909456))
        actual)
    (dotimes (_ 5)
      (setq state
            (nl-llm-agent-initialization--xorshift32 state))
      (push state actual))
    (should (equal (nreverse actual) expected))))

(ert-deftest nl-llm-agent-initialization-default-is-legacy-exact ()
  (let* ((constructor (symbol-function 'nl-llm-agent-improve-model))
         (legacy (nl-llm-agent-initialization-create
                  :dim 2 :ff 2 :vocab 256 :nblocks 1 :heads 1
                  :tokenizer "utf8-byte-v1"))
         (direct (nl-llm-agent-improve-model
                  2 2 256 1 1 "utf8-byte-v1")))
    (should (eq constructor (symbol-function 'nl-llm-agent-improve-model)))
    (should (equal legacy direct))
    (should (equal (nl-llm-agent-initialization-test--parameter-signature legacy)
                   (nl-llm-agent-initialization-test--parameter-signature direct)))))

(ert-deftest nl-llm-agent-initialization-no-args-default-is-legacy ()
  (let ((legacy (nl-llm-agent-initialization-create))
        (direct (nl-llm-agent-improve-model)))
    (should (equal legacy direct))))

(ert-deftest nl-llm-agent-initialization-xorshift32-reproducible-and-seed-sensitive ()
  (let ((left (nl-llm-agent-initialization-create
               :initializer 'xorshift32 :seed #x1A2B3C4D
               :dim 2 :ff 2 :vocab 256 :nblocks 1 :heads 1
               :tokenizer "utf8-byte-v1"))
        (right (nl-llm-agent-initialization-create
                :initializer 'xorshift32 :seed #x1A2B3C4D
                :dim 2 :ff 2 :vocab 256 :nblocks 1 :heads 1
                :tokenizer "utf8-byte-v1"))
        (other (nl-llm-agent-initialization-create
                :initializer 'xorshift32 :seed #x1A2B3C4E
                :dim 2 :ff 2 :vocab 256 :nblocks 1 :heads 1
                :tokenizer "utf8-byte-v1")))
    (should (equal (nl-llm-agent-initialization-test--parameter-signature left)
                   (nl-llm-agent-initialization-test--parameter-signature right)))
    (should-not (equal (nl-llm-agent-initialization-test--parameter-signature left)
                       (nl-llm-agent-initialization-test--parameter-signature other)))))

(ert-deftest nl-llm-agent-initialization-preserves-rank1-and-bounds-rank2 ()
  (let* ((model (nl-llm-agent-initialization-create
                 :initializer 'xorshift32 :seed #x1A2B3C4D
                 :dim 2 :ff 2 :vocab 256 :nblocks 1 :heads 1
                 :tokenizer "utf8-byte-v1"))
         (legacy (nl-llm-agent-improve-model
                  2 2 256 1 1 "utf8-byte-v1"))
         (scale (/ 1.0 (sqrt 2.0)))
         (parameters (nl-llm-agent--p5-params model))
         (rank1-before
          (mapcar
           (lambda (parameter)
             (let ((tensor (pav-value parameter)))
               (when (= (length (photon-tensor-shape tensor)) 1)
                 (copy-sequence (photon-tensor-data tensor)))))
           (nl-llm-agent--p5-params legacy))))
    (dolist (parameter parameters)
      (let ((tensor (pav-value parameter)))
        (if (= (length (photon-tensor-shape tensor)) 2)
            (dolist (value (append (photon-tensor-data tensor) nil))
              (should (and (numberp value) (< (abs value) scale)))))))
    (let ((rank1-after
           (mapcar
            (lambda (parameter)
              (let ((tensor (pav-value parameter)))
                (when (= (length (photon-tensor-shape tensor)) 1)
                  (photon-tensor-data tensor))))
            parameters)))
      (should (equal rank1-before rank1-after)))))

(ert-deftest nl-llm-agent-initialization-known-candidate-hash ()
  (let ((model (nl-llm-agent-initialization-create
                :initializer 'xorshift32 :seed #x1A2B3C4D
                :dim 32 :ff 64 :vocab 256 :nblocks 1 :heads 1
                :tokenizer "utf8-byte-v1")))
    (should (equal (nl-llm-agent-initialization-test--model-hash model)
                   "bc4ae38649deda046482b1d8eb70b369766cd2a848b905ba44bede75285629c3"))))

(ert-deftest nl-llm-agent-initialization-total-parameter-count-and-geometry ()
  (let* ((model (nl-llm-agent-initialization-create
                 :initializer 'xorshift32 :seed 1
                 :dim 32 :ff 64 :vocab 256 :nblocks 1 :heads 1
                 :tokenizer "utf8-byte-v1"))
         (total (cl-loop for parameter in (nl-llm-agent--p5-params model)
                         for tensor = (pav-value parameter)
                         sum (length (photon-tensor-data tensor)))))
    (should (= total 27264))
    (should (= (plist-get model :dim) 32))
    (should (= (plist-get model :ff) 64))
    (should (= (plist-get model :vocab) 256))
    (should (= (plist-get model :nblocks) 1))
    (should (= (plist-get model :heads) 1))))

(ert-deftest nl-llm-agent-initialization-validates-before-construction ()
  (let ((called nil))
    (cl-letf (((symbol-function 'nl-llm-agent-improve-model)
               (lambda (&rest _args) (setq called t) nil)))
      (dolist (arguments '((:initializer nope)
                           (:initializer legacy :seed 1)
                           (:initializer xorshift32)
                           (:initializer xorshift32 :seed 0)
                           (:initializer xorshift32 :seed -1)
                           (:initializer xorshift32 :seed #x100000000)
                           (:initializer xorshift32 :seed 1.0)
                           (:initializer legacy :unsupported t)))
        (setq called nil)
        (should-error (apply #'nl-llm-agent-initialization-create arguments))
        (should-not called)))))

(ert-run-tests-batch-and-exit)

;;; agent-initialization-test.el ends here
