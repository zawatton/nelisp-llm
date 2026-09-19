;;; agent-recur-artifact-test.el --- recurrent artifact provider tests -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'ert)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-recur-artifact)
(require 'nl-llm-agent-recur-provider)
(require 'nl-llm-agent-recur-supervised)

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../../nelisp-agent/lisp" here)))
(require 'nl-agent-service)

(defun nl-llm-agent-recur-artifact-test--last-logits (model tokens)
  "Return deterministic last-row logits for MODEL and TOKENS."
  (let ((photon-autograd--tape nil)
        (dim (plist-get model :dim)))
    (let* ((result
            (nl-llm-recur-forward
             model tokens 2 :k 2
             :s0
             (photon-autograd-const
              (photon-tensor
               (list (length tokens) dim)
               (nl-llm-recur-randn (* (length tokens) dim)
                                    (plist-get model :sigma) 1)))))
           (tensor (pav-value (plist-get result :logits)))
           (shape (photon-tensor-shape tensor))
           (data (photon-tensor-data tensor))
           (vocab (car (photon-tensor-shape
                        (pav-value (plist-get model :wte)))))
           (start (* (1- (length tokens)) vocab))
           (out (make-vector vocab 0.0)))
      (should (equal shape (list (length tokens) vocab)))
      (dotimes (index vocab)
        (aset out index (aref data (+ start index))))
      out)))

(defun nl-llm-agent-recur-artifact-test--parameter-data (model)
  "Return all MODEL parameter scalars as a detached list."
  (apply #'append
         (mapcar (lambda (parameter)
                   (append (photon-tensor-data (pav-value parameter)) nil))
                 (nl-llm-recur-params model))))

(defun nl-llm-agent-recur-artifact-test--all-zero-grad-p (model)
  "Return non-nil if all gradients in MODEL are zero."
  (cl-every
   (lambda (parameter)
     (cl-every (lambda (value) (= value 0.0))
               (append (photon-tensor-data (pav-grad parameter)) nil)))
   (nl-llm-recur-params model)))

(defun nl-llm-agent-recur-artifact-test--max-difference (left right)
  "Return the maximum absolute difference between equal-length vectors."
  (should (= (length left) (length right)))
  (let ((difference 0.0)
        (index 0))
    (while (< index (length left))
      (setq difference
            (max difference
                 (abs (- (aref left index) (aref right index)))))
      (setq index (1+ index)))
    difference))

(defconst nl-llm-agent-recur-artifact-test--digest
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

(defun nl-llm-agent-recur-artifact-test--model (seed)
  "Build a tiny recurrent model for provider tests."
  (nl-llm-recur-model-new
   :vocab 256 :dim 4 :heads 1 :kv-heads 1 :ff 8
   :n-prelude 1 :n-core 1 :n-coda 1 :seed seed :sigma 0.2))

(defun nl-llm-agent-recur-artifact-test--bundle (seed)
  "Build a fresh validated-looking bundle for SEED."
  (list :model (nl-llm-agent-recur-artifact-test--model seed)
        :tokenizer "utf8-byte-v1" :r 1 :s0-seed 104729 :step 7
        :family 'recurrent-depth
        :sha256 nl-llm-agent-recur-artifact-test--digest))

(defun nl-llm-agent-recur-artifact-test--spec (id path grammar &optional maxseq)
  (list :id id :path path
        :sha256 nl-llm-agent-recur-artifact-test--digest
        :grammar grammar :maxseq (or maxseq 128)))

(defun nl-llm-agent-recur-artifact-test--grammar (_emitted)
  '(:force ?A))

(ert-deftest nl-llm-agent-recur-artifact-provider-validates-and-hides-specs ()
  (let* ((path (concat "relative/" (make-string 12 ?x) ".sexp"))
         (grammar #'nl-llm-agent-recur-artifact-test--grammar)
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" path grammar 64))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)
                                                 "Recurrent artifacts"))
         (registry (nl-llm-agent-provider-registry-new)))
    ;; Caller-owned strings and grammar are not part of the public catalog.
    (nl-llm-agent-provider-register registry provider)
    (let ((public (car (nl-llm-agent-provider-models registry "recur"))))
      (should (equal (plist-get public :qualified-id) "recur/tiny"))
      (should (equal (plist-get public :name) "tiny"))
      (should (= (plist-get public :maxseq) 64))
      (should (equal (plist-get public :capabilities)
                     '(generate local constrained-decoding recurrent-depth)))
      (should-not (plist-member public :path))
      (should-not (plist-member public :sha256))
      (should-not (plist-member public :grammar)))
    (aset path 0 ?z)
    ;; The normalized path remains detached from the caller's mutable string.
    (should-error
     (nl-llm-agent-recur-provider
      "bad" (list (nl-llm-agent-recur-artifact-test--spec
                   "same" "a" grammar)
                   (nl-llm-agent-recur-artifact-test--spec
                    "same" "b" grammar))))
    (should-error
     (nl-llm-agent-recur-provider
      "bad" (list (append spec '(:unknown 1)))))
    (should-error
     (nl-llm-agent-recur-provider
      "bad" (list (append spec (list :id "tiny")))))))

(ert-deftest nl-llm-agent-recur-artifact-provider-opens-fresh-pinned-bundles ()
  (let* ((grammar #'nl-llm-agent-recur-artifact-test--grammar)
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" "artifact.sexp"
                                                        grammar))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (calls nil))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (path digest)
                 (push (list path digest) calls)
                 (nl-llm-agent-recur-artifact-test--bundle
                  (length calls)))))
      (let* ((first (nl-llm-agent-session-open registry "recur/tiny"))
             (second (nl-llm-agent-session-open registry "recur/tiny"))
             (one (nl-llm-agent-session-backend-state first))
             (two (nl-llm-agent-session-backend-state second)))
        (should (= (length calls) 2))
        (should (equal (cadar calls) nl-llm-agent-recur-artifact-test--digest))
        (should (equal (plist-get (plist-get one :bundle) :family)
                       'recurrent-depth))
        (should (not (eq (plist-get (plist-get one :bundle) :model)
                         (plist-get (plist-get two :bundle) :model))))
        (should (= (plist-get (plist-get one :bundle) :r) 1))
        (should (= (plist-get (plist-get one :bundle) :s0-seed) 104729))
        (should (= (plist-get (plist-get one :bundle) :step) 7))
        (should (equal (plist-get (plist-get one :bundle) :sha256)
                       nl-llm-agent-recur-artifact-test--digest))
        (should-error
         (nl-llm-agent-session-open
          registry "recur/tiny" :options '(:maxseq 0)))
        (should-error
         (nl-llm-agent-session-open
          registry "recur/tiny" :options '(:maxseq 64 :extra t)))))))

(ert-deftest nl-llm-agent-recur-artifact-provider-rejects-bad-bundles ()
  (let* ((grammar #'nl-llm-agent-recur-artifact-test--grammar)
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" "artifact.sexp"
                                                        grammar))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (bundle (nl-llm-agent-recur-artifact-test--bundle 1)))
    (nl-llm-agent-provider-register registry provider)
    (dolist (bad
             (list (plist-put (copy-tree bundle t) :family 'p5)
                   (plist-put (copy-tree bundle t) :sha256
                              (make-string 64 ?b))
                   (plist-put (copy-tree bundle t) :r 0)
                   (plist-put (copy-tree bundle t) :step -1)))
      (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
                 (lambda (_path _digest) bad)))
        (should-error (nl-llm-agent-session-open registry "recur/tiny"))))))

(ert-deftest nl-llm-agent-recur-artifact-provider-recomputes-full-prefix ()
  (let* ((grammar
          (lambda (emitted) (if (string-empty-p emitted)
                                '(:force ?A)
                              :stop)))
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" "artifact.sexp"
                                                        grammar 64))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (prefix-lengths nil) (forward-ks nil))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (nl-llm-agent-recur-artifact-test--bundle 3)))
              ((symbol-function 'nl-llm-recur-forward)
               (lambda (_model tokens r &rest keys)
                 (push (length tokens) prefix-lengths)
                 (push (plist-get keys :k) forward-ks)
                 (list :logits
                       (photon-autograd-const
                        (photon-tensor (list (length tokens) 256)
                                       (make-vector (* (length tokens) 256)
                                                    0.0)))
                       :states nil :e nil :r r))))
      (let* ((session (nl-llm-agent-session-open registry "recur/tiny"))
             (out (nl-llm-agent-session-complete
                   session '((user . "hello")))))
        (should (equal out "A"))
        (should (= (length prefix-lengths) 2))
        (should (= (car prefix-lengths) (1+ (cadr prefix-lengths))))
        (should (cl-every (lambda (k) (= k 0)) forward-ks))))))

(ert-deftest nl-llm-agent-recur-artifact-provider-service-switch-preserves-history ()
  (let* ((grammar (lambda (emitted) (if (string-empty-p emitted)
                                        '(:force ?A) :stop)))
         (specs (list
                 (nl-llm-agent-recur-artifact-test--spec "one" "one.sexp"
                                                          grammar)
                 (nl-llm-agent-recur-artifact-test--spec "two" "two.sexp"
                                                          grammar)))
         (provider (nl-llm-agent-recur-provider "recur" specs))
         (registry (nl-llm-agent-provider-registry-new))
         (load-count 0))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (setq load-count (1+ load-count))
                 (nl-llm-agent-recur-artifact-test--bundle load-count))))
      (let ((service (nl-agent-service-new registry "recur/one")))
        (should (equal (nl-agent-service-send service "hi") "A"))
        (nl-llm-agent-session-switch
         (nl-agent-service-session service) "recur/two")
        (should (equal (nl-agent-service-current-model service) "recur/two"))
        (should (= (length (nl-agent-service-messages service)) 2))
        ;; A failed switch is audited and leaves the current session usable.
        (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
                   (lambda (&rest _args) (error "corrupt recurrent artifact"))))
          (should-error
           (nl-llm-agent-session-switch
            (nl-agent-service-session service) "recur/one")))
        (should (equal (nl-agent-service-current-model service) "recur/two"))
        (should (eq (plist-get (car (nl-llm-agent-session-history
                                    (nl-agent-service-session service)))
                               :status)
                    'failed))))))

(ert-deftest nl-llm-agent-recur-artifact-roundtrip-preserves-model-and-loss ()
  "A saved recurrent model reloads with exact weights and fresh zero grads."
  (let* ((directory (make-temp-file "nl-llm-recur-artifact-test-" t))
         (path (expand-file-name "tiny.sexp" directory))
         (model (nl-llm-agent-recur-artifact-test--model 19))
         (examples [(:prompt "Q: " :completion "A\n")])
         (tokens (append (nl-llm-agent-tokenizer-encode "Q: " "utf8-byte-v1")
                         (nl-llm-agent-tokenizer-encode "A\n" "utf8-byte-v1")))
         (before-params (nl-llm-agent-recur-artifact-test--parameter-data model))
         (before-logits
          (nl-llm-agent-recur-artifact-test--last-logits
           model (butlast tokens)))
         (before-loss
          (nl-llm-agent-recur-supervised-loss
           model examples :r 2 :k 1 :s0-seed 1)))
    (unwind-protect
        (let* ((saved (nl-llm-agent-recur-artifact-save path model
                                                         :r 2 :s0-seed 1
                                                         :step 7))
               (bundle
                (nl-llm-agent-recur-artifact-load
                 path (plist-get saved :sha256)))
               (loaded (plist-get bundle :model))
               (after-logits
                (nl-llm-agent-recur-artifact-test--last-logits
                 loaded (butlast tokens)))
               (after-loss
                (nl-llm-agent-recur-supervised-loss
                 loaded examples :r 2 :k 1 :s0-seed 1)))
          (should (equal (plist-get bundle :family) 'recurrent-depth))
          (should (= (plist-get bundle :step) 7))
          (should (equal (plist-get bundle :sha256)
                         (plist-get saved :sha256)))
          (should (equal before-params
                         (nl-llm-agent-recur-artifact-test--parameter-data
                          loaded)))
          (should (< (nl-llm-agent-recur-artifact-test--max-difference
                      before-logits after-logits)
                     1.0e-12))
          (should (< (abs (- before-loss after-loss)) 1.0e-12))
          (should (nl-llm-agent-recur-artifact-test--all-zero-grad-p loaded))
          (aset (photon-tensor-data
                 (pav-value (plist-get loaded :wh))) 0 77.0)
          (should (= (car before-params)
                     (car (nl-llm-agent-recur-artifact-test--parameter-data
                           model)))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-recur-artifact-provider-service-real-files ()
  "Real saved artifacts work through send, switch, checkpoint, and restore."
  (let* ((directory (make-temp-file "nl-llm-recur-service-test-" t))
         (one-path (expand-file-name "one.sexp" directory))
         (two-path (expand-file-name "two.sexp" directory))
         (one (nl-llm-agent-recur-artifact-test--model 23))
         (two (nl-llm-agent-recur-artifact-test--model 29))
         (grammar (lambda (emitted)
                    (if (string-empty-p emitted)
                        (list :allow "ab")
                      :stop)))
         (service nil))
    ;; Make the constrained choice model-driven but deterministic by biasing
    ;; only the untied output head; the grammar still permits both choices.
    (aset (photon-tensor-data (pav-value (plist-get one :bh))) 97 100.0)
    (aset (photon-tensor-data (pav-value (plist-get one :bh))) 98 -100.0)
    (aset (photon-tensor-data (pav-value (plist-get two :bh))) 97 -100.0)
    (aset (photon-tensor-data (pav-value (plist-get two :bh))) 98 100.0)
    (unwind-protect
        (let* ((saved-one (nl-llm-agent-recur-artifact-save one-path one))
               (saved-two (nl-llm-agent-recur-artifact-save two-path two))
               (provider
                (nl-llm-agent-recur-provider
                 "recur"
                 (list
                  (list :id "one" :path one-path
                        :sha256 (plist-get saved-one :sha256)
                        :grammar grammar)
                  (list :id "two" :path two-path
                        :sha256 (plist-get saved-two :sha256)
                        :grammar grammar))))
               (registry (nl-llm-agent-provider-registry-new)))
          (nl-llm-agent-provider-register registry provider)
          (setq service (nl-agent-service-new registry "recur/one"))
          (should (equal (nl-agent-service-send service "hello") "a"))
          (nl-agent-service-switch service "recur/two")
          (should (equal (nl-agent-service-send service "again") "b"))
          (let ((checkpoint (nl-agent-service-checkpoint service)))
            (should (equal (plist-get checkpoint :model) "recur/two"))
            (nl-agent-service-restore service checkpoint)
            (should (equal (nl-agent-service-current-model service) "recur/two"))
            (should (equal (nl-agent-service-send service "restored") "b")))
          ;; Corrupting an inactive artifact makes a switch fail atomically.
          (with-temp-buffer
            (insert-file-contents-literally one-path)
            (goto-char (point-max))
            (insert "\n ")
            (write-region (point-min) (point-max) one-path nil 'silent))
          (should-error (nl-agent-service-switch service "recur/one"))
          (should (equal (nl-agent-service-current-model service) "recur/two"))
          (should (equal (nl-agent-service-messages service)
                         (list (cons 'user "hello") (cons 'assistant "a")
                               (cons 'user "again") (cons 'assistant "b")
                               (cons 'user "restored") (cons 'assistant "b"))))
          (should (equal (nl-agent-service-send service "retained") "b"))
          (should-not (equal (plist-get saved-one :sha256)
                             (secure-hash 'sha256
                                          (with-temp-buffer
                                            (insert-file-contents-literally one-path)
                                            (buffer-string)))))
          (should (stringp (plist-get saved-two :sha256))))
      (when (and service (nl-agent-service-p service))
        (condition-case nil
            (nl-agent-service-close service)
          (error nil)))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-recur-artifact-provider-rejects-overlong-prefix-before-commit ()
  "Prompt and generated-token bounds must not mutate service messages."
  (let* ((grammar (lambda (_emitted) '(:force ?A)))
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" "artifact.sexp"
                                                        grammar 8))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new)))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (nl-llm-agent-recur-artifact-test--bundle 4))))
      (let ((service (nl-agent-service-new registry "recur/tiny")))
        (should-error (nl-agent-service-send service "this prompt is too long"))
        (should-not (nl-agent-service-messages service))
        (nl-agent-service-close service)))))

(ert-deftest nl-llm-agent-recur-artifact-provider-rejects-generation-at-capacity ()
  "A completion token beyond maxseq is rejected before service commit."
  (let* ((grammar (lambda (_emitted) '(:force ?A)))
         (prompt (nl-llm-agent--render '((user . "hello"))))
         (capacity (1+ (length (nl-llm-agent-tokenizer-encode
                               prompt "utf8-byte-v1"))))
         (spec (nl-llm-agent-recur-artifact-test--spec
                "tiny" "artifact.sexp" grammar capacity))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (forwards 0))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (nl-llm-agent-recur-artifact-test--bundle 6)))
              ((symbol-function 'nl-llm-recur-forward)
               (let ((original (symbol-function 'nl-llm-recur-forward)))
                 (lambda (&rest args)
                   (setq forwards (1+ forwards))
                   (apply original args)))))
      (let ((service (nl-agent-service-new registry "recur/tiny")))
        (should-error (nl-agent-service-send service "hello"))
        (should (= forwards 2))
        (should-not (nl-agent-service-messages service))
        (nl-agent-service-close service)))))

(ert-deftest nl-llm-agent-recur-artifact-provider-rejects-utf8-token-overflow-before-forward ()
  "A multibyte prompt may fit the character bound but exceed token capacity."
  (let* ((grammar (lambda (_emitted) '(:force ?A)))
         (prompt (nl-llm-agent--render '((user . "é"))))
         (capacity (length prompt))
         (spec (nl-llm-agent-recur-artifact-test--spec
                "tiny" "artifact.sexp" grammar capacity))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (forwards 0))
    (should (> (length (nl-llm-agent-tokenizer-encode
                        prompt "utf8-byte-v1"))
               capacity))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (nl-llm-agent-recur-artifact-test--bundle 7)))
              ((symbol-function 'nl-llm-recur-forward)
               (let ((original (symbol-function 'nl-llm-recur-forward)))
                 (lambda (&rest args)
                   (setq forwards (1+ forwards))
                   (apply original args)))))
      (let ((service (nl-agent-service-new registry "recur/tiny")))
        (should-error (nl-agent-service-send service "é"))
        (should (= forwards 0))
        (should-not (nl-agent-service-messages service))
        (nl-agent-service-close service)))))

(ert-deftest nl-llm-agent-recur-artifact-provider-rejects-gpu-after-open ()
  "Dispatch changes after OPEN are rejected without invoking GPU work."
  (let* ((grammar (lambda (_emitted) '(:force ?A)))
         (spec (nl-llm-agent-recur-artifact-test--spec "tiny" "artifact.sexp"
                                                        grammar))
         (provider (nl-llm-agent-recur-provider "recur" (list spec)))
         (registry (nl-llm-agent-provider-registry-new))
         (calls 0)
         (gpu-linear (lambda (&rest _args)
                       (setq calls (1+ calls))
                       (error "unexpected GPU invocation"))))
    (nl-llm-agent-provider-register registry provider)
    (cl-letf (((symbol-function 'nl-llm-agent-recur-artifact-load)
               (lambda (_path _digest)
                 (nl-llm-agent-recur-artifact-test--bundle 5))))
      (let ((session (nl-llm-agent-session-open registry "recur/tiny")))
        ;; Install the swapped dispatch only after the artifact is opened.
        (cl-letf (((symbol-function 'photon-tensor-linear-gpu) gpu-linear)
                  ((symbol-function 'photon-tensor-linear) gpu-linear))
          (let ((caught nil))
            (condition-case err
                (nl-llm-agent-session-complete session '((user . "hello")))
              (error (setq caught err)))
            (should caught)
            (should (string-match-p "refuses an active GPU"
                                    (error-message-string caught)))
            (should (= calls 0))))))))

(ert-run-tests-batch-and-exit)

;;; agent-recur-artifact-test.el ends here
