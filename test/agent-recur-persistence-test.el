;;; agent-recur-persistence-test.el --- recurrent artifact persistence -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'ert)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-agent-supervised)

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here)))
(require 'nl-llm-agent-recur-artifact)
(require 'nl-llm-agent-recur-supervised)

(defvar read-eval)
(defvar read-circle)

(defun nl-llm-agent-recur-persistence-test--model ()
  "Build a small deterministic recurrent model for persistence tests."
  (nl-llm-recur-model-new
   :vocab 256 :dim 4 :heads 1 :kv-heads 1 :ff 8
   :n-prelude 1 :n-core 1 :n-coda 1 :seed 23 :sigma 0.2))

(defun nl-llm-agent-recur-persistence-test--snapshot (model)
  "Return values and gradients of MODEL as detached ordinary data."
  (mapcar
   (lambda (parameter)
     (list (copy-sequence (photon-tensor-shape (pav-value parameter)))
           (copy-sequence (photon-tensor-data (pav-value parameter)))
           (copy-sequence (photon-tensor-data (pav-grad parameter)))))
   (nl-llm-recur-params model)))

(defun nl-llm-agent-recur-persistence-test--examples ()
  "Return a fresh completion-only example set."
  [(:prompt "A " :completion "b\n")
   (:prompt "日本 " :completion "語\n")])

(defun nl-llm-agent-recur-persistence-test--loss (model)
  "Compute the deterministic loss used by round-trip assertions."
  (nl-llm-agent-recur-supervised-loss
   model (nl-llm-agent-recur-persistence-test--examples)
   :r 2 :k 1 :s0-seed 7 :tokenizer "utf8-byte-v1"))

(defun nl-llm-agent-recur-persistence-test--logits (model)
  "Compute deterministic logits for a fixed prefix and initial state."
  (let* ((tokens '(65 32))
         (s0 (nl-llm-agent-recur-supervised--s0
              (list :dim 4 :sigma 0.2) tokens 7)))
    (photon-tensor-data
     (pav-value
      (plist-get (nl-llm-recur-forward model tokens 2 :k 1 :s0 s0)
                 :logits)))))

(defun nl-llm-agent-recur-persistence-test--document ()
  "Return a fresh valid document."
  (nl-llm-agent-recur-artifact-export
   (nl-llm-agent-recur-persistence-test--model)
   :tokenizer "utf8-byte-v1" :r 2 :s0-seed 7 :step 11))

(defun nl-llm-agent-recur-persistence-test--write-bytes (path bytes)
  "Write unibyte BYTES to PATH for parser tests."
  (let ((coding-system-for-write 'binary))
    (write-region bytes nil path nil 'silent)))

(ert-deftest nl-llm-agent-recur-persistence-export-import-is-detached ()
  (let* ((model (nl-llm-agent-recur-persistence-test--model))
         (before (nl-llm-agent-recur-persistence-test--snapshot model))
         (document (nl-llm-agent-recur-artifact-export
                    model :tokenizer "utf8-byte-v1" :r 2 :s0-seed 7 :step 11))
         (bundle (nl-llm-agent-recur-artifact-import document))
         (imported (plist-get bundle :model))
         (raw (plist-get document :model))
         (raw-data (aref (plist-get raw :wte) 1))
         (format (plist-get document :format)))
    (should (equal format "nl-llm-recur-artifact-v1"))
    (should (equal (plist-get bundle :family) 'recurrent-depth))
    (should (= (plist-get bundle :step) 11))
    (should (not (eq model imported)))
    (should (equal before
                   (nl-llm-agent-recur-persistence-test--snapshot imported)))
    (should (cl-every
             (lambda (parameter)
               (cl-every (lambda (value) (= value 0.0))
                         (photon-tensor-data (pav-grad parameter))))
             (nl-llm-recur-params imported)))
    ;; Exported tensors and metadata do not alias the model or module constants.
    (let ((original (aref (photon-tensor-data (pav-value
                                               (car (nl-llm-recur-params model))))
                          0)))
      (aset raw-data 0 (+ original 1.0))
      (should (= original
                 (aref (photon-tensor-data (pav-value
                                            (car (nl-llm-recur-params model))))
                                           0))))
    (aset format 0 ?X)
    (should (equal nl-llm-agent-recur-artifact-format
                   "nl-llm-recur-artifact-v1"))))

(ert-deftest nl-llm-agent-recur-persistence-save-load-preserves-model-and-evaluation ()
  (let* ((model (nl-llm-agent-recur-persistence-test--model))
         (before (nl-llm-agent-recur-persistence-test--snapshot model))
         (loss-before (nl-llm-agent-recur-persistence-test--loss model))
         (logits-before (nl-llm-agent-recur-persistence-test--logits model))
         (directory (make-temp-file "recur-artifact-persist-" t))
         (path (expand-file-name "model.sexp" directory))
         saved loaded)
    (unwind-protect
        (progn
          (setq saved
                (nl-llm-agent-recur-artifact-save
                 path model :tokenizer "utf8-byte-v1" :r 2 :s0-seed 7 :step 11))
          (setq loaded
                (nl-llm-agent-recur-artifact-load
                 path (plist-get saved :sha256)))
          (let ((roundtrip (plist-get loaded :model)))
            (should (equal (plist-get loaded :sha256)
                           (plist-get saved :sha256)))
            (should (equal before
                           (nl-llm-agent-recur-persistence-test--snapshot
                            roundtrip)))
            (should (= loss-before
                       (nl-llm-agent-recur-persistence-test--loss roundtrip)))
            (should (equal logits-before
                           (nl-llm-agent-recur-persistence-test--logits roundtrip)))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-recur-persistence-hashes-before-parsing-and-disables-reader-eval ()
  (let* ((directory (make-temp-file "recur-artifact-reader-" t))
         (path (expand-file-name "model.sexp" directory))
         (document (nl-llm-agent-recur-persistence-test--document))
         (bytes (nl-llm-agent-recur-artifact--serialize document))
         (digest (secure-hash 'sha256 bytes))
         (parse-called nil)
         (read-seen nil)
         (original-read (symbol-function 'read))
         (original-parse
          (symbol-function 'nl-llm-agent-recur-artifact--parse-bytes)))
    ;; Start with permissive ambient reader flags; the artifact parser must
    ;; establish its own nil bindings before invoking `read'.
    (let ((read-eval t) (read-circle t))
      (unwind-protect
          (progn
            (nl-llm-agent-recur-persistence-test--write-bytes path bytes)
            (cl-letf (((symbol-function
                        'nl-llm-agent-recur-artifact--parse-bytes)
                       (lambda (&rest args)
                         (setq parse-called t)
                         (apply original-parse args)))
                      ((symbol-function 'read)
                       (lambda (&rest args)
                         (setq read-seen (list read-eval read-circle))
                         (apply original-read args))))
              (should-error
               (nl-llm-agent-recur-artifact-load
                path (make-string 64 ?0)))
              (should-not parse-called)
              (should
               (equal (nl-llm-agent-recur-artifact-load path digest)
                      (let ((expected
                             (nl-llm-agent-recur-artifact-import document)))
                        (plist-put expected :sha256 digest))))
              (should parse-called)
              (should (equal read-seen '(nil nil)))))
        (delete-directory directory t)))))

(ert-deftest nl-llm-agent-recur-persistence-rejects-malformed-data-before-pav-allocation ()
  (let ((bad-doc (nl-llm-agent-recur-persistence-test--document)))
    (should-error
     (nl-llm-agent-recur-artifact-import
      (append bad-doc '(:unknown 1))))
    (let ((duplicate (append (copy-sequence bad-doc)
                             (list :format (plist-get bad-doc :format)))))
      (should-error (nl-llm-agent-recur-artifact-import duplicate)))
    (let ((odd (append (copy-sequence bad-doc) '(:step))))
      (should-error (nl-llm-agent-recur-artifact-import odd)))
    (let ((dotted (cons :format (cons (plist-get bad-doc :format) 'broken))))
      (should-error (nl-llm-agent-recur-artifact-import dotted)))
    (let ((circular (copy-sequence bad-doc)))
      (setcdr (last circular) circular)
      (should-error (nl-llm-agent-recur-artifact-import circular)))
    (dolist (mutator
             (list
              (lambda (doc)
                (plist-put (plist-get doc :model) :family 'wrong))
              (lambda (doc)
                (let ((tensor (plist-get (plist-get doc :model) :wte)))
                  (aset tensor 0 '(256 5))
                  doc))
              (lambda (doc)
                (let ((tensor (plist-get (plist-get doc :model) :wte)))
                  (aset (aref tensor 1) 0 (string-to-number "1e+NaN"))
                  doc))
              (lambda (doc)
                (let ((tensor (plist-get (plist-get doc :model) :wte)))
                  (aset tensor 0 '(1 1000001))
                  doc))))
      (let ((doc (funcall mutator
                          (nl-llm-agent-recur-persistence-test--document)))
            (allocations 0)
            (original-const (symbol-function 'photon-autograd-const)))
        (cl-letf (((symbol-function 'photon-autograd-const)
                   (lambda (&rest args)
                     (setq allocations (1+ allocations))
                     (apply original-const args))))
          (should-error (nl-llm-agent-recur-artifact-import doc))
          (should (= allocations 0)))))))

(ert-deftest nl-llm-agent-recur-persistence-enforces-shared-bounds-before-allocation ()
  (let* ((model (nl-llm-agent-recur-persistence-test--model))
         (document (nl-llm-agent-recur-persistence-test--document))
         (directory (make-temp-file "recur-artifact-bound-" t))
         (path (expand-file-name "model.sexp" directory))
         (allocations 0)
         (original-const (symbol-function 'photon-autograd-const)))
    (unwind-protect
        (cl-letf (((symbol-value 'nl-llm-agent-recur-artifact-max-scalars) 10)
                  ((symbol-function 'photon-autograd-const)
                   (lambda (&rest args)
                     (setq allocations (1+ allocations))
                     (apply original-const args))))
          ;; The valid model is over the temporarily lowered aggregate bound;
          ;; neither import nor save may allocate or create the target.
          (should-error
           (nl-llm-agent-recur-artifact-import document))
          (should (= allocations 0))
          (should-error
           (nl-llm-agent-recur-artifact-save path model))
          (should-not (file-exists-p path)))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-recur-persistence-checks-file-bound-before-parser ()
  (let* ((directory (make-temp-file "recur-artifact-size-" t))
         (path (expand-file-name "model.sexp" directory))
         (bytes (nl-llm-agent-recur-artifact--serialize
                 (nl-llm-agent-recur-persistence-test--document)))
         (size (length bytes)))
    (unwind-protect
        (progn
          (nl-llm-agent-recur-persistence-test--write-bytes path bytes)
          (cl-letf (((symbol-value 'nl-llm-agent-recur-artifact-max-bytes)
                     size))
            (should (equal
                     (nl-llm-agent-recur-artifact--read-buffer path)
                     bytes)))
          (cl-letf (((symbol-value 'nl-llm-agent-recur-artifact-max-bytes)
                     (1- size)))
            (should-error
             (nl-llm-agent-recur-artifact--read-buffer path))))
      (delete-directory directory t))))

(ert-deftest nl-llm-agent-recur-persistence-rejects-reader-and-metadata-tampering ()
  (dolist (form
           (list "#.(error \"artifact evaluated\")"
                 "#1=(:format \"nl-llm-recur-artifact-v1\")"
                 "(:format \"nl-llm-recur-artifact-v1\") trailing"))
    (let* ((directory (make-temp-file "recur-artifact-bad-" t))
           (path (expand-file-name "bad.sexp" directory))
           (bytes (encode-coding-string (concat form "\n") 'utf-8))
           (digest (secure-hash 'sha256 bytes)))
      (unwind-protect
          (progn
            (nl-llm-agent-recur-persistence-test--write-bytes path bytes)
            (should-error
             (nl-llm-agent-recur-artifact-load path digest)))
        (delete-directory directory t))))
  (dolist (mutator
           (list
            (lambda (doc) (plist-put doc :format "wrong"))
            (lambda (doc) (plist-put doc :family 'p5))
            (lambda (doc) (plist-put doc :r 0))
            (lambda (doc) (plist-put doc :s0-seed -1))
            (lambda (doc) (plist-put doc :step -1))
            (lambda (doc) (let ((copy (copy-sequence doc)))
                            (setcdr (memq :model copy) nil)
                            copy))))
    (should-error
     (nl-llm-agent-recur-artifact-import
      (funcall mutator
               (nl-llm-agent-recur-persistence-test--document))))))

(ert-deftest nl-llm-agent-recur-persistence-save-does-not-overwrite-or-leak-temp-files ()
  (let* ((directory (make-temp-file "recur-artifact-save-" t))
         (path (expand-file-name "model.sexp" directory))
         (model (nl-llm-agent-recur-persistence-test--model))
         (saved (nl-llm-agent-recur-artifact-save path model))
         (original (with-temp-buffer
                     (insert-file-contents-literally path)
                     (buffer-string)))
         (files-before (directory-files directory nil
                                         "\\`\\.nl-llm-recur-artifact-")))
    (unwind-protect
        (progn
          (should-error (nl-llm-agent-recur-artifact-save path model))
          (should-error
           (nl-llm-agent-recur-artifact-save
            path model :tokenizer "unsupported-tokenizer"))
          (should-error
           (nl-llm-agent-recur-artifact-load path (make-string 64 ?a)))
          (should (equal original
                         (with-temp-buffer
                           (insert-file-contents-literally path)
                           (buffer-string))))
          (should (equal files-before
                         (directory-files directory nil
                                          "\\`\\.nl-llm-recur-artifact-")))
          (should (stringp (plist-get saved :sha256))))
      (delete-directory directory t))))

(ert-run-tests-batch-and-exit)

;;; agent-recur-persistence-test.el ends here
