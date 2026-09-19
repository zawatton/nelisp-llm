;;; bench-recurrent-inference.el --- reproducible recurrent CPU benchmark -*- lexical-binding: t; -*-

;; This benchmark compares the recurrent artifact provider's source and
;; in-memory byte-code paths.  It never enables a tensor backend and never
;; writes an .elc file.
;;
;; Required environment:
;;   NELISP_RECUR_ARTIFACT  local recurrent artifact path
;;   NELISP_RECUR_SHA256    lowercase SHA-256 of that artifact
;; Optional environment:
;;   NELISP_RECUR_LENGTHS   comma-separated lengths, default "64,256"
;;   NELISP_RECUR_REPEATS   repetitions per mode/length, default 3 (max 5)
;;   NELISP_RECUR_OUTPUT    new JSON output path; otherwise a project-local
;;                          temporary directory is created.
;;
;; Example:
;;   NELISP_RECUR_ARTIFACT=./models/recur.sexp \
;;   NELISP_RECUR_SHA256=... \
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
;;     -l examples/bench-recurrent-inference.el

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'nl-llm-agent-recur-artifact)
(require 'nl-llm-agent-recur-provider)
(require 'nl-llm-inference-runtime)
(require 'photon-tensor)

(declare-function nl-llm-agent-recur-provider--logits
                  "nl-llm-agent-recur-provider" (state sequence))

(defconst nl-llm-bench-recurrent-inference--examples-directory
  (file-name-directory (or load-file-name buffer-file-name)))
(defconst nl-llm-bench-recurrent-inference--repository-directory
  (file-name-as-directory
   (expand-file-name ".."
                     nl-llm-bench-recurrent-inference--examples-directory)))
(defconst nl-llm-bench-recurrent-inference--default-lengths '(64 256))
(defconst nl-llm-bench-recurrent-inference--default-repeats 3)
(defconst nl-llm-bench-recurrent-inference--max-length 1024)
(defconst nl-llm-bench-recurrent-inference--max-repeats 5)

(defun nl-llm-bench-recurrent-inference--stable-sha (value)
  "Return a stable SHA-256 for data VALUE, including full float precision."
  (let ((print-length nil)
        (print-level nil)
        (print-circle nil)
        (print-escape-nonascii t)
        (float-output-format nil))
    (secure-hash 'sha256 (prin1-to-string value))))

(defun nl-llm-bench-recurrent-inference--file-sha (path)
  "Return the literal SHA-256 of local source PATH."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert-file-contents-literally path)
    (secure-hash 'sha256 (current-buffer))))

(defun nl-llm-bench-recurrent-inference--parse-positive (text name maximum)
  "Parse bounded positive integer TEXT for NAME, up to MAXIMUM."
  (unless (and (stringp text)
               (string-match-p "\\`[0-9]+\\'" text))
    (error "%s must be a positive decimal integer" name))
  (let ((value (string-to-number text)))
    (unless (and (> value 0) (<= value maximum))
      (error "%s must be in 1..%d" name maximum))
    value))

(defun nl-llm-bench-recurrent-inference--lengths ()
  "Read and validate the configured sequence lengths."
  (let ((raw (or (getenv "NELISP_RECUR_LENGTHS")
                 (mapconcat #'number-to-string
                            nl-llm-bench-recurrent-inference--default-lengths
                            ","))))
    (unless (> (length raw) 0)
      (error "NELISP_RECUR_LENGTHS must not be empty"))
    (let ((parts (split-string raw "," t)) result)
      (unless parts
        (error "NELISP_RECUR_LENGTHS contains no lengths"))
      (dolist (part parts (nreverse result))
        (push
         (nl-llm-bench-recurrent-inference--parse-positive
          (string-trim part) "NELISP_RECUR_LENGTHS"
          nl-llm-bench-recurrent-inference--max-length)
         result)))))

(defun nl-llm-bench-recurrent-inference--repeats ()
  "Read and validate the configured repetition count."
  (nl-llm-bench-recurrent-inference--parse-positive
   (or (getenv "NELISP_RECUR_REPEATS")
       (number-to-string nl-llm-bench-recurrent-inference--default-repeats))
   "NELISP_RECUR_REPEATS"
   nl-llm-bench-recurrent-inference--max-repeats))

(defun nl-llm-bench-recurrent-inference--artifact ()
  "Return validated `(PATH . SHA256)' artifact settings from the environment."
  (let ((raw-path (getenv "NELISP_RECUR_ARTIFACT"))
        (digest (getenv "NELISP_RECUR_SHA256")))
    (unless (and (stringp raw-path) (> (length raw-path) 0)
                 (not (file-remote-p raw-path)))
      (error "NELISP_RECUR_ARTIFACT must name a local file"))
    (unless (and (stringp digest)
                 (let ((case-fold-search nil))
                   (string-match-p
                    (rx string-start (= 64 (in "a-f0-9")) string-end)
                    digest)))
      (error "NELISP_RECUR_SHA256 must be lowercase hexadecimal SHA-256"))
    (let ((path (expand-file-name raw-path)))
      (unless (and (file-regular-p path) (not (file-directory-p path)))
        (error "NELISP_RECUR_ARTIFACT must name a regular file"))
      (cons path (substring-no-properties digest)))))

(defun nl-llm-bench-recurrent-inference--tokens (length vocab)
  "Return deterministic LENGTH token IDs for VOCAB."
  (let ((index 0) result)
    (while (< index length)
      (push (mod (+ 17 (* index 37)) vocab) result)
      (setq index (1+ index)))
    (nreverse result)))

(defun nl-llm-bench-recurrent-inference--vocab (model)
  "Return MODEL's vocabulary size from its embedding tensor."
  (let ((shape (photon-tensor-shape
                (pav-value (plist-get model :wte)))))
    (unless (and (listp shape) (integerp (car shape)) (> (car shape) 0))
      (error "recurrent artifact model has invalid vocabulary shape"))
    (car shape)))

(defun nl-llm-bench-recurrent-inference--model-sha (model)
  "Return a detached parameter fingerprint for recurrent MODEL."
  (let ((values nil))
    (dolist (parameter (nl-llm-recur-params model))
      (let ((tensor (pav-value parameter)))
        (push (list (copy-sequence (photon-tensor-shape tensor))
                    (copy-sequence (photon-tensor-data tensor)))
              values)))
    (nl-llm-bench-recurrent-inference--stable-sha (nreverse values))))

(defun nl-llm-bench-recurrent-inference--targets-source-p (sources)
  "Assert all runtime targets still have their original SOURCE bindings."
  (dolist (symbol nl-llm-inference-runtime--targets t)
    (unless (and (fboundp symbol)
                 (eq (symbol-function symbol) (cdr (assq symbol sources))))
      (error "runtime target %s is not the original source definition" symbol))))

(defun nl-llm-bench-recurrent-inference--targets-byte-code-p ()
  "Assert all runtime targets are byte-code functions."
  (dolist (symbol nl-llm-inference-runtime--targets t)
    (unless (and (fboundp symbol)
                 (byte-code-function-p (symbol-function symbol)))
      (error "runtime target %s is not byte-code" symbol))))

(defun nl-llm-bench-recurrent-inference--phase
    (mode bundle tokens repeats sources)
  "Run MODE against BUNDLE and TOKENS REPEATS times, returning raw samples."
  (let ((nl-llm-inference-runtime-mode mode)
        effective (setup-seconds 0.0))
    (setq effective
          (progn
            (garbage-collect)
            (let ((started (float-time)))
              (prog1 (nl-llm-inference-runtime-prepare)
                (setq setup-seconds (- (float-time) started))))))
    (unless (eq effective mode)
      (error "runtime selected %S while %S was requested" effective mode))
    (if (eq mode 'source)
        (nl-llm-bench-recurrent-inference--targets-source-p sources)
      (nl-llm-bench-recurrent-inference--targets-byte-code-p))
    (let ((samples nil) (state (list :bundle bundle)))
      (dotimes (_ repeats)
        (garbage-collect)
        (let* ((started (float-time))
               (logits
                (nl-llm-agent-recur-provider--logits state tokens))
               (elapsed (- (float-time) started)))
          (push (list :seconds elapsed
                      :sha256
                      (nl-llm-bench-recurrent-inference--stable-sha logits)
                      :logits logits)
                samples)))
      (setq samples (nreverse samples))
      (let ((first-sha (plist-get (car samples) :sha256)))
        (unless (cl-every
                 (lambda (sample)
                   (equal first-sha (plist-get sample :sha256)))
                 samples)
          (error "recurrent %s logits changed between repetitions" mode)))
      (if (eq mode 'source)
          (nl-llm-bench-recurrent-inference--targets-source-p sources)
        (nl-llm-bench-recurrent-inference--targets-byte-code-p))
      (list :mode mode :effective-mode effective
            :prepare-seconds setup-seconds :samples samples))))

(defun nl-llm-bench-recurrent-inference--phase-json (phase)
  "Convert raw PHASE data to a JSON-safe alist without logits vectors."
  (list (cons "mode" (symbol-name (plist-get phase :mode)))
        (cons "effective-mode" (symbol-name
                                 (plist-get phase :effective-mode)))
        (cons "prepare-seconds" (plist-get phase :prepare-seconds))
        (cons "samples"
              (vconcat
               (mapcar (lambda (sample)
                         (list (cons "seconds" (plist-get sample :seconds))
                               (cons "logits-sha256"
                                     (plist-get sample :sha256))))
                       (plist-get phase :samples))))))

(defun nl-llm-bench-recurrent-inference--maxdiff (left right)
  "Return maximum absolute difference between logit vectors LEFT and RIGHT."
  (unless (= (length left) (length right))
    (error "source and byte-code vocabulary sizes differ"))
  (let ((maximum 0.0) (index 0))
    (while (< index (length left))
      (setq maximum (max maximum
                         (abs (- (aref left index) (aref right index)))))
      (setq index (1+ index)))
    maximum))

(defun nl-llm-bench-recurrent-inference--output-path ()
  "Return `(PATH . CREATED-DIRECTORY)' for a new output destination."
  (let ((configured (getenv "NELISP_RECUR_OUTPUT")))
    (if configured
        (progn
          (unless (and (> (length configured) 0)
                       (not (file-remote-p configured)))
            (error "NELISP_RECUR_OUTPUT must be a nonempty local path"))
          (let* ((path (expand-file-name configured default-directory))
                 (directory (file-name-directory path)))
            (unless (file-directory-p directory)
              (error "benchmark output directory does not exist"))
            (when (or (file-exists-p path) (file-symlink-p path))
              (error "benchmark output refuses to overwrite %s" path))
            (cons path nil)))
      (let* ((directory
              (make-temp-file
               (expand-file-name ".recurrent-inference-bench-"
                                 nl-llm-bench-recurrent-inference--repository-directory)
               t)))
        (cons (expand-file-name "result.json" directory) directory)))))

(defun nl-llm-bench-recurrent-inference--write (path report)
  "Write REPORT as a new immutable JSON file at PATH atomically."
  (let ((temporary nil)
        (directory (file-name-directory path)))
    (when (or (file-exists-p path) (file-symlink-p path))
      (error "benchmark output refuses to overwrite %s" path))
    (unwind-protect
        (progn
          (setq temporary
                (make-temp-file
                 (expand-file-name ".result-" directory)))
          (let ((coding-system-for-write 'utf-8-unix))
            (write-region (json-encode report) nil temporary nil 'silent))
          (rename-file temporary path nil)
          (setq temporary nil))
      (when (and temporary (file-exists-p temporary))
        (delete-file temporary)))))

;;;###autoload
(defun nl-llm-bench-recurrent-inference-run ()
  "Run the source/byte-code recurrent CPU benchmark and write its JSON result."
  (let* ((artifact (nl-llm-bench-recurrent-inference--artifact))
         (lengths (nl-llm-bench-recurrent-inference--lengths))
         (repeats (nl-llm-bench-recurrent-inference--repeats))
         (destination (nl-llm-bench-recurrent-inference--output-path))
         (output (car destination))
         (created-directory (cdr destination))
         (bundle nil) (report nil) (completed nil))
    (unwind-protect
        (progn
          ;; Artifact verification and model allocation happen once, before
          ;; either phase.  The provider's private logits entry point is used
          ;; only after this pinned bundle has been loaded.
          (setq bundle
                (nl-llm-agent-recur-artifact-load (car artifact) (cdr artifact)))
          (let* ((model (plist-get bundle :model))
                 (vocab (nl-llm-bench-recurrent-inference--vocab model))
                 (model-before
                  (nl-llm-bench-recurrent-inference--model-sha model))
                 (sources nil) (length-reports nil))
            ;; A source baseline is mandatory: comparing two already compiled
            ;; definitions would not measure the requested modes.
            (let ((nl-llm-inference-runtime-mode 'source))
              (nl-llm-inference-runtime-prepare))
            (setq sources
                  (mapcar (lambda (symbol)
                            (cons symbol (symbol-function symbol)))
                          nl-llm-inference-runtime--targets))
            (nl-llm-bench-recurrent-inference--targets-source-p sources)
            (dolist (length lengths)
              (let* ((tokens
                      (nl-llm-bench-recurrent-inference--tokens length vocab))
                     (source
                      (let ((nl-llm-inference-runtime-mode 'source))
                        (nl-llm-bench-recurrent-inference--phase
                         'source bundle tokens repeats sources)))
                     (byte-code
                      (let ((nl-llm-inference-runtime-mode 'byte-code))
                        (nl-llm-bench-recurrent-inference--phase
                         'byte-code bundle tokens repeats sources)))
                     (source-samples (plist-get source :samples))
                     (byte-samples (plist-get byte-code :samples))
                     (maximum 0.0) (index 0))
                (while (< index repeats)
                  (let ((difference
                         (nl-llm-bench-recurrent-inference--maxdiff
                          (plist-get (nth index source-samples) :logits)
                          (plist-get (nth index byte-samples) :logits))))
                    (setq maximum (max maximum difference)))
                  (setq index (1+ index)))
                (unless (= maximum 0.0)
                  (error "source/byte-code parity failed at length %d" length))
                (push
                 (list (cons "length" length)
                       (cons "tokens-sha256"
                             (nl-llm-bench-recurrent-inference--stable-sha tokens))
                       (cons "source"
                             (nl-llm-bench-recurrent-inference--phase-json source))
                       (cons "byte-code"
                             (nl-llm-bench-recurrent-inference--phase-json byte-code))
                       (cons "max-absolute-logit-difference" maximum))
                 length-reports)))
            (setq length-reports (vconcat (nreverse length-reports)))
            (let ((model-after
                   (nl-llm-bench-recurrent-inference--model-sha model)))
              (unless (equal model-before model-after)
                (error "recurrent inference mutated artifact model weights"))
              (setq report
                    (list
                     (cons "format" "nl-llm-recurrent-inference-benchmark-v1")
                     (cons "artifact-sha256" (cdr artifact))
                     (cons "lengths" (vconcat lengths))
                     (cons "repeats" repeats)
                     (cons "settings"
                           (list (cons "runtime-targets"
                                       (vconcat
                                        (mapcar #'symbol-name
                                                nl-llm-inference-runtime--targets)))
                                 (cons "runtime-mode-pairs"
                                       ["source" "byte-code"])
                                 (cons "cpu-only" t)))
                     (cons "metadata"
                           (list (cons "emacs-version" emacs-version)
                                 (cons "system-type" (symbol-name system-type))
                                 (cons "machine" (system-name))
                                 (cons "core-count"
                                       (and (fboundp 'num-processors)
                                            (num-processors)))))
                     (cons "source-file-sha256"
                           (mapcar
                            (lambda (relative)
                              (cons relative
                                    (nl-llm-bench-recurrent-inference--file-sha
                                     (expand-file-name relative
                                                       nl-llm-bench-recurrent-inference--repository-directory))))
                            '("lisp/nl-llm-inference-runtime.el"
                              "lisp/nl-llm-agent-recur-provider.el"
                              "lisp/nl-llm-agent-recur-artifact.el")))
                     (cons "model-sha256-before" model-before)
                     (cons "model-sha256-after" model-after)
                     (cons "length-results" length-reports))))))
          (let ((nl-llm-inference-runtime-mode 'source))
            (nl-llm-inference-runtime-prepare))
          (nl-llm-bench-recurrent-inference--write output report)
          (setq completed t)
          (princ (format "wrote %s\n" output))
          report)
      ;; Restore the shared runtime even when validation, parity, or output
      ;; creation fails.  An output directory created for a failed run is only
      ;; an empty private directory and is removed here.
      (let ((nl-llm-inference-runtime-mode 'source))
        (ignore-errors (nl-llm-inference-runtime-prepare)))
      (unless completed
        (when (and created-directory (file-directory-p created-directory))
          (delete-directory created-directory t)))))

(when noninteractive
  (nl-llm-bench-recurrent-inference-run))

(provide 'bench-recurrent-inference)
;;; bench-recurrent-inference.el ends here
