;;; nl-llm-distill.el --- build completion-only training data from a teacher  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 3.  Turns a locally served
;; open-weight teacher into a dataset the existing completion-only supervision
;; in `nl-llm-agent-supervised.el' accepts: a vector of (:prompt :completion),
;; prompts staying attention context while only completions carry loss.
;;
;; Licensing is the reason this takes a teacher function rather than a URL.  A
;; donor must be locally-run open weights (Apache-2.0 or MIT); outputs taken
;; from a hosted API may not be used to train a model, and the difference is a
;; matter of which endpoint was called, which the caller knows and this module
;; cannot.  `examples/distill-from-teacher.el' wires the Phase 0 provider in.
;;
;; The gates matter more than the generation.  A teacher's output is not data
;; until something has thrown the bad parts away, and each of these represents a
;; failure actually seen rather than a precaution:
;;
;;   empty      a reasoning model truncated mid-thought returns "" -- not an
;;              error, just a blank completion that would train on nothing
;;   echo       small instruct models sometimes hand the prompt back
;;   duplicate  repeated prompts or a degenerate sampler collapse the dataset
;;              to one example wearing many hats
;;   too-long   the supervised path caps a single example, so an over-long one
;;              would be rejected later, further from its cause
;;   budget     and caps the dataset, likewise
;;
;; Every rejection is counted and reported by reason.  A run that accepted 3 of
;; 200 prompts is a broken teacher, and that has to be visible without reading
;; the output file.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)

;; Mirrored from nl-llm-agent-supervised.el rather than required from it, so a
;; dataset can be built without loading the training path -- but kept identical
;; on purpose: rejecting here gives a message next to the prompt that caused it.
(defconst nl-llm-distill-max-examples 128
  "Most examples a dataset may hold, matching the supervised path.")
(defconst nl-llm-distill-max-example-chars 4096
  "Most characters one prompt and completion may total.")
(defconst nl-llm-distill-max-total-chars 65536
  "Most characters a whole dataset may total.")

(defconst nl-llm-distill-reasons
  '(empty echo duplicate too-long budget over-count error)
  "Why a candidate can be rejected.  `error' is the teacher signalling.")

(cl-defstruct (nl-llm-distill-report (:constructor nl-llm-distill-report--make))
  accepted    ; how many examples were kept
  rejected    ; how many candidates were thrown away
  counts      ; alist REASON -> count
  notes)      ; list of (REASON . PROMPT-PREFIX), newest first

(defun nl-llm-distill--normalize (s)
  "Return S folded for duplicate detection: trimmed, whitespace collapsed."
  (let ((out (replace-regexp-in-string "[ \t\n\r]+" " " (or s ""))))
    (downcase (string-trim out))))

(defun nl-llm-distill--bump (counts reason)
  "Return COUNTS with REASON incremented."
  (let ((cell (assq reason counts)))
    (if cell (progn (setcdr cell (1+ (cdr cell))) counts)
      (cons (cons reason 1) counts))))

;;;###autoload
(defun nl-llm-distill-gate (prompt completion seen total-chars count)
  "Return nil to accept, or the reason PROMPT/COMPLETION is rejected.
SEEN is a hash of normalized completions already accepted; TOTAL-CHARS and
COUNT are the dataset so far."
  (let* ((c (and (stringp completion) (string-trim completion)))
         (p (and (stringp prompt) (string-trim prompt)))
         (chars (+ (length (or prompt "")) (length (or completion "")))))
    (cond
     ((>= count nl-llm-distill-max-examples) 'over-count)
     ((or (null c) (string-empty-p c)) 'empty)
     ((or (null p) (string-empty-p p)) 'empty)
     ((equal (nl-llm-distill--normalize c) (nl-llm-distill--normalize p)) 'echo)
     ((gethash (nl-llm-distill--normalize c) seen) 'duplicate)
     ((> chars nl-llm-distill-max-example-chars) 'too-long)
     ((> (+ total-chars chars) nl-llm-distill-max-total-chars) 'budget)
     (t nil))))

;;;###autoload
(defun nl-llm-distill-collect (prompts teacher &optional progress)
  "Ask TEACHER for a completion to each of PROMPTS and keep what passes.
TEACHER is called with one prompt string and returns the completion text; it is
injected so a dataset can be built from any provider, and so the gates can be
tested without a model.  PROGRESS, when given, is called with (INDEX REASON),
REASON nil for an accepted example.

Returns (EXAMPLES . REPORT): EXAMPLES a list of (:prompt :completion) plists in
order, REPORT an `nl-llm-distill-report'."
  (let ((seen (make-hash-table :test 'equal))
        (examples nil) (counts nil) (notes nil)
        (total 0) (kept 0) (rejected 0) (index 0))
    (dolist (prompt prompts)
      (let* ((completion
              (condition-case err
                  (funcall teacher prompt)
                (error (push (cons 'error (format "%S" err)) notes) nil)))
             (reason (if (null completion) 'error
                       (nl-llm-distill-gate prompt completion seen total kept))))
        (if reason
            (progn
              (setq rejected (1+ rejected)
                    counts (nl-llm-distill--bump counts reason))
              (unless (eq reason 'error)
                (push (cons reason (substring prompt 0
                                              (min 40 (length prompt))))
                      notes)))
          (puthash (nl-llm-distill--normalize completion) t seen)
          (push (list :prompt (substring-no-properties prompt)
                      :completion (substring-no-properties
                                   (string-trim completion)))
                examples)
          (setq kept (1+ kept)
                total (+ total (length prompt) (length completion))))
        (when progress (funcall progress index reason))
        (setq index (1+ index))))
    (cons (nreverse examples)
          (nl-llm-distill-report--make
           :accepted kept :rejected rejected
           :counts (nreverse counts) :notes notes))))

;;;###autoload
(defun nl-llm-distill-report-summary (report)
  "Return a one-line human summary of REPORT."
  (format "%d accepted, %d rejected%s"
          (nl-llm-distill-report-accepted report)
          (nl-llm-distill-report-rejected report)
          (let ((c (nl-llm-distill-report-counts report)))
            (if (null c) ""
              (concat " ("
                      (mapconcat (lambda (cell)
                                   (format "%s %d" (car cell) (cdr cell)))
                                 c ", ")
                      ")")))))

;;;###autoload
(defun nl-llm-distill-write (examples path &optional metadata)
  "Write EXAMPLES to PATH as one sexp, with METADATA recorded alongside.
The file is a plist: :examples is the vector the supervised path takes, and the
rest is provenance -- which teacher, when, how many prompts were asked.  Kept
as a sexp so the training side needs no parser, and written with provenance
because a dataset whose origin is unrecorded cannot be reproduced or audited
for the licence it was collected under."
  (unless examples
    (error "nl-llm-distill-write: refusing to write an empty dataset"))
  (with-temp-buffer
    (insert ";; -*- lisp-data -*-  completion-only training data\n")
    (insert ";; Generated by nl-llm-distill from a locally served teacher.\n")
    (insert ";; :examples is the vector nl-llm-agent-supervised accepts.\n")
    (let ((print-length nil) (print-level nil))
      (prin1 (append metadata
                     (list :count (length examples)
                           :examples (vconcat examples)))
             (current-buffer)))
    (insert "\n")
    (let ((coding-system-for-write 'utf-8-unix))
      (write-region (point-min) (point-max) path nil 'silent)))
  path)

;;;###autoload
(defun nl-llm-distill-read (path)
  "Read the dataset at PATH and return its plist, :examples included."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8-unix))
      (insert-file-contents path))
    (goto-char (point-min))
    (let ((data (read (current-buffer))))
      (unless (vectorp (plist-get data :examples))
        (error "nl-llm-distill-read: %s has no :examples vector" path))
      data)))

(provide 'nl-llm-distill)
;;; nl-llm-distill.el ends here
