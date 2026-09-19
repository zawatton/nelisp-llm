;;; nl-llm-agent-action-grammar.el --- bounded file action syntax -*- lexical-binding: t; -*-

;; This grammar constrains syntax only.  Paths, edits, and final answers remain
;; model choices from a fixed caller-supplied alphabet.

;;; Code:

(require 'cl-lib)

(defconst nl-llm-agent-action-grammar--default-allow
  (let ((chars nil) (char 32))
    (while (<= char 126)
      (push char chars)
      (setq char (1+ char)))
    (apply #'string (nreverse chars)))
  "Default printable-ASCII alphabet for file action fields.")

(defconst nl-llm-agent-action-grammar--escapes "nrt\"\\"
  "Characters accepted after a backslash in a quoted field.")

(defun nl-llm-agent-action-grammar--contains-p (string char)
  "Return non-nil when STRING contains CHAR."
  (let ((index 0) found)
    (while (and (< index (length string)) (not found))
      (when (= (aref string index) char)
        (setq found t))
      (setq index (1+ index)))
    found))

(defun nl-llm-agent-action-grammar--scalar-p (char multibyte)
  "Return non-nil when CHAR from a MULTIBYTE string is a Unicode scalar."
  (and (integerp char)
       (>= char 0)
       (<= char #x10ffff)
       (not (and (>= char #xd800) (<= char #xdfff)))
       (or multibyte (< char 128))))

(defun nl-llm-agent-action-grammar--allow (value)
  "Validate VALUE as a bounded printable alphabet and return a plain copy."
  (unless (stringp value)
    (error "File action grammar allow alphabet must be a string"))
  (unless (<= 1 (length value) 512)
    (error "File action grammar allow alphabet must have 1..512 characters"))
  (let ((multibyte (multibyte-string-p value))
        (index 0))
    (while (< index (length value))
      (let ((char (aref value index)))
        (unless (nl-llm-agent-action-grammar--scalar-p char multibyte)
          (error "File action grammar allow value is not a Unicode scalar: %S"
                 char))
        ;; Unicode Cc consists of the C0 and C1 ranges.  Quoted fields provide
        ;; explicit escapes for the useful controls instead.
        (when (or (< char 32) (and (>= char 127) (<= char 159)))
          (error "File action grammar allow value is a control: U+%04X" char)))
      (setq index (1+ index))))
  ;; Rebuilding from character values both detaches the caller's storage and
  ;; drops text properties, while remaining available on standalone NeLisp.
  (apply #'string (string-to-list value)))

(defun nl-llm-agent-action-grammar--without-quote-slash (allow)
  "Return ALLOW without quote and backslash structural characters."
  (let ((chars nil) (index 0))
    (while (< index (length allow))
      (let ((char (aref allow index)))
        (unless (or (= char ?\") (= char ?\\))
          (push char chars)))
      (setq index (1+ index)))
    (apply #'string (nreverse chars))))

;;;###autoload
(defun nl-llm-agent-grammar-file-actions (max-field &optional allow)
  "Return a bounded syntax grammar for read, edit, or DONE actions.

MAX-FIELD is the maximum decoded character count of every model-chosen field
and must be an integer in 1..1024.  ALLOW is a fixed alphabet of 1..512
printable, non-control Unicode scalar characters; it defaults to ASCII 32..126.
The returned function implements the existing EMITTED to `:stop', `(:force
CHAR)', or `(:allow CHARS)' grammar protocol.  It accepts exactly one of:

  DONE ANSWER\n
  ```tool\n(:name \"read\" :arguments (:path \"PATH\"))\n```
  ```tool\n(:name \"edit\" :arguments
             (:path \"PATH\" :search \"SEARCH\" :replace \"REPLACE\"))\n```

PATH and SEARCH are nonempty.  REPLACE and ANSWER may be empty.  Quoted fields
support standard escaped newline, carriage return, tab, quote, and backslash
forms; ANSWER uses ALLOW literally.
The closure is stateless and can be reused for independent generations."
  (unless (and (integerp max-field) (<= 1 max-field 1024))
    (error "File action grammar max-field must be an integer in 1..1024"))
  (let* ((safe (nl-llm-agent-action-grammar--allow
                (or allow nl-llm-agent-action-grammar--default-allow)))
         (normal (nl-llm-agent-action-grammar--without-quote-slash safe))
         (quoted-more (concat normal "\\"))
         (quoted-close (concat quoted-more "\""))
         (done-more (concat safe "\n"))
         ;; Three maximally escaped edit fields plus fixed syntax.  Rejecting
         ;; beyond this before replay keeps malformed inputs bounded.
         (max-emitted (+ 256 (* 6 max-field))))
    (lambda (emitted)
      (unless (stringp emitted)
        (error "File action grammar emitted value must be a string"))
      (when (> (length emitted) max-emitted)
        (error "File action grammar emitted value exceeds its bound"))
      (let ((position 0)
            (length (length emitted)))
        (cl-labels
            ((next (value)
               (throw 'grammar-result value))
             (force (text)
               (let ((index 0))
                 (while (< index (length text))
                   (if (= position length)
                       (next (list :force (aref text index)))
                     (unless (= (aref emitted position) (aref text index))
                       (error "Invalid file action prefix at character %d"
                              position))
                     (setq position (1+ position)
                           index (1+ index))))))
             (choose (choices)
               (if (= position length)
                   (next (list :allow (copy-sequence choices)))
                 (let ((char (aref emitted position)))
                   (unless
                       (nl-llm-agent-action-grammar--contains-p choices char)
                     (error "Invalid file action choice at character %d"
                            position))
                   (setq position (1+ position))
                   char)))
             (quoted-field (minimum)
               (let ((count 0))
                 (catch 'field-closed
                   (while t
                     (when (= position length)
                       (if (= count max-field)
                           (next (list :force ?\"))
                         (next
                          (list :allow
                                (copy-sequence
                                 (if (>= count minimum)
                                     quoted-close
                                   quoted-more))))))
                     (let ((char (aref emitted position)))
                       (cond
                        ((= char ?\")
                         (unless (>= count minimum)
                           (error "File action field is shorter than required"))
                         (setq position (1+ position))
                         (throw 'field-closed t))
                        ((= char ?\\)
                         (when (= count max-field)
                           (error "File action field exceeds max-field"))
                         (setq position (1+ position))
                         (if (= position length)
                             (next
                              (list :allow
                                    (copy-sequence
                                     nl-llm-agent-action-grammar--escapes)))
                           (unless
                               (nl-llm-agent-action-grammar--contains-p
                                nl-llm-agent-action-grammar--escapes
                                (aref emitted position))
                             (error "Invalid file action string escape"))
                           (setq position (1+ position)
                                 count (1+ count))))
                        (t
                         (when (= count max-field)
                           (error "File action field exceeds max-field"))
                         (unless
                             (nl-llm-agent-action-grammar--contains-p normal char)
                           (error "Invalid file action field character"))
                         (setq position (1+ position)
                               count (1+ count)))))))))
             (done-field ()
               (let ((count 0))
                 (catch 'done-closed
                   (while t
                     (when (= position length)
                       (if (= count max-field)
                           (next (list :force ?\n))
                         (next (list :allow (copy-sequence done-more)))))
                     (let ((char (aref emitted position)))
                       (if (= char ?\n)
                           (progn
                             (setq position (1+ position))
                             (throw 'done-closed t))
                         (when (= count max-field)
                           (error "DONE answer exceeds max-field"))
                         (unless
                             (nl-llm-agent-action-grammar--contains-p safe char)
                           (error "Invalid DONE answer character"))
                         (setq position (1+ position)
                               count (1+ count)))))))))
          (catch 'grammar-result
            (let ((branch (choose "D`")))
              (if (= branch ?D)
                  (progn
                    (force "ONE ")
                    (done-field))
                (force "``tool\n(:name \"")
                (let ((tool (choose "re")))
                  (if (= tool ?r)
                      (progn
                        (force "ead\" :arguments (:path \"")
                        (quoted-field 1)
                        (force "))\n```"))
                    (force "dit\" :arguments (:path \"")
                    (quoted-field 1)
                    (force " :search \"")
                    (quoted-field 1)
                    (force " :replace \"")
                    (quoted-field 0)
                    (force "))\n```")))))
            (if (= position length)
                :stop
              (error "Trailing data after complete file action"))))))))

(provide 'nl-llm-agent-action-grammar)
;;; nl-llm-agent-action-grammar.el ends here
