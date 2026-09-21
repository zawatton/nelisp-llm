;;; nl-llm-token-table.el --- load the donor's byte-level vocabulary  -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)

(defun nl-llm-token-table-load (path)
  "Read PATH and return a hash table mapping token surfaces to ids."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8-unix))
      (insert-file-contents path))
    (goto-char (point-min))
    (let* ((data (read (current-buffer)))
           (tokens (plist-get data :tokens))
           (table (make-hash-table :test #'equal)))
      (dotimes (i (length tokens) table)
        (let ((entry (aref tokens i)))
          (puthash (car entry) (cdr entry) table))))))

(defun nl-llm-token-table--alphabet ()
  "Return the complete byte-to-surface alphabet."
  (let ((direct (append (number-sequence 33 126)
                        (number-sequence 161 172)
                        (number-sequence 174 255)))
        (next 256) out)
    (dotimes (byte 256 (nreverse out))
      (if (memq byte direct)
          (push (cons byte byte) out)
        (push (cons byte (prog1 next (setq next (1+ next)))) out)))))

(defun nl-llm-token-table-key (bytes)
  "Map UTF-8 BYTES through GPT-2's byte-level alphabet."
  (let ((alphabet (nl-llm-token-table--alphabet)))
    (apply #'concat
           (mapcar (lambda (byte)
                     (char-to-string (alist-get byte alphabet)))
                   bytes))))

(defun nl-llm-token-table-id (table position-or-alt)
  "Return TABLE's id for POSITION-OR-ALT, or nil when it is absent."
  (let* ((token (plist-get position-or-alt :token))
         (bytes (or (plist-get position-or-alt :bytes)
                    (string-to-list (encode-coding-string token 'utf-8 t))))
         (surface (nl-llm-token-table-key bytes)))
    (gethash surface table)))

(provide 'nl-llm-token-table)
;;; nl-llm-token-table.el ends here
