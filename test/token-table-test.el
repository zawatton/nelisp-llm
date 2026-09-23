;;; token-table-test.el --- donor byte-level vocabulary checks  -*- lexical-binding: t; -*-

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'nl-llm-token-table)

(defvar tt--fail 0)
(defvar tt--pass 0)

(defun tt--ck (name ok &optional detail)
  (if ok (setq tt--pass (1+ tt--pass)) (setq tt--fail (1+ tt--fail)))
  (princ (format "%-60s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(let ((okay (nl-llm-token-table-key '(79 107 97 121)))
      (space-i (nl-llm-token-table-key '(32 73))))
  (tt--ck "ASCII bytes map to their surface" (equal okay "Okay"))
  (tt--ck "space uses GPT-2's Ġ surface" (equal space-i "ĠI"))
  (tt--ck "control: the alphabet is not decoded text"
          (not (equal space-i " I"))))

(let ((path (expand-file-name "build/donor/qwen3-0.6b/tokenizer.json"))
      (table-path (expand-file-name "build/qwen-token-table.eld"))
      (data-path (expand-file-name "build/distilled-soft.eld")))
  (if (or (not (file-readable-p table-path))
          (not (file-readable-p data-path)))
      (princ (format "real token-table round trip: skip (file absent: %s)\n"
                     (if (file-readable-p path) "dataset or table" "donor")))
    (let ((table (nl-llm-token-table-load table-path))
          (data (with-temp-buffer
                  (insert-file-contents data-path)
                  (read (current-buffer))))
          (tokens (make-hash-table :test #'equal))
          unresolved)
      (dolist (example (append (plist-get data :examples) nil))
        (dolist (position (plist-get example :tokens))
          (dolist (alternative (cons position (plist-get position :top)))
            (puthash alternative t tokens))))
      (maphash (lambda (position _)
                 (unless (nl-llm-token-table-id table position)
                   (push (plist-get position :token) unresolved)))
               tokens)
      (let* ((total (hash-table-count tokens))
             (missing (length unresolved))
             (fraction (if (zerop total) 1.0
                         (/ (float (- total missing)) total))))
        (tt--ck "every token in the real dataset resolves"
                (> fraction 0.99)
                (format "%.6f (%d/%d), missing %S"
                        fraction (- total missing) total
                        (cl-subseq (nreverse unresolved) 0
                                   (min 5 missing))))
        (princ (format "resolve fraction: %.6f; %d distinct, %d unresolved\n"
                       fraction total missing)))
      (tt--ck "<|im_end|> resolves to 151645"
              (= (nl-llm-token-table-id
                  table (list :token "<|im_end|>" :bytes nil))
                 151645)))))

(princ (format "\ntoken-table: %d passed, %d failed\n" tt--pass tt--fail))
(kill-emacs (if (= tt--fail 0) 0 1))

;;; token-table-test.el ends here
