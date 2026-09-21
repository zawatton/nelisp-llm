;;; nl-llm-teacher-cache.el --- remember what the teacher already said  -*- lexical-binding: t; -*-

;; Doc 08 Phase 3.  A teacher call is the most expensive thing in a
;; distillation run and, at temperature 0, the most deterministic: the same
;; model asked the same prompt with the same options returns the same answer.
;; Paying for it twice is the single largest avoidable cost in the experiment
;; loop, because the loop is mostly *re-running* a dataset build after changing
;; something downstream of it.
;;
;; This wraps a teacher function rather than living inside `nl-llm-distill', so
;; the gates, the report and the dataset format are unchanged and a cached run
;; is indistinguishable from an uncached one except in how long it takes.
;;
;; What is stored is the answer *and the key material that produced it*.  A
;; cache that stores only the digest cannot tell a hash collision from a hit,
;; and -- far more likely in practice -- cannot tell that the entry was written
;; by a different model, a different base URL or a different temperature.  Every
;; read re-checks the material and treats a mismatch as a miss, so a stale or
;; mixed cache degrades to slowness rather than to wrong data.
;;
;; The value is a plist, not a string, because top-k logprob distillation needs
;; to keep more than the text and should not need a second cache.

;;; Code:

(require 'cl-lib)

(defconst nl-llm-teacher-cache-version 1
  "Bumped when the key scheme changes, which invalidates every entry.")

(defcustom nl-llm-teacher-cache-directory "build/teacher-cache"
  "Where cached teacher answers live, relative to `default-directory'."
  :type 'directory
  :group 'nl-llm-inference-runtime)

(defun nl-llm-teacher-cache--material (identity prompt)
  "Return the canonical key material for IDENTITY asking PROMPT.
IDENTITY names the teacher completely: model, endpoint and every option that
can change an answer.  It is written into the entry and re-checked on read."
  (list :version nl-llm-teacher-cache-version
        :identity identity
        :prompt prompt))

(defun nl-llm-teacher-cache--digest (material)
  (secure-hash 'sha256 (format "%S" material)))

(defun nl-llm-teacher-cache--path (dir digest)
  "Two-level fan-out, so a directory listing stays usable at 100k entries."
  (expand-file-name (concat (substring digest 0 2) "/" (substring digest 2)
                            ".eld")
                    dir))

(defun nl-llm-teacher-cache--read (path material)
  "Return the cached value at PATH when it was written for MATERIAL.
Any other outcome -- absent, unreadable, truncated, or written for different
material -- is nil, which the caller treats as a miss."
  (when (file-readable-p path)
    (condition-case nil
        (let ((entry (with-temp-buffer
                       (let ((coding-system-for-read 'utf-8-unix))
                         (insert-file-contents path))
                       (goto-char (point-min))
                       (read (current-buffer)))))
          (when (equal (plist-get entry :material) material)
            (plist-get entry :value)))
      (error nil))))

(defun nl-llm-teacher-cache--write (path material value)
  (make-directory (file-name-directory path) t)
  (let ((coding-system-for-write 'utf-8-unix)
        (print-level nil) (print-length nil))
    (with-temp-file path
      (insert ";; -*- lisp-data -*-  one cached teacher answer\n")
      (prin1 (list :material material :value value) (current-buffer))
      (insert "\n"))))

;;;###autoload
(defun nl-llm-teacher-cache-stats-new ()
  "Return a fresh mutable counter for `nl-llm-teacher-cache-wrap'."
  (list :hits 0 :misses 0 :errors 0))

;;;###autoload
(defun nl-llm-teacher-cache-wrap (teacher identity &optional dir stats)
  "Return TEACHER with a disk cache in front of it.

TEACHER takes a prompt and returns either a string or a plist; both are
stored as given and returned as given, so a caller that wants text keeps
getting text.  IDENTITY must name the teacher completely -- model, endpoint,
and every option that can change an answer -- because two teachers sharing an
IDENTITY would share answers.

DIR defaults to `nl-llm-teacher-cache-directory'.  STATS, when given, is a
plist from `nl-llm-teacher-cache-stats-new' and is updated in place."
  (unless (functionp teacher)
    (error "nl-llm-teacher-cache-wrap: TEACHER is not a function"))
  (let ((dir (or dir nl-llm-teacher-cache-directory)))
    (lambda (prompt)
      (let* ((material (nl-llm-teacher-cache--material identity prompt))
             (path (nl-llm-teacher-cache--path
                    dir (nl-llm-teacher-cache--digest material)))
             (hit (nl-llm-teacher-cache--read path material)))
        (if hit
            (progn (when stats
                     (plist-put stats :hits (1+ (plist-get stats :hits))))
                   hit)
          ;; A call that signals is counted and re-raised.  Counting it only
          ;; on the way out would leave a failed teacher invisible: a report of
          ;; "7 of 7 served" while three prompts never reached the teacher at
          ;; all reads as a complete run, and did.
          (let ((value (condition-case err
                           (funcall teacher prompt)
                         (error
                          (when stats
                            (plist-put stats :errors
                                       (1+ (plist-get stats :errors))))
                          (signal (car err) (cdr err))))))
            ;; A nil answer is not cached: it is the shape a failed call takes,
            ;; and remembering a failure forever is worse than repeating it.
            (when value
              (nl-llm-teacher-cache--write path material value))
            (when stats
              (plist-put stats :misses (1+ (plist-get stats :misses))))
            value))))))

;;;###autoload
(defun nl-llm-teacher-cache-report (stats)
  "Return a one-line summary of STATS."
  (let* ((h (plist-get stats :hits)) (m (plist-get stats :misses))
         (e (or (plist-get stats :errors) 0))
         (n (+ h m)))
    (format "teacher cache: %d hit, %d miss of %d (%.0f%% saved)%s"
            h m n (if (zerop n) 0.0 (* 100.0 (/ (float h) n)))
            (if (zerop e) "" (format ", %d call(s) failed" e)))))

(provide 'nl-llm-teacher-cache)
;;; nl-llm-teacher-cache.el ends here
