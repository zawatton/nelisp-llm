;;; teacher-cache-test.el --- the cache must save calls, never answers  -*- lexical-binding: t; -*-

;;   emacs -Q --batch -l test/teacher-cache-test.el
;;
;; A cache is only worth having if a hit is indistinguishable from a call, so
;; every check here has a control: the saving is real (the teacher stops being
;; called), and the key is real (changing any part of the identity, or the
;; prompt, or the stored material, produces a miss rather than someone else's
;; answer).

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'nl-llm-teacher-cache)

(defvar tc--fail 0)
(defvar tc--pass 0)

(defun tc--ck (name ok &optional detail)
  (if ok (setq tc--pass (1+ tc--pass)) (setq tc--fail (1+ tc--fail)))
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(defun tc--counting (calls answer)
  "Return a teacher that counts into CALLS and answers ANSWER."
  (lambda (prompt) (setcar calls (1+ (car calls)))
    (if (functionp answer) (funcall answer prompt) answer)))

(let* ((dir (make-temp-file "tc-" t))
       (id '(:model "qwen3:4b" :base-url "http://x/v1" :temperature 0.0)))
  (unwind-protect
      (progn
        ;; A cold call reaches the teacher; the next one does not.
        (let* ((calls (list 0))
               (stats (nl-llm-teacher-cache-stats-new))
               (f (nl-llm-teacher-cache-wrap
                   (tc--counting calls "ANSWER") id dir stats))
               (a (funcall f "P"))
               (b (funcall f "P")))
          (tc--ck "cold call reaches the teacher" (equal a "ANSWER"))
          (tc--ck "a repeat is served from disk"
                  (and (equal b "ANSWER") (= (car calls) 1))
                  (format "teacher called %d time(s) for 2 asks" (car calls)))
          (tc--ck "and the stats say so"
                  (and (= (plist-get stats :hits) 1)
                       (= (plist-get stats :misses) 1))
                  (nl-llm-teacher-cache-report stats)))

        ;; Control: without the cache the same teacher is called twice.  Without
        ;; this, "called once" could mean the wrapper swallowed the second ask.
        (let* ((calls (list 0)) (f (tc--counting calls "ANSWER")))
          (funcall f "P") (funcall f "P")
          (tc--ck "control: an unwrapped teacher is called twice"
                  (= (car calls) 2)))

        ;; A different prompt is a different entry.
        (let* ((calls (list 0))
               (f (nl-llm-teacher-cache-wrap
                   (tc--counting calls (lambda (p) (concat "A:" p))) id dir)))
          (funcall f "P")                        ; already cached above
          (tc--ck "a different prompt misses"
                  (equal (funcall f "Q") "A:Q")))

        ;; The identity is load-bearing: every field of it must split the cache,
        ;; or a run would silently serve another model's answers.
        (let ((split t) (detail ""))
          (dolist (other (list (plist-put (copy-sequence id) :model "other:8b")
                               (plist-put (copy-sequence id) :base-url "http://y/v1")
                               (plist-put (copy-sequence id) :temperature 0.7)))
            (let* ((calls (list 0))
                   (f (nl-llm-teacher-cache-wrap
                       (tc--counting calls "OTHER") other dir)))
              (unless (equal (funcall f "P") "OTHER")
                (setq split nil
                      detail (format "%S served the first teacher's answer"
                                     other)))))
          (tc--ck "each identity field splits the cache" split
                  (if split "model, base-url and temperature all checked"
                    detail)))

        ;; An entry whose stored material does not match is a miss, not a hit.
        ;; This is what makes a stale or mixed cache slow rather than wrong.
        ;; The rewritten entry must be the one the *same* prompt hashes to,
        ;; or a miss would prove nothing but that another prompt was asked.
        (let* ((dir2 (make-temp-file "tc2-" t))
               (calls (list 0))
               (f (nl-llm-teacher-cache-wrap
                   (tc--counting calls "FRESH") id dir2)))
          (funcall f "P")                       ; one entry, for "P"
          (let ((path (car (directory-files-recursively dir2 "\\.eld\\'"))))
            (with-temp-file path
              (insert ";; -*- lisp-data -*-\n"
                      (prin1-to-string
                       (list :material '(:version 0 :identity nil :prompt "P")
                             :value "STALE"))))
            (tc--ck "a mismatched entry is a miss, not a stale answer"
                    (and (equal (funcall f "P") "FRESH") (= (car calls) 2))
                    (format "same prompt, teacher called %d times" (car calls)))
            ;; Control: an untouched entry of that same prompt *is* served, so
            ;; the miss above is the rewrite and not the prompt.
            (setcar calls 0)
            (tc--ck "control: the re-written entry now hits"
                    (and (equal (funcall f "P") "FRESH") (= (car calls) 0))))
          (delete-directory dir2 t))

        ;; A truncated file must not take the run down with it.
        (let* ((dir3 (make-temp-file "tc3-" t))
               (calls (list 0))
               (f (nl-llm-teacher-cache-wrap
                   (tc--counting calls "RECOVERED") id dir3)))
          (funcall f "T")
          (let ((path (car (directory-files-recursively dir3 "\\.eld\\'"))))
            (with-temp-file path (insert "(:material (:version"))
            (tc--ck "a truncated entry is a miss, not an error"
                    (and (equal (funcall f "T") "RECOVERED") (= (car calls) 2))))
          (delete-directory dir3 t))

        ;; A failed call must not be remembered as an answer.
        (let* ((calls (list 0))
               (f (nl-llm-teacher-cache-wrap
                   (lambda (_p) (setcar calls (1+ (car calls)))
                     (if (= (car calls) 1) nil "SECOND"))
                   id dir)))
          (funcall f "flaky")
          (tc--ck "a nil answer is not cached"
                  (equal (funcall f "flaky") "SECOND")
                  "the retry reached the teacher"))

        ;; A teacher that signals must be counted and re-raised.  Without the
        ;; count, a run where three prompts never reached the teacher reports
        ;; the other seven as the whole of it -- which is how a broken server
        ;; looked like a finished dataset.
        (let* ((stats (nl-llm-teacher-cache-stats-new))
               (f (nl-llm-teacher-cache-wrap
                   (lambda (_p) (error "teacher exploded")) id dir stats))
               (raised (condition-case err (progn (funcall f "boom") nil)
                         (error (error-message-string err)))))
          (tc--ck "a signalling teacher is re-raised"
                  (and raised (string-match-p "exploded" raised)))
          (tc--ck "and counted as an error, not a miss"
                  (and (= (plist-get stats :errors) 1)
                       (= (plist-get stats :misses) 0))
                  (nl-llm-teacher-cache-report stats)))

        ;; Plists survive, which is what logprob distillation will store.
        (let* ((calls (list 0))
               (value '(:text "hi" :logprobs ((:token "h" :logprob -0.1))))
               (f (nl-llm-teacher-cache-wrap
                   (tc--counting calls value) id dir)))
          (funcall f "structured")
          (tc--ck "a structured answer round-trips"
                  (and (equal (funcall f "structured") value)
                       (= (car calls) 1)))))
    (delete-directory dir t)))

(princ (format "\nteacher-cache: %d passed, %d failed\n" tc--pass tc--fail))
(kill-emacs (if (= tc--fail 0) 0 1))

;;; teacher-cache-test.el ends here
