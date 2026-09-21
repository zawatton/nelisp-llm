;;; nl-llm-distill-soft.el --- keep the teacher's distribution, not just its pick  -*- lexical-binding: t; -*-

;; Doc 08 Phase 3.  `nl-llm-distill' keeps the token the teacher sampled.  This
;; keeps the top K the teacher was choosing between, which is the same teacher
;; call carrying K times the supervision.
;;
;; Everything about acceptance is reused: a soft target is only worth keeping
;; when the completion it belongs to would have been kept anyway, so the gates,
;; the reasons and the report are `nl-llm-distill's, and a dataset written here
;; carries the (:prompt :completion) pairs the existing supervised path already
;; takes.  The soft targets ride alongside; a reader that does not know about
;; them sees an ordinary dataset.
;;
;; `nl-llm-distill-soft-vocabulary' is the reason this module reports rather
;; than just writes.  Soft targets are only usable by a student that shares the
;; teacher's token ids, and a student carrying a 151936-entry embedding is not
;; a small model at all.  What makes the two compatible is that a corpus uses a
;; small fraction of a vocabulary, and the fraction is a measurement, not an
;; assumption -- so the writer reports it and the student is sized from it.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-distill)
(require 'nl-llm-teacher-logprobs)

;;;###autoload
(defun nl-llm-distill-soft-collect (prompts teacher &optional progress)
  "Like `nl-llm-distill-collect', but TEACHER returns a logprob answer.

TEACHER is called with one prompt and returns either a plist from
`nl-llm-teacher-logprobs-ask' or a plain string; a string simply carries no
soft targets, so a mixed or downgraded teacher degrades instead of failing.

Returns (EXAMPLES . REPORT) where each example is
(:prompt P :completion C :tokens TOKENS), TOKENS nil when none were offered."
  (let ((soft (make-hash-table :test 'equal)))
    (cl-destructuring-bind (examples . report)
        (nl-llm-distill-collect
         prompts
         (lambda (prompt)
           (let ((answer (funcall teacher prompt)))
             (when answer
               (let ((text (nl-llm-teacher-logprobs-text answer)))
                 ;; Keyed by prompt: the gates work on text, and re-attaching
                 ;; by text would lose a completion that two prompts share.
                 (puthash prompt (and (not (stringp answer))
                                      (plist-get answer :tokens))
                          soft)
                 text))))
         progress)
      (cons (mapcar (lambda (e)
                      (append e (list :tokens
                                      (gethash (plist-get e :prompt) soft))))
                    examples)
            report))))

;;;###autoload
(defun nl-llm-distill-soft-vocabulary (examples)
  "Return the token strings EXAMPLES mention, most frequent first.

Each element is (TOKEN . COUNT), counting both the token the teacher sampled
and every alternative it offered -- a student has to be able to represent an
alternative or the soft target for that position is unusable."
  (let ((counts (make-hash-table :test 'equal)))
    (dolist (e examples)
      (dolist (position (plist-get e :tokens))
        (dolist (alt (cons position (plist-get position :top)))
          (let ((tok (plist-get alt :token)))
            (when tok
              (puthash tok (1+ (gethash tok counts 0)) counts))))))
    (let (out)
      (maphash (lambda (k v) (push (cons k v) out)) counts)
      (sort out (lambda (a b) (> (cdr a) (cdr b)))))))

;;;###autoload
(defun nl-llm-distill-soft-coverage (vocabulary n)
  "Return the fraction of VOCABULARY's mass the N most frequent tokens carry.
This is what says whether a pruned student vocabulary loses anything: a
coverage of 1.0 at N means no soft target ever refers to a token outside it."
  (let ((total (apply #'+ 0 (mapcar #'cdr vocabulary)))
        (head (apply #'+ 0 (mapcar #'cdr (cl-subseq vocabulary 0
                                                    (min n (length vocabulary)))))))
    (if (zerop total) 0.0 (/ (float head) total))))

;;;###autoload
(defun nl-llm-distill-soft-position-coverage (examples n)
  "Return how many positions a vocabulary of the N most frequent tokens serves.

Mass coverage is the wrong number to size a student on.  A position whose
*sampled* token falls outside the kept vocabulary cannot be trained on at all
-- there is no target to point at -- while one whose sampled token is inside
and a rare alternative outside can still be trained against a renormalised
target.  Those are different losses and they are reported separately.

Returns (:positions P :sampled F :complete F): P positions in all, F the
fraction whose sampled token is representable, and F the fraction whose whole
top-k is."
  (let* ((keep (let ((h (make-hash-table :test 'equal)))
                 (dolist (entry (cl-subseq (nl-llm-distill-soft-vocabulary
                                            examples)
                                           0 (min n (length (nl-llm-distill-soft-vocabulary
                                                             examples)))))
                   (puthash (car entry) t h))
                 h))
         (total 0) (sampled 0) (complete 0))
    (dolist (e examples)
      (dolist (position (plist-get e :tokens))
        (setq total (1+ total))
        (when (gethash (plist-get position :token) keep)
          (setq sampled (1+ sampled))
          (when (cl-every (lambda (alt) (gethash (plist-get alt :token) keep))
                          (plist-get position :top))
            (setq complete (1+ complete))))))
    (list :positions total
          :sampled (if (zerop total) 0.0 (/ (float sampled) total))
          :complete (if (zerop total) 0.0 (/ (float complete) total)))))

;;;###autoload
(defun nl-llm-distill-soft-summary (examples)
  "Return a one-line description of the soft targets in EXAMPLES."
  (let* ((with (cl-remove-if-not (lambda (e) (plist-get e :tokens)) examples))
         (positions (apply #'+ 0 (mapcar (lambda (e) (length (plist-get e :tokens)))
                                         with)))
         (vocab (nl-llm-distill-soft-vocabulary examples)))
    (format "%d of %d example(s) with soft targets, %d position(s), \
%d distinct token(s)"
            (length with) (length examples) positions (length vocab))))

;;;###autoload
(defun nl-llm-distill-soft-write (examples path &optional metadata)
  "Write EXAMPLES to PATH, soft targets and vocabulary included.

The file stays readable by `nl-llm-distill-read': `:examples' holds the same
(:prompt :completion) plists, with `:tokens' as an extra key the completion-only
path ignores.  `:vocabulary' is written out because sizing a student from it is
the first thing anyone does with this file, and recomputing it means reading
every soft target back."
  (let* ((vocab (nl-llm-distill-soft-vocabulary examples))
         (coding-system-for-write 'utf-8-unix)
         (print-level nil) (print-length nil))
    (make-directory (file-name-directory (expand-file-name path)) t)
    (with-temp-file path
      (insert ";; -*- lisp-data -*-  completion data with the teacher's top-k\n"
              ";; :examples is the vector nl-llm-agent-supervised accepts;\n"
              ";; :tokens on each is the teacher's distribution per position.\n")
      (prin1 (append metadata
                     (list :count (length examples)
                           :vocabulary-size (length vocab)
                           :vocabulary (vconcat vocab)
                           :examples (vconcat examples)))
             (current-buffer))
      (insert "\n"))
    path))

(provide 'nl-llm-distill-soft)
;;; nl-llm-distill-soft.el ends here
