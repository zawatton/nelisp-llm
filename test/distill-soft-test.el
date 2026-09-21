;;; distill-soft-test.el --- soft targets ride along without changing the data  -*- lexical-binding: t; -*-

;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/distill-soft-test.el
;;
;; Two properties matter and each has a control.  A soft dataset must accept
;; and reject exactly what the completion-only one does -- so the same prompts
;; and the same teacher text go through both and the reports are compared.  And
;; the vocabulary must count what a student would actually have to represent,
;; which means the alternatives, not just the sampled token.

(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-distill)
(require 'nl-llm-distill-soft)

(defvar ds--fail 0)
(defvar ds--pass 0)

(defun ds--ck (name ok &optional detail)
  (if ok (setq ds--pass (1+ ds--pass)) (setq ds--fail (1+ ds--fail)))
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" "FAIL") (or detail ""))))

(defun ds--alt (tok lp) (list :token tok :logprob lp))

(defun ds--answer (text &rest tokens)
  (list :text text :tokens tokens))

(defun ds--position (tok lp &rest alts)
  (append (ds--alt tok lp) (list :top (cons (ds--alt tok lp) alts))))

;; One teacher, two collectors.  The prompts deliberately include a duplicate
;; completion and an echo so the gates have something to do.
(defvar ds--prompts '("greet" "again" "echo me" "count"))

(defun ds--text (prompt)
  (pcase prompt
    ("greet" "Hi there")
    ("again" "Hi there")                ; duplicate
    ("echo me" "echo me")               ; echo
    ("count" "one two")
    (_ "")))

(defun ds--rich (prompt)
  (pcase prompt
    ("greet" (ds--answer "Hi there"
                         (ds--position "Hi" -0.1 (ds--alt "Hey" -2.0))
                         (ds--position " there" -0.3 (ds--alt " you" -1.5))))
    ("count" (ds--answer "one two"
                         (ds--position "one" -0.2 (ds--alt "1" -1.1))
                         (ds--position " two" -0.2 (ds--alt "Hi" -3.0))))
    (_ (ds--answer (ds--text prompt)))))

(let* ((plain (nl-llm-distill-collect ds--prompts #'ds--text))
       (soft  (nl-llm-distill-soft-collect ds--prompts #'ds--rich))
       (pr (cdr plain)) (sr (cdr soft)))
  (ds--ck "the same prompts are accepted and rejected"
          (and (= (nl-llm-distill-report-accepted pr)
                  (nl-llm-distill-report-accepted sr))
               (equal (nl-llm-distill-report-counts pr)
                      (nl-llm-distill-report-counts sr)))
          (nl-llm-distill-report-summary sr))
  (ds--ck "control: the gates did reject something"
          (> (nl-llm-distill-report-rejected sr) 0)
          (format "%d rejected, so equality above is not vacuous"
                  (nl-llm-distill-report-rejected sr)))
  (ds--ck "the completion-only pairs are identical"
          (equal (mapcar (lambda (e) (list (plist-get e :prompt)
                                           (plist-get e :completion)))
                         (car plain))
                 (mapcar (lambda (e) (list (plist-get e :prompt)
                                           (plist-get e :completion)))
                         (car soft))))
  (ds--ck "each accepted example keeps its own soft targets"
          (equal (mapcar (lambda (e) (mapcar (lambda (p) (plist-get p :token))
                                             (plist-get e :tokens)))
                         (car soft))
                 '(("Hi" " there") ("one" " two")))
          (nl-llm-distill-soft-summary (car soft)))

  ;; The vocabulary has to include the alternatives.  Counting only sampled
  ;; tokens would report 4 and a student sized from it could not represent
  ;; "Hey", so the soft target at that position would be unusable.
  (let* ((vocab (nl-llm-distill-soft-vocabulary (car soft)))
         (tokens (mapcar #'car vocab)))
    (ds--ck "the vocabulary counts alternatives too"
            (and (member "Hey" tokens) (member "1" tokens))
            (format "%d distinct: %S" (length vocab) tokens))
    (ds--ck "and a shared token is counted once, with its total"
            (equal (assoc "Hi" vocab) '("Hi" . 3))
            "Hi: sampled once, offered twice")
    (ds--ck "coverage of the whole vocabulary is 1.0"
            (= (nl-llm-distill-soft-coverage vocab (length vocab)) 1.0))
    (ds--ck "control: a prefix covers less"
            (< (nl-llm-distill-soft-coverage vocab 1) 1.0)
            (format "top-1 covers %.2f"
                    (nl-llm-distill-soft-coverage vocab 1))))

  ;; Position coverage, which is what a student is actually sized on: a
  ;; position whose sampled token is unrepresentable cannot be trained at all.
  (let ((full (nl-llm-distill-soft-position-coverage (car soft) 99))
        (one  (nl-llm-distill-soft-position-coverage (car soft) 1)))
    (ds--ck "the whole vocabulary serves every position"
            (and (= (plist-get full :positions) 4)
                 (= (plist-get full :sampled) 1.0)
                 (= (plist-get full :complete) 1.0))
            (format "%S" full))
    (ds--ck "control: a one-token vocabulary serves almost none"
            (< (plist-get one :sampled) 1.0)
            (format "sampled %.2f complete %.2f"
                    (plist-get one :sampled) (plist-get one :complete)))
    ;; The two fractions must be able to differ, or reporting both is theatre.
    ;; "Hi" is sampled at one position whose alternative "Hey" is rarer, so a
    ;; vocabulary holding "Hi" but not "Hey" serves it partially.
    (let* ((vocab (nl-llm-distill-soft-vocabulary (car soft)))
           (n (1+ (cl-position "Hi" (mapcar #'car vocab) :test #'equal)))
           (part (nl-llm-distill-soft-position-coverage (car soft) n)))
      (ds--ck "sampled and complete coverage differ where they should"
              (> (plist-get part :sampled) (plist-get part :complete))
              (format "at N=%d: sampled %.2f, complete %.2f"
                      n (plist-get part :sampled)
                      (plist-get part :complete)))))

  ;; A soft dataset must still read as an ordinary one.
  (let* ((path (make-temp-file "ds-" nil ".eld")))
    (unwind-protect
        (progn
          (nl-llm-distill-soft-write (car soft) path '(:teacher "stub"))
          (let ((back (nl-llm-distill-read path)))
            (ds--ck "the completion-only reader still accepts the file"
                    (= (length (plist-get back :examples)) 2)
                    (format "vocabulary-size %S" (plist-get back :vocabulary-size)))
            (ds--ck "and the soft targets survived the round trip"
                    (equal (plist-get (aref (plist-get back :examples) 0) :tokens)
                           (plist-get (car (car soft)) :tokens)))))
      (delete-file path))))

;; A teacher that returns plain strings must degrade, not fail: that is what a
;; server without logprobs looks like from here.
(let ((soft (nl-llm-distill-soft-collect '("greet") #'ds--text)))
  (ds--ck "a string-only teacher yields examples with no soft targets"
          (and (= (length (car soft)) 1)
               (null (plist-get (car (car soft)) :tokens)))
          (nl-llm-distill-soft-summary (car soft))))

(princ (format "\ndistill-soft: %d passed, %d failed\n" ds--pass ds--fail))
(kill-emacs (if (= ds--fail 0) 0 1))

;;; distill-soft-test.el ends here
