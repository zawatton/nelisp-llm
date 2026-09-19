;;; distill-test.el --- the gates on teacher-generated training data  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/distill-test.el
;;
;; Doc 08 Phase 3.  A teacher's output is not training data until something has
;; thrown the bad parts away, so the gates are what this suite is about: each
;; one is driven with an input that must trip it, and one that must not.
;;
;; The teacher is injected, so none of this needs a model.  That is also the
;; point of the injection -- the gates are the part most likely to be wrong and
;; the part least in need of a GPU to check.
;;
;; The last checks are the ones that matter most: whatever survives the gates
;; must be accepted by `nl-llm-agent-supervised--prepare', the real consumer.
;; Limits duplicated in two files drift, so they are compared rather than
;; trusted.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-distill)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-supervised)

(defvar dt--fail 0)
(defun dt--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq dt--fail (1+ dt--fail)) "FAIL"))
                 (or extra ""))))

(defun dt--collect (pairs)
  "Run `nl-llm-distill-collect' over PAIRS, an alist of (PROMPT . COMPLETION)."
  (nl-llm-distill-collect
   (mapcar #'car pairs)
   (lambda (p) (cdr (assoc p pairs)))))

;;; --- each gate, with an input that must trip it ---------------------------

(let* ((long (make-string 5000 ?x))
       (cases
        ;; (name . ((prompt . completion) ...)) -> expected single reason
        (list
         (cons 'empty      '(("what is 2+2?" . "   ")))
         (cons 'echo       '(("repeat this" . "repeat this")))
         (cons 'too-long   (list (cons "summarise" long)))
         (cons 'duplicate  '(("first question" . "same answer")
                             ("second question" . "same answer"))))))
  (dolist (c cases)
    (let* ((reason (car c))
           (res (dt--collect (cdr c)))
           (report (cdr res))
           (counts (nl-llm-distill-report-counts report)))
      (dt--ck (format "gate %s fires" reason)
              (and (assq reason counts) (> (cdr (assq reason counts)) 0))
              (nl-llm-distill-report-summary report)))))

;; duplicate must keep the first and drop only the second
(let* ((res (dt--collect '(("q1" . "one answer") ("q2" . "one answer"))))
       (examples (car res)))
  (dt--ck "duplicate keeps the first occurrence"
          (and (= 1 (length examples))
               (equal (plist-get (car examples) :prompt) "q1"))
          (format "%d kept" (length examples))))

;; a teacher that signals must be counted, not abort the run
(let* ((res (nl-llm-distill-collect
             '("a" "b")
             (lambda (p) (if (equal p "a") (error "teacher exploded") "fine"))))
       (report (cdr res)))
  (dt--ck "a signalling teacher is counted, not fatal"
          (and (= 1 (nl-llm-distill-report-accepted report))
               (assq 'error (nl-llm-distill-report-counts report)))
          (nl-llm-distill-report-summary report)))

;; the total-characters budget
(let* ((chunk (make-string 2000 ?y))
       (prompts (cl-loop for i below 40 collect (format "prompt %d" i)))
       (res (nl-llm-distill-collect prompts
                                    (lambda (p) (concat p " " chunk))))
       (report (cdr res)))
  (dt--ck "gate budget fires before the dataset overruns"
          (assq 'budget (nl-llm-distill-report-counts report))
          (nl-llm-distill-report-summary report)))

;; and the example count
(let* ((prompts (cl-loop for i below 140 collect (format "q%d" i)))
       (res (nl-llm-distill-collect prompts (lambda (p) (format "a for %s" p))))
       (report (cdr res)))
  (dt--ck "gate over-count caps the dataset"
          (and (= nl-llm-distill-max-examples
                  (nl-llm-distill-report-accepted report))
               (assq 'over-count (nl-llm-distill-report-counts report)))
          (nl-llm-distill-report-summary report)))

;; clean input must pass untouched -- a gate that rejects everything is as
;; broken as one that rejects nothing
(let* ((pairs '(("what is the capital of France?" . "Paris.")
                ("name a prime above 10" . "11.")
                ("translate hello" . "Bonjour.")))
       (res (dt--collect pairs))
       (examples (car res))
       (report (cdr res)))
  (dt--ck "clean input passes every gate"
          (and (= 3 (nl-llm-distill-report-accepted report))
               (= 0 (nl-llm-distill-report-rejected report))
               (equal (plist-get (nth 1 examples) :completion) "11."))
          (nl-llm-distill-report-summary report))

  ;; --- the real consumer -------------------------------------------------
  (let* ((tmp (make-temp-file "distill" nil ".eld"))
         (_ (nl-llm-distill-write examples tmp
                                  (list :teacher "test-stub"
                                        :prompts (length pairs))))
         (back (nl-llm-distill-read tmp)))
    (dt--ck "write then read round-trips"
            (and (= 3 (length (plist-get back :examples)))
                 (equal (plist-get back :teacher) "test-stub")
                 (equal (plist-get back :count) 3))
            (format "%d examples, teacher %S"
                    (length (plist-get back :examples))
                    (plist-get back :teacher)))

    ;; The check that makes the rest worth having: does the training path
    ;; actually accept what came out?
    (let ((plan (condition-case err
                    (nl-llm-agent-supervised--prepare
                     (plist-get back :examples)
                     nl-llm-agent-tokenizer-utf8)
                  (error (cons 'failed err)))))
      (dt--ck "the supervised path accepts the dataset"
              (and (listp plan) (not (eq (car-safe plan) 'failed))
                   (plist-get plan :loss-starts))
              (if (eq (car-safe plan) 'failed)
                  (format "%S" (cdr plan))
                (format "%d completion tokens, loss starts %S"
                        (plist-get plan :completion-tokens)
                        (append (plist-get plan :loss-starts) nil)))))

    ;; Loss must start where the prompt ends, or "completion-only" is a label
    ;; rather than a fact.
    (let* ((plan (nl-llm-agent-supervised--prepare
                  (plist-get back :examples) nl-llm-agent-tokenizer-utf8))
           (starts (append (plist-get plan :loss-starts) nil))
           (want (mapcar (lambda (e)
                           (length (nl-llm-agent-tokenizer-encode
                                    (plist-get e :prompt)
                                    nl-llm-agent-tokenizer-utf8)))
                         examples)))
      (dt--ck "loss starts exactly where each prompt ends"
              (equal starts want)
              (format "%S vs %S" starts want)))
    (delete-file tmp))

  ;; Refusing to write nothing: an empty dataset is a failed run, and writing
  ;; it would turn that into a file someone later trains on.
  (dt--ck "writing an empty dataset is refused"
          (condition-case _ (progn (nl-llm-distill-write nil "/tmp/never") nil)
            (error t))))

;;; --- the duplicated limits must not drift ---------------------------------

(dt--ck "limits match the supervised path"
        (and (= nl-llm-distill-max-examples
                nl-llm-agent-supervised-max-examples)
             (= nl-llm-distill-max-example-chars
                nl-llm-agent-supervised-max-example-chars)
             (= nl-llm-distill-max-total-chars
                nl-llm-agent-supervised-max-total-chars))
        (format "%d / %d / %d"
                nl-llm-distill-max-examples
                nl-llm-distill-max-example-chars
                nl-llm-distill-max-total-chars))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop dt--fail) "distill OK" "distill") dt--fail))
(when (> dt--fail 0) (kill-emacs 1))
