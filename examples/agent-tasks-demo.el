;;; agent-tasks-demo.el --- richer reward: multi-task spec-conditioned self-improvement  -*- lexical-binding: t; -*-
;; The self-improvement reward is now RICHER: a BANK of synthesis tasks, each
;; specified by a prompt and graded by MULTIPLE test cases (which is what pins the
;; right program -- one case can't tell (* x 3) from (+ x 4) at x=2).  The model
;; (2 blocks -- enough depth to learn to copy the spec into the code) reads each
;; task's spec and synthesises a matching function; training on its OWN passing
;; programs with a replay buffer (keeps the curriculum balanced), the mean success
;; across the bank climbs from the ~1/8 random baseline.  This is the loop scaling
;; toward real coding (docs/design/05-agent-harness.org).  CPU; seeded.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/agent-tasks-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(random "p5-tasks-demo-seed")

(defun atd--reward (k)
  (let ((cases '(2 3 5)))
    (lambda (emitted)
      (let ((act (nl-llm-agent--parse emitted)))
        (if (not (eq (car act) 'elisp)) 0.0
          (condition-case nil
              (progn (fmakunbound 'f235) (eval (read (nth 1 act)) t)
                     (if (and (fboundp 'f235) (cl-every (lambda (x) (= (f235 x) (* x k))) cases)) 1.0 0.0))
            (error 0.0)))))))

(let* ((m (nl-llm-agent-improve-model 24 24 nl-llm-agent-char-vocab 2 2))
       (grammar (nl-llm-agent-grammar-template
                 (list "```elisp\n(defun f235 (x) (* x " '(:slot "23456789") "))\n```")))
       (tasks (list (cons "c=2\n" (atd--reward 2)) (cons "c=3\n" (atd--reward 3))
                    (cons "c=4\n" (atd--reward 4)) (cons "c=5\n" (atd--reward 5)))))
  (princ "=== richer reward: multi-task, spec-conditioned, multi-test-case synthesis ===\n")
  (princ "4 tasks (multiply by 2/3/4/5); each prompt gives the spec, reward runs 3 test cases.\n")
  (princ "the model must READ each spec and synthesise the matching function.\n\n")
  (dolist (tk tasks)
    (princ (format "  before: spec %S -> %S\n" (string-trim (car tk))
                   (string-trim (nth 1 (nl-llm-agent--parse (car (nl-llm-agent-p5-rollout m grammar 0.3 (car tk)))))))) )
  (princ "\n")
  (let ((rates (nl-llm-agent-improve-tasks m grammar tasks
                :rounds 6 :rollouts 40 :lr 0.5 :epochs 2 :temp 1.0 :eval-n 12
                :trace (lambda (r n rate) (princ (format "  round %d: replay %2d -> mean success %3.0f%%\n" r n (* 100 rate)))))))
    (princ (format "\nmean success across tasks: %s\n\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (dolist (tk tasks)
      (princ (format "  after : spec %S -> %S\n" (string-trim (car tk))
                     (string-trim (nth 1 (nl-llm-agent--parse (car (nl-llm-agent-p5-rollout m grammar 0.3 (car tk)))))))) )
    (let ((peak (apply #'max rates)))
      (princ (format "\nmulti-task self-improvement: %s  (%.0f%% -> peak %.0f%%)\n"
                     (if (> peak (+ (car rates) 0.25)) "YES" "NO") (* 100 (car rates)) (* 100 peak)))
      (kill-emacs (if (> peak (+ (car rates) 0.25)) 0 1)))))
;;; agent-tasks-demo.el ends here
