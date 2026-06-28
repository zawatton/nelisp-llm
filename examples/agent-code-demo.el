;;; agent-code-demo.el --- P5 with a real coding reward: synthesise code that passes tests  -*- lexical-binding: t; -*-
;; The self-improvement reward is now CODE EXECUTION, not a string match: the model
;; synthesises `(defun f235 (x) (OP x N))' by choosing OP and N under a grammar, and
;; the reward defines the function and runs it against test cases (f(2)=6, f(5)=15)
;; -- reward 1 iff the synthesised program passes (unique solution (* x 3)).
;; Training only on its OWN passing programs, the model's synthesis success rate
;; climbs from the ~1/9 random baseline toward 100%.  This is how the loop scales
;; from a toy to real coding ability (docs/design/05-agent-harness.org).  CPU; seeded.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/agent-code-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(random "p5-code-demo-seed")

;; reward = run the synthesised function against test cases
(defun acd--reward (emitted)
  (let ((act (nl-llm-agent--parse emitted)))
    (if (not (eq (car act) 'elisp)) 0.0
      (condition-case nil
          (progn (fmakunbound 'f235) (eval (read (nth 1 act)) t)
                 (if (and (fboundp 'f235) (= (f235 2) 6) (= (f235 5) 15)) 1.0 0.0))
        (error 0.0)))))

(let* ((m (nl-llm-agent-improve-model 16 32))
       (grammar (nl-llm-agent-grammar-template
                 (list "```elisp\n(defun f235 (x) (" '(:slot "+-*") " x " '(:slot "123") "))\n```"))))
  (princ "=== P5 real coding reward: synthesise a function that PASSES TESTS ===\n")
  (princ "spec (hidden from the model): f(2)=6 and f(5)=15   [unique solution: (* x 3)]\n")
  (princ "the model picks the operator + constant; the reward RUNS the code against the tests.\n\n")
  (princ (format "before: a synthesised program = %S  (reward %.0f)\n\n"
                 (let ((e (car (nl-llm-agent-p5-rollout m grammar 1.0)))) (string-trim (nth 1 (nl-llm-agent--parse e))))
                 (acd--reward (car (nl-llm-agent-p5-rollout m grammar 1.0)))))
  (let ((rates (nl-llm-agent-improve
                m grammar #'acd--reward
                :rounds 5 :rollouts 40 :lr 0.4 :epochs 2 :temp 1.0 :eval-n 40
                :trace (lambda (r n rate) (princ (format "  round %d: %2d programs passed tests -> synthesis success %3.0f%%\n" r n (* 100 rate)))))))
    (princ (format "\nsynthesis success: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (let ((after (car (nl-llm-agent-p5-rollout m grammar 0.5))))
      (princ (format "after : a synthesised program = %S  (reward %.0f)\n" (string-trim (nth 1 (nl-llm-agent--parse after))) (acd--reward after))))
    (let ((ok (> (car (last rates)) (+ (car rates) 0.10))))
      (princ (format "\nself-improvement from execution feedback: %s  (%.0f%% -> %.0f%%)\n"
                     (if ok "YES" "NO") (* 100 (car rates)) (* 100 (car (last rates)))))
      (kill-emacs (if ok 0 1)))))
;;; agent-code-demo.el ends here
