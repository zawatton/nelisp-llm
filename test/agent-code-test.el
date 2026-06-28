;;; agent-code-test.el --- P5 with a REAL coding reward (tests must pass)  -*- lexical-binding: t; -*-
;; Upgrades the self-improvement reward from a char match to actual code execution:
;; the model synthesises `(defun f235 (x) (OP x N))' by choosing the operator OP
;; and constant N under a grammar, and the reward DEFINES the function and runs it
;; against test cases (f(2)=6, f(5)=15) -- reward 1 iff the synthesised code passes
;; (the unique solution is (* x 3)).  After reward-filtered fine-tuning on its own
;; passing programs, the synthesis success rate rises well above the random
;; baseline (~1/9).  CPU; seeded RNG.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-code-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(defvar ac--fail 0)
(defun ac--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq ac--fail (1+ ac--fail)) "FAIL")) (or extra ""))))

(random "p5-code-reward-seed")

;; REAL coding reward: parse -> define the function -> run test cases.
(defun ac--reward (emitted)
  (let ((act (nl-llm-agent--parse emitted)))
    (if (not (eq (car act) 'elisp)) 0.0
      (condition-case nil
          (progn (fmakunbound 'f235)
                 (eval (read (nth 1 act)) t)
                 (if (and (fboundp 'f235) (= (f235 2) 6) (= (f235 5) 15)) 1.0 0.0))
        (error 0.0)))))

(let* ((m (nl-llm-agent-improve-model 16 32))
       (grammar (nl-llm-agent-grammar-template
                 (list "```elisp\n(defun f235 (x) (" '(:slot "+-*") " x " '(:slot "123") "))\n```")))
       (roll (nl-llm-agent-p5-rollout m grammar 1.0)))
  ;; the synthesised action is a real, parseable defun
  (ac--ck "synthesised action is a valid Elisp defun"
          (let ((a (nl-llm-agent--parse (car roll))))
            (and (eq (car a) 'elisp) (string-match-p "defun f235" (nth 1 a))))
          (format "%S" (string-trim (replace-regexp-in-string "\n" "\\\\n" (car roll)))))
  ;; reward is execution-based: a known-good program passes, a known-bad one fails
  (ac--ck "reward passes the correct program (* x 3)"
          (= (ac--reward "```elisp\n(defun f235 (x) (* x 3))\n```") 1.0))
  (ac--ck "reward fails an incorrect program (+ x 3)"
          (= (ac--reward "```elisp\n(defun f235 (x) (+ x 3))\n```") 0.0))
  ;; self-improvement under the coding reward
  (let ((rates (nl-llm-agent-improve m grammar #'ac--reward
                 :rounds 5 :rollouts 40 :lr 0.4 :epochs 2 :temp 1.0 :eval-n 40
                 :trace (lambda (r n rate) (princ (format "  round %d: %2d passing programs -> synth success %3.0f%%\n" r n (* 100 rate)))))))
    (princ (format "synthesis success: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (ac--ck "self-improvement: code-synthesis success rate rose"
            (> (car (last rates)) (+ (car rates) 0.10))
            (format "%.0f%% -> %.0f%%" (* 100 (car rates)) (* 100 (car (last rates)))))
    (ac--ck "ended well above the random baseline (~11%)" (> (car (last rates)) 0.45)
            (format "final %.0f%%" (* 100 (car (last rates)))))))

(princ (format "NL-LLM-AGENT-CODE %s (%d failures)\n" (if (= ac--fail 0) "ALL-PASS" "HAS-FAILURES") ac--fail))
(kill-emacs (if (= ac--fail 0) 0 1))
;;; agent-code-test.el ends here
