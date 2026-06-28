;;; agent-tasks-test.el --- richer reward: multi-task, spec-conditioned synthesis  -*- lexical-binding: t; -*-
;; Richer than the single fixed task: a BANK of synthesis tasks, each specified by
;; a prompt (op + constant) and graded by MULTIPLE test cases (which is what pins
;; the right program -- a single case can't tell (* x 3) from (+ x 4) at x=2).  The
;; model must read each task's spec and fill the template accordingly; the shared
;; skill transfers, so mean success across the bank climbs from the ~1/9 random
;; baseline.  CPU; seeded.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-tasks-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(defvar at--fail 0)
(defun at--ck (name ok &optional extra)
  (princ (format "%-56s %s  %s\n" name (if ok "PASS" (progn (setq at--fail (1+ at--fail)) "FAIL")) (or extra ""))))

(random "p5-tasks-seed")

;; reward: define the synthesised function (multiply-by-K) and check several cases
(defun at--reward (k)
  (let ((cases '(2 3 5)))
    (lambda (emitted)
      (let ((act (nl-llm-agent--parse emitted)))
        (if (not (eq (car act) 'elisp)) 0.0
          (condition-case nil
              (progn (fmakunbound 'f235) (eval (read (nth 1 act)) t)
                     (if (and (fboundp 'f235) (cl-every (lambda (x) (= (f235 x) (* x k))) cases)) 1.0 0.0))
            (error 0.0)))))))

(let* ((m (nl-llm-agent-improve-model 20 20 nl-llm-agent-char-vocab 2 2)) ; 2 blocks/heads -> can copy from prompt
       ;; one free constant slot, CONDITIONED on the spec prompt (c=K)
       (grammar (nl-llm-agent-grammar-template
                 (list "```elisp\n(defun f235 (x) (* x " '(:slot "23456789") "))\n```")))
       (tasks (list (cons "c=2\n" (at--reward 2)) (cons "c=3\n" (at--reward 3)))))
  ;; multi-case reward genuinely discriminates the right constant
  (at--ck "multi-case reward accepts the correct program"
          (= (funcall (at--reward 3) "```elisp\n(defun f235 (x) (* x 3))\n```") 1.0))
  (at--ck "multi-case reward rejects the wrong constant"
          (= (funcall (at--reward 3) "```elisp\n(defun f235 (x) (* x 4))\n```") 0.0))
  ;; lightweight here (fast for CI); examples/agent-tasks-demo.el runs the full 4-task convergence
  (let* ((rates (nl-llm-agent-improve-tasks m grammar tasks
                  :rounds 5 :rollouts 24 :lr 0.5 :epochs 2 :temp 1.0 :eval-n 12
                  :trace (lambda (r n rate) (princ (format "  round %d: replay %2d -> mean success %3.0f%%\n" r n (* 100 rate))))))
         (peak (apply #'max rates)))
    (princ (format "mean success across %d tasks: %s\n" (length tasks)
                   (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (at--ck "multi-task self-improvement: mean success rose"
            (> peak (+ (car rates) 0.20))
            (format "%.0f%% -> peak %.0f%%" (* 100 (car rates)) (* 100 peak)))
    (at--ck "reached above the random baseline (~12.5%)" (> peak 0.45)
            (format "peak %.0f%%" (* 100 peak)))))

(princ (format "NL-LLM-AGENT-TASKS %s (%d failures)\n" (if (= at--fail 0) "ALL-PASS" "HAS-FAILURES") at--fail))
(kill-emacs (if (= at--fail 0) 0 1))
;;; agent-tasks-test.el ends here
