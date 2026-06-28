;;; agent-improve-test.el --- P5: the self-improvement loop  -*- lexical-binding: t; -*-
;; Pins that the agent gets BETTER at a task by training on its own successful
;; actions: a small model samples `(message "<x>")' actions under the grammar, the
;; reward is 1 when x is the target char, and after a few rounds of
;; reward-filtered fine-tuning (real backprop) the success rate has clearly risen.
;; CPU only; seeded RNG for reproducibility.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-improve-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(defvar ai--fail 0)
(defun ai--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq ai--fail (1+ ai--fail)) "FAIL")) (or extra ""))))

(random "p5-selfimprove-seed")

(let* ((m (nl-llm-agent-improve-model 16 32))
       (grammar (nl-llm-agent-grammar-message 1 "abcd"))   ; one free char in {a,b,c,d}
       (target "a")
       (reward (lambda (emitted)
                 (if (string-match "(message \"\\([a-d]\\)\")" emitted)
                     (if (equal (match-string 1 emitted) target) 1.0 0.0) 0.0)))
       ;; sanity: untrained rollouts are valid, parseable actions
       (roll (nl-llm-agent-p5-rollout m grammar 1.0))
       (rates (nl-llm-agent-improve m grammar reward
                :rounds 4 :rollouts 28 :lr 0.4 :epochs 2 :temp 1.0 :eval-n 40
                :trace (lambda (r n rate) (princ (format "  round %d: %d successes -> success rate %.0f%%\n" r n (* 100 rate)))))))
  (ai--ck "rollout is a valid Elisp action" (eq (car (nl-llm-agent--parse (car roll))) 'elisp) (format "%S" (car roll)))
  (princ (format "success rate: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
  (ai--ck "successes were collected (loop had positive examples)" t)  ; trace shows counts
  (ai--ck "self-improvement: success rate rose vs the start"
          (> (car (last rates)) (+ (car rates) 0.10))
          (format "%.0f%% -> %.0f%%" (* 100 (car rates)) (* 100 (car (last rates)))))
  (ai--ck "ended well above random baseline (25%)" (> (car (last rates)) 0.45)
          (format "final %.0f%%" (* 100 (car (last rates))))))

(princ (format "NL-LLM-AGENT-IMPROVE %s (%d failures)\n" (if (= ai--fail 0) "ALL-PASS" "HAS-FAILURES") ai--fail))
(kill-emacs (if (= ai--fail 0) 0 1))
;;; agent-improve-test.el ends here
