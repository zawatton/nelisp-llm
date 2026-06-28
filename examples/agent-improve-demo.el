;;; agent-improve-demo.el --- P5: the model improves itself (STaR loop)  -*- lexical-binding: t; -*-
;; Closes the self-improvement loop: a small nelisp-llm model samples actions under
;; the agent grammar, a reward keeps only its successful ones, and it is fine-tuned
;; (real backprop, CPU autograd) on its OWN wins -- with no external labels.  The
;; measured success rate climbs round over round.  Toy task: emit `(message "<x>")'
;; with x the target char; reward = 1 when x matches.  The same mechanism, with a
;; richer grammar + real coding rewards (tests pass), scales to self-improving the
;; model's coding ability (docs/design/05-agent-harness.org).  CPU; seeded RNG.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/agent-improve-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(random "p5-demo-seed")

(let* ((m (nl-llm-agent-improve-model 16 32))
       (grammar (nl-llm-agent-grammar-message 1 "abcd"))
       (target "c")
       (reward (lambda (e) (if (and (string-match "(message \"\\([a-d]\\)\")" e) (equal (match-string 1 e) target)) 1.0 0.0))))
  (princ "=== P5: self-improvement loop -- the model trains on its own successful actions ===\n")
  (princ (format "toy task: emit (message \"x\") with x = %S; reward = 1 when it matches.\n" target))
  (princ "the model only ever sees its OWN wins -- no external labels.\n\n")
  (princ (format "before: a sampled action = %S\n\n" (car (nl-llm-agent-p5-rollout m grammar 1.0))))
  (let ((rates (nl-llm-agent-improve
                m grammar reward
                :rounds 5 :rollouts 28 :lr 0.4 :epochs 2 :temp 1.0 :eval-n 40
                :trace (lambda (r n rate) (princ (format "  round %d: kept %2d successes -> success rate %3.0f%%\n" r n (* 100 rate)))))))
    (princ (format "\nsuccess rate: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (princ (format "after : a sampled action = %S\n" (car (nl-llm-agent-p5-rollout m grammar 1.0))))
    (let ((ok (> (car (last rates)) (+ (car rates) 0.10))))
      (princ (format "\nself-improvement (final > start): %s  (%.0f%% -> %.0f%%)\n"
                     (if ok "YES" "NO") (* 100 (car rates)) (* 100 (car (last rates)))))
      (kill-emacs (if ok 0 1)))))
;;; agent-improve-demo.el ends here
