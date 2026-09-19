;;; agent-inference-runtime-test.el --- native policy runtime preparation  -*- lexical-binding: t; -*-
;; emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-inference-runtime-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent-model)

(defvar airt--fail 0)
(defun airt--ck (name ok &optional extra)
  (princ (format "%-62s %s  %s\n" name
                 (if ok "PASS" (progn (setq airt--fail (1+ airt--fail)) "FAIL"))
                 (or extra ""))))

(defun airt--error-message (thunk)
  (condition-case err
      (progn (funcall thunk) nil)
    (error (error-message-string err))))

(let* ((events nil)
       (policy (nl-llm-agent-model-policy
                '(:blocks (nil) :dim 4 :heads 2 :kvh 1)
                (lambda (_emitted) :stop) 3))
       msg)
  (cl-letf (((symbol-function 'nl-llm-inference-runtime-prepare)
             (lambda (&optional mode) (push (list 'prepare mode) events) 'source))
            ((symbol-function 'nl-llm-dcache-new)
             (lambda (&rest _args) (push '(cache) events) 'cache))
            ((symbol-function 'nl-llm-agent-model-step-fn)
             (lambda (&rest _args) (push '(step) events) (lambda (_id) []))))
    (setq msg (airt--error-message
               (lambda () (funcall policy '((user . "too long")))))))
  (airt--ck "prompt overflow fails before runtime preparation"
            (and msg (string-match-p "prompt length" msg) (null events)) msg))

(let* ((messages '((user . "ok")))
       (capacity (length (nl-llm-agent--render messages)))
       (events nil)
       (nl-llm-agent-model-inference-mode 'source)
       (policy (nl-llm-agent-model-policy
                '(:blocks (nil) :dim 4 :heads 2 :kvh 1)
                (lambda (_emitted) :stop) capacity))
       first second)
  (cl-letf (((symbol-function 'nl-llm-inference-runtime-prepare)
             (lambda (&optional mode) (push (list 'prepare mode) events) mode))
            ((symbol-function 'nl-llm-dcache-new)
             (lambda (&rest _args) (push '(cache) events) 'cache))
            ((symbol-function 'nl-llm-agent-model-step-fn)
             (lambda (&rest _args)
               (push '(step) events)
               (lambda (_id) (make-vector nl-llm-agent-char-vocab 0.0)))))
    (setq first (funcall policy messages))
    (setq second (funcall policy messages)))
  (setq events (nreverse events))
  (airt--ck "every valid invocation prepares before cache and model step"
            (and (equal first "") (equal second "")
                 (equal events
                        '((prepare source) (cache) (step)
                          (prepare source) (cache) (step))))
            (format "%S" events)))

(princ (format "NL-LLM-AGENT-INFERENCE-RUNTIME %s (%d failures)\n"
               (if (= airt--fail 0) "ALL-PASS" "HAS-FAILURES") airt--fail))
(kill-emacs (if (= airt--fail 0) 0 1))
;;; agent-inference-runtime-test.el ends here
