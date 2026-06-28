;;; agent-test.el --- nl-llm-agent harness (scripted policy, no model)  -*- lexical-binding: t; -*-
;; Drives the agent harness with a deterministic scripted policy (a stand-in for
;; the LLM) so the loop, action parsing, SEARCH/REPLACE edits and the linter
;; guardrail are all pinned without needing a model or a GPU.
;;   emacs -Q --batch -L lisp -l test/agent-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)

(defvar ag--fail 0)
(defun ag--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name (if ok "PASS" (progn (setq ag--fail (1+ ag--fail)) "FAIL")) (or extra ""))))

;; --- unit: action parsing ---------------------------------------------------
(ag--ck "parse: elisp CodeAct block"
        (equal (nl-llm-agent--parse "thinking...\n```elisp\n(+ 40 2)\n```\n") '(elisp "(+ 40 2)")))
(ag--ck "parse: SEARCH/REPLACE edit"
        (equal (nl-llm-agent--parse "foo.el\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n")
               '(edit "foo.el" "old" "new")))
(ag--ck "parse: shell block"
        (equal (nl-llm-agent--parse "```sh\nls -1\n```") '(shell "ls -1")))
(ag--ck "parse: DONE with answer" (equal (nl-llm-agent--parse "DONE all set") '(done "all set")))
(ag--ck "parse: plain prose -> none" (equal (nl-llm-agent--parse "I am thinking.") '(none)))

;; --- integration: a full scripted episode -----------------------------------
(let* ((dir (make-temp-file "nl-agent-" t))
       (file (expand-file-name "sample.el" dir))
       (obs nil)
       (responses
        (list
         ;; (1) run Elisp
         "First, compute.\n```elisp\n(+ 40 2)\n```"
         ;; (2) a valid edit
         "Now rename the greeting.\nsample.el\n<<<<<<< SEARCH\n(defun greet () \"hi\")\n=======\n(defun greet () \"hello\")\n>>>>>>> REPLACE"
         ;; (3) an edit whose result does NOT parse -> must be rejected by the linter
         "Break it (should be rejected).\nsample.el\n<<<<<<< SEARCH\n(defun greet () \"hello\")\n=======\n(defun greet () \"hello\"\n>>>>>>> REPLACE"
         ;; (4) finish
         "DONE renamed greet"))
       (_ (with-temp-file file (insert ";;; sample.el\n(defun greet () \"hi\")\n")))
       (res (nl-llm-agent-run "rename the greeting" (nl-llm-agent-scripted-policy responses)
              :workdir dir :max-steps 8
              :trace (lambda (_step role content) (when (eq role 'observation) (push content obs)))))
       (final (with-temp-buffer (insert-file-contents file) (buffer-string))))
  (setq obs (nreverse obs))
  (ag--ck "episode finished with DONE" (eq (plist-get res :status) 'done)
          (format "status=%s result=%S steps=%d" (plist-get res :status) (plist-get res :result) (plist-get res :steps)))
  (ag--ck "Elisp action evaluated (40+2=42)" (string-match-p "42" (nth 0 obs)))
  (ag--ck "valid edit applied" (string-match-p "(defun greet () \"hello\")" final))
  (ag--ck "linter guardrail rejected the breaking edit" (string-match-p "REJECTED" (nth 2 obs)))
  (ag--ck "file still parses (bad edit discarded)"
          (condition-case nil (with-temp-buffer (insert final) (goto-char (point-min)) (check-parens) t) (error nil)))
  ;; repo map sees the file + its def
  (ag--ck "repo map lists files + definitions" (string-match-p "greet" (nl-llm-agent-repo-map dir)))
  (delete-directory dir t))

(princ (format "NL-LLM-AGENT %s (%d failures)\n" (if (= ag--fail 0) "ALL-PASS" "HAS-FAILURES") ag--fail))
(kill-emacs (if (= ag--fail 0) 0 1))
;;; agent-test.el ends here
