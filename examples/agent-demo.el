;;; agent-demo.el --- the agent harness extending itself  -*- lexical-binding: t; -*-
;; Shows nl-llm-agent driving an episode where the "policy" (a scripted stand-in
;; for the nelisp-llm model) gives the agent a new capability by EDITING a tools
;; file -- and where the linter guardrail rejects a malformed edit first, so the
;; agent retries.  This is the mechanism by which an nelisp-llm-generated model,
;; plugged in as the policy, can strengthen its own toolset (Phase 4/5 wire the
;; real model + trajectory fine-tuning -- docs/design/05-agent-harness.org).
;; No model / GPU needed: the scripted policy makes it deterministic.
;;   emacs -Q --batch -L lisp -l examples/agent-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'nl-llm-agent)

(let* ((dir (make-temp-file "nl-agent-demo-" t))
       (tools (expand-file-name "my-tools.el" dir))
       ;; a tools file the agent will extend with a new tool
       (_ (with-temp-file tools
            (insert ";;; my-tools.el --- the agent's own tool library\n\n"
                    "(defun tool-add (a b) (+ a b))\n\n"
                    "(provide 'my-tools)\n")))
       ;; scripted policy: stand-in for the model.  It (1) inspects, (2) tries a
       ;; BROKEN edit (rejected by the linter), (3) fixes it, (4) verifies, (5) done.
       (policy
        (nl-llm-agent-scripted-policy
         (list
          ;; 1. look at what tools exist
          "Let me see the current tools.\n```elisp\n(nl-llm-agent-repo-map \"DIR\")\n```"
          ;; 2. add a new tool -- but with an unbalanced paren (linter must reject)
          "Add a multiply tool.\nmy-tools.el\n<<<<<<< SEARCH\n(defun tool-add (a b) (+ a b))\n=======\n(defun tool-add (a b) (+ a b))\n\n(defun tool-mul (a b) (* a b)\n>>>>>>> REPLACE"
          ;; 3. fix it -- balanced this time
          "Fix the parens.\nmy-tools.el\n<<<<<<< SEARCH\n(defun tool-add (a b) (+ a b))\n=======\n(defun tool-add (a b) (+ a b))\n\n(defun tool-mul (a b) (* a b))\n>>>>>>> REPLACE"
          ;; 4. load the extended library and use the NEW tool
          "Use the new tool.\n```elisp\n(progn (load \"DIR/my-tools.el\" nil t) (tool-mul 6 7))\n```"
          ;; 5. done
          "DONE added tool-mul and verified 6*7=42")))
       ;; substitute the temp dir into the scripted responses (they reference DIR)
       (policy (let ((p policy)) (lambda (msgs)
                 (replace-regexp-in-string "DIR" dir (funcall p msgs) t t)))))
  (princ "=== nl-llm-agent: the agent extends its own toolset ===\n\n")
  (nl-llm-agent-run
   "Add a multiply tool to my-tools.el and verify it works."
   policy :workdir dir :max-steps 8
   :trace (lambda (step role content)
            (when (> step 0)
              (princ (format "[step %d %s]\n%s\n\n" step role
                             (let ((s (string-trim content))) (if (> (length s) 400) (concat (substring s 0 400) " ...") s)))))))
  (princ "--- final my-tools.el ---\n")
  (princ (with-temp-buffer (insert-file-contents tools) (buffer-string)))
  (let ((ok (with-temp-buffer (insert-file-contents tools)
              (and (string-match-p "tool-mul" (buffer-string))
                   (condition-case nil (progn (goto-char (point-min)) (check-parens) t) (error nil))))))
    (princ (format "\nself-extension result: tool-mul added AND file parses : %s\n" (if ok "YES" "NO")))
    (delete-directory dir t)
    (kill-emacs (if ok 0 1))))
;;; agent-demo.el ends here
