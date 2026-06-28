;;; agent-sandbox-test.el --- P2: sandboxing + permission gating  -*- lexical-binding: t; -*-
;; Pins the safety layer for untrusted agent output: under safe permissions,
;; destructive Elisp is DENIED before it runs, shells are DENIED, edits outside the
;; workdir are DENIED, an infinite loop is killed by the timeout (not hung), and
;; safe Elisp still evaluates correctly in the isolated subprocess.
;;   emacs -Q --batch -L lisp -l test/agent-sandbox-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'cl-lib)
(require 'nl-llm-agent)

(defvar sb--fail 0)
(defun sb--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq sb--fail (1+ sb--fail)) "FAIL")) (or extra ""))))

;; unit: permission helpers
(sb--ck "denylist catches delete-directory"
        (equal (nl-llm-agent--denylisted "(delete-directory \"/x\")" nl-llm-agent-default-denylist) "delete-directory"))
(sb--ck "denylist passes harmless arithmetic"
        (null (nl-llm-agent--denylisted "(+ 1 2)" nl-llm-agent-default-denylist)))
(sb--ck "path confinement: inside workdir" (nl-llm-agent--path-confined-p "a/b.el" "/tmp/wd"))
(sb--ck "path confinement: ../ escape blocked" (not (nl-llm-agent--path-confined-p "../../etc/passwd" "/tmp/wd")))

;; integration: a malicious episode is fully contained under safe permissions
(let* ((dir (make-temp-file "nl-sb-" t))
       (victim (expand-file-name "keep.el" dir))
       (outside (expand-file-name "outside.txt" temporary-file-directory))
       (obs nil)
       (perms (nl-llm-agent-safe-permissions 2))
       (responses
        (list
         ;; 1. destructive Elisp -> must be DENIED (file must survive)
         (format "```elisp\n(delete-file %S)\n```" victim)
         ;; 2. shell -> DENIED
         "```sh\nrm -rf /\n```"
         ;; 3. edit outside the workdir -> DENIED
         (format "%s\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE" outside)
         ;; 4. safe Elisp in the sandbox -> correct value
         "```elisp\n(* 6 7)\n```"
         ;; 5. infinite loop -> killed by timeout, not hung
         "```elisp\n(while t)\n```"
         "DONE done"))
       (_ (with-temp-file victim (insert ";; keep me\n")))
       (res (nl-llm-agent-run "be evil" (nl-llm-agent-scripted-policy responses)
              :workdir dir :max-steps 7 :permissions perms
              :trace (lambda (_s role content) (when (eq role 'observation) (push content obs))))))
  (setq obs (nreverse obs))
  (sb--ck "destructive Elisp denied" (string-match-p "DENIED" (nth 0 obs)) (nth 0 obs))
  (sb--ck "victim file survived the delete attempt" (file-exists-p victim))
  (sb--ck "shell denied" (string-match-p "DENIED" (nth 1 obs)) (nth 1 obs))
  (sb--ck "out-of-workdir edit denied" (string-match-p "DENIED" (nth 2 obs)) (nth 2 obs))
  (sb--ck "safe Elisp evaluates in the sandbox (6*7=42)" (string-match-p "42" (nth 3 obs)) (nth 3 obs))
  (when (executable-find "timeout")
    (sb--ck "infinite loop killed by timeout (not hung)" (string-match-p "timed out" (nth 4 obs)) (nth 4 obs)))
  (delete-directory dir t))

;; permissive mode is unchanged (backward compatible, in-process)
(let* ((dir (make-temp-file "nl-sb2-" t))
       (res (nl-llm-agent-run "compute" (nl-llm-agent-scripted-policy (list "```elisp\n(+ 40 2)\n```" "DONE ok"))
              :workdir dir :max-steps 3))   ; default = permissive
       (msg (cl-find-if (lambda (m) (and (eq (car m) 'user) (string-match-p "OBSERVATION" (cdr m)))) (cdr (plist-get res :messages)))))
  (sb--ck "permissive (default) still evaluates in-process" (and msg (string-match-p "42" (cdr msg))))
  (delete-directory dir t))

(princ (format "NL-LLM-AGENT-SANDBOX %s (%d failures)\n" (if (= sb--fail 0) "ALL-PASS" "HAS-FAILURES") sb--fail))
(kill-emacs (if (= sb--fail 0) 0 1))
;;; agent-sandbox-test.el ends here
