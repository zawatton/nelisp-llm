;;; nl-llm-agent.el --- a self-hosting agent harness for nelisp-llm  -*- lexical-binding: t; -*-

;; A minimal, model-agnostic AI-agent scaffold that turns a text-generating policy
;; (an nelisp-llm model, or any LLM) into an agent that can edit files and run
;; code -- the "harness" the model needs in order to act on, and eventually
;; improve, itself.  It distils the parts that make the open-source coding agents
;; work, reimplemented in Elisp so the agent's native action language is Lisp:
;;
;;   * mini-swe-agent  -- a tiny loop over a COMPLETELY LINEAR message history;
;;     each action runs independently; no tool-calling API is required, so any
;;     model (however small) can drive it by emitting plain text.
;;   * smolagents CodeAct -- actions are written AS CODE.  Here the code is Elisp
;;     evaluated in Emacs, so every Emacs/anvil primitive is already a tool and
;;     the agent can extend itself by writing more Elisp.
;;   * Aider           -- file edits are SEARCH/REPLACE blocks (the model emits
;;     only the changed span, robust for small models).
;;   * SWE-agent ACI   -- a linter guardrail: an edit whose result does not parse
;;     is DISCARDED and the error is fed back, so the agent retries.
;;
;; The policy is a function (MESSAGES -> assistant-text).  Tests drive it with a
;; scripted policy (deterministic, no model needed); production wraps nelisp-llm
;; generation.  Trajectories this harness produces are also training data: feed
;; them back through nlga to fine-tune the model on its own agentic behaviour --
;; the self-improvement loop (docs/design/05-agent-harness.org).

;;; Code:

(require 'cl-lib)
(require 'subr-x)

(defconst nl-llm-agent-system-prompt
  "You are an agent that solves a task by emitting ONE action per message.
Available actions (use exactly one, then stop and wait for the observation):

1. Run Elisp (your primary tool -- Emacs is the computer):
```elisp
(your-elisp-form ...)
```

2. Edit a file with a SEARCH/REPLACE block (the SEARCH text must match exactly):
path/to/file.el
<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE

3. Run a shell command:
```sh
command
```

4. Finish, on its own line:
DONE <final answer>

After each action you receive an OBSERVATION.  Edits that do not parse are
rejected -- read the error and try again."
  "Default system prompt describing the action grammar to the policy.")

;; ---- action parsing: exactly one action per assistant message --------------

(defun nl-llm-agent--strip1-nl (s)
  "Strip a single leading and trailing newline from S."
  (let ((s (if (string-prefix-p "\n" s) (substring s 1) s)))
    (if (string-suffix-p "\n" s) (substring s 0 -1) s)))

(defun nl-llm-agent--fenced (lang)
  "Return the body of the first ```LANG fenced block in the current buffer, or nil."
  (goto-char (point-min))
  (when (re-search-forward (concat "^```" lang "[ \t]*$") nil t)
    (forward-line 1)
    (let ((s (point)))
      (when (re-search-forward "^```[ \t]*$" nil t)
        (nl-llm-agent--strip1-nl (buffer-substring s (match-beginning 0)))))))

(defun nl-llm-agent--parse (text)
  "Parse assistant TEXT into one action: (edit PATH SEARCH REPLACE) | (elisp CODE)
| (shell CMD) | (done ANSWER) | (none).  Concrete actions take priority over DONE."
  (with-temp-buffer
    (insert text)
    (cond
     ;; SEARCH/REPLACE edit
     ((progn (goto-char (point-min)) (re-search-forward "^<<<<<<< SEARCH[ \t]*$" nil t))
      (let ((after-mark (point)) (mark-bol (match-beginning 0)) path search replace)
        (save-excursion
          (goto-char mark-bol) (forward-line -1)
          (while (and (> (point) (point-min)) (looking-at "^[ \t]*$")) (forward-line -1))
          (setq path (string-trim (buffer-substring (line-beginning-position) (line-end-position)))))
        (goto-char after-mark)
        (when (re-search-forward "^=======[ \t]*$" nil t)
          (setq search (nl-llm-agent--strip1-nl (buffer-substring after-mark (match-beginning 0))))
          (let ((mid (point)))
            (when (re-search-forward "^>>>>>>> REPLACE[ \t]*$" nil t)
              (setq replace (nl-llm-agent--strip1-nl (buffer-substring mid (match-beginning 0)))))))
        (if (and path search replace) (list 'edit path search replace) (list 'none))))
     ;; Elisp CodeAct
     ((let ((c (or (nl-llm-agent--fenced "elisp") (nl-llm-agent--fenced "emacs-lisp")))) (and c (list 'elisp c))))
     ;; shell
     ((let ((c (or (nl-llm-agent--fenced "sh") (nl-llm-agent--fenced "bash")))) (and c (list 'shell c))))
     ;; finish
     ((progn (goto-char (point-min)) (re-search-forward "^DONE\\(?:[ \t]+\\(.*\\)\\)?$" nil t))
      (list 'done (string-trim (or (match-string 1) ""))))
     (t (list 'none)))))

;; ---- action execution ------------------------------------------------------

(defun nl-llm-agent--lint-ok (content file)
  "Return non-nil if CONTENT is acceptable for FILE (balanced sexps for *.el)."
  (if (not (string-suffix-p ".el" file)) t
    (condition-case nil
        (with-temp-buffer (insert content) (goto-char (point-min)) (check-parens) t)
      (error nil))))

(defun nl-llm-agent--apply-edit (path search replace workdir)
  "Apply a SEARCH/REPLACE edit to PATH under WORKDIR.  Linter guardrail: an *.el
result that does not parse is discarded.  Returns (OK . MESSAGE)."
  (let ((file (expand-file-name path workdir)))
    (cond
     ((not (file-exists-p file)) (cons nil (format "file not found: %s" path)))
     (t (let ((content (with-temp-buffer (insert-file-contents file) (buffer-string))))
          (let ((idx (string-search search content)))
            (cond
             ((null idx) (cons nil "SEARCH block not found (it must match the file exactly)"))
             ((string-search search content (1+ idx))
              (cons nil "SEARCH block is not unique (add surrounding context)"))
             (t (let ((new (concat (substring content 0 idx) replace
                                   (substring content (+ idx (length search))))))
                  (if (not (nl-llm-agent--lint-ok new file))
                      (cons nil "edit REJECTED: result has unbalanced parens / does not parse")
                    (progn (with-temp-file file (insert new)) (cons t (format "edit applied to %s" path)))))))))))))

(defun nl-llm-agent--eval-elisp (code)
  "Evaluate CODE (one or more forms) IN-PROCESS, capturing the value or error.
Fast but unsafe; use the sandbox (`nl-llm-agent-safe-permissions') for untrusted
policies."
  (condition-case err
      (let ((forms (car (read-from-string (concat "(progn\n" code "\n)")))))
        (format "%S" (eval forms t)))
    (error (format "ERROR: %S" err))))

;; ---- sandboxing + permission gating (P2) -----------------------------------
;; Untrusted model output should not run in, or be able to harm, the host session.
;; The sandbox runs Elisp in an isolated `emacs -Q --batch' subprocess with a wall
;; clock timeout (no hang, no host-state corruption, crashes contained); the
;; permission gate refuses destructive forms (denylist), shells, and edits outside
;; the workdir BEFORE anything runs.

(defconst nl-llm-agent-default-denylist
  '("delete-file" "delete-directory" "shell-command" "shell-command-to-string"
    "call-process" "call-process-region" "start-process" "make-process"
    "write-region" "rename-file" "copy-file" "make-symbolic-link" "set-file-modes"
    "kill-emacs" "server-start" "url-retrieve" "browse-url" "async-shell-command")
  "Symbols an agent's Elisp may not reference under a gated permission set.")

;;;###autoload
(defun nl-llm-agent-safe-permissions (&optional timeout)
  "Permissions for UNTRUSTED policies: sandboxed Elisp (subprocess + TIMEOUT, def
5s), no shell, edits confined to the workdir, destructive forms denied."
  (list :eval 'sandbox :eval-timeout (or timeout 5) :shell nil :confine-edits t
        :denylist nl-llm-agent-default-denylist))

;;;###autoload
(defun nl-llm-agent-permissive-permissions ()
  "Permissions for TRUSTED use: in-process Elisp, shell allowed, no confinement.
The default, for backward compatibility."
  (list :eval 'in-process :shell t :confine-edits nil :denylist nil))

(defun nl-llm-agent--denylisted (code denylist)
  "Return the first DENYLIST symbol referenced in CODE, or nil."
  (cl-find-if (lambda (s) (string-match-p (concat "\\_<" (regexp-quote s) "\\_>") code)) denylist))

(defun nl-llm-agent--path-confined-p (path workdir)
  "Non-nil if PATH resolves inside WORKDIR."
  (string-prefix-p (file-name-as-directory (expand-file-name workdir))
                   (expand-file-name path workdir)))

(defun nl-llm-agent--eval-elisp-sandbox (code timeout)
  "Evaluate CODE in an isolated `emacs -Q --batch' subprocess with a TIMEOUT
\(seconds).  The host session is untouched; an infinite loop is killed, not hung."
  (let ((tmp (make-temp-file "nl-agent-sb-" nil ".el"))
        (emacs (or (and (boundp 'invocation-name)
                        (expand-file-name invocation-name invocation-directory))
                   (executable-find "emacs") "emacs")))
    (unwind-protect
        (progn
          (with-temp-file tmp
            (insert "(prin1 (condition-case e (progn\n" code "\n) (error (format \"ERROR: %S\" e))))\n"))
          (with-temp-buffer
            (let* ((tcmd (executable-find "timeout"))
                   (prog (or tcmd emacs))
                   (args (if tcmd (list (number-to-string timeout) emacs "-Q" "--batch" "-l" tmp)
                           (list "-Q" "--batch" "-l" tmp)))
                   (status (apply #'call-process prog nil t nil args)))
              (cond ((and tcmd (eq status 124)) "ERROR: timed out")
                    ((not (equal status 0)) (format "ERROR: subprocess exit %s" status))
                    (t (nl-llm-agent--truncate (string-trim (buffer-string)) 2000))))))
      (ignore-errors (delete-file tmp)))))

(defun nl-llm-agent--shell (cmd workdir)
  "Run shell CMD in WORKDIR; return a short [exit N] + output observation."
  (let ((default-directory (file-name-as-directory (expand-file-name workdir))))
    (with-temp-buffer
      (let* ((status (call-process-shell-command cmd nil t))
             (out (string-trim (buffer-string))))
        (format "[exit %s] %s" status (if (string-empty-p out) "(no output)" (concat "\n" out)))))))

(defun nl-llm-agent--truncate (s n)
  (if (<= (length s) n) s (concat (substring s 0 n) (format "\n... [truncated %d chars]" (- (length s) n)))))

(defun nl-llm-agent--observe (action workdir perms)
  "Execute ACTION under PERMS; return (STATUS . OBSERVATION) where STATUS is one of
done/continue/none.  PERMS gates execution: denylisted Elisp, shells, and edits
outside WORKDIR are refused before anything runs; Elisp runs sandboxed or
in-process per PERMS' :eval."
  (pcase action
    (`(elisp ,code)
     (let ((deny (and (plist-get perms :denylist) (nl-llm-agent--denylisted code (plist-get perms :denylist)))))
       (cons 'continue
             (cond ((null (plist-get perms :eval)) "OBSERVATION: DENIED -- Elisp evaluation is not permitted")
                   (deny (format "OBSERVATION: DENIED -- forbidden operation `%s'" deny))
                   ((eq (plist-get perms :eval) 'sandbox)
                    (concat "OBSERVATION:\n" (nl-llm-agent--eval-elisp-sandbox code (or (plist-get perms :eval-timeout) 5))))
                   (t (concat "OBSERVATION:\n" (nl-llm-agent--eval-elisp code)))))))
    (`(shell ,cmd)
     (cons 'continue (if (plist-get perms :shell)
                         (concat "OBSERVATION:\n" (nl-llm-agent--shell cmd workdir))
                       "OBSERVATION: DENIED -- shell is not permitted")))
    (`(edit ,path ,search ,replace)
     (cons 'continue
           (if (and (plist-get perms :confine-edits) (not (nl-llm-agent--path-confined-p path workdir)))
               (format "OBSERVATION: DENIED -- edit outside the workdir (%s)" path)
             (concat "OBSERVATION: " (cdr (nl-llm-agent--apply-edit path search replace workdir))))))
    (`(done ,answer) (cons 'done answer))
    (_ (cons 'none "OBSERVATION: no recognized action found. Emit exactly one action block."))))

;; ---- the agent loop --------------------------------------------------------

;;;###autoload
(cl-defun nl-llm-agent-run (task policy &key (max-steps 12) (workdir default-directory) system trace permissions)
  "Run the agent on TASK using POLICY (a function MESSAGES -> assistant-text).
MESSAGES is the linear history as a list of (ROLE . CONTENT) in order.  Each step
the policy emits one action, which is executed under PERMISSIONS (default
`nl-llm-agent-permissive-permissions'; pass `nl-llm-agent-safe-permissions' for
untrusted policies); the OBSERVATION is appended and the loop repeats until DONE or
MAX-STEPS.  TRACE, if set, is called with (STEP ROLE CONTENT) for each turn.
Returns a plist (:status done/limit :steps N :result STRING :messages LIST)."
  (let ((messages (list (cons 'system (or system nl-llm-agent-system-prompt))
                        (cons 'user (concat "TASK: " task))))
        (perms (or permissions (nl-llm-agent-permissive-permissions)))
        (step 0) (status 'limit) (result nil))
    (when trace (funcall trace 0 'user task))
    (while (< step max-steps)
      (setq step (1+ step))
      (let* ((out (funcall policy messages))
             (action (nl-llm-agent--parse out)))
        (setq messages (append messages (list (cons 'assistant out))))
        (when trace (funcall trace step 'assistant out))
        (let ((res (nl-llm-agent--observe action workdir perms)))
          (when trace (funcall trace step 'observation (cdr res)))
          (cond
           ((eq (car res) 'done) (setq status 'done result (cdr res) step (1+ max-steps)))
           (t (setq messages (append messages (list (cons 'user (nl-llm-agent--truncate (cdr res) 4000))))))))))
    (list :status status :steps (min step max-steps) :result result :messages messages)))

;; ---- a tiny repo map (Aider-style, ranked context to feed the policy) -------

;;;###autoload
(defun nl-llm-agent-repo-map (dir &optional ext)
  "Return a compact map of DIR: each *.EXT (default \"el\") file with its top-level
definition names.  A cheap stand-in for Aider's tree-sitter/PageRank repo map, to
give the policy structure without dumping whole files."
  (let ((ext (or ext "el")) (lines nil))
    (dolist (f (sort (directory-files-recursively dir (concat "\\." ext "\\'")) #'string<))
      (let ((defs nil))
        (with-temp-buffer
          (insert-file-contents f)
          (goto-char (point-min))
          (while (re-search-forward "^(\\(?:cl-\\)?def\\(?:un\\|var\\|macro\\|custom\\|const\\)[* ]+\\([^ \t\n()]+\\)" nil t)
            (push (match-string 1) defs)))
        (push (format "%s: %s" (file-relative-name f dir)
                      (string-join (nreverse defs) " "))
              lines)))
    (string-join (nreverse lines) "\n")))

;; ---- scripted policy (for tests / demos: a stand-in for the model) ---------

;;;###autoload
(defun nl-llm-agent-scripted-policy (responses)
  "Return a policy that yields the queued RESPONSES (a list of strings) in order,
then repeats the last.  A deterministic stand-in for an LLM, used to exercise the
harness without a model."
  (let ((q (copy-sequence responses)) (last ""))
    (lambda (_messages)
      (if q (setq last (pop q)) last))))

(provide 'nl-llm-agent)
;;; nl-llm-agent.el ends here
