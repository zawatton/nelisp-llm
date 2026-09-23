;;; agent-action-grammar-test.el --- bounded file action grammar tests -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/agent-action-grammar-test.el

(setq load-prefer-newer t)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-action-grammar)

(defvar aag--fail 0)

(defun aag--check (name ok &optional detail)
  (princ (format "%-57s %s%s\n" name
                 (if ok "PASS" (progn (setq aag--fail (1+ aag--fail)) "FAIL"))
                 (if detail (concat "  " detail) ""))))

(defun aag--errors-p (function)
  (condition-case nil
      (progn (funcall function) nil)
    (error t)))

(defun aag--scripted-generate (grammar desired &optional tokenizer)
  "Generate DESIRED through GRAMMAR using deterministic scripted logits."
  (let* ((tokenizer (nl-llm-agent-tokenizer-id tokenizer))
         (ids (nl-llm-agent-tokenizer-encode desired tokenizer))
         (vocab (nl-llm-agent-tokenizer-vocab tokenizer))
         (index 0))
    (cl-labels
        ((logits ()
           (let ((values (make-vector vocab -100.0)))
             (when (< index (length ids))
               (aset values (nth index ids) 100.0))
             values))
         (step (id)
           (unless (and (< index (length ids)) (= id (nth index ids)))
             (error "Scripted grammar consumed unexpected token %S at %d"
                    id index))
           (setq index (1+ index))
           (logits)))
      (let ((actual
             (nl-llm-agent-constrained-generate
              (logits) #'step grammar tokenizer)))
        (unless (= index (length ids))
          (error "Scripted grammar stopped after %d of %d tokens"
                 index (length ids)))
        actual))))

(let* ((grammar (nl-llm-agent-grammar-file-actions 32))
       (read-text
        (concat "```tool\n(:name \"read\" :arguments (:path \""
                "x```y\\n```z\\\"q\\\\r"
                "\"))\n```"))
       (edit-text
        (concat "```tool\n(:name \"edit\" :arguments (:path \"f.el\""
                " :search \"a\\\"b\\n\\\\c\" :replace \"\"))\n```"))
       (done-text "DONE quote=\" fence=``` slash=\\\n")
       (read-out (aag--scripted-generate grammar read-text))
       (edit-out (aag--scripted-generate grammar edit-text))
       (done-out (aag--scripted-generate grammar done-text)))
  (aag--check "one reusable grammar generates exact read syntax"
              (equal read-out read-text))
  (aag--check
   "read action parses escaped newline and adversarial fence text"
   (equal (nl-llm-agent-parse-action read-out)
          (list 'tool "read" (list :path "x```y\n```z\"q\\r"))))
  (aag--check "same grammar generates exact edit syntax"
              (equal edit-out edit-text))
  (aag--check
   "edit action parses quote, newline, backslash, and empty replace"
   (equal (nl-llm-agent-parse-action edit-out)
          (list 'tool "edit"
                (list :path "f.el" :search "a\"b\n\\c" :replace ""))))
  (aag--check "same grammar generates exact DONE syntax"
              (equal done-out done-text))
  (aag--check "DONE parser accepts literal quote/backslash/backticks"
              (equal (nl-llm-agent-parse-action done-out)
                     (list 'done "quote=\" fence=``` slash=\\"))))

(let* ((grammar (nl-llm-agent-grammar-file-actions 4 "日🙂 "))
       (text "DONE 日🙂\n")
       (out (aag--scripted-generate grammar text "utf8-byte-v1")))
  (aag--check "UTF-8 scripted logits generate Unicode DONE answer"
              (equal out text))
  (aag--check "Unicode DONE output parses"
              (equal (nl-llm-agent-parse-action out) '(done "日🙂"))))

(let* ((open "```tool\n(:name \"read\" :arguments (:path \"")
       (grammar (nl-llm-agent-grammar-file-actions 2 "ab")))
  (aag--check "nonempty path cannot close at zero characters"
              (not (nl-llm-agent-action-grammar--contains-p
                    (nth 1 (funcall grammar open)) ?\")))
  (aag--check "path may close after its minimum"
              (nl-llm-agent-action-grammar--contains-p
               (nth 1 (funcall grammar (concat open "a"))) ?\"))
  (aag--check "incomplete escape offers only complete escape suffixes"
              (equal (funcall grammar (concat open "a\\"))
                     '(:allow "nrt\"\\")))
  (aag--check "escaped character counts once at max boundary"
              (equal (funcall grammar (concat open "a\\n"))
                     '(:force ?\")))
  (let ((text (concat open "a\\n\""))
        (suffix "))\n```"))
    (aag--check "closing quote is valid exactly at max field length"
                (equal
                 (funcall grammar text)
                 (list :force (aref suffix 0)))))
  (aag--check "DONE may close with an empty answer"
              (nl-llm-agent-action-grammar--contains-p
               (nth 1 (funcall grammar "DONE ")) ?\n))
  (aag--check "DONE newline is forced at max field length"
              (equal (funcall grammar "DONE ab") '(:force ?\n))))

;; The constructor and each returned :allow value detach the retained alphabet.
(let* ((source (copy-sequence "ab"))
       (_property
        (when (fboundp 'put-text-property)
          (put-text-property 0 1 'unsafe (lambda () t) source)))
       (grammar (nl-llm-agent-grammar-file-actions 3 source)))
  (aset source 0 ?z)
  (let* ((first (funcall grammar "DONE "))
         (choices (nth 1 first)))
    (aset choices 0 ?x)
    (let ((second (nth 1 (funcall grammar "DONE "))))
      (aag--check "caller and returned alphabet mutation cannot affect grammar"
                  (and (nl-llm-agent-action-grammar--contains-p second ?a)
                       (not (nl-llm-agent-action-grammar--contains-p second ?z))
                       (not (nl-llm-agent-action-grammar--contains-p second ?x))
                       (or (not (fboundp 'text-properties-at))
                           (null (text-properties-at 0 second))))))))

(dolist (thunk
         (list
          (lambda () (nl-llm-agent-grammar-file-actions 0))
          (lambda () (nl-llm-agent-grammar-file-actions 1025))
          (lambda () (nl-llm-agent-grammar-file-actions 2.0))
          (lambda () (nl-llm-agent-grammar-file-actions 2 ""))
          (lambda () (nl-llm-agent-grammar-file-actions 2 (make-string 513 ?a)))
          (lambda () (nl-llm-agent-grammar-file-actions 2 "a\n"))
          (lambda () (nl-llm-agent-grammar-file-actions 2 (string #x85)))
          (lambda () (nl-llm-agent-grammar-file-actions 2 (string #xd800)))
          (lambda () (nl-llm-agent-grammar-file-actions 2 '(?a)))))
  (aag--check "invalid constructor input is rejected" (aag--errors-p thunk)))

(let* ((grammar (nl-llm-agent-grammar-file-actions 2 "ab"))
       (open "```tool\n(:name \"read\" :arguments (:path \"")
       (bad
        (list "X" "DOX" "`x" (concat open "\"")
              (concat open "a\\q") (concat open "aba")
              "DONE abx" "DONE a\ntrailing"
              (concat "```tool\n(:name \"read\" :arguments (:path \"a\"))\n```x")
              (make-string 7000 ?D))))
  (dolist (emitted bad)
    (aag--check "malformed emitted prefix is rejected"
                (aag--errors-p (lambda () (funcall grammar emitted)))))
  (aag--check "non-string emitted value is rejected"
              (aag--errors-p (lambda () (funcall grammar nil)))))

(when (> aag--fail 0)
  (error "agent action grammar: %d failure(s)" aag--fail))
(princ "agent action grammar: all checks passed\n")

;;; agent-action-grammar-test.el ends here
