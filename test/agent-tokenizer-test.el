;;; agent-tokenizer-test.el --- native agent tokenizer tests -*- lexical-binding: t; -*-
;; emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-tokenizer-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-agent-tokenizer)

(defvar agent-tokenizer-test--fail 0)

(defun agent-tokenizer-test--ck (name ok &optional extra)
  (princ (format "%-60s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-tokenizer-test--fail
                         (1+ agent-tokenizer-test--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-tokenizer-test--errors (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(agent-tokenizer-test--ck
 "nil selects the canonical legacy tokenizer"
 (let* ((caller-id (copy-sequence "utf8-byte-v1"))
        (canonical (nl-llm-agent-tokenizer-id caller-id)))
   (aset caller-id 0 ?X)
   (let ((ok (and (equal canonical "utf8-byte-v1")
                  (= (nl-llm-agent-tokenizer-vocab) 96)
                  (= (nl-llm-agent-tokenizer-vocab "utf8-byte-v1") 256))))
     (aset canonical 0 ?X)
     (and ok
          (equal (nl-llm-agent-tokenizer-id "utf8-byte-v1")
                 "utf8-byte-v1")))))

(let ((ids (nl-llm-agent-tokenizer-encode " A\n~")))
  (agent-tokenizer-test--ck
   "legacy ASCII mapping and decoding remain exact"
   (and (equal ids '(0 33 95 94))
        (equal (nl-llm-agent-tokenizer-decode ids) " A\n~"))))

(let* ((text "日本語🙂")
       (ids (nl-llm-agent-tokenizer-encode text "utf8-byte-v1"))
       (again (copy-sequence ids)))
  (setcar ids 0)
  (agent-tokenizer-test--ck
   "Japanese and emoji round-trip as detached UTF-8 byte ids"
   (and (> (length again) (length text))
        (equal (nl-llm-agent-tokenizer-decode again "utf8-byte-v1") text)
        (equal (nl-llm-agent-tokenizer-encode text "utf8-byte-v1") again))))

(let* ((scalars '(0 #x7f #x80 #x7ff #x800 #xd7ff #xe000 #xffff
                    #x10000 #x10ffff))
       (text (apply #'string scalars))
       (expected (append (encode-coding-string text 'utf-8) nil))
       (actual (nl-llm-agent-tokenizer-encode text "utf8-byte-v1")))
  (agent-tokenizer-test--ck
   "Unicode scalar boundaries encode and strictly decode exactly"
   (and (equal actual expected)
        (equal (nl-llm-agent-tokenizer-decode actual "utf8-byte-v1")
               text))))

(agent-tokenizer-test--ck
 "ASCII explicitly rejects unsupported Unicode"
 (agent-tokenizer-test--errors
  (lambda () (nl-llm-agent-tokenizer-encode "日本語"))))

(agent-tokenizer-test--ck
 "raw unibyte characters and surrogate scalars are rejected"
 (and
  (agent-tokenizer-test--errors
   (lambda ()
     (nl-llm-agent-tokenizer-encode
      (unibyte-string #xff) "utf8-byte-v1")))
  (agent-tokenizer-test--errors
   (lambda ()
     (nl-llm-agent-tokenizer-encode
      (string #xd800) "utf8-byte-v1")))))

(let ((bad '((#xc0 #x80)
             (#xe3 #x81)
             (#xed #xa0 #x80)
             (#xf4 #x90 #x80 #x80)
             (#x80)
             (-1)
             (256)
             (1.5))))
  (agent-tokenizer-test--ck
   "malformed, overlong, truncated, and invalid byte ids are rejected"
   (let ((all t))
     (dolist (ids bad all)
       (unless (agent-tokenizer-test--errors
                (lambda ()
                  (nl-llm-agent-tokenizer-decode ids "utf8-byte-v1")))
         (setq all nil))))))

(agent-tokenizer-test--ck
 "unknown tokenizer identifiers and improper id lists are rejected"
 (and (agent-tokenizer-test--errors
       (lambda () (nl-llm-agent-tokenizer-id "unknown-v1")))
      (agent-tokenizer-test--errors
       (lambda () (nl-llm-agent-tokenizer-decode '(1 . 2))))))

(let* ((ids (list 1 2 3))
       (_ (setcdr (last ids) ids)))
  (agent-tokenizer-test--ck
   "circular token id lists are rejected without traversal"
   (agent-tokenizer-test--errors
    (lambda () (nl-llm-agent-tokenizer-decode ids "utf8-byte-v1")))))

(princ (format "NL-LLM-AGENT-TOKENIZER %s (%d failures)\n"
               (if (= agent-tokenizer-test--fail 0) "ALL-PASS" "HAS-FAILURES")
               agent-tokenizer-test--fail))
(kill-emacs (if (= agent-tokenizer-test--fail 0) 0 1))

;;; agent-tokenizer-test.el ends here
