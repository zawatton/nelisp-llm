;;; agent-native-provider-test.el --- NeLisp model provider adapter  -*- lexical-binding: t; -*-

;; The native model adapter exposes safe catalog metadata and turns the existing
;; nelisp-llm policy builder into the same session contract as remote providers.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
;;     -l test/agent-native-provider-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-model)

(defvar agent-native-provider--fail 0)

(defun agent-native-provider--ck (name ok &optional extra)
  (princ (format "%-58s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-native-provider--fail
                         (1+ agent-native-provider--fail))
                   "FAIL")
                 (or extra ""))))

(let ((builder-call nil))
  (cl-letf (((symbol-function 'nl-llm-agent-model-policy)
             (lambda (model grammar maxseq)
               (setq builder-call (list model grammar maxseq))
               (lambda (messages)
                 (format "native:%s:%d"
                         (plist-get model :label)
                         (length messages))))))
    (let* ((model '(:label "tiny"))
           (grammar (lambda (_emitted) :stop))
           (provider
            (nl-llm-agent-model-provider
             "nelisp"
             (list (list :id "tiny"
                         :name "Tiny local model"
                         :model model
                         :grammar grammar
                         :maxseq 256
                         :capabilities '(generate train evolve)))))
           (registry (nl-llm-agent-provider-registry-new)))
      (nl-llm-agent-provider-register registry provider)
      (let ((public (car (nl-llm-agent-provider-models registry))))
        (agent-native-provider--ck
         "native catalog exposes descriptive metadata"
         (and (equal (plist-get public :qualified-id) "nelisp/tiny")
              (equal (plist-get public :name) "Tiny local model")
              (equal (plist-get public :capabilities)
                     '(generate train evolve))))
        (agent-native-provider--ck
         "native catalog hides model and grammar objects"
         (and (not (plist-member public :model))
              (not (plist-member public :grammar)))))
      (let* ((session
              (nl-llm-agent-session-open
               registry "nelisp/tiny" :options '(:maxseq 512)))
             (policy (nl-llm-agent-session-policy session))
             (out (funcall policy '((user . "hello")))))
        (agent-native-provider--ck
         "native adapter delegates through the existing policy builder"
         (and (equal builder-call (list model grammar 512))
              (equal out "native:tiny:1")))
        (agent-native-provider--ck
         "native provider advertises local constrained generation"
         (equal (nl-llm-agent-provider-capabilities provider)
                '(generate local constrained-decoding)))))))

(princ (format "NL-LLM-AGENT-NATIVE-PROVIDER %s (%d failures)\n"
               (if (= agent-native-provider--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-native-provider--fail))
(kill-emacs (if (= agent-native-provider--fail 0) 0 1))

;;; agent-native-provider-test.el ends here
