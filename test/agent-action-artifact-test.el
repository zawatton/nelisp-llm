;;; agent-action-artifact-test.el --- file-action artifact grammar -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(setq load-prefer-newer t)
(require 'ert)
(require 'cl-lib)
(require 'json)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)

(defun nl-llm-agent-action-artifact-test--decode (grammar desired)
  "Decode DESIRED through GRAMMAR using deterministic scripted logits."
  (let* ((tokens (nl-llm-agent-tokenizer-encode desired))
         (vocab (nl-llm-agent-tokenizer-vocab))
         (position 0))
    (cl-labels
        ((logits (index)
           (let ((values (make-vector vocab -100.0)))
             (when (< index (length tokens))
               (aset values (nth index tokens) 100.0))
             values))
         (step (id)
           (unless (and (< position (length tokens))
                        (= id (nth position tokens)))
             (error "scripted decoder consumed unexpected token %S" id))
           (setq position (1+ position))
           (logits position)))
      (let ((decoded
             (nl-llm-agent-constrained-generate
              (logits 0) #'step grammar)))
        (unless (= position (length tokens))
          (error "scripted decoder did not consume the complete action"))
        decoded))))

(ert-deftest nl-llm-agent-action-artifact-normalizes-and-deep-detaches ()
  (let* ((type (propertize "file-actions-v1" 'face 'bold))
         (allow (propertize "日本 ok" 'secret t))
         (source (list :type type :max-field 12 :allow allow))
         (normalized
          (nl-llm-agent-artifact-normalize-grammar source "action test")))
    (should (equal normalized
                   '(:type "file-actions-v1" :max-field 12
                     :allow "日本 ok")))
    (should-not (eq type (plist-get normalized :type)))
    (should-not (eq allow (plist-get normalized :allow)))
    (should-not (text-properties-at 0 (plist-get normalized :type)))
    (should-not (text-properties-at 0 (plist-get normalized :allow)))
    (put-text-property 0 1 'changed t type)
    (put-text-property 0 1 'changed t allow)
    (should (equal (plist-get normalized :type) "file-actions-v1"))
    (should (equal (plist-get normalized :allow) "日本 ok"))
    (put-text-property 0 1 'normalized-only t
                       (plist-get normalized :allow))
    (should-not (get-text-property 0 'normalized-only allow)))
  (let ((without-allow
         (nl-llm-agent-artifact-normalize-grammar
          '(:type "file-actions-v1" :max-field 8))))
    (should (equal without-allow
                   '(:type "file-actions-v1" :max-field 8)))
    (should-not (plist-member without-allow :allow))))

(ert-deftest nl-llm-agent-action-artifact-rejects-malformed-profile ()
  (dolist
      (bad
       (list
        '(:type "file-actions-v1")
        '(:type "file-actions-v1" :max-field 0)
        '(:type "file-actions-v1" :max-field 1025)
        '(:type "file-actions-v1" :max-field 1.5)
        '(:type "file-actions-v1" :max-field 8 :allow nil)
        '(:type "file-actions-v1" :max-field 8 :allow "")
        '(:type "file-actions-v1" :max-field 8 :allow "a\n")
        '(:type "file-actions-v1" :max-field 8 :allow "a")
        (list :type "file-actions-v1" :max-field 8
              :allow (make-string 513 ?a))
        '(:type "file-actions-v1" :max-field 8 :length 2)
        '(:type "file-actions-v1" :max-field 8 :segments ["x"])
        '(:type "file-actions-v1" :max-field 8 :unknown t)
        '(:type "file-actions-v1" :type "file-actions-v1" :max-field 8)
        '(:type "file-actions-v1" :max-field 8 :max-field 9)
        '(:type "file-actions-v1" :max-field 8 :allow "a" :allow "b")))
    (should-error (nl-llm-agent-artifact-normalize-grammar bad)))
  (let ((cyclic (list :type "file-actions-v1" :max-field 8)))
    (setcdr (last cyclic) cyclic)
    (should-error (nl-llm-agent-artifact-normalize-grammar cyclic))))

(ert-deftest nl-llm-agent-action-artifact-publish-reload-provider-decodes ()
  (let* ((directory (make-temp-file "nl-llm-action-artifact-" t))
         (catalog (expand-file-name "catalog.json" directory))
         (grammar
          '(:type "file-actions-v1" :max-field 8 :allow "ok "))
         (model (nl-llm-agent-improve-model 2 2 nil 1 1))
         (registry (nl-llm-agent-provider-registry-new)))
    (unwind-protect
        (progn
          (nl-llm-agent-artifact-publish
           catalog (nl-llm-agent-artifact-export-pav model)
           :id "action-g0" :name "File action grammar"
           :grammar grammar :maxseq 256 :score 0.0 :generation 0)
          (let* ((raw
                  (with-temp-buffer
                    (insert-file-contents catalog)
                    (json-parse-buffer :object-type 'plist
                                       :array-type 'array)))
                 (stored
                  (plist-get (aref (plist-get raw :models) 0) :grammar))
                 (entry
                  (car (plist-get
                        (nl-llm-agent-artifact--read catalog) :models))))
            (should (equal stored grammar))
            (should (equal (plist-get entry :grammar) grammar)))
          (nl-llm-agent-provider-register
           registry (nl-llm-agent-artifact-provider "native" catalog))
          (cl-letf
              (((symbol-function 'nl-llm-agent-model-policy)
                (lambda (_loaded grammar-callback _maxseq)
                  (lambda (_messages)
                    (nl-llm-agent-action-artifact-test--decode
                     grammar-callback "DONE ok\n")))))
            (let* ((session
                    (nl-llm-agent-session-open registry "native/action-g0"))
                   (reply
                    (nl-llm-agent-session-complete
                     session '((user . "finish")))))
              (should (equal reply "DONE ok\n"))
              (nl-llm-agent-session-close session)))
          ;; This tiny random model proves only that the real native policy
          ;; receives and follows the reloaded syntax grammar.
          (let* ((session
                  (nl-llm-agent-session-open registry "native/action-g0"))
                 (reply
                  (nl-llm-agent-session-complete
                   session '((user . "x"))))
                 (action (nl-llm-agent--parse reply)))
            (should (memq (car action) '(done tool)))
            (when (eq (car action) 'tool)
              (should (member (cadr action) '("read" "edit"))))
            (nl-llm-agent-session-close session)))
      (delete-directory directory t))))

(ert-run-tests-batch-and-exit)

;;; agent-action-artifact-test.el ends here
