;;; agent-evolve-test.el --- P5 queue to promoted artifact pipeline  -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-agent-evolve)

(defvar agent-evolve--fail 0)

(defun agent-evolve--ck (name ok)
  (princ (format "%-69s %s\n" name
                 (if ok "PASS"
                   (setq agent-evolve--fail (1+ agent-evolve--fail))
                   "FAIL"))))

(defun agent-evolve--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(defun agent-evolve--score (model)
  "Return negative next-token loss on one fixed trusted benchmark."
  (let* ((tokens
          (mapcar #'nl-llm-agent--char->id (append " a" nil)))
         (loss
          (nl-llm-agent--p5-forward
           model (butlast tokens) (apply #'vector (cdr tokens)))))
    (- (aref (photon-tensor-data (pav-value loss)) 0))))

(let* ((directory (make-temp-file "nl-llm-agent-evolve-" t))
       (catalog-file (expand-file-name "catalog.json" directory))
       (model (nl-llm-agent-improve-model 2 2 96 1 1))
       (queue
        (nl-llm-agent-evolve-p5-queue
         model #'agent-evolve--score catalog-file
         '(:type "done" :length 4 :allow "ab ")
         :id-prefix "self" :min-delta 0.0 :maxseq 128))
       (initial
        (nl-llm-evolution-champion-score
         (nl-llm-evolve-queue-evolution queue))))
  (unwind-protect
      (progn
        (agent-evolve--ck
         "pipeline exposes only the bounded trajectory fine-tune proposal"
         (equal
          (nl-llm-evolve-queue-catalog queue)
          '((:kind "trajectory-finetune"
             :description
             "Fine-tune an isolated P5 challenger on bounded successful trajectory text"))))
        (nl-llm-evolve-queue-submit
         queue "trajectory-finetune"
         '(:examples [" a"] :lr 0.1 :epochs 4)
         :id "learn-a")
        (let ((result (nl-llm-evolve-queue-run queue "learn-a")))
          (agent-evolve--ck
           "successful trajectory training passes the fixed evaluation gate"
           (and (eq (plist-get result :status) 'promoted)
                (equal
                 (plist-get
                  (plist-get result :result) :publication)
                 (car (nl-llm-agent-artifact-catalog catalog-file)))
                (> (nl-llm-evolution-champion-score
                    (nl-llm-evolve-queue-evolution queue))
                   initial))))
        (agent-evolve--ck
         "accepted P5 challenger is published as a digest-addressed artifact"
         (let ((catalog (nl-llm-agent-artifact-catalog catalog-file)))
           (and (= (length catalog) 1)
                (equal (plist-get (car catalog) :id) "self-g1")
                (= (plist-get (car catalog) :generation) 1)
                (= (length (plist-get (car catalog) :sha256)) 64))))
        (let* ((provider
                (nl-llm-agent-artifact-provider "native" catalog-file))
               (registry (nl-llm-agent-provider-registry-new)))
          (nl-llm-agent-provider-register registry provider)
          (let* ((session
                  (nl-llm-agent-session-open registry "native/self-g1"))
                 (reply
                  (nl-llm-agent-session-complete
                   session '((user . "use the promoted generation")))))
            (agent-evolve--ck
             "promoted self-trained generation performs constrained inference"
             (and (string-prefix-p "DONE " reply)
                  (= (length reply) 9)))))
        (agent-evolve--ck
         "unrepresentable or oversized training payload is rejected before queueing"
         (and
          (agent-evolve--error-p
           (lambda ()
             (nl-llm-evolve-queue-submit
              queue "trajectory-finetune"
              '(:examples ["日本語"] :lr 0.1 :epochs 1))))
          (= (plist-get (nl-llm-evolve-queue-status queue) :pending) 0))))
    (delete-directory directory t)))

(princ (format "NL-LLM-AGENT-EVOLVE %s (%d failures)\n"
               (if (= agent-evolve--fail 0) "ALL-PASS" "HAS-FAILURES")
               agent-evolve--fail))
(kill-emacs (if (= agent-evolve--fail 0) 0 1))

;;; agent-evolve-test.el ends here
