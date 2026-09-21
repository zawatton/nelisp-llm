;;; nl-llm-soft-loss.el --- KL loss for teacher top-k targets  -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)

(defun nl-llm-soft-loss-targets (position table)
  "Return representable `(ID . LOGPROB)' targets for POSITION.

TABLE maps teacher token strings to student ids.  A position whose sampled
token is not representable has no usable target, even when alternatives are."
  (when (gethash (plist-get position :token) table)
    (let (targets)
      (dolist (alternative (plist-get position :top) (nreverse targets))
        (let ((id (gethash (plist-get alternative :token) table)))
          (when id
            (push (cons id (float (plist-get alternative :logprob))) targets)))))))

(defun nl-llm-soft-loss--normalised (values)
  "Return VALUES exponentiated and normalised after max subtraction."
  (let* ((maximum (apply #'max values))
         (weights (mapcar (lambda (value) (exp (- value maximum))) values))
         (sum (apply #'+ weights)))
    (mapcar (lambda (weight) (/ weight sum)) weights)))

(defun nl-llm-soft-loss--components (logits targets)
  "Return target ids, teacher probabilities and student probabilities."
  (when (null targets)
    (signal 'error '("soft KL loss requires at least one target")))
  (let* ((ids (mapcar #'car targets))
         (teacher-logprobs (mapcar #'cdr targets))
         (student-logits (mapcar (lambda (id) (aref logits id)) ids)))
    (list ids
          (nl-llm-soft-loss--normalised teacher-logprobs)
          (nl-llm-soft-loss--normalised student-logits))))

;;;###autoload
(defun nl-llm-soft-loss-kl (logits targets)
  "Return KL(Q || P), renormalised over the ids in TARGETS."
  (pcase-let ((`(,ids ,teacher ,student)
               (nl-llm-soft-loss--components logits targets)))
    (ignore ids)
    (let ((loss 0.0))
      (cl-mapc (lambda (q p)
                 (setq loss (+ loss (* q (- (log q) (log p))))))
               teacher student)
      (float loss))))

;;;###autoload
(defun nl-llm-soft-loss-kl-grad (logits targets)
  "Return the gradient of `nl-llm-soft-loss-kl' with respect to LOGITS."
  (pcase-let ((`(,ids ,teacher ,student)
               (nl-llm-soft-loss--components logits targets)))
    (let ((gradient (make-vector (length logits) 0.0)))
      (cl-mapc (lambda (id q p)
                 (aset gradient id (- p q)))
               ids teacher student)
      gradient)))

(provide 'nl-llm-soft-loss)
;;; nl-llm-soft-loss.el ends here
