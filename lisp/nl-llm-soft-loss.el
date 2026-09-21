;;; nl-llm-soft-loss.el --- KL loss for teacher top-k targets  -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'nl-llm-token-table)
(require 'photon-tensor)
(require 'photon-autograd)

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

;;;###autoload
(defun nl-llm-soft-loss-targets-bytes (position table)
  "Return representable targets for POSITION using TABLE's byte keys.

The sampled token must resolve, while missing alternatives are dropped just as
in `nl-llm-soft-loss-targets'."
  (when (nl-llm-token-table-id table position)
    (let (targets)
      (dolist (alternative (plist-get position :top) (nreverse targets))
        (let ((id (nl-llm-token-table-id table alternative)))
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

;;;###autoload
(defun nl-llm-ag-soft-kl (logits row-targets mask)
  "Mean top-k soft KL for the selected rows of LOGITS.
ROW-TARGETS and MASK are vectors with one entry per logit row.  A row is
included only when MASK selects it and its target is non-nil; this keeps an
unrepresentable teacher position out of both the value and the gradient."
  (unless (pav-p logits)
    (error "Soft KL logits must be an autograd value"))
  (let* ((lv (pav-value logits))
         (shape (photon-tensor-shape lv))
         (rows (and (consp shape) (car shape)))
         (vocab (and (consp (cdr shape)) (nth 1 shape))))
    (unless (and (vectorp lv) (= (length shape) 2)
                 (integerp rows) (> rows 0)
                 (integerp vocab) (> vocab 0)
                 (= (length (photon-tensor-data lv)) (* rows vocab)))
      (error "Soft KL logits must contain a nonempty 2D tensor"))
    (unless (and (vectorp row-targets) (= (length row-targets) rows))
      (error "Soft KL row targets must be a vector of length %d" rows))
    (unless (and (vectorp mask) (= (length mask) rows))
      (error "Soft KL mask must be a vector of length %d" rows))
    (let ((saved-targets (copy-sequence row-targets))
          (saved-mask (copy-sequence mask))
          (count 0)
          (loss 0.0)
          (row 0)
          (ld (photon-tensor-data lv)))
      (while (< row rows)
        (let ((bit (aref saved-mask row))
              (targets (aref saved-targets row)))
          (unless (and (integerp bit) (or (= bit 0) (= bit 1)))
            (error "Soft KL mask value %S at row %d is not 0 or 1" bit row))
          (when (and (= bit 1) targets)
            (setq count (1+ count))
            (let ((row-logits (make-vector vocab 0.0))
                  (base (* row vocab)))
              (dotimes (j vocab) (aset row-logits j (aref ld (+ base j))))
              (setq loss (+ loss (nl-llm-soft-loss-kl row-logits targets)))))
        (setq row (1+ row))))
      (when (= count 0)
        (error "Soft KL requires at least one selected row with targets"))
      (setq loss (/ loss (float count)))
      (photon-autograd--record
       (photon-tensor (list 1 1) (vector loss))
       (lambda (g)
         (let* ((upstream (aref (photon-tensor-data g) 0))
               (scale (/ upstream (float count)))
               (gradient (make-vector (* rows vocab) 0.0))
               (i 0))
           (while (< i rows)
             (let ((targets (aref saved-targets i)))
               (when (and (= (aref saved-mask i) 1) targets)
                 (let ((row-logits (make-vector vocab 0.0))
                       (base (* i vocab)))
                   (dotimes (j vocab)
                     (aset row-logits j (aref ld (+ base j))))
                   (let ((row-gradient
                          (nl-llm-soft-loss-kl-grad row-logits targets)))
                     (dotimes (j vocab)
                       (aset gradient (+ base j)
                            (* scale (aref row-gradient j))))))))
             (setq i (1+ i)))
           (photon-autograd--addgrad
            logits (photon-tensor (list rows vocab) gradient))))))))

(provide 'nl-llm-soft-loss)
;;; nl-llm-soft-loss.el ends here
