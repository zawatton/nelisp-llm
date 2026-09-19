;;; nl-llm-agent-completion-plan.el --- canonical completion training plans -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'nl-llm-agent-tokenizer)

(defconst nl-llm-agent-completion-plan-format
  "nl-llm-completion-plan-v1")
(defconst nl-llm-agent-completion-plan--max-trajectories 128)
(defconst nl-llm-agent-completion-plan--max-sequence 4096)
(defconst nl-llm-agent-completion-plan--uint32-mask #xffffffff)
(defconst nl-llm-agent-completion-plan--keys
  '(:format :tokenizer :vocab :sequence :pad-token :learning-rate :epochs
    :optimizer :transfer-mode :trajectories :loss-starts :loss-masks
    :shuffle-seed :digest))

(defun nl-llm-agent-completion-plan--ordinary-vector-p (value)
  "Return non-nil when VALUE is a normal vector, not a bool-vector."
  (and (vectorp value)
       (not (and (fboundp 'bool-vector-p) (bool-vector-p value)))))

(defun nl-llm-agent-completion-plan--elements (value maximum where)
  "Copy bounded sequence VALUE into a list, rejecting dotted/cyclic input."
  (cond
   ((nl-llm-agent-completion-plan--ordinary-vector-p value)
    (when (> (length value) maximum)
      (error "%s exceeds %d elements" where maximum))
    (let ((index 0) result)
      (while (< index (length value))
        (push (aref value index) result)
        (setq index (1+ index)))
      (nreverse result)))
   ((listp value)
    (let ((tail value) (slow value) (fast value) (result nil) (count 0))
      (while (consp tail)
        ;; Floyd's tortoise/hare check keeps hostile long lists linear rather
        ;; than repeatedly searching an ever-growing list of cons cells.
        (when (and (consp fast) (consp (cdr fast)))
          (setq slow (cdr slow)
                fast (cdr (cdr fast)))
          (when (eq slow fast)
            (error "%s must not be circular" where)))
        (setq count (1+ count))
        (when (> count maximum)
          (error "%s exceeds %d elements" where maximum))
        (push (car tail) result)
        (setq tail (cdr tail)))
      (unless (null tail)
        (error "%s must be a proper list or vector" where))
      (nreverse result)))
   (t (error "%s must be a proper list or vector" where))))

(defun nl-llm-agent-completion-plan--optimizer (value)
  "Normalize supported optimizer VALUE."
  (let ((value (if (stringp value) (intern value) value)))
    (unless (memq value '(sgd adam))
      (error "completion plan has unsupported optimizer %S" value))
    value))

(defun nl-llm-agent-completion-plan--transfer-mode (value)
  "Normalize dense or compact transfer VALUE."
  (let ((value (if (stringp value) (intern value) value)))
    (cond
     ((null value) nil)
     ;; Dense is the historical nil path; accept its descriptive spellings
     ;; but keep one canonical semantic representation in the plan.
     ((memq value '(dense dense->dense)) nil)
     ((eq value 'compact) 'compact)
     (t (error "completion plan has unsupported transfer mode %S" value)))))

(defun nl-llm-agent-completion-plan--finite-rate (value)
  "Normalize finite learning-rate VALUE in (0, 1]."
  (unless (and (numberp value) (= value value) (> value 0.0) (<= value 1.0))
    (error "completion plan learning rate must be finite and in (0, 1]"))
  (float value))

(defun nl-llm-agent-completion-plan--plist-keys (value)
  "Validate the bounded proper plist shape of VALUE and return its keys."
  (unless (listp value)
    (error "completion plan must be a plist"))
  (let ((tail value) (seen-cells nil) (keys nil) (count 0))
    (while tail
      (unless (consp tail)
        (error "completion plan must be a proper plist"))
      (when (cl-some (lambda (cell) (eq cell tail)) seen-cells)
        (error "completion plan must not be circular"))
      (push tail seen-cells)
      (setq count (1+ count))
      (when (> count (* 2 (length nl-llm-agent-completion-plan--keys)))
        (error "completion plan has too many fields"))
      (let ((key (car tail)))
        (unless (memq key nl-llm-agent-completion-plan--keys)
          (error "completion plan contains unknown key %S" key))
        (when (memq key keys)
          (error "completion plan contains duplicate key %S" key))
        (push key keys))
      (setq tail (cdr tail))
      (unless tail
        (error "completion plan has an odd number of plist elements"))
      (setq tail (cdr tail)))
    (dolist (key nl-llm-agent-completion-plan--keys)
      (unless (memq key keys)
        (error "completion plan is missing key %S" key)))
    keys))

(defun nl-llm-agent-completion-plan--digest (plan)
  "Return the stable SHA-256 digest of semantic PLAN fields."
  (let ((print-length nil)
        (print-level nil)
        (print-circle nil)
        (print-escape-nonascii t)
        (float-output-format "%.17g"))
    (secure-hash
     'sha256
     (prin1-to-string
      (list :format (plist-get plan :format)
            :tokenizer (plist-get plan :tokenizer)
            :vocab (plist-get plan :vocab)
            :sequence (plist-get plan :sequence)
            :pad-token (plist-get plan :pad-token)
            :learning-rate (plist-get plan :learning-rate)
            :epochs (plist-get plan :epochs)
            :optimizer (plist-get plan :optimizer)
            :transfer-mode (plist-get plan :transfer-mode)
            :trajectories (plist-get plan :trajectories)
            :loss-starts (plist-get plan :loss-starts)
            :loss-masks (plist-get plan :loss-masks)
            :shuffle-seed (plist-get plan :shuffle-seed))))))

;;;###autoload
(cl-defun nl-llm-agent-completion-plan-make
    (trajectories loss-starts
     &key (tokenizer "ascii-char-v1") sequence learning-rate epochs
     (optimizer 'sgd) transfer-mode loss-masks shuffle-seed)
  "Build a detached canonical completion-only training plan.
This function validates data and bindings only.  It does not allocate a model,
GPU context, callback, or training worker."
  (let* ((tokenizer
          (copy-sequence (nl-llm-agent-tokenizer-id tokenizer)))
         (vocab (nl-llm-agent-tokenizer-vocab tokenizer))
         (trajectory-list
          (nl-llm-agent-completion-plan--elements
           trajectories nl-llm-agent-completion-plan--max-trajectories
           "completion trajectories"))
         (count (length trajectory-list)))
    (unless (and (> count 0) (<= count nl-llm-agent-completion-plan--max-trajectories))
      (error "completion trajectories must contain 1..%d entries"
             nl-llm-agent-completion-plan--max-trajectories))
    (unless (and (integerp sequence)
                 (<= 2 sequence)
                 (<= sequence nl-llm-agent-completion-plan--max-sequence))
      (error "completion plan sequence must be an integer in [2, %d]"
             nl-llm-agent-completion-plan--max-sequence))
    (unless (and (nl-llm-agent-completion-plan--ordinary-vector-p loss-starts)
                 (= (length loss-starts) count))
      (error "completion loss starts must be an ordinary vector of length %d"
             count))
    (let ((starts (make-vector count 0))
          (out-trajectories (make-vector count nil))
          (total 0))
      (dotimes (index count)
        (let* ((items
                (nl-llm-agent-completion-plan--elements
                 (nth index trajectory-list) sequence
                 (format "completion trajectory %d" index)))
               (length (length items))
               (start (aref loss-starts index))
               (tokens (make-vector length 0)))
          (unless (and (<= 2 length) (<= length sequence))
            (error "completion trajectory %d length must be in [2, %d]"
                   index sequence))
          (unless (and (integerp start) (<= 1 start) (< start length))
            (error "completion loss start %S for trajectory %d must be in [1, %d]"
                   start index (1- length)))
          (let ((token-index 0))
            (dolist (token items)
              (unless (and (integerp token) (<= 0 token) (< token vocab))
                (error "completion trajectory %d has invalid token %S"
                       index token))
              (aset tokens token-index token)
              (setq token-index (1+ token-index))))
          (setq total (+ total length))
          (when (> total (* nl-llm-agent-completion-plan--max-trajectories
                            nl-llm-agent-completion-plan--max-sequence))
            (error "completion plan aggregate token bound exceeded"))
          (aset starts index start)
          (aset out-trajectories index tokens)))
      (let ((masks nil))
        (when loss-masks
          (unless (and (nl-llm-agent-completion-plan--ordinary-vector-p loss-masks)
                       (= (length loss-masks) count))
            (error "completion loss masks must be an ordinary vector of length %d"
                   count))
          (setq masks (make-vector count nil))
          (dotimes (index count)
            (let* ((mask (aref loss-masks index))
                   (trajectory (aref out-trajectories index))
                   (start (aref starts index))
                   (length (length trajectory))
                   (copy (make-vector length 0))
                   (selected 0))
              (unless (nl-llm-agent-completion-plan--ordinary-vector-p mask)
                (error "completion loss mask %d must be an ordinary vector" index))
              (unless (= (length mask) length)
                (error "completion loss mask %d must have length %d" index length))
              (dotimes (mask-index length)
                (let ((value (aref mask mask-index)))
                  (unless (and (integerp value) (or (= value 0) (= value 1)))
                    (error "completion loss mask %d has invalid value %S"
                           index value))
                  (when (and (< mask-index start) (= value 1))
                    (error "completion loss mask %d selects prompt index %d"
                           index mask-index))
                  (when (and (>= mask-index start) (= value 1))
                    (setq selected (1+ selected)))
                  (aset copy mask-index value)))
              (when (= selected 0)
                (error "completion loss mask %d selects no completion target"
                       index))
              (aset masks index copy))))
        (let* ((pad-token (car (nl-llm-agent-tokenizer-encode " " tokenizer)))
               (plan
                (list :format (copy-sequence nl-llm-agent-completion-plan-format)
                      :tokenizer tokenizer :vocab vocab :sequence sequence
                      :pad-token pad-token
                      :learning-rate
                      (nl-llm-agent-completion-plan--finite-rate learning-rate)
                      :epochs (if (and (integerp epochs) (<= 1 epochs) (<= epochs 32))
                                  epochs
                                (error "completion plan epochs must be in [1, 32]"))
                      :optimizer
                      (nl-llm-agent-completion-plan--optimizer optimizer)
                      :transfer-mode
                      (nl-llm-agent-completion-plan--transfer-mode transfer-mode)
                      :trajectories out-trajectories :loss-starts starts
                      :loss-masks masks :shuffle-seed
                      (when shuffle-seed
                        (unless (and (integerp shuffle-seed)
                                     (<= 1 shuffle-seed)
                                     (<= shuffle-seed
                                         nl-llm-agent-completion-plan--uint32-mask))
                          (error "completion plan shuffle seed must be in [1, #xffffffff]"))
                        shuffle-seed)
                      :digest nil)))
          (setf (plist-get plan :digest)
                (nl-llm-agent-completion-plan--digest plan))
          plan)))))

;;;###autoload
(defun nl-llm-agent-completion-plan-validate (plan)
  "Validate PLAN, reject tampering, and return a detached canonical copy."
  (nl-llm-agent-completion-plan--plist-keys plan)
  (unless (equal (plist-get plan :format)
                 nl-llm-agent-completion-plan-format)
    (error "unsupported completion plan format"))
  (let ((canonical
         (nl-llm-agent-completion-plan-make
          (plist-get plan :trajectories)
          (plist-get plan :loss-starts)
          :tokenizer (plist-get plan :tokenizer)
          :sequence (plist-get plan :sequence)
          :learning-rate (plist-get plan :learning-rate)
          :epochs (plist-get plan :epochs)
          :optimizer (plist-get plan :optimizer)
          :transfer-mode (plist-get plan :transfer-mode)
          :loss-masks (plist-get plan :loss-masks)
          :shuffle-seed (plist-get plan :shuffle-seed))))
    (dolist (key nl-llm-agent-completion-plan--keys)
      (unless (equal (plist-get plan key) (plist-get canonical key))
        (error "completion plan field %S was tampered" key)))
    canonical))

(provide 'nl-llm-agent-completion-plan)
;;; nl-llm-agent-completion-plan.el ends here
