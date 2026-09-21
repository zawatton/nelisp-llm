;;; nl-llm-student.el --- student models for teacher soft targets  -*- lexical-binding: t; -*-

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-soft-loss)
(require 'nl-llm-token-table)

;;;###autoload
(cl-defun nl-llm-student-new (&key dim ff blocks heads vocab tokenizer)
  "Build a trainable P5 student with VOCAB supplied independently of TOKENIZER."
  (let* ((dim (or dim 24))
         (ff (or ff dim))
         (blocks (or blocks 1))
         (heads (or heads 1))
         (tokenizer (or tokenizer 'qwen-byte))
         (vocab (or vocab
                    (and (stringp tokenizer)
                         (ignore-errors
                           (nl-llm-agent-tokenizer-vocab tokenizer))))))
    (unless (and (integerp vocab) (> vocab 0))
      (error "student vocab must be a positive integer"))
    (list :wte (nl-llm-agent--p5-p
                (list vocab dim) 1 (/ 1.0 (sqrt (float dim))))
          :blocks (cl-loop for i below blocks
                           collect (nl-llm-agent--p5-block dim ff (* 20 (1+ i))))
          :lnfg (nl-llm-agent--p5-c dim 1.0)
          :wh (nl-llm-agent--p5-p
               (list vocab dim) 9 (/ 1.0 (sqrt (float dim))))
          :bh (nl-llm-agent--p5-c vocab 0.0)
          :dim dim :ff ff :vocab vocab :heads heads :nblocks blocks
          :tokenizer tokenizer)))

(defun nl-llm-student--example-data (example tokenize table)
  "Return encoded training data for one soft-target EXAMPLE."
  (let* ((prompt (plist-get example :prompt))
         (completion (plist-get example :completion))
         (prompt-ids (funcall tokenize prompt))
         (completion-ids (funcall tokenize completion))
         (trajectory (append prompt-ids completion-ids))
         (rows (length (butlast trajectory)))
         (targets (make-vector rows nil))
         (mask (make-vector rows 0))
         (positions (plist-get example :tokens))
         (with-targets 0)
         (position-count (length positions)))
    (dotimes (i (length completion-ids))
      (let ((row (+ (length prompt-ids) i -1)))
        (when (and (>= row 0) (< row rows))
          (aset mask row 1)
          (when (< i position-count)
            (let ((target (nl-llm-soft-loss-targets-bytes
                          (nth i positions) table)))
              (when target
                (aset targets row target)
                (setq with-targets (1+ with-targets))))))))
    (list :tokens (butlast trajectory) :targets targets :mask mask
          :positions position-count :positions-with-targets with-targets
          :rows rows)))

(defun nl-llm-student--loss-summary (model data)
  "Return (LOSS . COUNT) for encoded DATA without changing MODEL."
  (let ((total 0.0) count)
    (dolist (item data)
      (when (> (plist-get item :positions-with-targets) 0)
        (let ((loss (nl-llm-ag-soft-kl
                     (nl-llm-agent--p5-forward model (plist-get item :tokens))
                     (plist-get item :targets) (plist-get item :mask)))
              (n (plist-get item :positions-with-targets)))
          (setq total (+ total (* n (aref (photon-tensor-data (pav-value loss)) 0)))
                count (+ (or count 0) n)))))
    (cons (if (and count (> count 0)) (/ total (float count)) 0.0)
          (or count 0))))

;;;###autoload
(cl-defun nl-llm-soft-train (model examples &key lr epochs tokenize table)
  "Train MODEL on completion-only teacher soft targets in EXAMPLES."
  (let* ((lr (or lr 0.05))
         (epochs (or epochs 1))
         (tokenize (or tokenize (lambda (text) (string-to-list text))))
         (data (mapcar (lambda (example)
                         (nl-llm-student--example-data example tokenize table))
                       (append examples nil)))
         (parameters (nl-llm-agent--p5-params model))
         (before (nl-llm-student--loss-summary model data))
         (positions (apply #'+ 0 (mapcar (lambda (x) (plist-get x :positions)) data)))
         (with-targets
          (apply #'+ 0 (mapcar (lambda (x) (plist-get x :positions-with-targets)) data))))
    (dotimes (_ epochs)
      (dolist (item data)
        (when (> (plist-get item :positions-with-targets) 0)
          (let ((loss (nl-llm-ag-soft-kl
                       (nl-llm-agent--p5-forward model (plist-get item :tokens))
                       (plist-get item :targets) (plist-get item :mask))))
            (photon-autograd-zero-grad parameters)
            (photon-autograd-backward loss)
            (photon-autograd-sgd parameters lr)))))
    (let ((after (nl-llm-student--loss-summary model data)))
      (list :examples (length data)
            :positions positions
            :positions-with-targets with-targets
            :loss-before (car before)
            :loss-after (car after)))))

(provide 'nl-llm-student)
;;; nl-llm-student.el ends here
