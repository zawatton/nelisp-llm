;;; nl-llm-lora-ckpt.el --- LoRA adapter checkpoint save/load for nelisp-llm  -*- lexical-binding: t; -*-

;; Persist LoRA adapter sets separately from the base model checkpoint.  Each
;; adapter stores only its low-rank factors plus metadata, so one base model on
;; disk can be paired with many small task-specific adapters.  photon-tensors
;; are plain [shape data] vectors and floats round-trip exactly through
;; prin1/read, so adapter checkpoints serialise directly as sexps.

;;; Code:

(require 'photon-tensor)

(defconst nl-llm-lora-ckpt-format "nl-llm-lora-ckpt-v1"
  "Format tag for standalone LoRA adapter checkpoints.")

(defun nl-llm-lora-ckpt--shape-size (shape)
  "Return the flat element count for SHAPE."
  (let ((n 1))
    (dolist (dim shape n)
      (setq n (* n dim)))))

(defun nl-llm-lora-ckpt--require-key (adapter key site)
  "Return ADAPTER's KEY value, or signal an error naming SITE."
  (unless (plist-member adapter key)
    (error "nl-llm-lora-ckpt: adapter %S is missing %S" site key))
  (plist-get adapter key))

(defun nl-llm-lora-ckpt--validate-tensor (site label tensor expected-shape)
  "Validate TENSOR for SITE LABEL against EXPECTED-SHAPE."
  (unless (and (vectorp tensor) (= (length tensor) 2))
    (error "nl-llm-lora-ckpt: adapter %S %s is not a photon-tensor: %S"
           site label tensor))
  (let ((shape (photon-tensor-shape tensor))
        (data (photon-tensor-data tensor)))
    (unless (equal shape expected-shape)
      (error "nl-llm-lora-ckpt: adapter %S %s has shape %S, expected %S"
             site label shape expected-shape))
    (unless (vectorp data)
      (error "nl-llm-lora-ckpt: adapter %S %s data is not a vector: %S"
             site label data))
    (unless (= (length data) (nl-llm-lora-ckpt--shape-size expected-shape))
      (error "nl-llm-lora-ckpt: adapter %S %s data length %d does not match shape %S"
             site label (length data) expected-shape))))

(defun nl-llm-lora-ckpt--validate-adapter (site adapter)
  "Validate one ADAPTER at SITE."
  (unless (or (symbolp site) (stringp site))
    (error "nl-llm-lora-ckpt: invalid adapter site key %S" site))
  (unless (listp adapter)
    (error "nl-llm-lora-ckpt: adapter %S is not a plist: %S" site adapter))
  (let* ((a     (nl-llm-lora-ckpt--require-key adapter :a site))
         (b     (nl-llm-lora-ckpt--require-key adapter :b site))
         (rank  (nl-llm-lora-ckpt--require-key adapter :rank site))
         (alpha (nl-llm-lora-ckpt--require-key adapter :alpha site))
         (in    (nl-llm-lora-ckpt--require-key adapter :in site))
         (out   (nl-llm-lora-ckpt--require-key adapter :out site)))
    (unless (integerp rank)
      (error "nl-llm-lora-ckpt: adapter %S rank is not an integer: %S" site rank))
    (unless (>= rank 1)
      (error "nl-llm-lora-ckpt: adapter %S rank must be >= 1, got %S" site rank))
    (unless (numberp alpha)
      (error "nl-llm-lora-ckpt: adapter %S alpha is not numeric: %S" site alpha))
    (unless (and (integerp in) (>= in 1))
      (error "nl-llm-lora-ckpt: adapter %S in must be a positive integer, got %S"
             site in))
    (unless (and (integerp out) (>= out 1))
      (error "nl-llm-lora-ckpt: adapter %S out must be a positive integer, got %S"
             site out))
    (unless (<= rank (min in out))
      (error "nl-llm-lora-ckpt: adapter %S rank %d exceeds min(in,out)=%d"
             site rank (min in out)))
    (nl-llm-lora-ckpt--validate-tensor site "A" a (list rank in))
    (nl-llm-lora-ckpt--validate-tensor site "B" b (list out rank)))
  adapter)

(defun nl-llm-lora-ckpt--validate (adapters)
  "Validate ADAPTERS and return them."
  (unless (listp adapters)
    (error "nl-llm-lora-ckpt: adapters must be an alist, got %S" adapters))
  (let ((rest adapters))
    (while rest
      (let ((cell (car rest)))
        (unless (consp cell)
          (error "nl-llm-lora-ckpt: bad adapter entry %S" cell))
        (nl-llm-lora-ckpt--validate-adapter (car cell) (cdr cell)))
      (setq rest (cdr rest))))
  adapters)

;;;###autoload
(defun nl-llm-lora-ckpt-save (path adapters &optional meta)
  "Save ADAPTERS to PATH as a standalone LoRA checkpoint.

ADAPTERS is an alist mapping opaque site keys to adapter plists.
META is written verbatim under the :meta key."
  (nl-llm-lora-ckpt--validate adapters)
  (let ((form (list :format nl-llm-lora-ckpt-format
                    :meta meta
                    :adapters adapters)))
    (with-temp-buffer
      (let ((print-length nil)
            (print-level nil))
        (prin1 form (current-buffer)))
      (let ((coding-system-for-write 'utf-8))
        (write-region (point-min) (point-max) path nil 'silent)))
    path))

;;;###autoload
(defun nl-llm-lora-ckpt-load (path)
  "Load the LoRA checkpoint at PATH and return its plist."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8))
      (insert-file-contents path))
    (goto-char (point-min))
    (let* ((ckpt (read (current-buffer)))
           (found (plist-get ckpt :format)))
      (unless (equal found nl-llm-lora-ckpt-format)
        (error "nl-llm-lora-ckpt: bad/unknown format %S (expected %S)"
               found nl-llm-lora-ckpt-format))
      (unless (plist-member ckpt :adapters)
        (error "nl-llm-lora-ckpt: checkpoint missing :adapters"))
      (nl-llm-lora-ckpt--validate (plist-get ckpt :adapters))
      ckpt)))

;;;###autoload
(defun nl-llm-lora-ckpt-describe (ckpt-or-path)
  "Return a human-readable summary of CKPT-OR-PATH.

CKPT-OR-PATH may be a loaded checkpoint plist or a file path.  When called
interactively, also display the summary with `message'."
  (interactive "fLoRA checkpoint: ")
  (let* ((ckpt (if (stringp ckpt-or-path)
                   (nl-llm-lora-ckpt-load ckpt-or-path)
                 ckpt-or-path))
         (adapters (plist-get ckpt :adapters))
         (lines nil)
         (adapter-floats 0)
         (full-floats 0))
    (nl-llm-lora-ckpt--validate adapters)
    (dolist (cell adapters)
      (let* ((site (car cell))
             (adapter (cdr cell))
             (rank (plist-get adapter :rank))
             (alpha (plist-get adapter :alpha))
             (in (plist-get adapter :in))
             (out (plist-get adapter :out))
             (trainable (* rank (+ in out))))
        (push (format "Site %S: shape (%d x %d), rank %d, alpha %s, trainable %d"
                      site out in rank alpha trainable)
              lines)
        (setq adapter-floats (+ adapter-floats trainable))
        (setq full-floats (+ full-floats (* in out)))))
    (let* ((ratio (if (= full-floats 0)
                      0.0
                    (* 100.0 (/ (float adapter-floats) full-floats))))
           (text (mapconcat #'identity
                            (append (nreverse lines)
                                    (list (format
                                           "Total: adapter floats %d, full-weight floats %d, ratio %.6f%%"
                                           adapter-floats full-floats ratio)))
                            "\n")))
      (when (called-interactively-p 'interactive)
        (message "%s" text))
      text)))

;;;###autoload
(defun nl-llm-lora-ckpt-merge-plists (a b)
  "Return a new adapter alist combining A and B, with B winning duplicates."
  (let ((merged (mapcar #'copy-tree b))
        (rest (reverse a)))
    (while rest
      (unless (assoc (caar rest) merged)
        (push (copy-tree (car rest)) merged))
      (setq rest (cdr rest)))
    merged))

(provide 'nl-llm-lora-ckpt)
;;; nl-llm-lora-ckpt.el ends here
