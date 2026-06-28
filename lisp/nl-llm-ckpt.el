;;; nl-llm-ckpt.el --- model checkpoint save/load for nelisp-llm  -*- lexical-binding: t; -*-

;; Persist a trained modern model (embedding, blocks with biases, final RMSNorm
;; gain, tied-head bias) plus its config and step to a sexp file, and load it
;; back -- so training can be checkpointed and resumed, and a trained model
;; shared or used for inference later.  photon-tensors are plain [shape data]
;; vectors and floats round-trip exactly through prin1/read (Emacs 26+), so the
;; whole model plist serialises directly.

;;; Code:

(defconst nl-llm-ckpt-format "nl-llm-ckpt-v1")

;;;###autoload
(defun nl-llm-ckpt-save (path model)
  "Save MODEL to PATH as a sexp checkpoint.  MODEL is a plist with :config (a
plist of hyperparameters), :step (integer), :wte, :lnfg, :bh (tensors), :blocks
(a list of block plists of tensors, with biases), and optionally :opt (the Adam
optimiser state from `nlga-adam-state', a list of (m . v) tensor pairs in
parameter order) for a complete, seamless resume."
  (let ((form (list :format nl-llm-ckpt-format
                    :config (plist-get model :config)
                    :step   (or (plist-get model :step) 0)
                    :wte    (plist-get model :wte)
                    :lnfg   (plist-get model :lnfg)
                    :bh     (plist-get model :bh)
                    :blocks (plist-get model :blocks)
                    :opt    (plist-get model :opt))))
    (with-temp-buffer
      (let ((print-length nil) (print-level nil)) (prin1 form (current-buffer)))
      (let ((coding-system-for-write 'utf-8))
        (write-region (point-min) (point-max) path nil 'silent)))
    path))

;;;###autoload
(defun nl-llm-ckpt-load (path)
  "Load the checkpoint at PATH; return the model plist (:config :step :wte :lnfg
:bh :blocks).  Errors if the format tag does not match."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8)) (insert-file-contents path))
    (goto-char (point-min))
    (let ((p (read (current-buffer))))
      (unless (equal (plist-get p :format) nl-llm-ckpt-format)
        (error "nl-llm-ckpt: bad/unknown format %S" (plist-get p :format)))
      p)))

(provide 'nl-llm-ckpt)
;;; nl-llm-ckpt.el ends here
