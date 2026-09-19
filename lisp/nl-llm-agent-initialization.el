;;; nl-llm-agent-initialization.el --- opt-in deterministic model creation -*- lexical-binding: t; -*-

;; This API keeps the established P5 constructor as the source of model
;; geometry and plist shape.  The optional deterministic initializer only
;; replaces dense matrix values after construction; it does not alter global
;; random state or add metadata to the model.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)

(declare-function nl-llm-agent--p5-params "nl-llm-agent-improve" (model))

(defconst nl-llm-agent-initialization--uint32-mask #xffffffff
  "Mask used to keep the local xorshift state in the uint32 range.")

(defun nl-llm-agent-initialization--xorshift32 (state)
  "Return the next uint32 value from local xorshift32 STATE.

STATE is expected to be a non-zero uint32.  Masking after every stage keeps
the sequence independent of the host integer width and of Emacs' global
`random' state."
  (setq state
        (logand nl-llm-agent-initialization--uint32-mask
                (logxor state (ash state 13))))
  (setq state
        (logand nl-llm-agent-initialization--uint32-mask
                (logxor state (ash state -17))))
  (logand nl-llm-agent-initialization--uint32-mask
          (logxor state (ash state 5))))

(defun nl-llm-agent-initialization--valid-seed (seed)
  "Return SEED when it is a non-zero uint32, otherwise signal an error."
  (unless (and (integerp seed) (<= 1 seed)
               (<= seed nl-llm-agent-initialization--uint32-mask))
    (error "initializer seed must be a non-zero uint32: %S" seed))
  seed)

(defun nl-llm-agent-initialization--xorshift32-model (model seed)
  "Replace MODEL's rank-2 parameter values using xorshift32 SEED.

Parameters are visited in the canonical order returned by
`nl-llm-agent--p5-params'.  Rank-1 values and every model plist entry remain
unchanged.  Return MODEL after in-place replacement."
  (let* ((dim (plist-get model :dim))
         (state seed)
         (scale (/ 1.0 (sqrt (float dim)))))
    (dolist (parameter (nl-llm-agent--p5-params model))
      (let ((tensor (pav-value parameter)))
        (when (= (length (photon-tensor-shape tensor)) 2)
          (let ((data (photon-tensor-data tensor)))
            (dotimes (index (length data))
              (setq state
                    (nl-llm-agent-initialization--xorshift32 state))
              (let ((unit (/ (float state) 4294967296.0)))
                (aset data index (* scale (- (* 2.0 unit) 1.0)))))))))
    model))

;;;###autoload
(cl-defun nl-llm-agent-initialization-create
    (&key (initializer 'legacy) seed dim ff vocab nblocks heads tokenizer)
  "Create a fresh agent model with an opt-in weight INITIALIZER.

With the default `legacy' initializer, this is exactly the existing
`nl-llm-agent-improve-model' constructor.  `xorshift32' first constructs that
same model, then replaces rank-2 parameter values in canonical parameter order
using the explicit non-zero uint32 SEED.  Rank-1 values, model geometry, and
the model plist are preserved.

Callers must record the initializer and seed in their run metadata.  Runtime
hot-reload is not automatic, and this function does not reinitialize a trained
model; it only creates a fresh model."
  (unless (memq initializer '(legacy xorshift32))
    (error "unsupported model initializer: %S" initializer))
  (when (and (eq initializer 'legacy) seed)
    (error "legacy initializer does not accept a seed"))
  (when (eq initializer 'xorshift32)
    (setq seed (nl-llm-agent-initialization--valid-seed seed)))
  (let ((model (nl-llm-agent-improve-model
                dim ff vocab nblocks heads tokenizer)))
    (if (eq initializer 'xorshift32)
        (nl-llm-agent-initialization--xorshift32-model model seed)
      model)))

(provide 'nl-llm-agent-initialization)
;;; nl-llm-agent-initialization.el ends here
