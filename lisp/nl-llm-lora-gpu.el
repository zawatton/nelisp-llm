;;; nl-llm-lora-gpu.el --- LoRA for the resident GPU autograd builder  -*- lexical-binding: t; -*-

;; For a base (out x in) weight, a rank-R LoRA adapter trains only
;; R*(in + out) floats in A (R x in) and B (out x R) instead of in*out.
;; The resident-autograd path needs no new kernels: the LoRA branch is just
;; matmul, transpose, vadd and scale over the existing Vulkan dispatch set.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)

(defun nl-llm-lora-gpu--shape2 (tensor who)
  "Return TENSOR's 2D shape for WHO or signal a readable error."
  (let ((sh (photon-tensor-shape tensor)))
    (unless (= (length sh) 2)
      (error "%s: expected a rank-2 tensor, got shape %S" who sh))
    sh))

(defun nl-llm-lora-gpu--rt-shape (rt)
  "Return RT's shape as a list."
  (list (nlga-rt-rows rt) (nlga-rt-cols rt)))

(defun nl-llm-lora-gpu--check-adapter (lora who)
  "Validate CPU adapter LORA for WHO; return (A B RANK ALPHA IN OUT)."
  (let* ((a (plist-get lora :a))
         (b (plist-get lora :b))
         (rank (plist-get lora :rank))
         (alpha (plist-get lora :alpha))
         (in (plist-get lora :in))
         (out (plist-get lora :out))
         (ash (and a (nl-llm-lora-gpu--shape2 a who)))
         (bsh (and b (nl-llm-lora-gpu--shape2 b who))))
    (unless a (error "%s: adapter is missing :a" who))
    (unless b (error "%s: adapter is missing :b" who))
    (unless (and (integerp rank) (> rank 0))
      (error "%s: adapter :rank must be a positive integer, got %S" who rank))
    (unless (numberp alpha)
      (error "%s: adapter :alpha must be numeric, got %S" who alpha))
    (unless (and (integerp in) (> in 0))
      (error "%s: adapter :in must be a positive integer, got %S" who in))
    (unless (and (integerp out) (> out 0))
      (error "%s: adapter :out must be a positive integer, got %S" who out))
    (unless (equal ash (list rank in))
      (error "%s: adapter :a shape %S does not match (:rank %d :in %d)"
             who ash rank in))
    (unless (equal bsh (list out rank))
      (error "%s: adapter :b shape %S does not match (:out %d :rank %d)"
             who bsh out rank))
    (list a b rank alpha in out)))

(defun nl-llm-lora-gpu--copy-rt-into-tensor (rt tensor who)
  "Copy resident RT into host TENSOR for WHO."
  (let* ((sh (nl-llm-lora-gpu--shape2 tensor who))
         (rows (nlga-rt-rows rt))
         (cols (nlga-rt-cols rt))
         (n (* rows cols))
         (src (nelisp-gpu-server-read-resident (nlga-rt-handle rt) n))
         (dst (photon-tensor-data tensor))
         (i 0))
    (unless (equal sh (list rows cols))
      (error "%s: shape mismatch, resident %S vs tensor %S"
             who (list rows cols) sh))
    (while (< i n)
      (aset dst i (aref src i))
      (setq i (1+ i)))))

(defun nl-llm-lora-gpu--check-linear-shapes (x w bias lora-rts)
  "Validate X/W/BIAS/LORA-RTS shapes for `nlga-lora-linear'."
  (let* ((x-cols (nlga-rt-cols x))
         (w-rows (nlga-rt-rows w))
         (w-cols (nlga-rt-cols w))
         (a (plist-get lora-rts :a))
         (brt (plist-get lora-rts :b))
         (scale (plist-get lora-rts :scale))
         (rank (plist-get lora-rts :rank))
         (in (plist-get lora-rts :in))
         (out (plist-get lora-rts :out))
         (a-rows (and a (nlga-rt-rows a)))
         (a-cols (and a (nlga-rt-cols a)))
         (b-rows (and brt (nlga-rt-rows brt)))
         (b-cols (and brt (nlga-rt-cols brt))))
    (unless bias
      (error "nlga-lora-linear: BIAS rt is required"))
    (unless (= x-cols w-cols)
      (error "nlga-lora-linear: X cols %d do not match W cols %d" x-cols w-cols))
    (unless (and (= (nlga-rt-rows bias) w-rows) (= (nlga-rt-cols bias) 1))
      (error "nlga-lora-linear: BIAS shape %S must be (%d x 1)"
             (nl-llm-lora-gpu--rt-shape bias) w-rows))
    (unless a
      (error "nlga-lora-linear: LORA-RTS is missing :a"))
    (unless brt
      (error "nlga-lora-linear: LORA-RTS is missing :b"))
    (unless scale
      (error "nlga-lora-linear: LORA-RTS is missing :scale"))
    (unless (and (= (nlga-rt-rows scale) 1) (= (nlga-rt-cols scale) 1))
      (error "nlga-lora-linear: :scale shape %S must be (1 x 1)"
             (nl-llm-lora-gpu--rt-shape scale)))
    (unless (= a-cols x-cols)
      (error "nlga-lora-linear: LoRA A cols %d do not match X/W input %d"
             a-cols x-cols))
    (unless (= b-rows w-rows)
      (error "nlga-lora-linear: LoRA B rows %d do not match W output %d"
             b-rows w-rows))
    (unless (= a-rows b-cols)
      (error "nlga-lora-linear: LoRA rank mismatch, A rows %d vs B cols %d"
             a-rows b-cols))
    (when (and rank (/= rank a-rows))
      (error "nlga-lora-linear: LORA-RTS :rank %S does not match A/B rank %d"
             rank a-rows))
    (when (and in (/= in a-cols))
      (error "nlga-lora-linear: LORA-RTS :in %S does not match A cols %d"
             in a-cols))
    (when (and out (/= out b-rows))
      (error "nlga-lora-linear: LORA-RTS :out %S does not match B rows %d"
             out b-rows))))

;;;###autoload
(cl-defun nl-llm-lora-gpu-attach (b lora &optional (trainable t))
  "Upload CPU adapter LORA into builder B and return resident LoRA rts.
When TRAINABLE is non-nil, upload A/B with `nlga-param'; when nil, upload them
with `nlga-const'.  The return plist is
\(:a RT :b RT :scale RT :rank RANK :in IN :out OUT)."
  (let* ((checked (nl-llm-lora-gpu--check-adapter lora "nl-llm-lora-gpu-attach"))
         (a (nth 0 checked))
         (bt (nth 1 checked))
         (rank (nth 2 checked))
         (alpha (nth 3 checked))
         (in (nth 4 checked))
         (out (nth 5 checked))
         (upload (if trainable #'nlga-param #'nlga-const)))
    (list :a (funcall upload b a)
          :b (funcall upload b bt)
          :scale (nlga-scalar b (/ (float alpha) (float rank)))
          :rank rank :in in :out out)))

;;;###autoload
(defun nlga-lora-linear (b x w bias lora-rts)
  "Affine `nlga-linear' with an optional LoRA branch.
With LORA-RTS nil, this reduces to plain `nlga-linear'.  Otherwise it adds
scale * ((X . A^T) . B^T) using only the existing resident ops."
  (if (not lora-rts)
      (nlga-linear b x w bias)
    (nl-llm-lora-gpu--check-linear-shapes x w bias lora-rts)
    (let* ((a (plist-get lora-rts :a))
           (brt (plist-get lora-rts :b))
           (scale (plist-get lora-rts :scale))
           (base (nlga-linear b x w bias))
           (down (nlga-matmul b x (nlga-transpose b a)))
           (up (nlga-matmul b down (nlga-transpose b brt))))
      (nlga-add b base (nlga-scale b up scale)))))

;;;###autoload
(defun nl-llm-lora-gpu-params (lora-rts)
  "Return LORA-RTS's parameter rts in explicit optimiser order."
  (list (plist-get lora-rts :a) (plist-get lora-rts :b)))

;;;###autoload
(defun nl-llm-lora-gpu-pull (lora-rts lora)
  "Copy LORA-RTS's resident A/B back into CPU adapter LORA and return LORA."
  (let* ((checked (nl-llm-lora-gpu--check-adapter lora "nl-llm-lora-gpu-pull"))
         (a-tensor (nth 0 checked))
         (b-tensor (nth 1 checked))
         (a-rt (plist-get lora-rts :a))
         (b-rt (plist-get lora-rts :b)))
    (unless a-rt
      (error "nl-llm-lora-gpu-pull: LORA-RTS is missing :a"))
    (unless b-rt
      (error "nl-llm-lora-gpu-pull: LORA-RTS is missing :b"))
    (nl-llm-lora-gpu--copy-rt-into-tensor a-rt a-tensor "nl-llm-lora-gpu-pull/:a")
    (nl-llm-lora-gpu--copy-rt-into-tensor b-rt b-tensor "nl-llm-lora-gpu-pull/:b")
    lora))

(provide 'nl-llm-lora-gpu)
;;; nl-llm-lora-gpu.el ends here
