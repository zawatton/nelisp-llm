;;; nl-llm-weights-lora.el --- train a LoRA over a frozen int8 base  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 3.  Removes the barrier that
;; stood in front of adapting an imported model: a frozen quantized weight has
;; to participate in the backward pass even though nothing about it is being
;; trained.
;;
;; The reason is the chain rule rather than anything to do with LoRA.  For
;; y = W.x + s * B.(A.x), the gradients of the trainable parts are local --
;;
;;   dL/dB = s * g (x) u          where u = A.x
;;   dL/du = s * B^T.g
;;   dL/dA = dL/du (x) x
;;
;; -- but dL/dx, which anything upstream of this linear needs, is
;;
;;   dL/dx = W^T.g + A^T.(dL/du)
;;
;; and that first term is a transposed multiply against the frozen base.  So
;; "the base is frozen" does not mean "the base is not needed in the backward";
;; it means only that W collects no gradient of its own.  `nl-llm-weights-apply-t'
;; supplies that term straight off the int8 table, so the 14 GB of boxed floats
;; a dequantized base would cost are never paid, in either direction.
;;
;; The adapter itself is `nl-llm-lora-make's, unchanged: A hashed-deterministic,
;; B zero so the delta is exactly zero at step 0, scale alpha/rank.  Sharing that
;; with the f32 autograd path is deliberate -- two notions of what a LoRA is
;; would be one too many -- and it gives this module a free invariant to be
;; tested against: a fresh adapter must leave the base's output untouched.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)
(require 'photon-tensor)
(require 'nl-llm-weights)
(require 'nl-llm-lora)

(defun nl-llm-wlora--matvec (m rows cols x &optional out)
  "Return M (ROWS x COLS, row-major float vector) applied to X."
  (let ((y (or out (make-vector rows 0.0))) (o 0))
    (while (< o rows)
      (let ((base (* o cols)) (acc 0.0) (i 0))
        (while (< i cols)
          (setq acc (+ acc (* (aref m (+ base i)) (aref x i))))
          (setq i (1+ i)))
        (aset y o acc))
      (setq o (1+ o)))
    y))

(defun nl-llm-wlora--matvec-t (m rows cols g &optional out)
  "Return M^T (COLS long) applied to G (ROWS long)."
  (let ((y (or out (make-vector cols 0.0))) (o 0))
    (dotimes (i cols) (aset y i 0.0))
    (while (< o rows)
      (let ((s (aref g o)))
        (unless (= s 0.0)
          (let ((base (* o cols)) (i 0))
            (while (< i cols)
              (aset y i (+ (aref y i) (* (aref m (+ base i)) s)))
              (setq i (1+ i))))))
      (setq o (1+ o)))
    y))

;;;###autoload
(defun nl-llm-wlora-forward (lin lora x &optional base)
  "Apply LIN with LORA to the COLS-long slice of X at BASE.
Returns (Y U): Y the ROWS-long output, U the rank-long A.x kept so the backward
pass does not recompute it.  With a fresh adapter B is zero, so Y equals the
base's output exactly -- an invariant rather than an approximation."
  (let* ((rows (nl-llm-weights-lin-rows lin))
         (cols (nl-llm-weights-lin-cols lin))
         (rank (plist-get lora :rank))
         (a (photon-tensor-data (plist-get lora :a)))
         (b (photon-tensor-data (plist-get lora :b)))
         (s (nl-llm-lora-scale lora))
         (off (or base 0))
         (xs (if (and (= off 0) (= (length x) cols))
                 x
               (let ((v (make-vector cols 0.0)))
                 (dotimes (i cols) (aset v i (aref x (+ off i))))
                 v)))
         (y (nl-llm-weights-apply lin x off)))
    (unless (and (= (plist-get lora :out) rows) (= (plist-get lora :in) cols))
      (error "nl-llm-wlora-forward: adapter is %dx%d, weight is %dx%d"
             (plist-get lora :out) (plist-get lora :in) rows cols))
    (let* ((u (nl-llm-wlora--matvec a rank cols xs))
           (v (nl-llm-wlora--matvec b rows rank u)))
      (dotimes (o rows) (aset y o (+ (aref y o) (* s (aref v o)))))
      (list y u xs))))

;;;###autoload
(defun nl-llm-wlora-backward (lin lora xs u g &optional wt-fn)
  "Gradients of LIN+LORA at XS (from `nl-llm-wlora-forward') for output grad G.
Returns a plist (:da :db :dx), each a flat float vector: :da is RANK x COLS,
:db is ROWS x RANK, :dx is COLS long and includes the frozen base's
contribution W^T.G, which is what makes anything upstream trainable.

WT-FN, when given, computes that W^T.G instead of `nl-llm-weights-apply-t' --
which is how the same backward runs on the GPU without this file knowing there
is one.  It is called with (LIN G) and must return a COLS-long vector."
  (let* ((rows (nl-llm-weights-lin-rows lin))
         (cols (nl-llm-weights-lin-cols lin))
         (rank (plist-get lora :rank))
         (a (photon-tensor-data (plist-get lora :a)))
         (b (photon-tensor-data (plist-get lora :b)))
         (s (nl-llm-lora-scale lora))
         (db (make-vector (* rows rank) 0.0))
         (da (make-vector (* rank cols) 0.0)))
    (unless (= (length g) rows)
      (error "nl-llm-wlora-backward: G is %d long, weight has %d rows"
             (length g) rows))
    ;; dL/dB = s * g (x) u
    (dotimes (o rows)
      (let ((go (* s (aref g o))))
        (unless (= go 0.0)
          (dotimes (r rank)
            (aset db (+ (* o rank) r) (* go (aref u r)))))))
    ;; dL/du = s * B^T.g   then   dL/dA = dL/du (x) xs
    (let ((du (nl-llm-wlora--matvec-t b rows rank g)))
      (dotimes (r rank) (aset du r (* s (aref du r))))
      (dotimes (r rank)
        (let ((dr (aref du r)))
          (unless (= dr 0.0)
            (dotimes (i cols)
              (aset da (+ (* r cols) i) (* dr (aref xs i)))))))
      ;; dL/dx = W^T.g + A^T.(dL/du)
      (let ((dx (if wt-fn (funcall wt-fn lin g) (nl-llm-weights-apply-t lin g)))
            (dxa (nl-llm-wlora--matvec-t a rank cols du)))
        (dotimes (i cols) (aset dx i (+ (aref dx i) (aref dxa i))))
        (list :da da :db db :dx dx)))))

;;;###autoload
(defun nl-llm-wlora-sgd (lora grads lr)
  "Update LORA in place by LR along GRADS from `nl-llm-wlora-backward'.
Only A and B move; the base is never touched, which is the whole point of
adapting an imported model rather than fine-tuning one."
  (let ((a (photon-tensor-data (plist-get lora :a)))
        (b (photon-tensor-data (plist-get lora :b)))
        (da (plist-get grads :da))
        (db (plist-get grads :db)))
    (dotimes (i (length a)) (aset a i (- (aref a i) (* lr (aref da i)))))
    (dotimes (i (length b)) (aset b i (- (aref b i) (* lr (aref db i)))))
    lora))

;;;###autoload
(defun nl-llm-wlora-sq-loss (y target)
  "Return (LOSS . GRAD) for half the squared error between Y and TARGET."
  (let ((n (length y)) (acc 0.0) (g (make-vector (length y) 0.0)))
    (dotimes (i n)
      (let ((d (- (aref y i) (aref target i))))
        (aset g i d)
        (setq acc (+ acc (* 0.5 d d)))))
    (cons acc g)))

;;;###autoload
(defun nl-llm-wlora-fit (lin lora x target steps lr &optional progress)
  "Take STEPS gradient steps of LORA at LR so LIN+LORA maps X toward TARGET.
Returns the list of losses, oldest first.  A frozen quantized base with a
trainable adapter, end to end -- the smallest thing that demonstrates the
backward path actually learns rather than merely type-checking.

Plain SGD with no clipping and no schedule, deliberately: it is a demonstration
that the gradients drive the loss down, not a trainer.  That also means LR and
the scale of X are not independent -- the first version of
`test/weights-lora-test.el' fed it inputs around +-39, which made the loss
start at 1.1e7 and diverge to NaN in sixty steps at LR 0.002, with correct
gradients throughout.  Real adaptation wants Adam, clipping and a schedule,
which `nl-llm-gpu-ag.el' already has for the f32 path."
  (let ((losses nil))
    (dotimes (step steps)
      (let* ((fw (nl-llm-wlora-forward lin lora x))
             (y (nth 0 fw)) (u (nth 1 fw)) (xs (nth 2 fw))
             (lg (nl-llm-wlora-sq-loss y target))
             (grads (nl-llm-wlora-backward lin lora xs u (cdr lg))))
        (push (car lg) losses)
        (nl-llm-wlora-sgd lora grads lr)
        (when progress (funcall progress step (car lg)))))
    (nreverse losses)))

(provide 'nl-llm-weights-lora)
;;; nl-llm-weights-lora.el ends here
