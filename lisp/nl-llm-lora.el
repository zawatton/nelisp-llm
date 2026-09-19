;;; nl-llm-lora.el --- LoRA adapters for CPU autograd linear layers  -*- lexical-binding: t; -*-

;; LoRA trains RANK*(IN+OUT) floats instead of IN*OUT, a ratio of
;; RANK*(IN+OUT)/(IN*OUT), while keeping the base weight frozen and mergeable.

;;; Code:

(require 'nl-llm-compat)
(require 'photon-tensor)
(require 'photon-autograd)

(defconst nl-llm-lora--sqrt3 1.7320508075688772
  "Square root of 3, used to scale hashed uniform init to stddev 1/sqrt(IN).")

(defconst nl-llm-lora--u32-mask #xffffffff
  "Mask used to keep hashed arithmetic in unsigned 32-bit range.")

(defun nl-llm-lora--copy-tensor (tensor)
  "Return a fresh tensor copy of TENSOR."
  (photon-tensor (copy-sequence (photon-tensor-shape tensor))
                 (copy-sequence (photon-tensor-data tensor))))

(defun nl-llm-lora--u32 (x)
  "Return X reduced modulo 2^32."
  (logand x nl-llm-lora--u32-mask))

(defun nl-llm-lora--mul32 (a b)
  "Return the low 32 bits of A times B without widening to bignums."
  (let* ((a (nl-llm-lora--u32 a))
         (b (nl-llm-lora--u32 b))
         (a0 (logand a #xffff))
         (a1 (ash a -16))
         (b0 (logand b #xffff))
         (b1 (ash b -16))
         (lo (* a0 b0))
         (mid (+ (* a0 b1) (* a1 b0))))
    (nl-llm-lora--u32 (+ lo (ash mid 16)))))

(defun nl-llm-lora--mix32 (x)
  "Return a Murmur3-style avalanche mix of 32-bit X."
  (setq x (nl-llm-lora--u32 x))
  (setq x (logxor x (ash x -16)))
  (setq x (nl-llm-lora--mul32 x #x85ebca6b))
  (setq x (logxor x (ash x -13)))
  (setq x (nl-llm-lora--mul32 x #xc2b2ae35))
  (setq x (logxor x (ash x -16)))
  (nl-llm-lora--u32 x))

(defun nl-llm-lora--hash-unit (index seed)
  "Deterministic pseudo-random float in [0,1) for INDEX and SEED."
  (let* ((ix (nl-llm-lora--mix32
              (logxor (nl-llm-lora--u32 index) #x9e3779b9)))
         (sd (nl-llm-lora--mix32
              (logxor (nl-llm-lora--u32 seed) #x7f4a7c15)))
         (x (logxor ix sd)))
    (/ (float (nl-llm-lora--mix32 (nl-llm-lora--u32 (+ x #x52dce729))))
       4294967296.0)))

(defun nl-llm-lora--check-dims (out in rank alpha where)
  "Validate OUT, IN, RANK and ALPHA for WHERE."
  (unless (and (integerp out) (> out 0))
    (error "%s: OUT must be a positive integer, got %S" where out))
  (unless (and (integerp in) (> in 0))
    (error "%s: IN must be a positive integer, got %S" where in))
  (unless (and (integerp rank) (>= rank 1))
    (error "%s: RANK must be >= 1, got %S" where rank))
  (unless (<= rank (min in out))
    (error "%s: RANK %S must be <= min(IN, OUT) = %S" where rank (min in out)))
  (unless (numberp alpha)
    (error "%s: ALPHA must be numeric, got %S" where alpha)))

(defun nl-llm-lora--validate (lora where)
  "Validate plain adapter LORA for WHERE and return it."
  (let ((a (plist-get lora :a)) (b (plist-get lora :b))
        (rank (plist-get lora :rank)) (alpha (plist-get lora :alpha))
        (in (plist-get lora :in)) (out (plist-get lora :out)))
    (nl-llm-lora--check-dims out in rank alpha where)
    (unless (equal (photon-tensor-shape a) (list rank in))
      (error "%s: A must have shape %S, got %S" where (list rank in) (photon-tensor-shape a)))
    (unless (equal (photon-tensor-shape b) (list out rank))
      (error "%s: B must have shape %S, got %S" where (list out rank) (photon-tensor-shape b)))
    lora))

(defun nl-llm-lora--check-weight (tensor out in where)
  "Validate that TENSOR is a weight of shape (OUT x IN) for WHERE."
  (unless (equal (photon-tensor-shape tensor) (list out in))
    (error "%s: expected weight shape %S, got %S" where (list out in)
           (photon-tensor-shape tensor))))

(defun nl-llm-lora--check-input (tensor in where)
  "Validate that TENSOR is a 2D input whose width is IN for WHERE."
  (let ((shape (photon-tensor-shape tensor)))
    (unless (and (= (length shape) 2) (= (nth 1 shape) in))
      (error "%s: expected input shape (m x %d), got %S" where in shape))))

(defun nl-llm-lora--init-a (rank in seed)
  "Return a deterministic hashed LoRA A tensor of shape (RANK x IN)."
  (let* ((n (* rank in))
         (scale (/ nl-llm-lora--sqrt3 (sqrt (float in))))
         (data (make-vector n 0.0))
         (i 0))
    (while (< i n)
      (aset data i (* scale 2.0 (- (nl-llm-lora--hash-unit i seed) 0.5)))
      (setq i (1+ i)))
    (photon-tensor (list rank in) data)))

(defun nl-llm-lora--attached-shape (attached where)
  "Return (OUT IN RANK) for ATTACHED, validating internal leaf shapes for WHERE."
  (let* ((a (plist-get attached :a)) (b (plist-get attached :b))
         (ash (photon-tensor-shape (pav-value a)))
         (bsh (photon-tensor-shape (pav-value b)))
         (rank (car ash)) (in (nth 1 ash)) (out (car bsh)) (rank2 (nth 1 bsh)))
    (unless (= rank rank2)
      (error "%s: attached A/B rank mismatch %S %S" where ash bsh))
    (list out in rank)))

;;;###autoload
(defun nl-llm-lora-make (out in rank &optional alpha seed)
  "Make a plain LoRA adapter for an (OUT x IN) weight at RANK.
ALPHA defaults to RANK and SEED defaults to 0.  A is initialized with small
deterministic hashed values with standard deviation roughly 1/sqrt(IN), while
B is initialized to all zeros so DELTA-W is exactly zero at step 0: attaching a
fresh adapter is the identity and fine-tuning starts from the base model."
  (let ((alpha0 (or alpha rank))
        (seed0 (or seed 0)))
    (nl-llm-lora--check-dims out in rank alpha0 "nl-llm-lora-make")
    (list :a (nl-llm-lora--init-a rank in seed0)
          :b (photon-tensor-create (list out rank) 0.0)
          :rank rank :alpha alpha0 :in in :out out)))

;;;###autoload
(defun nl-llm-lora-scale (lora)
  "Return the effective LoRA scale ALPHA / RANK for LORA as a float."
  (nl-llm-lora--validate lora "nl-llm-lora-scale")
  (/ (float (plist-get lora :alpha)) (float (plist-get lora :rank))))

;;;###autoload
(defun nl-llm-lora-attach (lora)
  "Wrap LORA's plain tensors as autograd leaves for one training run.
The returned plist is (:a PAV :b PAV :scale FLOAT).  Call this once per run so
the leaves persist across steps and accumulate gradients."
  (nl-llm-lora--validate lora "nl-llm-lora-attach")
  (list :a (photon-autograd-const (nl-llm-lora--copy-tensor (plist-get lora :a)))
        :b (photon-autograd-const (nl-llm-lora--copy-tensor (plist-get lora :b)))
        :scale (nl-llm-lora-scale lora)))

;;;###autoload
(defun nl-llm-lora-params (attached)
  "Return ATTACHED's trainable autograd leaves as (A-pav B-pav)."
  (list (plist-get attached :a) (plist-get attached :b)))

;;;###autoload
(defun nl-llm-lora-detach (attached lora)
  "Copy ATTACHED's current leaf values back into plain adapter LORA and return it."
  (nl-llm-lora--validate lora "nl-llm-lora-detach")
  (setq lora (plist-put lora :a (nl-llm-lora--copy-tensor (pav-value (plist-get attached :a)))))
  (setq lora (plist-put lora :b (nl-llm-lora--copy-tensor (pav-value (plist-get attached :b)))))
  lora)

;;;###autoload
(defun nl-llm-ag-lora-linear (x w bias attached)
  "Autograd linear layer with optional LoRA branch over X, W and BIAS.
X is (m x in), W is (out x in), BIAS is (out), all as pav leaves or vars.
If ATTACHED is nil this degrades exactly to `photon-autograd-linear'."
  (if (null attached)
      (photon-autograd-linear x w bias)
    (let* ((shape (nl-llm-lora--attached-shape attached "nl-llm-ag-lora-linear"))
           (out (nth 0 shape)) (in (nth 1 shape))
           (a (plist-get attached :a)) (b (plist-get attached :b)))
      (nl-llm-lora--check-weight (pav-value w) out in "nl-llm-ag-lora-linear")
      (nl-llm-lora--check-input (pav-value x) in "nl-llm-ag-lora-linear")
      (photon-autograd-add
       (photon-autograd-linear x w bias)
       (photon-autograd-scale
        (photon-autograd-matmul
         (photon-autograd-matmul x (photon-autograd-transpose a))
         (photon-autograd-transpose b))
        (plist-get attached :scale))))))

;;;###autoload
(defun nl-llm-lora-delta (lora)
  "Return LORA's plain delta weight SCALE * (B * A) as an (out x in) tensor."
  (nl-llm-lora--validate lora "nl-llm-lora-delta")
  (photon-tensor-scale
   (photon-tensor-matmul (plist-get lora :b) (plist-get lora :a))
   (nl-llm-lora-scale lora)))

;;;###autoload
(defun nl-llm-lora-merge (w lora)
  "Return a fresh weight tensor W + DELTA for LORA without mutating W."
  (nl-llm-lora--validate lora "nl-llm-lora-merge")
  (nl-llm-lora--check-weight w (plist-get lora :out) (plist-get lora :in)
                             "nl-llm-lora-merge")
  (photon-tensor-add w (nl-llm-lora-delta lora)))

;;;###autoload
(defun nl-llm-lora-unmerge (w lora)
  "Return a fresh weight tensor W - DELTA for LORA without mutating W."
  (nl-llm-lora--validate lora "nl-llm-lora-unmerge")
  (nl-llm-lora--check-weight w (plist-get lora :out) (plist-get lora :in)
                             "nl-llm-lora-unmerge")
  (photon-tensor-add w (photon-tensor-scale (nl-llm-lora-delta lora) -1.0)))

;;;###autoload
(defun nl-llm-lora-param-count (lora)
  "Return LORA's trainable parameter count, RANK * (IN + OUT)."
  (nl-llm-lora--validate lora "nl-llm-lora-param-count")
  (* (plist-get lora :rank) (+ (plist-get lora :in) (plist-get lora :out))))

;;;###autoload
(defun nl-llm-lora-full-count (lora)
  "Return the full dense parameter count, IN * OUT, for LORA's target weight."
  (nl-llm-lora--validate lora "nl-llm-lora-full-count")
  (* (plist-get lora :in) (plist-get lora :out)))

(provide 'nl-llm-lora)
;;; nl-llm-lora.el ends here
