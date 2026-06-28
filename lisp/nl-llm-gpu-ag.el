;;; nl-llm-gpu-ag.el --- resident-tensor autograd: build on-device train steps  -*- lexical-binding: t; -*-

;; A thin deferred-execution autograd whose tensors live on the GPU.  Each op
;; emitter appends its forward dispatch(es) to a fused command buffer and
;; records a backward thunk; calling the thunks in reverse emits the backward
;; dispatches, accumulating gradients into per-tensor resident grad slots; a
;; final pass emits an on-device `sgd' dispatch per parameter.  The whole
;; forward + backward + SGD then runs as ONE batch per step with the weights
;; (and the optimiser update) resident on the GPU -- no weight round-trips.
;;
;; This generalises the hand-wired MLP trainer (nl-llm-gpu-train.el): a model
;; is expressed once with the op emitters and is trained on-device automatically.
;;
;; Conventions: a resident tensor (`nlga-rt') is (slot rows cols grad); 2D row
;; major; a "vector" (bias, gain) is rows=n cols=1.  Gradients ACCUMULATE
;; (vadd into a zeroed tmp slot) so fan-out / residuals are correct.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)   ; nelisp-gpu-server + bin path

(defsubst nlga--g (n) (/ (+ n 63) 64))

(cl-defstruct (nlga (:constructor nlga--make))
  (slots nil) (nslot 0) (disp nil) (bwd nil) (params nil) (nout 0) (compiled nil) (adam nil))
(cl-defstruct (nlga-rt (:constructor nlga-rt--make)) slot rows cols grad handle gwritten)
(defsubst nlga-rt-size (x) (* (nlga-rt-rows x) (nlga-rt-cols x)))

(defun nlga-new () (nlga--make))

(defun nlga--slot (b spec)
  "Append SPEC to B's slot list; return its index."
  (push spec (nlga-slots b)) (prog1 (nlga-nslot b) (cl-incf (nlga-nslot b))))
(defun nlga--res (b vec) (nlga--slot b (list 'res (nelisp-gpu-server-upload vec) (length vec))))
(defun nlga--tmp (b n) (nlga--slot b (cons 'tmp n)))
(defun nlga--out (b n) (nlga--slot b (cons 'out n)))
(defun nlga--d (b d) (push d (nlga-disp b)))

(defun nlga--grad (b x)
  "Resident grad slot for rt X (a zeroed tmp), allocated on first use."
  (or (nlga-rt-grad x) (setf (nlga-rt-grad x) (nlga--tmp b (nlga-rt-size x)))))
(defun nlga--accum (b dst-grad src-slot n)
  "Accumulate SRC-SLOT into grad slot DST-GRAD in place (vadd)."
  (nlga--d b (list 'vadd (list dst-grad src-slot dst-grad) (list n) (nlga--g n))))
(defun nlga--accum-into (b rt n producer)
  "PRODUCER is a fn of a destination slot that emits a dispatch writing RT's full
N-element gradient contribution there.  The FIRST contribution to RT's grad is
written directly into the grad slot (which is zeroed each run), saving a tmp +
vadd; later contributions (fan-out) go through a tmp + vadd.  Only correct when
every writer of RT's grad uses this helper -- used for single-writer leaf weight
gradients (W/bias/gamma/embedding)."
  (let ((g (nlga--grad b rt)))
    (if (nlga-rt-gwritten rt)
        (let ((tmp (nlga--tmp b n)))
          (funcall producer tmp)
          (nlga--d b (list 'vadd (list g tmp g) (list n) (nlga--g n))))
      (funcall producer g)
      (setf (nlga-rt-gwritten rt) t))))
;; backward thunks are stored most-recent-first; running them in stored order
;; is reverse-chronological = correct backprop order.
(defun nlga--bwd-push (b thunk) (push thunk (nlga-bwd b)))

;; --- leaves ----------------------------------------------------------
(defun nlga-const (b tensor)
  "Upload TENSOR resident as a non-trainable input; return an rt.
The rt records its resident handle so it can be refreshed with `nlga-update'
(e.g. per training window) without recompiling the batch."
  (let* ((sh (photon-tensor-shape tensor)) (data (photon-tensor-data tensor))
         (h (nelisp-gpu-server-upload data))
         (slot (nlga--slot b (list 'res h (length data)))))
    (nlga-rt--make :slot slot :rows (car sh) :cols (or (nth 1 sh) 1) :handle h)))
(defun nlga-param (b tensor)
  "Upload TENSOR resident as a trainable parameter (records it for SGD)."
  (let* ((sh (photon-tensor-shape tensor)) (data (photon-tensor-data tensor))
         (h (nelisp-gpu-server-upload data))
         (slot (nlga--slot b (list 'res h (length data))))
         (rt (nlga-rt--make :slot slot :rows (car sh) :cols (or (nth 1 sh) 1) :handle h)))
    (push (list :rt rt :tensor tensor :handle h) (nlga-params b))
    rt))

(defun nlga-update (rt tensor)
  "Overwrite the resident buffer backing RT with TENSOR's data, in place.
RT must be a resident input created by `nlga-const' (same shape).  Use between
`nlga-step' calls to feed the next training window without recompiling."
  (nelisp-gpu-server-write-resident (nlga-rt-handle rt) (photon-tensor-data tensor)))
(defun nlga-scalar (b v) "Resident 1-element rt holding V." (nlga-const b (photon-tensor '(1) (vector v))))

;; --- ops -------------------------------------------------------------
(defun nlga-linear (b x w bias)
  "Affine Y = X (seq x in) . W^T (W out x in) + BIAS (out)."
  (let* ((seq (nlga-rt-rows x)) (in (nlga-rt-cols x)) (out (nlga-rt-rows w))
         (ys (nlga--tmp b (* seq out))) (y (nlga-rt--make :slot ys :rows seq :cols out)))
    (nlga--d b (list 'linear (list (nlga-rt-slot x) (nlga-rt-slot w) (nlga-rt-slot bias) ys)
                     (list seq in out) (nlga--g (* seq out))))
    (nlga--bwd-push b
      (lambda ()
        (let ((gy (nlga--grad b y)))
          ;; dx = gy . W
          (let ((dx (nlga--tmp b (* seq in))))
            (nlga--d b (list 'matmul (list gy (nlga-rt-slot w) dx) (list seq out in) (nlga--g (* seq in))))
            (nlga--accum b (nlga--grad b x) dx (* seq in)))
          ;; dW = gy^T . x  (one matmul-at; written straight into the weight grad)
          (nlga--accum-into b w (* out in)
            (lambda (dst) (nlga--d b (list 'matmul-at (list gy (nlga-rt-slot x) dst) (list out seq in) (nlga--g (* out in))))))
          ;; db = colsum(gy)  (straight into the bias grad)
          (nlga--accum-into b bias out
            (lambda (dst) (nlga--d b (list 'colsum (list gy dst) (list seq out) (nlga--g out))))))))
    y))

(defun nlga-quant-weight (b w)
  "Ternary (absmean) quantization of weight W for BitNet b1.58 (QAT, Phase A).
Forward: WQ = beta * round(clip(W/beta, -1, 1)) with beta = mean|W|, so WQ takes
values in {-beta, 0, +beta}.  Backward is straight-through (STE): the quantizer is
treated as identity, so the full-precision latent W receives WQ's gradient
unchanged and the optimizer trains W.  Returns the WQ rt (same shape as W)."
  (let* ((n (nlga-rt-size w))
         (ss (nlga--tmp b 1))                 ; sum|W| accumulator (zeroed each run)
         (wqs (nlga--tmp b n))
         (wq (nlga-rt--make :slot wqs :rows (nlga-rt-rows w) :cols (nlga-rt-cols w))))
    (nlga--d b (list 'absmean-acc (list (nlga-rt-slot w) ss) (list n) (nlga--g 1)))
    (nlga--d b (list 'quant-w (list (nlga-rt-slot w) ss wqs) (list n) (nlga--g n)))
    (nlga--bwd-push b
      (lambda ()                              ; STE: dW_latent += dWQ (identity)
        (nlga--accum b (nlga--grad b w) (nlga--grad b wq) n)))
    wq))

(defun nlga-quant-act (b x)
  "Per-row absmax int8 activation quantization for BitNet b1.58 (QAT, Phase A).
Forward: each row is scaled by gamma = max|row|/127 and rounded to an int8 grid,
XQ = gamma * clip(round(X/gamma), -127, 127).  Backward is straight-through (STE):
the gradient passes to X unchanged.  Returns the quantized activation rt."
  (let* ((seq (nlga-rt-rows x)) (cols (nlga-rt-cols x)) (n (* seq cols))
         (xqs (nlga--tmp b n))
         (xq (nlga-rt--make :slot xqs :rows seq :cols cols)))
    (nlga--d b (list 'quant-act (list (nlga-rt-slot x) xqs) (list seq cols) (nlga--g seq)))
    (nlga--bwd-push b
      (lambda ()                              ; STE: dX += dXQ (identity)
        (nlga--accum b (nlga--grad b x) (nlga--grad b xq) n)))
    xq))

(defun nlga-bitlinear (b x w bias)
  "BitLinear (BitNet b1.58, Phase A): ternary weights + int8 activations via
QAT + STE.  Forward uses WQ = `nlga-quant-weight' and XQ = `nlga-quant-act';
backward (straight-through) trains the full-precision latent weight.  For a
ternary-weight / fp-activation (W1.58A16) variant, compose `nlga-quant-weight'
with `nlga-linear' directly instead."
  (nlga-linear b (nlga-quant-act b x) (nlga-quant-weight b w) bias))

(defun nlga-gelu (b x)
  (let* ((n (nlga-rt-size x)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows x) :cols (nlga-rt-cols x))))
    (nlga--d b (list 'gelu (list (nlga-rt-slot x) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b n)))
        (nlga--d b (list 'gelu-bwd (list go (nlga-rt-slot x) dx) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b x) dx n))))
    o))

(defun nlga-add (b a c)
  "Elementwise A + C (same shape); gradient flows to both."
  (let* ((n (nlga-rt-size a)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows a) :cols (nlga-rt-cols a))))
    (nlga--d b (list 'vadd (list (nlga-rt-slot a) (nlga-rt-slot c) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)))
        (nlga--accum b (nlga--grad b a) go n)
        (nlga--accum b (nlga--grad b c) go n))))
    o))

(defun nlga-rmsnorm (b x gamma)
  "Row-wise RMSNorm of X (M x N) with trainable GAMMA (N)."
  (let* ((m (nlga-rt-rows x)) (n (nlga-rt-cols x)) (istd (nlga--tmp b m))
         (os (nlga--tmp b (* m n))) (o (nlga-rt--make :slot os :rows m :cols n)))
    (nlga--d b (list 'rmsnorm-istd (list (nlga-rt-slot x) istd) (list m n) (nlga--g m)))
    (nlga--d b (list 'rmsnorm-fwd (list (nlga-rt-slot x) istd (nlga-rt-slot gamma) os)
                     (list m n) (nlga--g (* m n))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b (* m n))))
        (nlga--d b (list 'rmsnorm-dx (list go (nlga-rt-slot x) istd (nlga-rt-slot gamma) dx)
                         (list m n) (nlga--g m)))
        (nlga--accum b (nlga--grad b x) dx (* m n))
        (nlga--accum-into b gamma n
          (lambda (dst) (nlga--d b (list 'rmsnorm-dgamma (list go (nlga-rt-slot x) istd dst) (list m n) (nlga--g n))))))))
    o))

(defun nlga-silu (b x)
  (let* ((n (nlga-rt-size x)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows x) :cols (nlga-rt-cols x))))
    (nlga--d b (list 'silu (list (nlga-rt-slot x) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b n)))
        (nlga--d b (list 'silu-bwd (list go (nlga-rt-slot x) dx) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b x) dx n))))
    o))

(defun nlga-mul (b a c)
  "Elementwise product of same-shape A and C."
  (let* ((n (nlga-rt-size a)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows a) :cols (nlga-rt-cols a))))
    (nlga--d b (list 'mul (list (nlga-rt-slot a) (nlga-rt-slot c) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (da (nlga--tmp b n)) (dc (nlga--tmp b n)))
        (nlga--d b (list 'mul (list go (nlga-rt-slot c) da) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b a) da n)
        (nlga--d b (list 'mul (list go (nlga-rt-slot a) dc) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b c) dc n))))
    o))

(defun nlga-dropout (b x mask)
  "Apply dropout to X by elementwise-multiplying a (refreshable) MASK rt.
MASK is a resident const built with `nl-llm-dropout-mask', refreshed each step
via `nlga-update' (feed an all-ones mask, or omit, at eval).  Backward passes the
gradient through the mask.  (Thin wrapper over `nlga-mul'.)"
  (nlga-mul b x mask))

(defun nlga-rope (b x heads cosr sinr spos sneg)
  "Per-head RoPE on X (M x cols) with HEADS heads; COSR/SINR are resident
cos/sin tables (M x hd/2), SPOS/SNEG are resident [+1]/[-1] sign scalars.
RoPE is orthogonal so the backward is the inverse rotation (SNEG)."
  (let* ((m (nlga-rt-rows x)) (cols (nlga-rt-cols x)) (total (/ (* m cols) 2))
         (os (nlga--tmp b (* m cols))) (o (nlga-rt--make :slot os :rows m :cols cols)))
    (nlga--d b (list 'rope-apply (list (nlga-rt-slot x) (nlga-rt-slot cosr) (nlga-rt-slot sinr)
                                       (nlga-rt-slot spos) os)
                     (list m cols heads) (nlga--g total)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b (* m cols))))
        (nlga--d b (list 'rope-apply (list go (nlga-rt-slot cosr) (nlga-rt-slot sinr)
                                           (nlga-rt-slot sneg) dx)
                         (list m cols heads) (nlga--g total)))
        (nlga--accum b (nlga--grad b x) dx (* m cols)))))
    o))

(defun nlga-transpose (b x)
  (let* ((m (nlga-rt-rows x)) (n (nlga-rt-cols x)) (os (nlga--tmp b (* m n)))
         (o (nlga-rt--make :slot os :rows n :cols m)))
    (nlga--d b (list 'transpose (list (nlga-rt-slot x) os) (list m n) (nlga--g (* m n))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b (* m n))))
        (nlga--d b (list 'transpose (list go dx) (list n m) (nlga--g (* m n))))
        (nlga--accum b (nlga--grad b x) dx (* m n)))))
    o))

(defun nlga-matmul (b a c)
  "Matmul A (M x K) by C (K x N)."
  (let* ((mm (nlga-rt-rows a)) (kk (nlga-rt-cols a)) (nn (nlga-rt-cols c))
         (os (nlga--tmp b (* mm nn))) (o (nlga-rt--make :slot os :rows mm :cols nn)))
    (nlga--d b (list 'matmul (list (nlga-rt-slot a) (nlga-rt-slot c) os) (list mm kk nn) (nlga--g (* mm nn))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)))
        ;; dA = go (M x N) . C^T (N x K)
        (let ((ct (nlga--tmp b (* kk nn))) (da (nlga--tmp b (* mm kk))))
          (nlga--d b (list 'transpose (list (nlga-rt-slot c) ct) (list kk nn) (nlga--g (* kk nn))))
          (nlga--d b (list 'matmul (list go ct da) (list mm nn kk) (nlga--g (* mm kk))))
          (nlga--accum b (nlga--grad b a) da (* mm kk)))
        ;; dC = A^T (K x M) . go (M x N)
        (let ((at (nlga--tmp b (* mm kk))) (dc (nlga--tmp b (* kk nn))))
          (nlga--d b (list 'transpose (list (nlga-rt-slot a) at) (list mm kk) (nlga--g (* mm kk))))
          (nlga--d b (list 'matmul (list at go dc) (list kk mm nn) (nlga--g (* kk nn))))
          (nlga--accum b (nlga--grad b c) dc (* kk nn))))))
    o))

(defun nlga-scale (b x s)
  "Scale X by resident scalar S ([s])."
  (let* ((n (nlga-rt-size x)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows x) :cols (nlga-rt-cols x))))
    (nlga--d b (list 'scale (list (nlga-rt-slot x) (nlga-rt-slot s) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b n)))
        (nlga--d b (list 'scale (list go (nlga-rt-slot s) dx) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b x) dx n))))
    o))

(defun nlga-softmax (b x)
  "Row-wise softmax of X (M x N)."
  (let* ((m (nlga-rt-rows x)) (n (nlga-rt-cols x)) (os (nlga--tmp b (* m n)))
         (o (nlga-rt--make :slot os :rows m :cols n)))
    (nlga--d b (list 'softmax (list (nlga-rt-slot x) os) (list m n) (nlga--g m)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (ds (nlga--tmp b (* m n))))
        (nlga--d b (list 'softmax-bwd (list os go ds) (list m n) (nlga--g m)))
        (nlga--accum b (nlga--grad b x) ds (* m n)))))
    o))

(defun nlga-slice-cols (b x c0 w)
  "Extract W columns from X (seq x dim) starting at C0 -> (seq x W)."
  (let* ((seq (nlga-rt-rows x)) (dim (nlga-rt-cols x)) (os (nlga--tmp b (* seq w)))
         (o (nlga-rt--make :slot os :rows seq :cols w)))
    (nlga--d b (list 'slice-cols (list (nlga-rt-slot x) os) (list seq dim w c0) (nlga--g (* seq w))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dx (nlga--tmp b (* seq dim))))
        (nlga--d b (list 'set-cols (list dx go) (list seq dim w c0) (nlga--g (* seq w))))
        (nlga--accum b (nlga--grad b x) dx (* seq dim)))))
    o))

(defun nlga-concat-cols (b rts)
  "Column-concatenate a list of (seq x wi) rts into (seq x sum wi)."
  (let* ((seq (nlga-rt-rows (car rts))) (dim (apply #'+ (mapcar #'nlga-rt-cols rts)))
         (os (nlga--tmp b (* seq dim))) (o (nlga-rt--make :slot os :rows seq :cols dim)) (c0 0))
    (dolist (r rts)
      (let ((w (nlga-rt-cols r)))
        (nlga--d b (list 'set-cols (list os (nlga-rt-slot r)) (list seq dim w c0) (nlga--g (* seq w))))
        (setq c0 (+ c0 w))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (c 0))
        (dolist (r rts)
          (let ((w (nlga-rt-cols r)) (sl (nlga--tmp b (* seq (nlga-rt-cols r)))))
            (nlga--d b (list 'slice-cols (list go sl) (list seq dim w c) (nlga--g (* seq w))))
            (nlga--accum b (nlga--grad b r) sl (* seq w))
            (setq c (+ c w)))))))
    o))

(defun nlga-scale-rows (b y s)
  "Multiply each row of Y (M x N) by per-row scalar S (M x 1)."
  (let* ((m (nlga-rt-rows y)) (n (nlga-rt-cols y)) (os (nlga--tmp b (* m n)))
         (o (nlga-rt--make :slot os :rows m :cols n)))
    (nlga--d b (list 'scale-rows (list (nlga-rt-slot y) (nlga-rt-slot s) os) (list m n) (nlga--g (* m n))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (dy (nlga--tmp b (* m n))) (ds (nlga--tmp b m)))
        (nlga--d b (list 'scale-rows (list go (nlga-rt-slot s) dy) (list m n) (nlga--g (* m n))))
        (nlga--accum b (nlga--grad b y) dy (* m n))
        (nlga--d b (list 'rowdot (list go (nlga-rt-slot y) ds) (list m n) (nlga--g m)))
        (nlga--accum b (nlga--grad b s) ds m))))
    o))

(defun nlga-topk-mask (b logits top-k)
  "Top-K additive selection mask from LOGITS (M x E); forward-only (the hard
selection is held constant -- straight-through, no gradient to LOGITS)."
  (let* ((m (nlga-rt-rows logits)) (e (nlga-rt-cols logits)) (os (nlga--tmp b (* m e))))
    (nlga--d b (list 'topk-mask (list (nlga-rt-slot logits) os) (list m e top-k) (nlga--g (* m e))))
    (nlga-rt--make :slot os :rows m :cols e)))

;; --- composite: GQA attention + SwiGLU FFN + MoE + full block --------
(defun nlga-attn-scores (b q k seq dim heads kvheads)
  "Fused causal scaled scores for all heads at once: S laid out
\(heads*seq x seq), S[h,i,j] = (1/sqrt(hd))<q_i^h,k_j^kv> with -inf for j>i."
  (let* ((kvdim (nlga-rt-cols k)) (ss (* heads (* seq seq)))
         (os (nlga--tmp b ss)) (o (nlga-rt--make :slot os :rows (* heads seq) :cols seq)))
    (nlga--d b (list 'attn-scores (list (nlga-rt-slot q) (nlga-rt-slot k) os)
                     (list seq dim heads kvheads) (nlga--g ss)))
    (nlga--bwd-push b (lambda ()
      (let ((gs (nlga--grad b o)) (dq (nlga--tmp b (* seq dim))) (dk (nlga--tmp b (* seq kvdim))))
        (nlga--d b (list 'attn-sc-dq (list gs (nlga-rt-slot k) dq) (list seq dim heads kvheads) (nlga--g (* seq dim))))
        (nlga--accum b (nlga--grad b q) dq (* seq dim))
        (nlga--d b (list 'attn-sc-dk (list gs (nlga-rt-slot q) dk) (list seq dim heads kvheads) (nlga--g (* seq kvdim))))
        (nlga--accum b (nlga--grad b k) dk (* seq kvdim)))))
    o))

(defun nlga-attn-context (b p v seq dim heads kvheads)
  "Fused per-head context: C (seq x dim), C[i,h*hd+t] = sum_j P[h,i,j] V[j,kv]."
  (let* ((kvdim (nlga-rt-cols v)) (os (nlga--tmp b (* seq dim)))
         (o (nlga-rt--make :slot os :rows seq :cols dim)))
    (nlga--d b (list 'attn-context (list (nlga-rt-slot p) (nlga-rt-slot v) os)
                     (list seq dim heads kvheads) (nlga--g (* seq dim))))
    (nlga--bwd-push b (lambda ()
      (let ((gc (nlga--grad b o)) (dp (nlga--tmp b (* heads (* seq seq)))) (dv (nlga--tmp b (* seq kvdim))))
        (nlga--d b (list 'attn-ctx-dp (list gc (nlga-rt-slot v) dp) (list seq dim heads kvheads) (nlga--g (* heads (* seq seq)))))
        (nlga--accum b (nlga--grad b p) dp (* heads (* seq seq)))
        (nlga--d b (list 'attn-ctx-dv (list gc (nlga-rt-slot p) dv) (list seq dim heads kvheads) (nlga--g (* seq kvdim))))
        (nlga--accum b (nlga--grad b v) dv (* seq kvdim)))))
    o))

(defun nlga-gqa (b x wq bq wk bk wv bv wo bo heads kvheads cosr sinr spos sneg _scl _mask)
  "Autograd causal GQA over X (seq x dim), fused: all heads are computed in a
single dispatch per stage (scores / softmax / context) -- the scale and causal
mask are folded into the scores kernel, so the per-head loop (and its O(heads)
dispatches) is gone.  COSR/SINR are RoPE tables, SPOS/SNEG sign scalars.  The
old SCL/MASK args are accepted for API compatibility but unused."
  (let* ((seq (nlga-rt-rows x)) (dim (nlga-rt-cols x))
         (q (nlga-rope b (nlga-linear b x wq bq) heads cosr sinr spos sneg))
         (k (nlga-rope b (nlga-linear b x wk bk) kvheads cosr sinr spos sneg))
         (v (nlga-linear b x wv bv))
         (p (nlga-softmax b (nlga-attn-scores b q k seq dim heads kvheads)))
         (ctx (nlga-attn-context b p v seq dim heads kvheads)))
    (nlga-linear b ctx wo bo)))

(defun nlga-silu-mul (b a c)
  "Fused SwiGLU gate: silu(A) (*) C (one kernel instead of silu + mul)."
  (let* ((n (nlga-rt-size a)) (os (nlga--tmp b n))
         (o (nlga-rt--make :slot os :rows (nlga-rt-rows a) :cols (nlga-rt-cols a))))
    (nlga--d b (list 'silu-mul (list (nlga-rt-slot a) (nlga-rt-slot c) os) (list n) (nlga--g n)))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)) (da (nlga--tmp b n)) (db (nlga--tmp b n)))
        (nlga--d b (list 'silu-mul-bwd (list go (nlga-rt-slot a) (nlga-rt-slot c) da db) (list n) (nlga--g n)))
        (nlga--accum b (nlga--grad b a) da n)
        (nlga--accum b (nlga--grad b c) db n))))
    o))

(defun nlga-swiglu (b x wg bg wu bu wd bd)
  "SwiGLU FFN: (silu(X.Wg^T+bg) (*) (X.Wu^T+bu)) . Wd^T + bd  (fused gate)."
  (nlga-linear b (nlga-silu-mul b (nlga-linear b x wg bg) (nlga-linear b x wu bu)) wd bd))

(defun nlga-moe (b x router brouter experts top-k)
  "Top-K sparse MoE over X (seq x dim).  ROUTER (E x dim) param, BROUTER (E)
param, EXPERTS a list of E plists each with :wg :bg :wu :bu :wd :bd param rts.
The top-K selection is computed on-GPU and held constant; the gate softmax and
the expert SwiGLUs are differentiated."
  (let* ((logits (nlga-linear b x router brouter))
         (gate (nlga-softmax b (nlga-add b logits (nlga-topk-mask b logits top-k))))
         (acc nil) (e 0) (ne (length experts)))
    (while (< e ne)
      (let* ((ex (nth e experts))
             (ge (nlga-slice-cols b gate e 1))
             (ye (nlga-swiglu b x (plist-get ex :wg) (plist-get ex :bg)
                              (plist-get ex :wu) (plist-get ex :bu)
                              (plist-get ex :wd) (plist-get ex :bd)))
             (contrib (nlga-scale-rows b ye ge)))
        (setq acc (if acc (nlga-add b acc contrib) contrib)))
      (setq e (1+ e)))
    acc))

(defun nlga-block (b x blk heads kvheads cosr sinr spos sneg scl mask)
  "Full pre-norm modern block (RMSNorm + GQA/RoPE + FFN) on X (seq x dim).
BLK is a plist of param rts: :ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g and a
feed-forward: either (:router :brouter :experts :top-k) for MoE, or
(:wg :bg :wu :bu :wd :bd) for a single SwiGLU."
  (let* ((x1 (nlga-add b x (nlga-gqa b (nlga-rmsnorm b x (plist-get blk :ln1g))
                                    (plist-get blk :wq) (plist-get blk :bq)
                                    (plist-get blk :wk) (plist-get blk :bk)
                                    (plist-get blk :wv) (plist-get blk :bv)
                                    (plist-get blk :wo) (plist-get blk :bo)
                                    heads kvheads cosr sinr spos sneg scl mask)))
         (bn (nlga-rmsnorm b x1 (plist-get blk :ln2g)))
         (ffn (if (plist-get blk :router)
                  (nlga-moe b bn (plist-get blk :router) (plist-get blk :brouter)
                            (plist-get blk :experts) (or (plist-get blk :top-k) 1))
                (nlga-swiglu b bn (plist-get blk :wg) (plist-get blk :bg)
                             (plist-get blk :wu) (plist-get blk :bu)
                             (plist-get blk :wd) (plist-get blk :bd)))))
    (nlga-add b x1 ffn)))

(defun nlga-embed (b tok wte)
  "Gather embedding: x[i,:] = WTE[TOK[i],:].  TOK is a resident (seq) index rt
\(refreshable per window via `nlga-update'), WTE a (vocab x dim) param.  Returns
\(seq x dim).  Cheaper than one-hot @ WTE: forward is O(seq*dim) and the per-step
input is seq indices, not a seq x vocab one-hot.  WTE is trained on-device by the
scatter-add backward."
  (let* ((seq (nlga-rt-rows tok)) (vocab (nlga-rt-rows wte)) (dim (nlga-rt-cols wte))
         (os (nlga--tmp b (* seq dim))) (o (nlga-rt--make :slot os :rows seq :cols dim)))
    (nlga--d b (list 'embed-gather (list (nlga-rt-slot tok) (nlga-rt-slot wte) os)
                     (list seq dim) (nlga--g (* seq dim))))
    (nlga--bwd-push b (lambda ()
      (let ((go (nlga--grad b o)))
        (nlga--accum-into b wte (* vocab dim)
          (lambda (dst) (nlga--d b (list 'embed-bwd (list (nlga-rt-slot tok) go dst)
                                        (list seq dim vocab) (nlga--g (* vocab dim)))))))))
    o))

(defun nlga-model (b onehot wte blks lnfg wh bh heads kvheads cosr sinr spos sneg scl mask)
  "Stacked model: embed (ONEHOT (seq x vocab) @ WTE (vocab x dim)) -> each block
in BLKS -> final RMSNorm (LNFG) -> linear head (WH, BH).  Returns the logits rt.
Embedding is a matmul against a one-hot token matrix, so WTE is a normal
resident parameter trained on-device by the matmul backward."
  (let ((x (nlga-matmul b onehot wte)))
    (dolist (blk blks)
      (setq x (nlga-block b x blk heads kvheads cosr sinr spos sneg scl mask)))
    (nlga-linear b (nlga-rmsnorm b x lnfg) wh bh)))

;; RoPE cos/sin tables (seq x hd/2), row-major by (position, pair).
(defun nl-llm-gpu-rope-tables (seq hd &optional base)
  "Return (cos-tensor . sin-tensor), each (seq x hd/2), for RoPE."
  (let* ((half (/ hd 2)) (bb (or base 10000.0))
         (co (make-vector (* seq half) 0.0)) (si (make-vector (* seq half) 0.0)) (p 0))
    (while (< p seq)
      (let ((m 0))
        (while (< m half)
          (let ((theta (/ (float p) (expt bb (/ (* 2.0 m) (float hd))))))
            (aset co (+ (* p half) m) (cos theta))
            (aset si (+ (* p half) m) (sin theta)))
          (setq m (1+ m))))
      (setq p (1+ p)))
    (cons (photon-tensor (list seq half) co) (photon-tensor (list seq half) si))))

;; --- loss seeds ------------------------------------------------------
(defun nlga-seed-ce (b logits onehot)
  "Seed LOGITS' gradient with softmax cross-entropy: (softmax - ONEHOT)/M.
ONEHOT is a resident (M x V) one-hot target rt."
  (let ((gl (nlga--grad b logits)) (m (nlga-rt-rows logits)) (v (nlga-rt-cols logits)))
    (nlga--d b (list 'ce-grad (list (nlga-rt-slot logits) (nlga-rt-slot onehot) gl)
                     (list m v) (nlga--g m)))))

(defun nlga-seed-ce-idx (b logits tgt)
  "Seed LOGITS' gradient with softmax cross-entropy using a target-index rt TGT
\(a resident (seq) buffer of target token ids), avoiding a one-hot target."
  (let ((gl (nlga--grad b logits)) (m (nlga-rt-rows logits)) (v (nlga-rt-cols logits)))
    (nlga--d b (list 'ce-grad-idx (list (nlga-rt-slot logits) (nlga-rt-slot tgt) gl)
                     (list m v) (nlga--g m)))))

(defun nlga-seed-mse (b y target invn)
  "Seed Y's gradient with (Y - TARGET) * INVN (MSE); TARGET, INVN are rt."
  (let ((gy (nlga--grad b y)) (n (nlga-rt-size y)))
    (nlga--d b (list 'sub-scale (list (nlga-rt-slot y) (nlga-rt-slot target) (nlga-rt-slot invn) gy)
                     (list n) (nlga--g n)))))

;; --- finalize / run --------------------------------------------------
(defun nlga-keep (b x one)
  "Mark rt X to be returned from the batch; ONE is a resident [1.0] rt.
Returns the ordinal of X among the batch's out slots (index into `nlga-step')."
  (let ((n (nlga-rt-size x)) (os (nlga--out b (nlga-rt-size x))))
    (nlga--d b (list 'scale (list (nlga-rt-slot x) (nlga-rt-slot one) os) (list n) (nlga--g n)))
    (prog1 (nlga-nout b) (cl-incf (nlga-nout b)))))

(defun nlga--clip-scale (b clip)
  "Return a slot holding the SGD/Adam gradient scale.  If CLIP (a positive float)
is given, emit a global-norm clip: sum-of-squares over every parameter gradient
into one slot, then scale = min(1, CLIP / sqrt(sumsq)).  Else a resident [1.0]
\(no scaling).  Call after the backward thunks (grads must be final)."
  (if (not clip)
      (nlga--slot b (list 'res (nelisp-gpu-server-upload (vector 1.0)) 1))
    (let ((gsq (nlga--tmp b 1))
          (cfg (nlga--slot b (list 'res (nelisp-gpu-server-upload (vector clip)) 1)))
          (scl (nlga--tmp b 1)))
      (dolist (p (nlga-params b))
        (let* ((rt (plist-get p :rt)) (n (nlga-rt-size rt)))
          (nlga--d b (list 'sumsq-acc (list (nlga--grad b rt) gsq) (list n) 1))))
      (nlga--d b (list 'clip-scale (list gsq cfg scl) (list 1) 1))
      scl)))

(defun nlga-finish (b lr &optional clip)
  "Run the backward thunks and emit an on-device SGD dispatch per parameter.
LR is a resident scalar rt; CLIP (optional positive float) enables global-norm
gradient clipping.  Call after the forward + loss seed are built."
  (dolist (th (nlga-bwd b)) (funcall th))
  (let ((sslot (nlga--clip-scale b clip)))
    (dolist (p (nlga-params b))
      (let* ((rt (plist-get p :rt)) (n (nlga-rt-size rt)) (gs (nlga--grad b rt)))
        (nlga--d b (list 'sgd (list (nlga-rt-slot rt) gs (nlga-rt-slot lr) sslot) (list n) (nlga--g n)))))))

(defun nlga-compile (b)
  "Compile the assembled batch into a persistent command buffer on the server.
Subsequent `nlga-step' calls then just re-submit it (the graph is fixed across
training steps; only resident weights change in place), avoiding per-step
protocol re-send / buffer alloc / descriptor rebuild.  Call after `nlga-finish'."
  (setf (nlga-compiled b)
        (nelisp-gpu-server-compile (reverse (nlga-slots b)) (reverse (nlga-disp b)))))

(defun nlga-finish-adam (b lr &optional beta1 beta2 eps clip)
  "Like `nlga-finish' but with an on-device Adam optimiser.  Per-parameter first
and second moment buffers (m, v) are kept resident and updated in place by the
`adam' kernel; a shared resident hyperparameter buffer holds [lr_t, b1, b2, eps]
and must be refreshed each step with `nlga-adam-update-t'.  CLIP (optional
positive float) enables global-norm gradient clipping.  Call after the forward +
loss seed (before `nlga-compile')."
  (let ((b1 (or beta1 0.9)) (b2 (or beta2 0.999)) (ep (or eps 1.0e-8)))
    (dolist (th (nlga-bwd b)) (funcall th))
    (let* ((sslot (nlga--clip-scale b clip))
           (hh (nelisp-gpu-server-upload (vector lr b1 b2 ep)))
           (hslot (nlga--slot b (list 'res hh 4))) (mv nil))
      (dolist (p (nlga-params b))
        (let* ((rt (plist-get p :rt)) (n (nlga-rt-size rt)) (gs (nlga--grad b rt))
               (mh (nelisp-gpu-server-upload (make-vector n 0.0)))
               (vh (nelisp-gpu-server-upload (make-vector n 0.0)))
               (ms (nlga--slot b (list 'res mh n))) (vs (nlga--slot b (list 'res vh n))))
          (push mh mv) (push vh mv)
          (nlga--d b (list 'adam (list (nlga-rt-slot rt) gs ms vs hslot sslot) (list n) (nlga--g n)))))
      (setf (nlga-adam b) (list :h hh :lr lr :b1 b1 :b2 b2 :eps ep :mv mv)))))

(defun nlga-adam-state (b)
  "Read back the Adam optimiser state as a list of (m-tensor . v-tensor) in
parameter order (matching `nlga-params').  Server must be up; call before
`nlga-free'.  Pair with the saved step for a complete, seamless resume."
  (let* ((rest (reverse (plist-get (nlga-adam b) :mv))) (out nil))
    (dolist (p (nlga-params b))
      (let* ((sh (photon-tensor-shape (plist-get p :tensor))) (n (nlga-rt-size (plist-get p :rt)))
             (mh (car rest)) (vh (cadr rest)))
        (push (cons (photon-tensor sh (nelisp-gpu-server-read-resident mh n))
                    (photon-tensor sh (nelisp-gpu-server-read-resident vh n)))
              out)
        (setq rest (cddr rest))))
    (nreverse out)))

(defun nlga-adam-restore (b opt)
  "Overwrite each parameter's resident Adam m/v with OPT, a list of (m . v)
tensors in parameter order (from `nlga-adam-state' / a checkpoint).  Call after
`nlga-finish-adam' (which allocated the m/v buffers) and before stepping."
  (let ((rest (reverse (plist-get (nlga-adam b) :mv))) (ol opt))
    (dolist (_ (nlga-params b))
      (nelisp-gpu-server-write-resident (car rest) (photon-tensor-data (car (car ol))))
      (nelisp-gpu-server-write-resident (cadr rest) (photon-tensor-data (cdr (car ol))))
      (setq rest (cddr rest) ol (cdr ol)))))

(defun nlga-adam-update-t (b tstep &optional base-lr)
  "Refresh the Adam hyperparameter buffer for 1-based timestep TSTEP:
lr_t = LR * sqrt(1 - b2^t) / (1 - b1^t), where LR is BASE-LR if given (e.g. a
learning-rate schedule value) else the base lr set at `nlga-finish-adam'.
Call before each `nlga-step'."
  (let* ((a (nlga-adam b)) (lr (or base-lr (plist-get a :lr))) (b1 (plist-get a :b1))
         (b2 (plist-get a :b2)) (ep (plist-get a :eps))
         (lr-t (* lr (/ (sqrt (- 1.0 (expt b2 tstep))) (- 1.0 (expt b1 tstep))))))
    (nelisp-gpu-server-write-resident (plist-get a :h) (vector lr-t b1 b2 ep))))

;; --- learning-rate schedules (pure host math; feed to the lr buffer) --
(defun nl-llm-lr-warmup-cosine (step total warmup base &optional min-lr)
  "Learning rate at 1-based STEP: linear warmup over WARMUP steps up to BASE,
then cosine decay to MIN-LR (default 0) over the remaining TOTAL-WARMUP steps.
For SGD, write (vector lr) into the lr scalar with `nlga-update'; for Adam, pass
it as BASE-LR to `nlga-adam-update-t'."
  (let ((mn (or min-lr 0.0)))
    (cond ((<= step warmup) (* base (/ (float step) (float (max 1 warmup)))))
          ((>= step total) mn)
          (t (let ((p (/ (float (- step warmup)) (float (max 1 (- total warmup))))))
               (+ mn (* 0.5 (- base mn) (+ 1.0 (cos (* float-pi p))))))))))

(defun nl-llm-lr-inv-sqrt (step warmup base)
  "Noam-style inverse-sqrt schedule: BASE * min(1/sqrt(step), step/warmup^1.5)."
  (* base (min (/ 1.0 (sqrt (float step))) (/ (float step) (expt (float warmup) 1.5)))))

(defun nlga-step (b)
  "Run the assembled batch once (one training step); return the out vectors.
Uses the compiled command buffer when `nlga-compile' has been called."
  (if (nlga-compiled b)
      (nelisp-gpu-server-run-compiled (car (nlga-compiled b)) (cdr (nlga-compiled b)))
    (nelisp-gpu-server-batch (reverse (nlga-slots b)) (reverse (nlga-disp b)))))

(defun nlga-readback (b)
  "Copy each parameter's trained resident buffer back into its host tensor."
  (dolist (p (nlga-params b))
    (let* ((rt (plist-get p :rt)) (n (nlga-rt-size rt)) (h (plist-get p :handle))
           (dst (photon-tensor-data (plist-get p :tensor)))
           (v (car (nelisp-gpu-server-run2
                    'scale (list (list 'res h n) (cons 'in (vector 1.0)) (cons 'out n))
                    (list n) (nlga--g n))))
           (i 0))
      (while (< i n) (aset dst i (aref v i)) (setq i (1+ i))))))

(defun nlga-free (b)
  "Free the compiled batch (if any) and all resident parameter handles."
  (when (nlga-compiled b)
    (ignore-errors (nelisp-gpu-server-free-compiled (car (nlga-compiled b))))
    (setf (nlga-compiled b) nil))
  (when (nlga-adam b)
    (ignore-errors (nelisp-gpu-server-free (plist-get (nlga-adam b) :h)))
    (dolist (h (plist-get (nlga-adam b) :mv)) (ignore-errors (nelisp-gpu-server-free h)))
    (setf (nlga-adam b) nil))
  (dolist (p (nlga-params b)) (ignore-errors (nelisp-gpu-server-free (plist-get p :handle)))))

(provide 'nl-llm-gpu-ag)
;;; nl-llm-gpu-ag.el ends here
