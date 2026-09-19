;;; nl-llm-weights-backward.el --- vjps for the imported-model block  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 3.  The pieces between one
;; linear and the next, differentiated: RMSNorm, the half-split rotation,
;; QK-norm, causal grouped-query attention and SwiGLU.  With
;; `nl-llm-weights-lora.el' supplying the linear -- including the frozen base's
;; W^T.g -- these complete a backward pass over an imported model, so a LoRA
;; anywhere in a block can be trained.
;;
;; Each is a vector-Jacobian product over flat float vectors rather than a node
;; in an autograd graph, for the same reason the forward is written out: the
;; weights are int8 bytes and `photon-autograd' wants tensors of boxed floats.
;;
;; Every one of them is checked against finite differences individually in
;; test/weights-backward-test.el before anything is composed.  That order is
;; deliberate.  A composed check is stronger -- if a gradient is right through a
;; whole block then every vjp inside it is right -- but when it fails it says
;; only "somewhere in here", and these are functions where a sign or a factor
;; hides comfortably.
;;
;; Gains are treated as frozen: these return gradients with respect to
;; activations, not with respect to the RMSNorm or QK-norm weights.  That is
;; what adapting an imported model needs, and it is a deliberate limit rather
;; than an oversight -- training the gains would need their gradients too.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)

;;; --- RMSNorm --------------------------------------------------------------

(defun nl-llm-wb-rmsnorm-vjp (x base n gain eps dy &optional out)
  "Gradient of RMSNorm at X[BASE..BASE+N) with GAIN, for output gradient DY.

Forward is y_j = x_j * inv * g_j with inv = 1/sqrt(mean(x^2) + eps), so inv
depends on every x_j and the Jacobian is not diagonal:

  dL/dx_j = g_j * dy_j * inv - x_j * inv^3 / n * sum_k x_k g_k dy_k

The second term is the one that gets dropped by accident, and dropping it still
leaves a plausible descent direction, which is why the suite checks this
against finite differences rather than by inspection."
  (let ((ss 0.0) (dot 0.0) (dx (or out (make-vector n 0.0))))
    (dotimes (j n)
      (let ((v (aref x (+ base j))))
        (setq ss (+ ss (* v v)))))
    (let* ((ms (+ (/ ss (float n)) eps))
           (inv (/ 1.0 (sqrt ms)))
           (inv3 (/ (* inv inv inv) (float n))))
      (dotimes (j n)
        (setq dot (+ dot (* (aref x (+ base j)) (aref gain j) (aref dy j)))))
      (dotimes (j n)
        (aset dx j (- (* (aref gain j) (aref dy j) inv)
                      (* (aref x (+ base j)) inv3 dot))))
      dx)))

;;; --- the rotation ---------------------------------------------------------

(defun nl-llm-wb-rope-vjp (dy rowbase nheads hd pos rbase &optional style)
  "Apply the transpose of the RoPE rotation to DY, in place.
A rotation's Jacobian is the rotation itself, so its vjp is the inverse
rotation -- the same angles with the sine negated.  STYLE selects the pairing,
as in `nl-llm--rope-heads': `half' pairs j with j + HD/2, otherwise adjacent."
  (let ((half (/ hd 2)))
    (dotimes (h nheads)
      (let ((base (+ rowbase (* h hd)))
            (orig (make-vector hd 0.0)))
        (dotimes (j hd) (aset orig j (aref dy (+ base j))))
        (dotimes (m half)
          (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 m) (float hd)))))
                 (c (cos theta)) (s (sin theta)))
            (if (eq style 'half)
                (let ((a (aref orig m)) (b (aref orig (+ m half))))
                  ;; forward: [c -s; s c]; transpose: [c s; -s c]
                  (aset dy (+ base m) (+ (* a c) (* b s)))
                  (aset dy (+ base m half) (- (* b c) (* a s))))
              (let ((a (aref orig (* 2 m))) (b (aref orig (1+ (* 2 m)))))
                (aset dy (+ base (* 2 m)) (+ (* a c) (* b s)))
                (aset dy (+ base (1+ (* 2 m))) (- (* b c) (* a s)))))))))
    dy))

;;; --- QK-norm --------------------------------------------------------------

(defun nl-llm-wb-rmsnorm-heads-vjp (x rowbase nheads hd gain dy eps)
  "Gradient of a per-head RMSNorm over NHEADS blocks of HD, in place on DY.
X must be the value the forward saw, before normalisation.  GAIN nil is a
no-op, matching `nl-llm--rmsnorm-heads'."
  (when gain
    (dotimes (h nheads)
      (let* ((base (+ rowbase (* h hd)))
             (slice (make-vector hd 0.0)))
        (dotimes (j hd) (aset slice j (aref dy (+ base j))))
        (let ((dx (nl-llm-wb-rmsnorm-vjp x base hd gain eps slice)))
          (dotimes (j hd) (aset dy (+ base j) (aref dx j)))))))
  dy)

;;; --- SwiGLU ---------------------------------------------------------------

(defun nl-llm-wb-silu-mul-vjp (g u dy n)
  "Gradient of silu(G) * U for output gradient DY.  Returns (DG DU).
silu(z) = z * sigmoid(z), so silu'(z) = sig * (1 + z * (1 - sig))."
  (let ((dg (make-vector n 0.0)) (du (make-vector n 0.0)))
    (dotimes (i n)
      (let* ((z (aref g i))
             (sig (/ 1.0 (+ 1.0 (exp (- z)))))
             (silu (* z sig))
             (dsilu (* sig (+ 1.0 (* z (- 1.0 sig))))))
        (aset dg i (* (aref dy i) (aref u i) dsilu))
        (aset du i (* (aref dy i) silu))))
    (list dg du)))

;;; --- causal grouped-query attention --------------------------------------

(defun nl-llm-wb-attend-vjp (q k v seq heads kv-heads hd dctx)
  "Gradient of `nl-llm-wf--attend' for context gradient DCTX.
Returns (DQ DK DV).  Q is SEQ x HEADS*HD, K and V are SEQ x KV-HEADS*HD, all
as the forward saw them -- after QK-norm and the rotation.  The softmax
Jacobian appears as ds = p * (dp - <p, dp>), the part that is wrong if the
subtraction is forgotten and still looks like a gradient."
  (let* ((qdim (* heads hd)) (kvdim (* kv-heads hd))
         (grp (/ heads kv-heads)) (scale (/ 1.0 (sqrt (float hd))))
         (dq (make-vector (* seq qdim) 0.0))
         (dk (make-vector (* seq kvdim) 0.0))
         (dv (make-vector (* seq kvdim) 0.0)))
    (dotimes (h heads)
      (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
        (dotimes (i seq)
          ;; recompute the forward softmax for this (head, position)
          (let ((p (make-vector (1+ i) 0.0)) (mx -1.0e30) (sm 0.0))
            (dotimes (j (1+ i))
              (let ((acc 0.0))
                (dotimes (t0 hd)
                  (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                      (aref k (+ (* j kvdim) kc t0))))))
                (aset p j (* acc scale))
                (when (> (aref p j) mx) (setq mx (aref p j)))))
            (dotimes (j (1+ i))
              (aset p j (exp (- (aref p j) mx)))
              (setq sm (+ sm (aref p j))))
            (dotimes (j (1+ i)) (aset p j (/ (aref p j) sm)))
            ;; dp_j = <dctx_i, v_j>, and dv_j += p_j * dctx_i
            (let ((dp (make-vector (1+ i) 0.0)) (pdp 0.0))
              (dotimes (j (1+ i))
                (let ((acc 0.0))
                  (dotimes (t0 hd)
                    (setq acc (+ acc (* (aref dctx (+ (* i qdim) qc t0))
                                        (aref v (+ (* j kvdim) kc t0)))))
                    (aset dv (+ (* j kvdim) kc t0)
                          (+ (aref dv (+ (* j kvdim) kc t0))
                             (* (aref p j)
                                (aref dctx (+ (* i qdim) qc t0))))))
                  (aset dp j acc)
                  (setq pdp (+ pdp (* (aref p j) acc)))))
              ;; ds_j = p_j * (dp_j - <p, dp>), then into q and k
              (dotimes (j (1+ i))
                (let ((ds (* (aref p j) (- (aref dp j) pdp) scale)))
                  (unless (= ds 0.0)
                    (dotimes (t0 hd)
                      (aset dq (+ (* i qdim) qc t0)
                            (+ (aref dq (+ (* i qdim) qc t0))
                               (* ds (aref k (+ (* j kvdim) kc t0)))))
                      (aset dk (+ (* j kvdim) kc t0)
                            (+ (aref dk (+ (* j kvdim) kc t0))
                               (* ds (aref q (+ (* i qdim) qc t0))))))))))))))
    (list dq dk dv)))

;;; --- the whole block -----------------------------------------------------
;;
;; Composing the vjps above with the linear's, which means a forward that keeps
;; what the backward needs.  Two values are easy to get wrong here and neither
;; is recoverable afterwards: QK-norm's vjp needs q and k as they were *before*
;; normalisation, while attention's needs them *after* the rotation.  So both
;; are saved, under names that say which is which.
;;
;; `nl-llm-wb-block-forward' is a separate function from `nl-llm-wf-block'
;; rather than a flag on it, because that one is the CPU oracle the whole import
;; is verified against to 1e-14 and is not worth disturbing.  The suite pins the
;; two together instead: the taped forward must equal the oracle exactly.

(require 'nl-llm-weights)
(require 'nl-llm-weights-lora)

;; The forward's own helpers.  Required at load time rather than declared,
;; because the taped forward must be the same arithmetic as the oracle's -- a
;; declaration would let the two drift apart silently.
(require 'nl-llm-weights-forward)

(defvar nl-llm-wb-forward-fn nil
  "When non-nil, a function (LIN X BASE) applying a linear in place of the CPU.
Bound alongside `nl-llm-wb-transpose-fn' to put a block's forward on the GPU
too.  Note that it is a *different computation*, not merely a faster one: the
GPU path quantizes the activation to int8, so a tape taken through it and one
taken through the f32 path do not agree to rounding.  Bind it only when the
comparison you intend is against the same W8A8 arithmetic.")

(defun nl-llm-wb--apply (lin x base)
  "Apply LIN to X at BASE, through `nl-llm-wb-forward-fn' if one is bound."
  (if nl-llm-wb-forward-fn
      (funcall nl-llm-wb-forward-fn lin x base)
    (nl-llm-weights-apply lin x base)))

(defvar nl-llm-wb-forward-seq-fn nil
  "When non-nil, a function (LIN X SEQ STRIDE) applying LIN to SEQ slices.
Slice P is COLS wide at P*STRIDE, and the result is SEQ x ROWS.  Bound
alongside the per-call hooks when the backing store can carry a batch of
positions in one call, which is where the cost of these calls actually is.")

(defun nl-llm-wb--apply-seq (lin x seq stride)
  "Apply LIN to SEQ slices of X, returning SEQ x ROWS.
Falls back to SEQ separate applications -- through `nl-llm-wb--apply', so the
per-call hook still applies -- when no batched one is bound.  The two agree
exactly: batching changes how the arithmetic is carried, not what it is."
  (if nl-llm-wb-forward-seq-fn
      (funcall nl-llm-wb-forward-seq-fn lin x seq stride)
    (let* ((rows (nl-llm-weights-lin-rows lin))
           (out (make-vector (* seq rows) 0.0)))
      (dotimes (p seq)
        (let ((y (nl-llm-wb--apply lin x (* p stride))))
          (dotimes (o rows) (aset out (+ (* p rows) o) (aref y o)))))
      out)))

(defun nl-llm-wb--lin-forward-seq (lay role loras x seq stride)
  "LAY's ROLE over SEQ slices of X; returns a list of SEQ (Y U XS) triples.
The base is applied once for all positions and the adapter is then run per
position through `nl-llm-wlora-forward' itself, handed the precomputed base
output.  Reusing that function rather than repeating its arithmetic is
deliberate: the adapter's forward and its backward have to agree about what U
and XS are, and a second copy of it here is exactly the kind of thing that
drifts."
  (let* ((lin (nl-llm-wf-layer-lin lay role))
         (lora (plist-get loras role))
         (rows (nl-llm-weights-lin-rows lin))
         (base (nl-llm-wb--apply-seq lin x seq stride))
         (out nil))
    (dotimes (p seq)
      (let ((yp (make-vector rows 0.0)))
        (dotimes (o rows) (aset yp o (aref base (+ (* p rows) o))))
        (push (if lora
                  (nl-llm-wlora-forward lin lora x (* p stride)
                                        (lambda (_lin _x _base) yp))
                (list yp nil nil))
              out)))
    (nreverse out)))

(defun nl-llm-wb--lin-forward (lay role loras x base)
  "Apply LAY's ROLE to X at BASE, through a LoRA from LORAS if one is attached.
Returns (Y U XS), U and XS nil when there is no adapter."
  (let ((lin (nl-llm-wf-layer-lin lay role))
        (lora (plist-get loras role)))
    (if lora
        (nl-llm-wlora-forward lin lora x base #'nl-llm-wb--apply)
      (list (nl-llm-wb--apply lin x base) nil nil))))

(defvar nl-llm-wb-transpose-fn nil
  "When non-nil, a function (LIN G) computing W^T.G in place of the CPU loop.
Bound around a backward to route every frozen base's transpose somewhere else
-- the GPU, in practice.  A dynamic variable rather than an argument threaded
through six call sites, and nil restores the CPU path exactly, which is what
keeps the verified reference available for comparison.")

(defun nl-llm-wb--transpose (lin g)
  "Return W^T.G for LIN, through `nl-llm-wb-transpose-fn' if one is bound."
  (if nl-llm-wb-transpose-fn
      (funcall nl-llm-wb-transpose-fn lin g)
    (nl-llm-weights-apply-t lin g)))

(defvar nl-llm-wb-transpose-seq-fn nil
  "When non-nil, a function (LIN G SEQ) computing W^T.G for SEQ gradients.
G is SEQ x ROWS and the result SEQ x COLS.  Same reason as the forward\='s
batched hook: the per-call cost is a round trip and an Elisp-side float
encoding of G, so the number of calls is what there is to save.")

(defun nl-llm-wb--transpose-seq (lin g seq)
  "W^T.G for SEQ gradients at once; SEQ x ROWS in, SEQ x COLS out.
Falls back to SEQ separate transposes -- through `nl-llm-wb--transpose\=', so
the per-call hook still applies -- when no batched one is bound."
  (if nl-llm-wb-transpose-seq-fn
      (funcall nl-llm-wb-transpose-seq-fn lin g seq)
    (let* ((rows (nl-llm-weights-lin-rows lin))
           (cols (nl-llm-weights-lin-cols lin))
           (out (make-vector (* seq cols) 0.0)))
      (dotimes (p seq)
        (let ((gp (make-vector rows 0.0)))
          (dotimes (o rows) (aset gp o (aref g (+ (* p rows) o))))
          (let ((dx (nl-llm-wb--transpose lin gp)))
            (dotimes (i cols) (aset out (+ (* p cols) i) (aref dx i))))))
      out)))

(defun nl-llm-wb--lin-backward-seq (lay role loras saved-list g-all seq acc-all)
  "Accumulate ROLE\='s input gradients for SEQ output gradients into ACC-ALL.
G-ALL is SEQ x ROWS, ACC-ALL is SEQ x COLS and is added to in place.  Returns
ROLE\='s LoRA gradients summed over positions, or nil.

The base transpose runs once for the batch; the adapter then runs per position
through `nl-llm-wlora-backward\=' itself, handed the precomputed column through
its wt-fn -- the same arrangement as the forward, and for the same reason: the
adapter\='s two halves have to agree about U and XS, and a second copy of that
arithmetic here is what would drift.

Accumulation into ACC-ALL is per position and in call order, so a caller that
invokes this for :wq then :wk then :wv adds them in that order at every
position, exactly as a position loop would.  That is not pedantry -- float
addition is not associative, and the point of this function is to be
indistinguishable from the loop it replaces."
  (let* ((lin (nl-llm-wf-layer-lin lay role))
         (lora (plist-get loras role))
         (cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (base (nl-llm-wb--transpose-seq lin g-all seq))
         (merged nil))
    (dotimes (p seq)
      (if (null lora)
          (dotimes (i cols)
            (aset acc-all (+ (* p cols) i)
                  (+ (aref acc-all (+ (* p cols) i)) (aref base (+ (* p cols) i)))))
        (let ((gp (make-vector rows 0.0))
              (dxp (make-vector cols 0.0)))
          (dotimes (o rows) (aset gp o (aref g-all (+ (* p rows) o))))
          (dotimes (i cols) (aset dxp i (aref base (+ (* p cols) i))))
          (let* ((sv (nth p saved-list))
                 (grads (nl-llm-wlora-backward lin lora (nth 2 sv) (nth 1 sv) gp
                                               (lambda (_lin _g) dxp)))
                 (dx (plist-get grads :dx)))
            (dotimes (i cols)
              (aset acc-all (+ (* p cols) i)
                    (+ (aref acc-all (+ (* p cols) i)) (aref dx i))))
            (setq merged (nl-llm-wb--merge-grads merged role grads))))))
    (plist-get merged role)))

(defun nl-llm-wb--lin-backward (lay role loras saved g acc)
  "Accumulate ROLE's input gradient for output gradient G into ACC.
Returns the plist of LoRA gradients for ROLE, or nil.  SAVED is the (Y U XS)
from `nl-llm-wb--lin-forward'."
  (let* ((lin (nl-llm-wf-layer-lin lay role))
         (lora (plist-get loras role)))
    (if (null lora)
        (let ((dx (nl-llm-wb--transpose lin g)))
          (dotimes (i (length dx)) (aset acc i (+ (aref acc i) (aref dx i))))
          nil)
      (let ((grads (nl-llm-wlora-backward lin lora (nth 2 saved) (nth 1 saved) g
                                          #'nl-llm-wb--transpose)))
        (let ((dx (plist-get grads :dx)))
          (dotimes (i (length dx)) (aset acc i (+ (aref acc i) (aref dx i)))))
        grads))))

;;;###autoload
(defun nl-llm-wf-layer-lin (lay role)
  "Return LAY's linear for ROLE, for either layer representation.
Accepts the CPU `nl-llm-wf-layer' and the GPU `nl-llm-wgpu-layer', so a
backward can be taken over a layer loaded either way."
  (cond
   ((nl-llm-wf-layer-p lay)
    (pcase role
      (:wq (nl-llm-wf-layer-wq lay)) (:wk (nl-llm-wf-layer-wk lay))
      (:wv (nl-llm-wf-layer-wv lay)) (:wo (nl-llm-wf-layer-wo lay))
      (:wg (nl-llm-wf-layer-wg lay)) (:wu (nl-llm-wf-layer-wu lay))
      (:wd (nl-llm-wf-layer-wd lay))
      (_ (error "nl-llm-wf-layer-lin: unknown role %S" role))))
   (t (error "nl-llm-wf-layer-lin: unsupported layer %S" (type-of lay)))))

;;;###autoload
(defun nl-llm-wb-block-forward (lay x seq cfg &optional loras)
  "Run one imported block over X, keeping what the backward needs.
Returns (OUT TAPE).  LORAS is a plist ROLE -> adapter; roles without one use
the frozen base alone.  OUT is identical to `nl-llm-wf-block\='s output, which
the suite checks bit for bit.

The linears go a *role* at a time rather than a position at a time, so the
seven of them are seven calls whatever SEQ is.  That reads less naturally than
the position loop it replaces, and it is the point: the cost of one of these
calls is a round trip plus an Elisp-side float encoding rather than the kernel,
so a block at seq 6 was paying 42 of them to apply 7 matrices.  The tape is
built exactly as before, which is what leaves the backward untouched."
  (let* ((dim (plist-get cfg :dim)) (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads)) (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ff (nl-llm-weights-lin-rows (nl-llm-wf-layer-lin lay :wg)))
         (q (make-vector (* seq qdim) 0.0))
         (k (make-vector (* seq kvdim) 0.0))
         (v (make-vector (* seq kvdim) 0.0))
         (as nil))
    (dotimes (i seq)
      (push (nl-llm-wf--rmsnorm x (* i dim) dim
                                (nl-llm-wf-layer-ln1g lay) eps)
            as))
    (setq as (nreverse as))
    (let* ((a-all (apply #'vconcat as))
           (qsaved (nl-llm-wb--lin-forward-seq lay :wq loras a-all seq dim))
           (ksaved (nl-llm-wb--lin-forward-seq lay :wk loras a-all seq dim))
           (vsaved (nl-llm-wb--lin-forward-seq lay :wv loras a-all seq dim)))
      (dotimes (i seq)
        (let ((fq (nth i qsaved)) (fk (nth i ksaved)) (fv (nth i vsaved)))
          (dotimes (t0 qdim) (aset q (+ (* i qdim) t0) (aref (nth 0 fq) t0)))
          (dotimes (t0 kvdim) (aset k (+ (* i kvdim) t0) (aref (nth 0 fk) t0)))
          (dotimes (t0 kvdim) (aset v (+ (* i kvdim) t0) (aref (nth 0 fv) t0)))))
      ;; QK-norm's vjp needs these; the rotation's does not, but attention's
      ;; needs the post-rotation values, so both are kept.
      (let ((q-pre (copy-sequence q)) (k-pre (copy-sequence k)))
        (dotimes (i seq)
          (nl-llm--rmsnorm-heads q (* i qdim) heads hd
                                 (photon-tensor (list hd)
                                                (nl-llm-wf-layer-q-norm lay)) eps)
          (nl-llm--rmsnorm-heads k (* i kvdim) kv-heads hd
                                 (photon-tensor (list hd)
                                                (nl-llm-wf-layer-k-norm lay)) eps)
          (nl-llm--rope-heads q (* i qdim) heads hd i rbase 'half)
          (nl-llm--rope-heads k (* i kvdim) kv-heads hd i rbase 'half))
        (let* ((ctx (nl-llm-wf--attend q k v seq heads kv-heads hd))
               (osaved (nl-llm-wb--lin-forward-seq lay :wo loras ctx seq qdim))
               (x1 (make-vector (* seq dim) 0.0)))
          (dotimes (i seq)
            (let ((yo (nth 0 (nth i osaved))))
              (dotimes (t0 dim)
                (aset x1 (+ (* i dim) t0)
                      (+ (aref x (+ (* i dim) t0)) (aref yo t0))))))
          (let ((bs nil))
            (dotimes (i seq)
              (push (nl-llm-wf--rmsnorm x1 (* i dim) dim
                                        (nl-llm-wf-layer-ln2g lay) eps)
                    bs))
            (setq bs (nreverse bs))
            (let* ((b-all (apply #'vconcat bs))
                   (gs (nl-llm-wb--lin-forward-seq lay :wg loras b-all seq dim))
                   (us (nl-llm-wb--lin-forward-seq lay :wu loras b-all seq dim))
                   (hs nil))
              (dotimes (i seq)
                (push (nl-llm-wf--silu-mul (nth 0 (nth i gs))
                                           (nth 0 (nth i us)) ff)
                      hs))
              (setq hs (nreverse hs))
              (let* ((h-all (apply #'vconcat hs))
                     (ds (nl-llm-wb--lin-forward-seq lay :wd loras h-all seq ff))
                     (out (make-vector (* seq dim) 0.0)))
                (dotimes (i seq)
                  (let ((yd (nth 0 (nth i ds))))
                    (dotimes (t0 dim)
                      (aset out (+ (* i dim) t0)
                            (+ (aref x1 (+ (* i dim) t0)) (aref yd t0))))))
                ;; Every list here is reversed exactly once, above, and handed
                ;; over as it stands.  `nreverse' is destructive, so reversing
                ;; one twice -- which an earlier version did, filling both :g
                ;; and :fg -- corrupts it.
                (list out
                      (list :x x :a as :q-pre q-pre :k-pre k-pre
                            :q q :k k :v v :ctx ctx :x1 x1
                            :b bs :h hs
                            :fq qsaved :fk ksaved :fv vsaved :fo osaved
                            :fg gs :fu us :fd ds
                            :ff ff))))))))))
(defun nl-llm-wb--merge-grads (into role grads)
  "Accumulate GRADS for ROLE into the plist INTO.  Returns the new plist."
  (if (null grads) into
    (let ((have (plist-get into role)))
      (if (null have) (plist-put into role (copy-tree grads))
        (let ((da (plist-get have :da)) (db (plist-get have :db)))
          (dotimes (i (length da))
            (aset da i (+ (aref da i) (aref (plist-get grads :da) i))))
          (dotimes (i (length db))
            (aset db i (+ (aref db i) (aref (plist-get grads :db) i))))
          into)))))

;;;###autoload
(defun nl-llm-wb-block-backward (lay tape dout seq cfg &optional loras)
  "Gradients of one imported block for output gradient DOUT.
TAPE is from `nl-llm-wb-block-forward'.  Returns (DX . LORA-GRADS): DX the
gradient with respect to the block's input, LORA-GRADS a plist ROLE -> (:da :db)
summed over positions.  The base collects nothing, by construction.

Like the forward, this goes a role at a time: the seven transposes are seven
calls whatever SEQ is.  It has to be staged rather than simply reordered,
because within a position the feed-forward's gradient runs wd, then the
SwiGLU's vjp, then wg and wu -- so the batch for wd must complete before the
batch for wg exists.  Four stages come out of that, and the accumulation order
at each position is unchanged, which is what keeps the result identical rather
than merely close."
  (let* ((dim (plist-get cfg :dim)) (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads)) (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ff (plist-get tape :ff))
         (dx (make-vector (* seq dim) 0.0))
         (dx1 (make-vector (* seq dim) 0.0))
         (dctx (make-vector (* seq qdim) 0.0))
         (dh-all (make-vector (* seq ff) 0.0))
         (db-all (make-vector (* seq dim) 0.0))
         (da-all (make-vector (* seq dim) 0.0))
         (lg nil))
    ;; the feed-forward half: wd for every position, then the SwiGLU's vjp,
    ;; then wg and wu, then the norm and the residual into x1
    (setq lg (nl-llm-wb--merge-grads
              lg :wd (nl-llm-wb--lin-backward-seq
                      lay :wd loras (plist-get tape :fd) dout seq dh-all)))
    (let ((sg0 (make-vector (* seq ff) 0.0))
          (sg1 (make-vector (* seq ff) 0.0)))
      (dotimes (i seq)
        (let* ((dh (make-vector ff 0.0)))
          (dotimes (t0 ff) (aset dh t0 (aref dh-all (+ (* i ff) t0))))
          (let ((sg (nl-llm-wb-silu-mul-vjp
                     (nth 0 (nth i (plist-get tape :fg)))
                     (nth 0 (nth i (plist-get tape :fu)))
                     dh ff)))
            (dotimes (t0 ff)
              (aset sg0 (+ (* i ff) t0) (aref (nth 0 sg) t0))
              (aset sg1 (+ (* i ff) t0) (aref (nth 1 sg) t0))))))
      (setq lg (nl-llm-wb--merge-grads
                lg :wg (nl-llm-wb--lin-backward-seq
                        lay :wg loras (plist-get tape :fg) sg0 seq db-all)))
      (setq lg (nl-llm-wb--merge-grads
                lg :wu (nl-llm-wb--lin-backward-seq
                        lay :wu loras (plist-get tape :fu) sg1 seq db-all))))
    (dotimes (i seq)
      (let ((db (make-vector dim 0.0)))
        (dotimes (t0 dim) (aset db t0 (aref db-all (+ (* i dim) t0))))
        (let ((dnorm (nl-llm-wb-rmsnorm-vjp
                      (plist-get tape :x1) (* i dim) dim
                      (nl-llm-wf-layer-ln2g lay) eps db)))
          (dotimes (t0 dim)
            (aset dx1 (+ (* i dim) t0)
                  (+ (aref dx1 (+ (* i dim) t0)) (aref dnorm t0)
                     (aref dout (+ (* i dim) t0))))))))
    ;; o_i = Wo.ctx_i, and the residual into x
    (setq lg (nl-llm-wb--merge-grads
              lg :wo (nl-llm-wb--lin-backward-seq
                      lay :wo loras (plist-get tape :fo) dx1 seq dctx)))
    (dotimes (i seq)
      (dotimes (t0 dim)
        (aset dx (+ (* i dim) t0)
              (+ (aref dx (+ (* i dim) t0)) (aref dx1 (+ (* i dim) t0))))))
    ;; attention, then the rotation and QK-norm in reverse
    (let* ((av (nl-llm-wb-attend-vjp (plist-get tape :q) (plist-get tape :k)
                                     (plist-get tape :v)
                                     seq heads kv-heads hd dctx))
           (dq (nth 0 av)) (dk (nth 1 av)) (dv (nth 2 av)))
      (dotimes (i seq)
        (nl-llm-wb-rope-vjp dq (* i qdim) heads hd i rbase 'half)
        (nl-llm-wb-rope-vjp dk (* i kvdim) kv-heads hd i rbase 'half)
        (nl-llm-wb-rmsnorm-heads-vjp (plist-get tape :q-pre) (* i qdim)
                                     heads hd (nl-llm-wf-layer-q-norm lay)
                                     dq eps)
        (nl-llm-wb-rmsnorm-heads-vjp (plist-get tape :k-pre) (* i kvdim)
                                     kv-heads hd (nl-llm-wf-layer-k-norm lay)
                                     dk eps))
      (setq lg (nl-llm-wb--merge-grads
                lg :wq (nl-llm-wb--lin-backward-seq
                        lay :wq loras (plist-get tape :fq) dq seq da-all)))
      (setq lg (nl-llm-wb--merge-grads
                lg :wk (nl-llm-wb--lin-backward-seq
                        lay :wk loras (plist-get tape :fk) dk seq da-all)))
      (setq lg (nl-llm-wb--merge-grads
                lg :wv (nl-llm-wb--lin-backward-seq
                        lay :wv loras (plist-get tape :fv) dv seq da-all)))
      (dotimes (i seq)
        (let ((da (make-vector dim 0.0)))
          (dotimes (t0 dim) (aset da t0 (aref da-all (+ (* i dim) t0))))
          (let ((dnorm (nl-llm-wb-rmsnorm-vjp
                        (plist-get tape :x) (* i dim) dim
                        (nl-llm-wf-layer-ln1g lay) eps da)))
            (dotimes (t0 dim)
              (aset dx (+ (* i dim) t0)
                    (+ (aref dx (+ (* i dim) t0)) (aref dnorm t0))))))))
    (cons dx lg)))

(provide 'nl-llm-weights-backward)
;;; nl-llm-weights-backward.el ends here
