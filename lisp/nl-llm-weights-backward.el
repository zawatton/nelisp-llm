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

(provide 'nl-llm-weights-backward)
;;; nl-llm-weights-backward.el ends here
