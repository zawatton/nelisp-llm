;;; nl-llm-deltanet.el --- the gated delta rule, forward and backward  -*- lexical-binding: t; -*-

;; Qwen3.6 and Qwen3.8 are hybrid: three quarters of their blocks are not
;; attention at all but Gated DeltaNet, a linear-attention recurrence.  Nothing
;; in this project had an analogue -- `nl-llm-wf--attend' is causal softmax
;; attention -- so 48 of 64 blocks in those models cannot run without this.
;;
;; The recurrence, per head, with state S of shape (d_k x d_v):
;;
;;     g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))     the decay
;;     b_t = sigmoid(b_t)                                   the write strength
;;     P   = g_t * S_{t-1}
;;     m   = P^T k~_t                                       what is already there
;;     d   = b_t * (v_t - m)                                what to change by
;;     S_t = P + k~_t (x) d                                 erase and write
;;     o_t = S_t^T q~_t
;;
;; where q~ and k~ are the L2-normalised query and key.
;;
;; One convention is worth stating because it is invisible in the shapes: the
;; reference divides the query by sqrt(head_dim) *before* the L2 normalisation,
;; and normalises in the kernel.  Dividing afterwards instead is off by a factor
;; of 11.3 on a 128-wide head, with correct shapes and no error.
;;
;; It is tempting to conclude that the division is therefore a no-op, since
;; normalising discards scale.  It nearly is, and the difference is the reason
;; this is done here in the same order rather than skipped: the norm is applied
;; as x / (|x| + 1e-6), so the epsilon does not scale with the vector and a
;; pre-divided input lands about 1e-06 away in relative terms.  That is exactly
;; the size of disagreement the fixture caught when this omitted the division
;; on the reasoning above.
;;
;; The forward keeps every state, because the backward reads S_{t-1} at each
;; step.  Inverting the recurrence instead would divide by g_t, which is a
;; decay in (0,1) and can be small.  That costs SEQ * d_k * d_v per head, which
;; is fine at the sequence lengths this is verified at and is the first thing
;; that will need attention at long ones.

;;; Code:

(require 'cl-lib)

(defconst nl-llm-dn-eps 1.0e-6
  "The epsilon the reference adds to an L2 norm before dividing.")

(defun nl-llm-dn--l2norm (v n)
  "L2-normalise the first N elements of V into a fresh vector."
  (let ((ss 0.0) (out (make-vector n 0.0)))
    (dotimes (i n) (setq ss (+ ss (* (aref v i) (aref v i)))))
    (let ((d (+ (sqrt ss) nl-llm-dn-eps)))
      (dotimes (i n) (aset out i (/ (aref v i) d))))
    out))

(defun nl-llm-dn--scaled (v base n f)
  "N elements of V from BASE, each times F, in a fresh vector."
  (let ((out (make-vector n 0.0)))
    (dotimes (i n) (aset out i (* f (aref v (+ base i)))))
    out))

(defun nl-llm-dn--softplus (x)
  "log(1 + exp(X)), taking the linear branch where exp would overflow."
  (if (> x 20.0) x (log (+ 1.0 (exp x)))))

(defun nl-llm-dn-gate (a a-log dt-bias)
  "The decay for one position: exp(-exp(A-LOG) * softplus(A + DT-BIAS))."
  (exp (- (* (exp a-log) (nl-llm-dn--softplus (+ a dt-bias))))))

(defun nl-llm-dn-beta (b) (/ 1.0 (+ 1.0 (exp (- b)))))

(defvar nl-llm-dn-scale-after-norm nil
  "Non-nil applies 1/sqrt(d_k) to the query AFTER the L2 norm, and not to the key.

The default is nil, which is transformers\=' order: the scale goes on before
the norm, where the norm undoes it, so the query entering the recurrence is
exactly l2norm(q).  That is also what the equations want -- o_t = S_t\=' q~ is
a retrieval by a unit key, and there is no softmax here for an attention scale
to live in.

The variable exists because the other order measures better on Ternary Bonsai
2 27B -- 9.63 nats against 11.56, and the DeltaNet halves stop degrading with
depth -- and that is worth recording rather than burying.  It is not adopted,
because it wins by attenuating the recurrence\='s output eleven-fold (the
residual rms drops from 2.34 to 1.09), and those blocks are currently harmful,
so muting them helps whatever the truth is.  When the DeltaNet defect is
found, this is one of the first things to re-measure.")

;;;###autoload
(defun nl-llm-dn-forward (q k v a b a-log dt-bias seq dk dv)
  "Gated delta rule over SEQ positions for one head; return (OUT TAPE).
Q and K are SEQ x DK flat vectors before normalisation, V is SEQ x DV, A and B
are SEQ long.  OUT is SEQ x DV.  TAPE carries what the backward needs: the
states, the normalised queries and keys, the gates, the write strengths and the
retrieved values."
  (let ((s (make-vector (* dk dv) 0.0))
        (out (make-vector (* seq dv) 0.0))
        (states nil) (qs nil) (ks nil) (gs nil) (bs nil) (mems nil) (deltas nil))
    (dotimes (tt seq)
      (let* ((g (nl-llm-dn-gate (aref a tt) a-log dt-bias))
             (bt (nl-llm-dn-beta (aref b tt)))
             (isq (/ 1.0 (sqrt (float dk))))
             (qt (if nl-llm-dn-scale-after-norm
                     (nl-llm-dn--scaled
                      (nl-llm-dn--l2norm (nl-llm-dn--scaled q (* tt dk) dk 1.0) dk)
                      0 dk isq)
                   (nl-llm-dn--l2norm (nl-llm-dn--scaled q (* tt dk) dk isq) dk)))
             (kt (if nl-llm-dn-scale-after-norm
                     (nl-llm-dn--l2norm (nl-llm-dn--scaled k (* tt dk) dk 1.0) dk)
                   (nl-llm-dn--l2norm (nl-llm-dn--scaled k (* tt dk) dk isq) dk)))
             (mem (make-vector dv 0.0))
             (delta (make-vector dv 0.0)))
        (push (copy-sequence s) states)   ; S_{t-1}, which the backward reads
        ;; P = g * S, in place, then m = P^T k
        (dotimes (i (* dk dv)) (aset s i (* g (aref s i))))
        (dotimes (j dv)
          (let ((acc 0.0))
            (dotimes (i dk) (setq acc (+ acc (* (aref s (+ (* i dv) j)) (aref kt i)))))
            (aset mem j acc)))
        (dotimes (j dv)
          (aset delta j (* bt (- (aref v (+ (* tt dv) j)) (aref mem j)))))
        (dotimes (i dk)
          (dotimes (j dv)
            (aset s (+ (* i dv) j) (+ (aref s (+ (* i dv) j))
                                      (* (aref kt i) (aref delta j))))))
        (dotimes (j dv)
          (let ((acc 0.0))
            (dotimes (i dk) (setq acc (+ acc (* (aref s (+ (* i dv) j)) (aref qt i)))))
            (aset out (+ (* tt dv) j) acc)))
        (push qt qs) (push kt ks) (push g gs) (push bt bs)
        (push mem mems) (push delta deltas)))
    (list out
          (list :states (nreverse states) :q (nreverse qs) :k (nreverse ks)
                :g (nreverse gs) :beta (nreverse bs)
                :mem (nreverse mems) :delta (nreverse deltas)))))


;;; --- the backward ---------------------------------------------------------
;;
;; Reverse over the scan, carrying dS.  At each step the state *after* the
;; update is needed for dq and the state *before* it for dg; the tape holds the
;; latter, and the former is reconstructed as g*S_prev + k (x) delta rather
;; than stored, which keeps the tape at one state per position instead of two.

(defun nl-llm-dn--l2norm-vjp (x n dy)
  "Gradient of X / (|X| + eps) for output gradient DY, both N long.
The epsilon is in the denominator and does not scale with X, which is why it
cannot be dropped: it is the whole reason the pre-division by sqrt(d_k) is not
a no-op."
  (let ((ss 0.0) (dot 0.0) (out (make-vector n 0.0)))
    (dotimes (i n) (setq ss (+ ss (* (aref x i) (aref x i)))))
    (let* ((nrm (sqrt ss)) (d (+ nrm nl-llm-dn-eps)))
      (dotimes (i n) (setq dot (+ dot (* (aref dy i) (aref x i)))))
      (dotimes (i n)
        (aset out i (/ (- (aref dy i)
                          (/ (* (aref x i) dot) (* d (max nrm 1.0e-30))))
                       d)))
      out)))

;;;###autoload
(defun nl-llm-dn-backward (q k v a b a-log dt-bias seq dk dv tape dout)
  "Gradients of `nl-llm-dn-forward' for output gradient DOUT.
Returns a plist (:dq :dk :dv :da :db :da-log :ddt-bias).  TAPE is the second
value the forward returned."
  (let* ((states (vconcat (plist-get tape :states)))
         (qs (vconcat (plist-get tape :q)))
         (ks (vconcat (plist-get tape :k)))
         (gs (vconcat (plist-get tape :g)))
         (bs (vconcat (plist-get tape :beta)))
         (mems (vconcat (plist-get tape :mem)))
         (deltas (vconcat (plist-get tape :delta)))
         (ds (make-vector (* dk dv) 0.0))
         (dp (make-vector (* dk dv) 0.0))
         (st (make-vector (* dk dv) 0.0))
         (dq (make-vector (* seq dk) 0.0))
         (dkv (make-vector (* seq dk) 0.0))
         (dv-out (make-vector (* seq dv) 0.0))
         (da (make-vector seq 0.0))
         (db (make-vector seq 0.0))
         (inv-sqrt (/ 1.0 (sqrt (float dk))))
         (ea (exp a-log))
         (da-log 0.0) (ddt 0.0))
    (cl-loop for tt downfrom (1- seq) to 0 do
      (let* ((g (aref gs tt)) (bt (aref bs tt))
             (qt (aref qs tt)) (kt (aref ks tt))
             (mem (aref mems tt)) (delta (aref deltas tt))
             (sprev (aref states tt))
             (dqn (make-vector dk 0.0)) (dkn (make-vector dk 0.0))
             (ddelta (make-vector dv 0.0)) (dmem (make-vector dv 0.0))
             (dbeta 0.0) (dg 0.0))
        ;; S_t, rebuilt rather than stored
        (dotimes (i dk)
          (dotimes (j dv)
            (aset st (+ (* i dv) j)
                  (+ (* g (aref sprev (+ (* i dv) j)))
                     (* (aref kt i) (aref delta j))))))
        ;; o_t = S_t^T q~
        (dotimes (i dk)
          (let ((acc 0.0))
            (dotimes (j dv)
              (setq acc (+ acc (* (aref st (+ (* i dv) j))
                                  (aref dout (+ (* tt dv) j)))))
              (aset ds (+ (* i dv) j)
                    (+ (aref ds (+ (* i dv) j))
                       (* (aref qt i) (aref dout (+ (* tt dv) j))))))
            (aset dqn i acc)))
        ;; S_t = P + k~ (x) delta
        (dotimes (i dk)
          (let ((acck 0.0))
            (dotimes (j dv)
              (let ((d (aref ds (+ (* i dv) j))))
                (aset dp (+ (* i dv) j) d)
                (setq acck (+ acck (* d (aref delta j))))
                (aset ddelta j (+ (aref ddelta j) (* d (aref kt i))))))
            (aset dkn i (+ (aref dkn i) acck))))
        ;; delta = beta * (v - mem)
        (dotimes (j dv)
          (setq dbeta (+ dbeta (* (aref ddelta j)
                                  (- (aref v (+ (* tt dv) j)) (aref mem j)))))
          (aset dv-out (+ (* tt dv) j) (* bt (aref ddelta j)))
          (aset dmem j (- (* bt (aref ddelta j)))))
        ;; mem = P^T k~,  P = g * S_prev
        (dotimes (i dk)
          (let ((acc 0.0))
            (dotimes (j dv)
              (let ((pij (* g (aref sprev (+ (* i dv) j)))))
                (setq acc (+ acc (* pij (aref dmem j))))
                (aset dp (+ (* i dv) j)
                      (+ (aref dp (+ (* i dv) j)) (* (aref kt i) (aref dmem j))))))
            (aset dkn i (+ (aref dkn i) acc))))
        (dotimes (i (* dk dv))
          (setq dg (+ dg (* (aref dp i) (aref sprev i))))
          (aset ds i (* g (aref dp i))))
        ;; back through the normalisations and the parameterisations
        (if nl-llm-dn-scale-after-norm
            (let* ((qsc (nl-llm-dn--scaled q (* tt dk) dk 1.0))
                   (ksc (nl-llm-dn--scaled k (* tt dk) dk 1.0))
                   (dqs (nl-llm-dn--l2norm-vjp
                         qsc dk (nl-llm-dn--scaled dqn 0 dk inv-sqrt)))
                   (dks (nl-llm-dn--l2norm-vjp ksc dk dkn)))
              (dotimes (i dk)
                (aset dq (+ (* tt dk) i) (aref dqs i))
                (aset dkv (+ (* tt dk) i) (aref dks i))))
          (let* ((qsc (nl-llm-dn--scaled q (* tt dk) dk inv-sqrt))
                 (ksc (nl-llm-dn--scaled k (* tt dk) dk inv-sqrt))
                 (dqs (nl-llm-dn--l2norm-vjp qsc dk dqn))
                 (dks (nl-llm-dn--l2norm-vjp ksc dk dkn)))
            (dotimes (i dk)
              (aset dq (+ (* tt dk) i) (* inv-sqrt (aref dqs i)))
              (aset dkv (+ (* tt dk) i) (* inv-sqrt (aref dks i))))))
        (let* ((z (+ (aref a tt) dt-bias))
               (sp (nl-llm-dn--softplus z))
               (sig (/ 1.0 (+ 1.0 (exp (- z)))))
               (common (* dg g (- ea))))
          (aset da tt (* common sig))
          (setq da-log (+ da-log (* common sp))   ; d/dA_log of -exp(A_log)*sp
                ddt (+ ddt (* common sig))))
        (aset db tt (* dbeta bt (- 1.0 bt)))))
    (list :dq dq :dk dkv :dv dv-out :da da :db db
          :da-log da-log :ddt-bias ddt)))


;;; --- the block around the recurrence --------------------------------------
;;
;; Two pieces sit either side of it.  Before: a causal depthwise convolution
;; over q, k and v with a kernel of four, then SiLU -- the reference fuses the
;; activation into the convolution, so it is fused here too and the
;; pre-activation is kept for the backward.  After: an RMSNorm gated by another
;; projection of the input.
;;
;; The gated norm's order is worth stating because all three orders type-check
;; and two of them are wrong: the reference normalises, *then* applies the
;; weight, *then* multiplies by silu(gate).  Gating before normalising would
;; put the gate inside the variance.

(defun nl-llm-dn--silu (z) (/ z (+ 1.0 (exp (- z)))))

;;;###autoload
(defun nl-llm-dn--rmsnorm-into (vec base n gain eps)
  "RMSNorm the N-long block of VEC at BASE by GAIN, in place."
  (let ((ss 0.0))
    (dotimes (i n) (setq ss (+ ss (* (aref vec (+ base i)) (aref vec (+ base i))))))
    (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
      (dotimes (i n)
        (aset vec (+ base i) (* (aref vec (+ base i)) inv (aref gain i))))))
  vec)

(defun nl-llm-dn--silu-d (z)
  "Derivative of SiLU at Z."
  (let ((sg (/ 1.0 (+ 1.0 (exp (- z))))))
    (* sg (+ 1.0 (* z (- 1.0 sg))))))

;;;###autoload
(defun nl-llm-dn-conv (x w bias seq ch kern)
  "Causal depthwise convolution over X (SEQ x CH) then SiLU; return (Y PRE).
W is CH x KERN, BIAS is CH.  Position T sees T-KERN+1..T, with everything
before the sequence treated as zero.  PRE is the pre-activation, which the
backward needs and the forward would otherwise throw away."
  (let ((pre (make-vector (* seq ch) 0.0))
        (y (make-vector (* seq ch) 0.0)))
    (dotimes (tt seq)
      (dotimes (c ch)
        (let ((acc (aref bias c)))
          (dotimes (r kern)
            (let ((src (+ tt r (- kern) 1)))
              (when (>= src 0)
                (setq acc (+ acc (* (aref w (+ (* c kern) r))
                                    (aref x (+ (* src ch) c))))))))
          (aset pre (+ (* tt ch) c) acc)
          (aset y (+ (* tt ch) c) (nl-llm-dn--silu acc)))))
    (list y pre)))

;;;###autoload
(defun nl-llm-dn-conv-vjp (x w pre dy seq ch kern)
  "Gradient of `nl-llm-dn-conv'.  Returns (:dx :dw :dbias)."
  (let ((dx (make-vector (* seq ch) 0.0))
        (dw (make-vector (* ch kern) 0.0))
        (dbias (make-vector ch 0.0))
        (dpre (make-vector (* seq ch) 0.0)))
    (dotimes (i (* seq ch))
      (aset dpre i (* (aref dy i) (nl-llm-dn--silu-d (aref pre i)))))
    (dotimes (tt seq)
      (dotimes (c ch)
        (let ((d (aref dpre (+ (* tt ch) c))))
          (aset dbias c (+ (aref dbias c) d))
          (dotimes (r kern)
            (let ((src (+ tt r (- kern) 1)))
              (when (>= src 0)
                (aset dw (+ (* c kern) r)
                      (+ (aref dw (+ (* c kern) r))
                         (* d (aref x (+ (* src ch) c)))))
                (aset dx (+ (* src ch) c)
                      (+ (aref dx (+ (* src ch) c))
                         (* d (aref w (+ (* c kern) r)))))))))))
    (list :dx dx :dw dw :dbias dbias)))

;;;###autoload
(defvar nl-llm-dn-gate-before-norm t
  "Non-nil applies the output gate BEFORE the norm, which is Mamba2's order.

`RMSNormGated' in Mamba2 -- and in the Qwen3-Next line that Ternary Bonsai
descends from -- multiplies by silu(gate) and then normalises, so the result
is bounded however large the gate grows.  Normalising first and gating after
gives an output that scales with the gate, which is a different function.

Both run, both are stable, and both produce a plausible residual stream, so
nothing short of the model\='s own output distinguishes them.  It is a variable
rather than a decision because that is the honest state of the knowledge: the
convention is not written down in the weight file.")

(defun nl-llm-dn-norm-gated (x gate weight n eps)
  "Gated RMSNorm of X (N long) by WEIGHT; return (OUT NRM).
NRM is the normalised value before the weight, which the backward reads.
`nl-llm-dn-gate-before-norm' picks which side of the norm the gate falls on."
  (let ((ss 0.0) (nrm (make-vector n 0.0)) (out (make-vector n 0.0)))
    (if nl-llm-dn-gate-before-norm
        (let ((h (make-vector n 0.0)))
          (dotimes (i n)
            (aset h i (* (aref x i) (nl-llm-dn--silu (aref gate i))))
            (setq ss (+ ss (* (aref h i) (aref h i)))))
          (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
            (dotimes (i n)
              (aset nrm i (* (aref h i) inv))
              (aset out i (* (aref weight i) (aref nrm i))))))
      (dotimes (i n) (setq ss (+ ss (* (aref x i) (aref x i)))))
      (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
        (dotimes (i n)
          (aset nrm i (* (aref x i) inv))
          (aset out i (* (aref weight i) (aref nrm i)
                         (nl-llm-dn--silu (aref gate i)))))))
    (list out nrm)))

;;;###autoload
(defun nl-llm-dn-norm-gated-vjp (x gate weight nrm dout n eps)
  "Gradient of `nl-llm-dn-norm-gated'.  Returns (:dx :dgate :dweight)."
  (let ((dx (make-vector n 0.0)) (dgate (make-vector n 0.0))
        (dweight (make-vector n 0.0)) (dnrm (make-vector n 0.0))
        (ss 0.0) (dot 0.0))
    (if nl-llm-dn-gate-before-norm
        (let ((h (make-vector n 0.0)) (dh (make-vector n 0.0)))
          (dotimes (i n)
            (aset h i (* (aref x i) (nl-llm-dn--silu (aref gate i))))
            (setq ss (+ ss (* (aref h i) (aref h i)))))
          (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
            (dotimes (i n)
              (aset dweight i (* (aref dout i) (aref nrm i)))
              (aset dnrm i (* (aref dout i) (aref weight i))))
            (dotimes (i n) (setq dot (+ dot (* (aref dnrm i) (aref h i)))))
            (dotimes (i n)
              (aset dh i (- (* (aref dnrm i) inv)
                            (/ (* inv inv inv (aref h i) dot) (float n)))))
            (dotimes (i n)
              (aset dx i (* (aref dh i) (nl-llm-dn--silu (aref gate i))))
              (aset dgate i (* (aref dh i) (aref x i)
                               (nl-llm-dn--silu-d (aref gate i)))))))
      (dotimes (i n) (setq ss (+ ss (* (aref x i) (aref x i)))))
      (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
        (dotimes (i n)
          (let* ((sg (nl-llm-dn--silu (aref gate i)))
                 (yi (* (aref weight i) (aref nrm i))))
            (aset dgate i (* yi (aref dout i) (nl-llm-dn--silu-d (aref gate i))))
            (let ((dy (* (aref dout i) sg)))
              (aset dweight i (* dy (aref nrm i)))
              (aset dnrm i (* dy (aref weight i))))))
        (dotimes (i n) (setq dot (+ dot (* (aref dnrm i) (aref x i)))))
        (dotimes (i n)
          (aset dx i (- (* (aref dnrm i) inv)
                        (/ (* inv inv inv (aref x i) dot) (float n)))))))
    (list :dx dx :dgate dgate :dweight dweight)))


;;; --- the whole block ------------------------------------------------------
;;
;; in_proj -> causal conv + SiLU -> L2 norm -> the recurrence -> gated norm ->
;; out_proj, which is the order the reference runs them in.  q and k have fewer
;; heads than v and z and are repeated to match; at Qwen3.8-27B that is 16 QK
;; heads against 48 value heads, so each key head serves three value heads.
;;
;; The projections are taken as plain f32 matrices rather than read from a
;; weight file, so the block can be verified before anything can read the
;; formats these models actually ship in.  Whatever supplies them later --
;; ternary with group scales, or int8, or bf16 -- does not change what is
;; below it.

(defun nl-llm-dn--matmul (x w rows in out)
  "X (ROWS x IN) times W (IN x OUT)."
  (let ((y (make-vector (* rows out) 0.0)))
    (dotimes (r rows)
      (dotimes (o out)
        (let ((acc 0.0))
          (dotimes (i in)
            (setq acc (+ acc (* (aref x (+ (* r in) i)) (aref w (+ (* i out) o))))))
          (aset y (+ (* r out) o) acc))))
    y))

(defun nl-llm-dn--matmul-vjp (x w dy rows in out)
  "Gradients of `nl-llm-dn--matmul'.  Returns (DX DW)."
  (let ((dx (make-vector (* rows in) 0.0))
        (dw (make-vector (* in out) 0.0)))
    (dotimes (r rows)
      (dotimes (i in)
        (let ((acc 0.0))
          (dotimes (o out)
            (let ((d (aref dy (+ (* r out) o))))
              (setq acc (+ acc (* d (aref w (+ (* i out) o)))))
              (aset dw (+ (* i out) o)
                    (+ (aref dw (+ (* i out) o))
                       (* d (aref x (+ (* r in) i)))))))
          (aset dx (+ (* r in) i) acc))))
    (list dx dw)))

(defun nl-llm-dn--slice (x rows stride off n)
  "ROWS slices of N columns at OFF from X, which has STRIDE columns a row."
  (let ((out (make-vector (* rows n) 0.0)))
    (dotimes (r rows)
      (dotimes (i n) (aset out (+ (* r n) i) (aref x (+ (* r stride) off i)))))
    out))

(defun nl-llm-dn--unslice (dst rows stride off src n)
  "Add SRC (ROWS x N) into DST at column OFF, DST having STRIDE columns."
  (dotimes (r rows)
    (dotimes (i n)
      (aset dst (+ (* r stride) off i)
            (+ (aref dst (+ (* r stride) off i)) (aref src (+ (* r n) i))))))
  dst)

;;;###autoload
(defun nl-llm-dn-block (x cfg wts)
  "One Gated DeltaNet block over X (SEQ x HIDDEN); return (OUT TAPE).
CFG is (:seq :hidden :nk :nv :hd :kern :eps).  WTS is (:wqkvz :wba :conv-w
:conv-b :a-log :dt-bias :norm-w :wout)."
  (let* ((seq (plist-get cfg :seq)) (hidden (plist-get cfg :hidden))
         (nk (plist-get cfg :nk)) (nv (plist-get cfg :nv))
         (hd (plist-get cfg :hd)) (kern (plist-get cfg :kern))
         (eps (or (plist-get cfg :eps) 1.0e-6))
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd))
         (qkvz-out (+ cd vd)) (grp (/ nv nk))
         (qkvz (nl-llm-dn--matmul x (plist-get wts :wqkvz) seq hidden qkvz-out))
         (ba (nl-llm-dn--matmul x (plist-get wts :wba) seq hidden (* 2 nv)))
         ;; q, k and v are the first CD columns, laid out exactly as the
         ;; convolution wants them, so no copy is needed to build its input
         (mixed (nl-llm-dn--slice qkvz seq qkvz-out 0 cd))
         (cv (nl-llm-dn-conv mixed (plist-get wts :conv-w) (plist-get wts :conv-b)
                             seq cd kern))
         (conv-out (nth 0 cv)) (conv-pre (nth 1 cv))
         (z (nl-llm-dn--slice qkvz seq qkvz-out cd vd))
         (ctx (make-vector (* seq vd) 0.0))
         (heads nil))
    ;; one recurrence per value head, its q and k coming from the key head it
    ;; shares with GRP-1 others
    (dotimes (h nv)
      (let* ((kh (/ h grp))
             (qh (make-vector (* seq hd) 0.0))
             (kh-v (make-vector (* seq hd) 0.0))
             (vh (make-vector (* seq hd) 0.0))
             (ah (make-vector seq 0.0)) (bh (make-vector seq 0.0)))
        (dotimes (tt seq)
          (dotimes (i hd)
            (aset qh (+ (* tt hd) i) (aref conv-out (+ (* tt cd) (* kh hd) i)))
            (aset kh-v (+ (* tt hd) i) (aref conv-out (+ (* tt cd) kd (* kh hd) i)))
            (aset vh (+ (* tt hd) i)
                  (aref conv-out (+ (* tt cd) kd kd (* h hd) i))))
          (aset ah tt (aref ba (+ (* tt 2 nv) h)))
          (aset bh tt (aref ba (+ (* tt 2 nv) nv h))))
        (let* ((fw (nl-llm-dn-forward qh kh-v vh ah bh
                                      (aref (plist-get wts :a-log) h)
                                      (aref (plist-get wts :dt-bias) h)
                                      seq hd hd))
               (oh (nth 0 fw)))
          (push (list :h h :tape (nth 1 fw) :q qh :k kh-v :v vh :a ah :b bh
                      :out oh)
                heads)
          (dotimes (tt seq)
            (dotimes (i hd)
              (aset ctx (+ (* tt vd) (* h hd) i) (aref oh (+ (* tt hd) i))))))))
    (setq heads (nreverse heads))
    ;; the gated norm, per (position, value head), then the output projection
    (let ((gated (make-vector (* seq vd) 0.0)) (nrms nil))
      (dotimes (tt seq)
        (dotimes (h nv)
          (let* ((xs (make-vector hd 0.0)) (gs (make-vector hd 0.0)))
            (dotimes (i hd)
              (aset xs i (aref ctx (+ (* tt vd) (* h hd) i)))
              (aset gs i (aref z (+ (* tt vd) (* h hd) i))))
            (let ((r (nl-llm-dn-norm-gated xs gs (plist-get wts :norm-w) hd eps)))
              (push (list tt h xs gs (nth 1 r)) nrms)
              (dotimes (i hd)
                (aset gated (+ (* tt vd) (* h hd) i) (aref (nth 0 r) i)))))))
      (let ((out (nl-llm-dn--matmul gated (plist-get wts :wout) seq vd hidden)))
        (list out
              (list :qkvz qkvz :ba ba :mixed mixed :conv-out conv-out
                    :conv-pre conv-pre :z z :ctx ctx :gated gated
                    :heads heads :nrms (nreverse nrms)))))))


;;;###autoload
(defun nl-llm-dn-block-backward (x cfg wts tape dout)
  "Gradient of `nl-llm-dn-block' for output gradient DOUT.
Returns a plist (:dx :dwqkvz :dwba :dconv-w :dconv-b :da-log :ddt-bias
:dnorm-w :dwout).  Everything is the reverse of the forward in order; the
only part that is not a straight composition is that q and k are shared
across GRP value heads, so their gradients accumulate rather than assign."
  (let* ((seq (plist-get cfg :seq)) (hidden (plist-get cfg :hidden))
         (nk (plist-get cfg :nk)) (nv (plist-get cfg :nv))
         (hd (plist-get cfg :hd)) (kern (plist-get cfg :kern))
         (eps (or (plist-get cfg :eps) 1.0e-6))
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd))
         (qkvz-out (+ cd vd)) (grp (/ nv nk))
         (mm (nl-llm-dn--matmul-vjp (plist-get tape :gated) (plist-get wts :wout)
                                    dout seq vd hidden))
         (dgated (nth 0 mm)) (dwout (nth 1 mm))
         (dctx (make-vector (* seq vd) 0.0))
         (dz (make-vector (* seq vd) 0.0))
         (dnorm-w (make-vector hd 0.0))
         (dconv-out (make-vector (* seq cd) 0.0))
         (dba (make-vector (* seq 2 nv) 0.0))
         (da-log (make-vector nv 0.0)) (ddt (make-vector nv 0.0)))
    ;; the gated norm, per (position, head)
    (dolist (rec (plist-get tape :nrms))
      (let* ((tt (nth 0 rec)) (h (nth 1 rec)) (xs (nth 2 rec)) (gs (nth 3 rec))
             (nrm (nth 4 rec))
             (dd (make-vector hd 0.0)))
        (dotimes (i hd) (aset dd i (aref dgated (+ (* tt vd) (* h hd) i))))
        (let ((g (nl-llm-dn-norm-gated-vjp xs gs (plist-get wts :norm-w)
                                           nrm dd hd eps)))
          (dotimes (i hd)
            (aset dctx (+ (* tt vd) (* h hd) i) (aref (plist-get g :dx) i))
            (aset dz (+ (* tt vd) (* h hd) i) (aref (plist-get g :dgate) i))
            (aset dnorm-w i (+ (aref dnorm-w i) (aref (plist-get g :dweight) i)))))))
    ;; each head's recurrence
    (dolist (hr (plist-get tape :heads))
      (let* ((h (plist-get hr :h)) (kh (/ h grp))
             (doh (make-vector (* seq hd) 0.0)))
        (dotimes (tt seq)
          (dotimes (i hd)
            (aset doh (+ (* tt hd) i) (aref dctx (+ (* tt vd) (* h hd) i)))))
        (let ((g (nl-llm-dn-backward (plist-get hr :q) (plist-get hr :k)
                                     (plist-get hr :v) (plist-get hr :a)
                                     (plist-get hr :b)
                                     (aref (plist-get wts :a-log) h)
                                     (aref (plist-get wts :dt-bias) h)
                                     seq hd hd (plist-get hr :tape) doh)))
          (dotimes (tt seq)
            (dotimes (i hd)
              ;; q and k are shared by GRP value heads: accumulate
              (aset dconv-out (+ (* tt cd) (* kh hd) i)
                    (+ (aref dconv-out (+ (* tt cd) (* kh hd) i))
                       (aref (plist-get g :dq) (+ (* tt hd) i))))
              (aset dconv-out (+ (* tt cd) kd (* kh hd) i)
                    (+ (aref dconv-out (+ (* tt cd) kd (* kh hd) i))
                       (aref (plist-get g :dk) (+ (* tt hd) i))))
              (aset dconv-out (+ (* tt cd) kd kd (* h hd) i)
                    (aref (plist-get g :dv) (+ (* tt hd) i))))
            (aset dba (+ (* tt 2 nv) h) (aref (plist-get g :da) tt))
            (aset dba (+ (* tt 2 nv) nv h) (aref (plist-get g :db) tt)))
          (aset da-log h (plist-get g :da-log))
          (aset ddt h (plist-get g :ddt-bias)))))
    ;; the convolution, then the two projections
    (let* ((cg (nl-llm-dn-conv-vjp (plist-get tape :mixed) (plist-get wts :conv-w)
                                   (plist-get tape :conv-pre) dconv-out
                                   seq cd kern))
           (dqkvz (make-vector (* seq qkvz-out) 0.0)))
      (nl-llm-dn--unslice dqkvz seq qkvz-out 0 (plist-get cg :dx) cd)
      (nl-llm-dn--unslice dqkvz seq qkvz-out cd dz vd)
      (let* ((m1 (nl-llm-dn--matmul-vjp x (plist-get wts :wqkvz) dqkvz
                                        seq hidden qkvz-out))
             (m2 (nl-llm-dn--matmul-vjp x (plist-get wts :wba) dba
                                        seq hidden (* 2 nv)))
             (dx (make-vector (* seq hidden) 0.0)))
        (dotimes (i (* seq hidden))
          (aset dx i (+ (aref (nth 0 m1) i) (aref (nth 0 m2) i))))
        (list :dx dx :dwqkvz (nth 1 m1) :dwba (nth 1 m2)
              :dconv-w (plist-get cg :dw) :dconv-b (plist-get cg :dbias)
              :da-log da-log :ddt-bias ddt
              :dnorm-w dnorm-w :dwout dwout)))))

(provide 'nl-llm-deltanet)
;;; nl-llm-deltanet.el ends here
