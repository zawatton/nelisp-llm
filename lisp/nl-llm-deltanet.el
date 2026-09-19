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
             (qt (nl-llm-dn--l2norm
                  (nl-llm-dn--scaled q (* tt dk) dk (/ 1.0 (sqrt (float dk)))) dk))
             (kt (nl-llm-dn--l2norm
                  (nl-llm-dn--scaled k (* tt dk) dk (/ 1.0 (sqrt (float dk)))) dk))
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
        (let* ((qsc (nl-llm-dn--scaled q (* tt dk) dk inv-sqrt))
               (ksc (nl-llm-dn--scaled k (* tt dk) dk inv-sqrt))
               (dqs (nl-llm-dn--l2norm-vjp qsc dk dqn))
               (dks (nl-llm-dn--l2norm-vjp ksc dk dkn)))
          (dotimes (i dk)
            (aset dq (+ (* tt dk) i) (* inv-sqrt (aref dqs i)))
            (aset dkv (+ (* tt dk) i) (* inv-sqrt (aref dks i)))))
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
(defun nl-llm-dn-norm-gated (x gate weight n eps)
  "RMSNorm X (N long) by WEIGHT, then multiply by silu(GATE); return (OUT NRM).
NRM is the normalised value before the weight, which the backward reads."
  (let ((ss 0.0) (nrm (make-vector n 0.0)) (out (make-vector n 0.0)))
    (dotimes (i n) (setq ss (+ ss (* (aref x i) (aref x i)))))
    (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
      (dotimes (i n)
        (aset nrm i (* (aref x i) inv))
        (aset out i (* (aref weight i) (aref nrm i)
                       (nl-llm-dn--silu (aref gate i))))))
    (list out nrm)))

;;;###autoload
(defun nl-llm-dn-norm-gated-vjp (x gate weight nrm dout n eps)
  "Gradient of `nl-llm-dn-norm-gated'.  Returns (:dx :dgate :dweight)."
  (let ((dx (make-vector n 0.0)) (dgate (make-vector n 0.0))
        (dweight (make-vector n 0.0)) (dnrm (make-vector n 0.0))
        (ss 0.0) (dot 0.0))
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
                      (/ (* inv inv inv (aref x i) dot) (float n))))))
    (list :dx dx :dgate dgate :dweight dweight)))

(provide 'nl-llm-deltanet)
;;; nl-llm-deltanet.el ends here
