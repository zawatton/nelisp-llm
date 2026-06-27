;;; nl-llm-autograd.el --- autograd ops for the modern transformer block  -*- lexical-binding: t; -*-

;; Reverse-mode autograd ops for the modern primitives (RMSNorm, SiLU,
;; elementwise mul, RoPE, SwiGLU) on photon-autograd's tape, so a modern
;; (Llama/Qwen-style) block can be trained directly -- not only run forward.
;; Each op records a backward closure; correctness is checked numerically by
;; test/autograd-test.el.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-moe)   ; for nl-llm--topk-indices

;;;###autoload
(defun nl-llm-ag-rmsnorm (x gamma &optional eps)
  "Autograd row-wise RMSNorm of X (m x n) with trainable GAMMA (n), no mean sub."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (m (car sh)) (n (nth 1 sh))
         (e (or eps 1.0e-6)) (xd (photon-tensor-data xv)) (gd (photon-tensor-data (pav-value gamma)))
         (istd (make-vector m 0.0)) (od (make-vector (* m n) 0.0)) (ninv (/ 1.0 (float n))) (i 0))
    (while (< i m)
      (let ((base (* i n)) (ss 0.0) (j 0))
        (while (< j n) (let ((v (aref xd (+ base j)))) (setq ss (+ ss (* v v)))) (setq j (1+ j)))
        (let ((is (/ 1.0 (sqrt (+ (* ss ninv) e)))) (j2 0))
          (aset istd i is)
          (while (< j2 n) (aset od (+ base j2) (* (aref xd (+ base j2)) is (aref gd j2))) (setq j2 (1+ j2)))))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (list m n) od)
     (lambda (g)
       (let* ((ggd (photon-tensor-data g)) (dx (make-vector (* m n) 0.0))
              (dgamma (make-vector n 0.0)) (i2 0))
         (while (< i2 m)
           (let ((base (* i2 n)) (is (aref istd i2)) (a 0.0) (j 0))
             (while (< j n)
               (setq a (+ a (* (aref ggd (+ base j)) (aref gd j) (aref xd (+ base j)))))
               (setq j (1+ j)))
             (setq j 0)
             (while (< j n)
               (let ((xk (aref xd (+ base j))) (dyk (aref ggd (+ base j))) (gk (aref gd j)))
                 (aset dgamma j (+ (aref dgamma j) (* dyk xk is)))
                 (aset dx (+ base j) (- (* is dyk gk) (* is is is xk ninv a))))
               (setq j (1+ j))))
           (setq i2 (1+ i2)))
         (photon-autograd--addgrad x (photon-tensor (list m n) dx))
         (photon-autograd--addgrad gamma (photon-tensor (list n) dgamma)))))))

;;;###autoload
(defun nl-llm-ag-silu (x)
  "Autograd SiLU / swish: x * sigmoid(x)."
  (let* ((xv (pav-value x)) (xd (photon-tensor-data xv)) (n (length xd))
         (od (make-vector n 0.0)) (i 0))
    (while (< i n)
      (let* ((v (aref xd i)) (sg (/ 1.0 (+ 1.0 (exp (- v)))))) (aset od i (* v sg)))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (photon-tensor-shape xv) od)
     (lambda (g)
       (let* ((gd (photon-tensor-data g)) (dx (make-vector n 0.0)) (j 0))
         (while (< j n)
           (let* ((v (aref xd j)) (sg (/ 1.0 (+ 1.0 (exp (- v)))))
                  (d (* sg (+ 1.0 (* v (- 1.0 sg))))))
             (aset dx j (* (aref gd j) d)))
           (setq j (1+ j)))
         (photon-autograd--addgrad x (photon-tensor (photon-tensor-shape xv) dx)))))))

;;;###autoload
(defun nl-llm-ag-mul (a b)
  "Autograd elementwise product of same-shape A and B."
  (let ((out (photon-tensor-hadamard (pav-value a) (pav-value b))))
    (photon-autograd--record
     out
     (lambda (g)
       (photon-autograd--addgrad a (photon-tensor-hadamard g (pav-value b)))
       (photon-autograd--addgrad b (photon-tensor-hadamard g (pav-value a)))))))

(defun nl-llm--rope-apply (src seq dim heads base transpose)
  "Return a fresh copy of float vector SRC with per-head RoPE applied.
TRANSPOSE non-nil applies the inverse rotation (Jacobian transpose)."
  (let* ((hd (/ dim heads)) (half (/ hd 2)) (out (copy-sequence src)) (p 0))
    (while (< p seq)
      (let ((h 0))
        (while (< h heads)
          (let ((bb (+ (* p dim) (* h hd))) (m 0))
            (while (< m half)
              (let* ((theta (/ (float p) (expt base (/ (* 2.0 m) (float hd)))))
                     (c (cos theta)) (s (if transpose (- (sin theta)) (sin theta)))
                     (i0 (+ bb (* 2 m))) (i1 (+ bb (* 2 m) 1))
                     (a0 (aref src i0)) (a1 (aref src i1)))
                (aset out i0 (- (* a0 c) (* a1 s)))
                (aset out i1 (+ (* a0 s) (* a1 c))))
              (setq m (1+ m))))
          (setq h (1+ h))))
      (setq p (1+ p)))
    out))

;;;###autoload
(defun nl-llm-ag-rope (x heads &optional base)
  "Autograd per-head RoPE on X (seq x dim).  RoPE is an orthogonal rotation,
so the backward pass is the inverse rotation of the upstream gradient."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (seq (car sh)) (dim (nth 1 sh))
         (b (or base 10000.0)))
    (photon-autograd--record
     (photon-tensor (list seq dim)
                    (nl-llm--rope-apply (photon-tensor-data xv) seq dim heads b nil))
     (lambda (g)
       (photon-autograd--addgrad
        x (photon-tensor (list seq dim)
                         (nl-llm--rope-apply (photon-tensor-data g) seq dim heads b t)))))))

;;;###autoload
(defun nl-llm-ag-swiglu (x wg bg wu bu wd bd)
  "Autograd SwiGLU FFN over X: (silu(X.Wg^T+bg) (*) (X.Wu^T+bu)) . Wd^T + bd."
  (photon-autograd-linear
   (nl-llm-ag-mul (nl-llm-ag-silu (photon-autograd-linear x wg bg))
                  (photon-autograd-linear x wu bu))
   wd bd))

;; --- column slice / concat (per-head split for multi-head attention) ---

;;;###autoload
(defun nl-llm-ag-slice-cols (x c0 w)
  "Autograd: extract W columns starting at C0 from X (m x n) -> (m x w).
Backward scatters the upstream gradient back into those columns."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (m (car sh)) (n (nth 1 sh))
         (xd (photon-tensor-data xv)) (od (make-vector (* m w) 0.0)) (i 0))
    (while (< i m)
      (let ((sb (+ (* i n) c0)) (db (* i w)) (j 0))
        (while (< j w) (aset od (+ db j) (aref xd (+ sb j))) (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (list m w) od)
     (lambda (g)
       (let* ((gd (photon-tensor-data g)) (dx (make-vector (* m n) 0.0)) (i2 0))
         (while (< i2 m)
           (let ((sb (+ (* i2 n) c0)) (db (* i2 w)) (j 0))
             (while (< j w) (aset dx (+ sb j) (aref gd (+ db j))) (setq j (1+ j))))
           (setq i2 (1+ i2)))
         (photon-autograd--addgrad x (photon-tensor (list m n) dx)))))))

;;;###autoload
(defun nl-llm-ag-concat-cols (tensors)
  "Autograd: column-concatenate a list of pav TENSORS, all with the same rows.
Backward slices the upstream gradient back to each input's columns."
  (let* ((vals (mapcar #'pav-value tensors))
         (m (car (photon-tensor-shape (car vals))))
         (widths (mapcar (lambda (v) (nth 1 (photon-tensor-shape v))) vals))
         (n (apply #'+ widths)) (od (make-vector (* m n) 0.0)) (coff 0))
    (dolist (v vals)
      (let* ((w (nth 1 (photon-tensor-shape v))) (vd (photon-tensor-data v)) (i 0))
        (while (< i m)
          (let ((db (+ (* i n) coff)) (sb (* i w)) (j 0))
            (while (< j w) (aset od (+ db j) (aref vd (+ sb j))) (setq j (1+ j))))
          (setq i (1+ i)))
        (setq coff (+ coff w))))
    (photon-autograd--record
     (photon-tensor (list m n) od)
     (lambda (g)
       (let ((gd (photon-tensor-data g)) (coff2 0))
         (dolist (tns tensors)
           (let* ((w (nth 1 (photon-tensor-shape (pav-value tns))))
                  (dx (make-vector (* m w) 0.0)) (i 0))
             (while (< i m)
               (let ((sb (+ (* i n) coff2)) (db (* i w)) (j 0))
                 (while (< j w) (aset dx (+ db j) (aref gd (+ sb j))) (setq j (1+ j))))
               (setq i (1+ i)))
             (photon-autograd--addgrad tns (photon-tensor (list m w) dx))
             (setq coff2 (+ coff2 w)))))))))

;;;###autoload
(defun nl-llm-ag-scale-rows (y s)
  "Autograd: multiply each row i of Y (m x n) by scalar S[i] (S is m x 1).
Used to gate per-token expert outputs in the MoE backward."
  (let* ((yv (pav-value y)) (sh (photon-tensor-shape yv)) (m (car sh)) (n (nth 1 sh))
         (yd (photon-tensor-data yv)) (sd (photon-tensor-data (pav-value s)))
         (od (make-vector (* m n) 0.0)) (i 0))
    (while (< i m)
      (let ((si (aref sd i)) (base (* i n)) (j 0))
        (while (< j n) (aset od (+ base j) (* (aref yd (+ base j)) si)) (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-autograd--record
     (photon-tensor (list m n) od)
     (lambda (g)
       (let* ((gd (photon-tensor-data g)) (dy (make-vector (* m n) 0.0))
              (ds (make-vector m 0.0)) (i2 0))
         (while (< i2 m)
           (let ((si (aref sd i2)) (base (* i2 n)) (acc 0.0) (j 0))
             (while (< j n)
               (aset dy (+ base j) (* (aref gd (+ base j)) si))
               (setq acc (+ acc (* (aref gd (+ base j)) (aref yd (+ base j)))))
               (setq j (1+ j)))
             (aset ds i2 acc))
           (setq i2 (1+ i2)))
         (photon-autograd--addgrad y (photon-tensor (list m n) dy))
         (photon-autograd--addgrad s (photon-tensor (list m 1) ds)))))))

;; --- multi-head / grouped-query attention (fully autograd) -----------

(defun nl-llm--causal-mask (seq)
  "Const additive causal mask (seq x seq): 0 on/below the diagonal, -inf above."
  (let ((md (make-vector (* seq seq) 0.0)) (i 0))
    (while (< i seq)
      (let ((j (1+ i)))
        (while (< j seq) (aset md (+ (* i seq) j) -1.0e30) (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-autograd-const (photon-tensor (list seq seq) md))))

;;;###autoload
(defun nl-llm-ag-gqa (x wq bq wk bk wv bv wo bo heads kv-heads &optional rope-base mask)
  "Autograd causal grouped-query attention over X (seq x dim).
WQ/BQ project queries (dim wide, HEADS heads); WK/BK WV/BV project keys/values
\(KV-HEADS*hd wide); WO/BO is the output projection.  Per-head RoPE is applied
to Q and K.  MASK is an additive (seq x seq) mask const; defaults to causal.
KV-HEADS must divide HEADS (MHA is KV-HEADS = HEADS)."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (seq (car sh)) (dim (nth 1 sh))
         (hd (/ dim heads)) (grp (/ heads kv-heads)) (scale (/ 1.0 (sqrt (float hd))))
         (m (or mask (nl-llm--causal-mask seq)))
         (q (nl-llm-ag-rope (photon-autograd-linear x wq bq) heads rope-base))
         (k (nl-llm-ag-rope (photon-autograd-linear x wk bk) kv-heads rope-base))
         (v (photon-autograd-linear x wv bv))
         (ctxs nil) (h 0))
    (while (< h heads)
      (let* ((qh (nl-llm-ag-slice-cols q (* h hd) hd))
             (kvc (* (/ h grp) hd))
             (kh (nl-llm-ag-slice-cols k kvc hd))
             (vh (nl-llm-ag-slice-cols v kvc hd))
             (scores (photon-autograd-scale
                      (photon-autograd-matmul qh (photon-autograd-transpose kh)) scale))
             (probs (photon-autograd-softmax-rows (photon-autograd-add scores m)))
             (ctxh (photon-autograd-matmul probs vh)))
        (push ctxh ctxs))
      (setq h (1+ h)))
    (photon-autograd-linear (nl-llm-ag-concat-cols (nreverse ctxs)) wo bo)))

;; --- sparse mixture-of-experts (fully autograd) ----------------------

;;;###autoload
(defun nl-llm-ag-moe (x router brouter experts top-k)
  "Autograd top-K sparse MoE over X (seq x dim).
ROUTER is pav (E x dim), BROUTER pav (E) bias; EXPERTS is a list of E plists
each with :wg :bg :wu :bu :wd :bd pav for a SwiGLU expert.  The top-K *selection*
per token is read from the forward router logits and held constant; the gate
weights (softmax over the selected experts) and the expert FFNs are fully
differentiated.  Returns pav (seq x dim)."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv)) (seq (car sh)) (dim (nth 1 sh))
         (ne (length experts))
         (logits (photon-autograd-linear x router brouter))   ; pav (seq x E)
         (ld (photon-tensor-data (pav-value logits)))
         (maskd (make-vector (* seq ne) -1.0e30)) (i 0))
    (while (< i seq)
      (let* ((row (let ((rv (make-vector ne 0.0)) (e 0))
                    (while (< e ne) (aset rv e (aref ld (+ (* i ne) e))) (setq e (1+ e))) rv))
             (idxs (nl-llm--topk-indices row top-k)))
        (dolist (e idxs) (aset maskd (+ (* i ne) e) 0.0)))
      (setq i (1+ i)))
    (let* ((selmask (photon-autograd-const (photon-tensor (list seq ne) maskd)))
           (gate (photon-autograd-softmax-rows (photon-autograd-add logits selmask)))
           (acc nil) (e 0))
      (while (< e ne)
        (let* ((ge (nl-llm-ag-slice-cols gate e 1))   ; (seq x 1) gate weight col
               (ex (nth e experts))
               (ye (nl-llm-ag-swiglu x (plist-get ex :wg) (plist-get ex :bg)
                                     (plist-get ex :wu) (plist-get ex :bu)
                                     (plist-get ex :wd) (plist-get ex :bd)))
               (contrib (nl-llm-ag-scale-rows ye ge)))
          (setq acc (if acc (photon-autograd-add acc contrib) contrib)))
        (setq e (1+ e)))
      acc)))

;; --- full pre-norm block (multi-head + optional MoE) -----------------

;;;###autoload
(defun nl-llm-ag-block (x block heads kv-heads &optional rope-base mask)
  "Autograd pre-norm transformer BLOCK over X (seq x dim).
BLOCK is a plist of pav weights: :ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g
and a feed-forward, either (:router :brouter :experts :top-k) for MoE or
\(:wg :bg :wu :bu :wd :bd) for a single SwiGLU.  Computes
x1 = x + GQA(RMSNorm(x)); returns x1 + FFN(RMSNorm(x1))."
  (let* ((a (nl-llm-ag-rmsnorm x (plist-get block :ln1g)))
         (x1 (photon-autograd-add
              x (nl-llm-ag-gqa a (plist-get block :wq) (plist-get block :bq)
                               (plist-get block :wk) (plist-get block :bk)
                               (plist-get block :wv) (plist-get block :bv)
                               (plist-get block :wo) (plist-get block :bo)
                               heads kv-heads rope-base mask)))
         (bb (nl-llm-ag-rmsnorm x1 (plist-get block :ln2g)))
         (ffn (if (plist-get block :router)
                  (nl-llm-ag-moe bb (plist-get block :router) (plist-get block :brouter)
                                 (plist-get block :experts) (or (plist-get block :top-k) 1))
                (nl-llm-ag-swiglu bb (plist-get block :wg) (plist-get block :bg)
                                  (plist-get block :wu) (plist-get block :bu)
                                  (plist-get block :wd) (plist-get block :bd)))))
    (photon-autograd-add x1 ffn)))

(provide 'nl-llm-autograd)
;;; nl-llm-autograd.el ends here
