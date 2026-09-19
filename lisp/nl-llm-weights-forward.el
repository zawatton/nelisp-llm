;;; nl-llm-weights-forward.el --- reference forward over an imported table  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 2c.  Runs an imported donor
;; model forward on the CPU, in pure Elisp, straight off the int8 table -- the
;; first point at which "the import works" is a measurement rather than a
;; property of the file format.
;;
;; It is deliberately slow.  The job is to be the oracle the GPU path is checked
;; against, so it prefers the obvious arrangement over the fast one: weights
;; stay int8 and are accumulated lane by lane (`nl-llm-weights-apply'), one
;; tensor is read per linear per layer, and attention is written out rather than
;; routed through `nl-llm-gqa', which wants photon-tensors this model cannot
;; afford.  That last choice is a divergence risk, so
;; test/weights-forward-test.el pins this attention against `nl-llm-gqa' on a
;; synthetic model small enough for both.
;;
;; All three donor conventions apply here and all three came from finding them
;; wrong first: the head width is `:head-dim' rather than (/ dim heads), the
;; rotation is half-split rather than interleaved, and q/k are RMSNormed per
;; head before rotating.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)
(require 'nl-llm-weights)
(require 'nl-llm-attn)   ; --rope-heads, --rmsnorm-heads

(defun nl-llm-wf--rmsnorm (x base n gain eps)
  "Return the RMSNorm of X[BASE..BASE+N) scaled by GAIN, as a fresh vector."
  (let ((ss 0.0) (i 0) (out (make-vector n 0.0)))
    (while (< i n)
      (let ((v (aref x (+ base i)))) (setq ss (+ ss (* v v))))
      (setq i (1+ i)))
    (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))) (j 0))
      (while (< j n)
        (aset out j (* (aref x (+ base j)) inv (aref gain j)))
        (setq j (1+ j))))
    out))

(defun nl-llm-wf--silu-mul (g u n)
  "Return silu(G) * U elementwise over N elements, into a fresh vector."
  (let ((out (make-vector n 0.0)) (i 0))
    (while (< i n)
      (let ((v (aref g i)))
        (aset out i (* (/ v (+ 1.0 (exp (- v)))) (aref u i))))
      (setq i (1+ i)))
    out))

(cl-defstruct (nl-llm-wf-layer (:constructor nl-llm-wf-layer--make))
  wq wk wv wo wg wu wd          ; nl-llm-weights-lin
  ln1g ln2g q-norm k-norm)      ; float vectors

;;;###autoload
(defun nl-llm-wf-load-layer (wts layer)
  "Load LAYER of WTS into an `nl-llm-wf-layer'.
Holds int8 bytes for the seven matrices and floats only for the four small
gains, so a layer costs about 12 MB rather than a share of 14 GB."
  (nl-llm-wf-layer--make
   :wq (nl-llm-weights-linear wts :wq layer)
   :wk (nl-llm-weights-linear wts :wk layer)
   :wv (nl-llm-weights-linear wts :wv layer)
   :wo (nl-llm-weights-linear wts :wo layer)
   :wg (nl-llm-weights-linear wts :wg layer)
   :wu (nl-llm-weights-linear wts :wu layer)
   :wd (nl-llm-weights-linear wts :wd layer)
   :ln1g (nl-llm-weights-row wts (nl-llm-weights-tensor wts :ln1g layer) 0)
   :ln2g (nl-llm-weights-row wts (nl-llm-weights-tensor wts :ln2g layer) 0)
   :q-norm (nl-llm-weights-row wts (nl-llm-weights-tensor wts :q-norm layer) 0)
   :k-norm (nl-llm-weights-row wts (nl-llm-weights-tensor wts :k-norm layer) 0)))

(defun nl-llm-wf--attend (q k v seq heads kv-heads hd)
  "Causal grouped-query attention over prepared Q, K and V.
Q is SEQ x HEADS*HD, K and V are SEQ x KV-HEADS*HD, all flat vectors with RoPE
and QK-norm already applied.  Returns the SEQ x HEADS*HD context."
  (let* ((qdim (* heads hd)) (kvdim (* kv-heads hd))
         (grp (/ heads kv-heads)) (scale (/ 1.0 (sqrt (float hd))))
         (ctx (make-vector (* seq qdim) 0.0)))
    (dotimes (h heads)
      (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((scores (make-vector (1+ i) 0.0)) (mx -1.0e30))
            (dotimes (j (1+ i))
              (let ((acc 0.0) (t0 0))
                (while (< t0 hd)
                  (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                      (aref k (+ (* j kvdim) kc t0)))))
                  (setq t0 (1+ t0)))
                (aset scores j (* acc scale))
                (when (> (aref scores j) mx) (setq mx (aref scores j)))))
            (let ((sm 0.0))
              (dotimes (j (1+ i))
                (aset scores j (exp (- (aref scores j) mx)))
                (setq sm (+ sm (aref scores j))))
              (dotimes (t0 hd)
                (let ((acc 0.0))
                  (dotimes (j (1+ i))
                    (setq acc (+ acc (* (/ (aref scores j) sm)
                                        (aref v (+ (* j kvdim) kc t0))))))
                  (aset ctx (+ (* i qdim) qc t0) acc))))))))
    ctx))

;;;###autoload
(defun nl-llm-wf-block (lay x seq cfg)
  "Run one imported BLOCK over X (flat SEQ x dim) and return the new hidden.
LAY is an `nl-llm-wf-layer', CFG the plist from `nl-llm-weights-config'."
  (let* ((dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ff (nl-llm-weights-lin-rows (nl-llm-wf-layer-wg lay)))
         (q (make-vector (* seq qdim) 0.0))
         (k (make-vector (* seq kvdim) 0.0))
         (v (make-vector (* seq kvdim) 0.0)))
    ;; attention projections, row by row
    (dotimes (i seq)
      (let ((a (nl-llm-wf--rmsnorm x (* i dim) dim
                                   (nl-llm-wf-layer-ln1g lay) eps)))
        (let ((qi (nl-llm-weights-apply (nl-llm-wf-layer-wq lay) a))
              (ki (nl-llm-weights-apply (nl-llm-wf-layer-wk lay) a))
              (vi (nl-llm-weights-apply (nl-llm-wf-layer-wv lay) a)))
          (dotimes (t0 qdim) (aset q (+ (* i qdim) t0) (aref qi t0)))
          (dotimes (t0 kvdim) (aset k (+ (* i kvdim) t0) (aref ki t0)))
          (dotimes (t0 kvdim) (aset v (+ (* i kvdim) t0) (aref vi t0))))))
    ;; QK-norm then the half-split rotation, per position
    (dotimes (i seq)
      (nl-llm--rmsnorm-heads q (* i qdim) heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wf-layer-q-norm lay))
                             eps)
      (nl-llm--rmsnorm-heads k (* i kvdim) kv-heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wf-layer-k-norm lay))
                             eps)
      (nl-llm--rope-heads q (* i qdim) heads hd i rbase 'half)
      (nl-llm--rope-heads k (* i kvdim) kv-heads hd i rbase 'half))
    (let ((ctx (nl-llm-wf--attend q k v seq heads kv-heads hd))
          (x1 (make-vector (* seq dim) 0.0)))
      (dotimes (i seq)
        (let ((o (nl-llm-weights-apply (nl-llm-wf-layer-wo lay) ctx (* i qdim))))
          (dotimes (t0 dim)
            (aset x1 (+ (* i dim) t0) (+ (aref x (+ (* i dim) t0))
                                         (aref o t0))))))
      ;; feed-forward
      (let ((out (make-vector (* seq dim) 0.0)))
        (dotimes (i seq)
          (let* ((b (nl-llm-wf--rmsnorm x1 (* i dim) dim
                                        (nl-llm-wf-layer-ln2g lay) eps))
                 (g (nl-llm-weights-apply (nl-llm-wf-layer-wg lay) b))
                 (u (nl-llm-weights-apply (nl-llm-wf-layer-wu lay) b))
                 (h (nl-llm-wf--silu-mul g u ff))
                 (d (nl-llm-weights-apply (nl-llm-wf-layer-wd lay) h)))
            (dotimes (t0 dim)
              (aset out (+ (* i dim) t0) (+ (aref x1 (+ (* i dim) t0))
                                            (aref d t0))))))
        out))))

;;;###autoload
(defun nl-llm-wf-hidden (wts tokens &optional nlayers progress)
  "Run TOKENS through the first NLAYERS blocks of WTS and return every hidden.
Returns a list of flat SEQ x dim vectors: the embedding first, then the output
of each block, so a comparison can name the first layer that diverges instead
of only reporting that the logits are wrong.  PROGRESS, when non-nil, is called
with the layer index as each finishes.  NLAYERS defaults to all of them, which
on this hardware is minutes per token -- that is the price of being the oracle."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (seq (length tokens))
         (n (min (or nlayers (plist-get cfg :layers)) (plist-get cfg :layers)))
         (x (make-vector (* seq dim) 0.0))
         (states nil)
         (i 0))
    (dolist (tk tokens)
      (let ((row (nl-llm-weights-embed wts tk)))
        (dotimes (t0 dim) (aset x (+ (* i dim) t0) (aref row t0))))
      (setq i (1+ i)))
    (push (copy-sequence x) states)
    (dotimes (ly n)
      (setq x (nl-llm-wf-block (nl-llm-wf-load-layer wts ly) x seq cfg))
      (push (copy-sequence x) states)
      (when progress (funcall progress ly)))
    (nreverse states)))

;;;###autoload
(defun nl-llm-wf-final-norm (wts x seq)
  "Apply WTS's final RMSNorm to the flat SEQ x dim hidden X."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (gain (nl-llm-weights-row wts (nl-llm-weights-tensor wts :lnf) 0))
         (out (make-vector (* seq dim) 0.0)))
    (dotimes (i seq)
      (let ((r (nl-llm-wf--rmsnorm x (* i dim) dim gain eps)))
        (dotimes (t0 dim) (aset out (+ (* i dim) t0) (aref r t0)))))
    out))

;;;###autoload
(defun nl-llm-wf-logits (wts hidden seq pos tokens)
  "Return the logits of TOKENS at position POS of HIDDEN, as a float vector.
Only the requested rows of the tied head are read: the full head is 151936 rows
over 1024 columns, so scoring the whole vocabulary here would cost more than
the rest of the forward put together."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (tn (nl-llm-weights-tensor wts :wte))
         (scales (nl-llm-weights-scales wts tn))
         (base (* pos dim))
         (out (make-vector (length tokens) 0.0))
         (i 0))
    (unless (< pos seq)
      (error "nl-llm-wf-logits: position %d outside 0..%d" pos (1- seq)))
    (dolist (tk tokens)
      (let ((lanes (nl-llm-weights-lanes wts tn tk)) (acc 0.0) (j 0))
        (while (< j dim)
          (setq acc (+ acc (* (aref lanes j) (aref hidden (+ base j)))))
          (setq j (1+ j)))
        (aset out i (* acc (aref scales tk))))
      (setq i (1+ i)))
    out))

;;;###autoload
(defun nl-llm-wf-logits-all (wts hidden seq pos)
  "Return every logit at position POS of HIDDEN, as a float vector of :vocab.
Reads the tied head once as bytes -- 155 MB for Qwen3-0.6B, which Emacs holds
comfortably as a unibyte string -- and accumulates over int8 lanes, so the
151936 x 1024 head costs one read rather than 151936 of them.  That is the
difference between scoring the whole vocabulary in seconds and in an hour."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (head (nl-llm-weights-linear wts :wte)))
    (unless (< pos seq)
      (error "nl-llm-wf-logits-all: position %d outside 0..%d" pos (1- seq)))
    (nl-llm-weights-apply head hidden (* pos dim))))

;;;###autoload
(defun nl-llm-wf-argmax (v)
  "Return (INDEX . VALUE) of the largest element of float vector V."
  (let ((best 0) (bv (aref v 0)) (i 1) (n (length v)))
    (while (< i n)
      (when (> (aref v i) bv) (setq bv (aref v i) best i))
      (setq i (1+ i)))
    (cons best bv)))

;;;###autoload
(defun nl-llm-wf-next-token (wts tokens &optional nlayers progress)
  "Run TOKENS through WTS and return (ID . LOGIT) for the greedy next token.
Uses every layer unless NLAYERS says otherwise.  This is the end-to-end path:
the donor's weights, the donor's conventions, and nothing but Elisp."
  (let* ((states (nl-llm-wf-hidden wts tokens nlayers progress))
         (seq (length tokens))
         (final (nl-llm-wf-final-norm wts (car (last states)) seq)))
    (nl-llm-wf-argmax (nl-llm-wf-logits-all wts final seq (1- seq)))))

(provide 'nl-llm-weights-forward)
;;; nl-llm-weights-forward.el ends here
