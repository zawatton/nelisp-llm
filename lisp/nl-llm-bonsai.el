;;; nl-llm-bonsai.el --- run a hybrid qwen35 model from an imported table  -*- lexical-binding: t; -*-

;; Ternary Bonsai 2 27B is Qwen3.5/3.8-27B's architecture: 64 blocks in which
;; every fourth is gated full attention and the other three are Gated DeltaNet.
;; `nl-llm-deltanet.el' has the recurrence and its block; this drives them from
;; an `nl-llm-wts-v1' table written by tools/bonsai-export.py, and supplies the
;; two things that table brings and Qwen3 did not: a folded orthogonal rotation
;; that has to be undone on activations, and a second block type.
;;
;; The weights are int8 per output row here, quantized by the exporter from the
;; model's F16 build.  That is this project's format, not the model's: the
;; shipped PTQ1_0 and PQ2_0 builds are PrismML's own ternary packings with
;; group scales, which stock tooling refuses and which nothing here reads.

;;; Code:

(require 'nl-llm-weights)
(require 'nl-llm-deltanet)
(require 'nl-llm-deltanet-gpu)
(require 'nl-llm-hadamard)
(require 'nl-llm-weights-forward)   ; rmsnorm and silu-mul

(defvar nl-llm-bonsai-folded-gains t
  "Non-nil treats the RMSNorm gains before a linear as already in the weights.

`general.basename' in Ternary Bonsai 2 27B\='s file is \"folded\", and it is not
only the rotation that is folded.  A gain sitting immediately before a linear
can be folded into it on the input axis -- W\=' = W.G.A\=' makes W\='.(A.x_hat)
equal W.G.x_hat -- and applying it at run time as well applies it twice.  That
is `attn_norm\=', `post_attention_norm\=', `ssm_norm\=' and `output_norm\='.
`attn_q_norm\=' and `attn_k_norm\=' are followed by the rotary and a dot
product, not by a linear, so they cannot be folded into anything and are
always applied.

Measured on 86 tokens of prose: 9.42 nats against 11.22 with the gains
applied, where chance is 12.42 and the same measurement gives 3.12 on the
Qwen3-0.6B donor.  The largest single improvement found, and invisible in
every norm -- the gains are near 1, so double-applying them is a mild
distortion that only compounds over sixty-four blocks.")

(defun nl-llm-bonsai--gate (z)
  "The attention block\='s output gate: a sigmoid, not a SiLU.

Qwen3-Next gates its attention output with `sigmoid\=' and its Gated DeltaNet
norm with `silu\='; the two are easy to interchange and the shapes do not
object.  SiLU is negative below zero and unbounded above it, so using it here
does not gate the context, it distorts it.  Measured on 86 tokens of prose:
9.42 nats with the sigmoid against 12.68 with the SiLU, where chance is 12.42."
  (/ 1.0 (+ 1.0 (exp (- z)))))

(defun nl-llm-bonsai--gain (sess role layer n)
  "The gain for ROLE at LAYER, or ones when the weights already carry it."
  (if (and nl-llm-bonsai-folded-gains (plist-get sess :folded))
      (make-vector n 1.0)
    (let ((wts (plist-get sess :wts)))
      (nl-llm-weights-row wts (nl-llm-weights-tensor wts role layer) 0))))

(defun nl-llm-bonsai--interval (cfg)
  "Blocks per group; 1 -- every block is full attention -- when unstated."
  (or (plist-get cfg :full-attention-interval) 1))

(defun nl-llm-bonsai--gated-q-p (lins cfg)
  "Non-nil when `:wq' carries an output gate beside the query.
Self-describing: the gated form is twice as wide as HEADS * HEAD-DIM, so the
tensor says which it is and no header key has to."
  (> (nl-llm-weights-lin-rows (plist-get lins :wq))
     (* (plist-get cfg :heads) (plist-get cfg :head-dim))))

(defvar nl-llm-bonsai-apply-seq-fn nil
  "When non-nil, a function (LIN X SEQ STRIDE) applying LIN at every position.

A projection called once a position is one round trip a position, and the
round trip is most of what it costs: profiling a DeltaNet block found the
projections at 5.46s of 7.37s across forty calls of 136ms, for arithmetic the
device does in a fraction of that.  The kernels already take the count of
positions, so the batching is a change here rather than there.")

(defvar nl-llm-bonsai-scan-fn nil
  "When non-nil, a function running the gated delta rule for every head.

Called with (QN KN V GATES BETAS SEQ NV DK DV) and returning SEQ x NV x DV,
which is `nl-llm-dngpu-scan\='s shape.  The recurrence is where a DeltaNet
block\='s time goes -- 48 heads, a 128x128 state each, updated at every
position -- and it is sequential only in position, so it belongs on the
device.  The hook keeps the CPU path, which the gradient suite checks, rather
than replacing it.")

(defvar nl-llm-bonsai-apply-fn nil
  "When non-nil, a function (LIN X BASE) applying a linear in place of the CPU.
The same hook shape the Qwen3 path uses, so `nl-llm-wgpu-apply-resident' drops
straight in.  A block is about 385M multiply-accumulates a position and nearly
all of them are in these projections, so this is where the time is.")

(defun nl-llm-bonsai-linears (sess layer)
  "The LAYER's quantized linears, loaded once and cached on SESS.

Cached because identity matters, not only cost.  A GPU handle table is keyed
by `eq' on the linear object, so a block that called `nl-llm-weights-linear'
again would hand the lookup an equal-but-not-eq object, miss every time, and
fall back to the CPU -- which looks like \"the GPU did not help\" rather than
like a defect.  That has now happened three times in this work; a cache is the
structural answer to it."
  (let* ((tbl (plist-get sess :lins))
         (hit (gethash layer tbl)))
    (or hit
        (let* ((wts (plist-get sess :wts))
               (cfg (plist-get sess :cfg))
               (iv (nl-llm-bonsai--interval cfg))
               (roles (if (= (mod layer iv) (1- iv))
                          '(:wq :wk :wv :wo :wg :wu :wd)
                        '(:wqkv :wz :walpha :wbeta :wout :wg :wu :wd)))
               (pl nil))
          (dolist (r roles)
            (setq pl (plist-put pl r (nl-llm-weights-linear wts r layer))))
          (puthash layer pl tbl)
          pl))))

(defun nl-llm-bonsai-forget-layer (sess layer)
  "Drop LAYER's cached linears, so a long run does not hold all 64."
  (remhash layer (plist-get sess :lins)))

(defun nl-llm-bonsai--apply-seq (lin x seq stride)
  "LIN applied to SEQ slices of X strided by STRIDE; returns SEQ x ROWS.
Without the hook this is the per-position loop written out, so the two agree
element for element and a block can be read without knowing which is in use."
  (if nl-llm-bonsai-apply-seq-fn
      (funcall nl-llm-bonsai-apply-seq-fn lin x seq stride)
    (let* ((rows (nl-llm-weights-lin-rows lin))
           (out (make-vector (* seq rows) 0.0)))
      (dotimes (p seq)
        (let ((y (nl-llm-bonsai--apply lin x (* p stride))))
          (dotimes (o rows) (aset out (+ (* p rows) o) (aref y o)))))
      out)))

(defun nl-llm-bonsai--apply (lin x base)
  "LIN applied to the COLS-long slice of X at BASE, dequantizing per row."
  (if nl-llm-bonsai-apply-fn
      (funcall nl-llm-bonsai-apply-fn lin x base)
    (nl-llm-weights-apply lin x base)))

;;;###autoload
(defun nl-llm-bonsai-open (path &optional invert)
  "Open the table at PATH and return a session plist.
INVERT selects the other of the two orthogonal transforms.  Which one the
weights were folded with is not in the file: the Sylvester-Walsh matrix is
symmetric, so \"signs then Hadamard\" and \"Hadamard then signs\" are exact
transposes of each other, and every norm-based instrument is blind to the
difference because both are orthogonal."
  (let* ((wts (nl-llm-weights-open path))
         (cfg (nl-llm-weights-config wts))
         (widths (plist-get cfg :hadamard-widths))
         (vals (vconcat (plist-get cfg :hadamard-signs)))
         (signs nil))
    (dolist (w widths)
      (setq signs (plist-put signs w (nl-llm-had-signs vals widths w))))
    (list :wts wts :cfg cfg :signs signs :invert invert
          :block (or (plist-get cfg :hadamard-block) nl-llm-had-block)
          :rotate (and widths t)
          :folded (if (plist-member cfg :folded-gains)
                      (plist-get cfg :folded-gains)
                    (and widths t))
          :lins (make-hash-table :test 'eql))))

;;;###autoload
(defun nl-llm-bonsai-rotate (sess x n &optional back)
  "Apply the model's folded rotation to the N-long X in place.
BACK pulls a gradient through it instead, which is the inverse transform --
the rotation is orthogonal, so that is all a pullback is.  The session's
`:invert' says which of the two orthogonal candidates is the forward one.

A model that declares no `:hadamard-widths' has no folded rotation and this
is the identity, which is what lets the same driver run an ordinary
transformer."
  (if (not (plist-get sess :rotate))
      x
    (let ((s (plist-get (plist-get sess :signs) n))
          (nl-llm-had-block (or (plist-get sess :block) nl-llm-had-block))
          (inv (plist-get sess :invert)))
      (unless s (error "nl-llm-bonsai-rotate: no signs for width %d" n))
      (nl-llm-had-rotate x n s (if back (not inv) inv)))))

;;;###autoload
(defun nl-llm-bonsai-deltanet-block (sess layer x seq)
  "One Gated DeltaNet block of LAYER over X (SEQ x dim); return the output.
The projections are int8, so each is applied row by row rather than as a dense
matrix; everything between them is `nl-llm-deltanet.el'."
  (let* ((wts (plist-get sess :wts)) (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (nk (plist-get cfg :ssm-groups)) (nv (plist-get cfg :ssm-heads))
         (hd (plist-get cfg :ssm-state)) (kern (plist-get cfg :ssm-conv-kernel))
         (ff (plist-get cfg :ff))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd))
         (ln1 (nl-llm-bonsai--gain sess :ln1g layer dim))
         (ln2 (nl-llm-bonsai--gain sess :ln2g layer dim))
         (snorm (nl-llm-bonsai--gain sess :ssm-norm layer hd))
         (alog (nl-llm-weights-row wts (nl-llm-weights-tensor wts :a-log layer) 0))
         (dtb (nl-llm-weights-row wts (nl-llm-weights-tensor wts :dt-bias layer) 0))

         (lins (nl-llm-bonsai-linears sess layer))
         (wqkv (plist-get lins :wqkv)) (wz (plist-get lins :wz))
         (wa (plist-get lins :walpha)) (wb (plist-get lins :wbeta))
         (wout (plist-get lins :wout)) (wg (plist-get lins :wg))
         (wu (plist-get lins :wu)) (wd (plist-get lins :wd))
         ;; ssm_conv1d is [4, 10240] in GGUF's fastest-first order, which the
         ;; exporter reshapes to 10240 rows of 4 -- already channel-major, the
         ;; layout the block wants, so this reads rows and does not transpose
         (convw (nl-llm-weights-f32-flat
                 wts (nl-llm-weights-tensor wts :conv-w layer)))
         (convb (make-vector cd 0.0))
         (mixed (make-vector (* seq cd) 0.0))
         (z (make-vector (* seq vd) 0.0))
         (ba (make-vector (* seq 2 nv) 0.0))
         (out (make-vector (* seq dim) 0.0)))
    ;; Normalise, then feed each projection the basis ITS weight was folded
    ;; for.  The file lists which weights carry the rotation, and ssm_alpha and
    ;; ssm_beta are not among them -- only attn_qkv, attn_gate, ssm_out and the
    ;; three feed-forward matrices are.  Handing the gates a rotated activation
    ;; runs clean and poisons the recurrence, because alpha is the decay and
    ;; beta the write strength: an early version did exactly that and the
    ;; block's output came out 1300 times its input.
    (let ((pl (make-vector (* seq dim) 0.0))
          (ar (make-vector (* seq dim) 0.0)))
      (dotimes (tt seq)
        (let ((p (nl-llm-wf--rmsnorm x (* tt dim) dim ln1 eps)))
          (dotimes (i dim) (aset pl (+ (* tt dim) i) (aref p i)))
          (nl-llm-bonsai-rotate sess p dim)
          (dotimes (i dim) (aset ar (+ (* tt dim) i) (aref p i)))))
      (setq mixed (nl-llm-bonsai--apply-seq wqkv ar seq dim))
      (setq z (nl-llm-bonsai--apply-seq wz ar seq dim))
      (let ((ya (nl-llm-bonsai--apply-seq wa pl seq dim))
            (yb (nl-llm-bonsai--apply-seq wb pl seq dim)))
        (dotimes (tt seq)
          (dotimes (i nv)
            (aset ba (+ (* tt 2 nv) i) (aref ya (+ (* tt nv) i)))
            (aset ba (+ (* tt 2 nv) nv i) (aref yb (+ (* tt nv) i)))))))
    (let* ((cv (nl-llm-dn-conv mixed convw convb seq cd kern))
           (conv-out (nth 0 cv))
           (ctx (make-vector (* seq vd) 0.0))
           (grp (/ nv nk)))
      (if nl-llm-bonsai-scan-fn
          (nl-llm-bonsai--scan-heads ctx conv-out ba alog dtb
                                     seq nv hd kd grp cd)
        (dotimes (h nv)
          (let* ((kh (/ h grp))
                 (qh (make-vector (* seq hd) 0.0)) (khv (make-vector (* seq hd) 0.0))
                 (vh (make-vector (* seq hd) 0.0))
                 (ah (make-vector seq 0.0)) (bh (make-vector seq 0.0)))
            (dotimes (tt seq)
              (dotimes (i hd)
                (aset qh (+ (* tt hd) i) (aref conv-out (+ (* tt cd) (* kh hd) i)))
                (aset khv (+ (* tt hd) i) (aref conv-out (+ (* tt cd) kd (* kh hd) i)))
                (aset vh (+ (* tt hd) i)
                      (aref conv-out (+ (* tt cd) kd kd (* h hd) i))))
              (aset ah tt (aref ba (+ (* tt 2 nv) h)))
              (aset bh tt (aref ba (+ (* tt 2 nv) nv h))))
            (let ((oh (nth 0 (nl-llm-dn-forward qh khv vh ah bh (aref alog h)
                                                (aref dtb h) seq hd hd))))
              (dotimes (tt seq)
                (dotimes (i hd)
                  (aset ctx (+ (* tt vd) (* h hd) i)
                        (aref oh (+ (* tt hd) i)))))))))
      ;; gated norm per (position, head), then out_proj, then the residual
      (let ((gr (make-vector (* seq vd) 0.0)))
        (dotimes (tt seq)
          (let ((g (make-vector vd 0.0)))
            (dotimes (h nv)
              (let ((xs (make-vector hd 0.0)) (gs (make-vector hd 0.0)))
                (dotimes (i hd)
                  (aset xs i (aref ctx (+ (* tt vd) (* h hd) i)))
                  (aset gs i (aref z (+ (* tt vd) (* h hd) i))))
                (let ((r (nth 0 (nl-llm-dn-norm-gated xs gs snorm hd eps))))
                  (dotimes (i hd) (aset g (+ (* h hd) i) (aref r i))))))
            (nl-llm-bonsai-rotate sess g vd)
            (dotimes (i vd) (aset gr (+ (* tt vd) i) (aref g i)))))
        (let ((o (nl-llm-bonsai--apply-seq wout gr seq vd)))
          (dotimes (tt seq)
            (dotimes (i dim)
              (aset out (+ (* tt dim) i)
                    (+ (aref x (+ (* tt dim) i)) (aref o (+ (* tt dim) i))))))))
      ;; the feed-forward half
      (nl-llm-bonsai--ffn sess out seq dim ff ln2 eps wg wu wd)
      out)))

(defun nl-llm-bonsai--ffn (sess out seq dim ff ln2 eps wg wu wd)
  "Add the feed-forward half to OUT in place, every position at once.
The same half in both block types, and the only place three of a block's
eight projections live."
  (let ((br (make-vector (* seq dim) 0.0))
        (hr (make-vector (* seq ff) 0.0)))
    (dotimes (tt seq)
      (let ((b (nl-llm-wf--rmsnorm out (* tt dim) dim ln2 eps)))
        (nl-llm-bonsai-rotate sess b dim)
        (dotimes (i dim) (aset br (+ (* tt dim) i) (aref b i)))))
    (let ((gg (nl-llm-bonsai--apply-seq wg br seq dim))
          (uu (nl-llm-bonsai--apply-seq wu br seq dim)))
      (dotimes (tt seq)
        (let ((g1 (make-vector ff 0.0)) (u1 (make-vector ff 0.0)))
          (dotimes (i ff)
            (aset g1 i (aref gg (+ (* tt ff) i)))
            (aset u1 i (aref uu (+ (* tt ff) i))))
          (let ((hh (nl-llm-wf--silu-mul g1 u1 ff)))
            (nl-llm-bonsai-rotate sess hh ff)
            (dotimes (i ff) (aset hr (+ (* tt ff) i) (aref hh i)))))))
    (let ((dd (nl-llm-bonsai--apply-seq wd hr seq ff)))
      (dotimes (tt seq)
        (dotimes (i dim)
          (aset out (+ (* tt dim) i)
                (+ (aref out (+ (* tt dim) i)) (aref dd (+ (* tt dim) i)))))))
    out))


;;; --- the full-attention block, every fourth one ---------------------------
;;
;; Not the Qwen3 attention this project already has.  Three differences, none
;; of which changes a shape:
;;
;;   * head_dim is 256 against Qwen3-0.6B's 128, and 24 query heads to 4 key
;;     heads rather than 16 to 8;
;;   * the rotation covers 64 of those 256 dimensions, not all of them, and
;;     the base is 10,000,000 rather than 1,000,000;
;;   * the query projection is 24 * 256 * 2 wide because it carries an output
;;     gate alongside the query.
;;
;; A partial rotation is the kind of thing that runs clean and answers wrong:
;; rotating all 256 dimensions of a head whose model rotates 64 produces a
;; correctly shaped, entirely incorrect key.

(defun nl-llm-bonsai--scan-heads (ctx conv-out ba alog dtb seq nv hd kd grp cd)
  "Fill CTX by running every head\='s recurrence through `nl-llm-bonsai-scan-fn\='.

The per-head slicing the CPU path does one head at a time is done once here,
into the flat [position][head][dim] layout the device wants.  Queries and keys
come from the key GROUP a value head belongs to, which is why they are copied
rather than pointed at: three value heads share one key head."
  (let ((q (make-vector (* seq nv hd) 0.0))
        (k (make-vector (* seq nv hd) 0.0))
        (v (make-vector (* seq nv hd) 0.0))
        (a (make-vector (* seq nv) 0.0))
        (b (make-vector (* seq nv) 0.0)))
    (dotimes (tt seq)
      (dotimes (h nv)
        (let ((kh (/ h grp)) (base (* (+ (* tt nv) h) hd)) (cb (* tt cd)))
          (dotimes (i hd)
            (aset q (+ base i) (aref conv-out (+ cb (* kh hd) i)))
            (aset k (+ base i) (aref conv-out (+ cb kd (* kh hd) i)))
            (aset v (+ base i) (aref conv-out (+ cb kd kd (* h hd) i))))
          (aset a (+ (* tt nv) h) (aref ba (+ (* tt 2 nv) h)))
          (aset b (+ (* tt nv) h) (aref ba (+ (* tt 2 nv) nv h))))))
    (let* ((prep (nl-llm-dngpu-prepare q k a b alog dtb seq nv hd))
           (out (funcall nl-llm-bonsai-scan-fn
                         (nth 0 prep) (nth 1 prep) v (nth 2 prep) (nth 3 prep)
                         seq nv hd hd)))
      (dotimes (i (length out)) (aset ctx i (aref out i))))))

(defun nl-llm-bonsai--rope-partial (vec base hd rdims pos rbase)
  "Rotate the first RDIMS of the HD-long block at BASE, half-split, in place.
The remaining HD - RDIMS dimensions pass through, which is what a partial
rotary does and what rotating the whole head would silently not do."
  (unless (<= rdims hd)
    (error "nl-llm-bonsai: rotary covers %d of a %d-wide head" rdims hd))
  (let* ((half (/ rdims 2)) (orig (make-vector rdims 0.0)))
    (dotimes (i rdims) (aset orig i (aref vec (+ base i))))
    (dotimes (i half)
      (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 i) (float rdims)))))
             (c (cos theta)) (s (sin theta))
             (a (aref orig i)) (b (aref orig (+ i half))))
        (aset vec (+ base i) (- (* a c) (* b s)))
        (aset vec (+ base i half) (+ (* b c) (* a s)))))
    vec))

;;;###autoload
(defun nl-llm-bonsai-attn-block (sess layer x seq)
  "One gated full-attention block of LAYER over X (SEQ x dim)."
  (let* ((wts (plist-get sess :wts)) (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads)) (kvh (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim)) (ff (plist-get cfg :ff))
         ;; no partial rotary declared means the whole head
         (rdims (or (plist-get cfg :rope-dims) hd))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kvh hd))
         (ln1 (nl-llm-bonsai--gain sess :ln1g layer dim))
         (ln2 (nl-llm-bonsai--gain sess :ln2g layer dim))
         (qn (nl-llm-weights-row wts (nl-llm-weights-tensor wts :q-norm layer) 0))
         (kn (nl-llm-weights-row wts (nl-llm-weights-tensor wts :k-norm layer) 0))
         (lins (nl-llm-bonsai-linears sess layer))
         (wq (plist-get lins :wq)) (wk (plist-get lins :wk))
         (wv (plist-get lins :wv)) (wo (plist-get lins :wo))
         (wg (plist-get lins :wg)) (wu (plist-get lins :wu))
         (wd (plist-get lins :wd))
         (has-gate (nl-llm-bonsai--gated-q-p lins cfg))
         (q (make-vector (* seq qdim) 0.0)) (k (make-vector (* seq kvdim) 0.0))
         (v (make-vector (* seq kvdim) 0.0)) (gate (make-vector (* seq qdim) 0.0))
         (out (make-vector (* seq dim) 0.0)))
    (let ((ar (make-vector (* seq dim) 0.0))
          (qw (if has-gate (* 2 qdim) qdim)))
      (dotimes (tt seq)
        (let ((a (nl-llm-wf--rmsnorm x (* tt dim) dim ln1 eps)))
          (nl-llm-bonsai-rotate sess a dim)
          (dotimes (i dim) (aset ar (+ (* tt dim) i) (aref a i)))))
      (let ((yq (nl-llm-bonsai--apply-seq wq ar seq dim))
            (yk (nl-llm-bonsai--apply-seq wk ar seq dim))
            (yv (nl-llm-bonsai--apply-seq wv ar seq dim)))
        ;; the query projection is [query | gate], each HEADS * HD wide,
        ;; when the model has an output gate at all
        (dotimes (tt seq)
          (dotimes (i qdim)
            (aset q (+ (* tt qdim) i) (aref yq (+ (* tt qw) i)))
            (when has-gate
              (aset gate (+ (* tt qdim) i) (aref yq (+ (* tt qw) qdim i)))))
          (dotimes (i kvdim)
            (aset k (+ (* tt kvdim) i) (aref yk (+ (* tt kvdim) i)))
            (aset v (+ (* tt kvdim) i) (aref yv (+ (* tt kvdim) i)))))))
    ;; QK-norm per head, then the partial rotation
    (dotimes (tt seq)
      (dotimes (h heads)
        (let ((b (+ (* tt qdim) (* h hd))))
          (nl-llm-dn--rmsnorm-into q b hd qn eps)
          (nl-llm-bonsai--rope-partial q b hd rdims tt rbase)))
      (dotimes (h kvh)
        (let ((b (+ (* tt kvdim) (* h hd))))
          (nl-llm-dn--rmsnorm-into k b hd kn eps)
          (nl-llm-bonsai--rope-partial k b hd rdims tt rbase))))
    (let ((ctx (nl-llm-wf--attend q k v seq heads kvh hd)))
      ;; the gate, then the output projection and the residual
      (let ((gr (make-vector (* seq qdim) 0.0)))
        (dotimes (tt seq)
          (let ((g (make-vector qdim 0.0)))
            (dotimes (i qdim)
              (aset g i (if has-gate
                            (* (aref ctx (+ (* tt qdim) i))
                               (nl-llm-bonsai--gate (aref gate (+ (* tt qdim) i))))
                          (aref ctx (+ (* tt qdim) i)))))
            (nl-llm-bonsai-rotate sess g qdim)
            (dotimes (i qdim) (aset gr (+ (* tt qdim) i) (aref g i)))))
        (let ((o (nl-llm-bonsai--apply-seq wo gr seq qdim)))
          (dotimes (tt seq)
            (dotimes (i dim)
              (aset out (+ (* tt dim) i)
                    (+ (aref x (+ (* tt dim) i)) (aref o (+ (* tt dim) i))))))))
      (nl-llm-bonsai--ffn sess out seq dim ff ln2 eps wg wu wd)
      out)))



;;;###autoload
(defun nl-llm-bonsai-block (sess layer x seq)
  "Dispatch LAYER to the block type the model's interval says it is."
  (let ((iv (nl-llm-bonsai--interval (plist-get sess :cfg))))
    (if (= (mod layer iv) (1- iv))
        (nl-llm-bonsai-attn-block sess layer x seq)
      (nl-llm-bonsai-deltanet-block sess layer x seq))))

(defun nl-llm-bonsai-embed (sess token)
  "TOKEN's embedding, in the basis the rest of the model works in.

`token_embd.weight' is the one tensor the file lists under
`prism.hadamard.inverse_weight_names'; the other 401 are under
`weight_names'.  The asymmetry is what makes one runtime operation correct
everywhere: a consuming projection was folded as W' = W.A\=' so that W'.(A.x)
is W.x, while the embedding was folded as E' = E.A so that applying the same
A to a stored row returns the true one.  So the rotation the blocks apply to
every normalised activation is applied here too, once, and the residual stream
is then in the unrotated basis the RMSNorm gains expect.

Leaving it out costs nothing visible: the transform is orthogonal, so the
embedding has the right norm either way and the stack runs to completion and
produces a token.  It is simply the wrong token, from the first block on."
  (nl-llm-bonsai-rotate
   sess (nl-llm-weights-embed (plist-get sess :wts) token)
   (plist-get (plist-get sess :cfg) :dim)))

(defun nl-llm-bonsai-head (sess)
  "The output projection, loaded once and cached on SESS.
Cached for the same reason a block's linears are: the GPU handle table is keyed
by `eq', so a second `nl-llm-weights-linear' would silently miss."
  (or (plist-get sess :head-lin)
      (let ((lin (nl-llm-weights-linear
                  (plist-get sess :wts)
                  (if (plist-get (plist-get sess :cfg) :tied-head) :wte :head))))
        (plist-put sess :head-lin lin)
        lin)))

;;;###autoload
(defun nl-llm-bonsai-logits (sess x seq &optional pos)
  "Logits at POS (default the last position) of the SEQ-long stream X.
The final norm, then the rotation -- `output.weight' is one of the 401 tensors
the model lists as rotated, so the head expects a rotated activation exactly as
the blocks' projections do -- then the head itself."
  (let* ((cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (wts (plist-get sess :wts))
         (lnf (nl-llm-bonsai--gain sess :lnf nil dim))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (at (or pos (1- seq)))
         (h (nl-llm-wf--rmsnorm x (* at dim) dim lnf eps)))
    (nl-llm-bonsai-rotate sess h dim)
    (nl-llm-bonsai--apply (nl-llm-bonsai-head sess) h 0)))

;;;###autoload
(defun nl-llm-bonsai-argmax (v)
  "Index of the largest element of V, and its value, as (INDEX . VALUE)."
  (let ((bi 0) (bv (aref v 0)))
    (dotimes (i (length v))
      (when (> (aref v i) bv) (setq bv (aref v i) bi i)))
    (cons bi bv)))

(provide 'nl-llm-bonsai)
;;; nl-llm-bonsai.el ends here
