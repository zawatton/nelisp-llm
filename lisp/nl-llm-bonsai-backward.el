;;; nl-llm-bonsai-backward.el --- gradients through a hybrid block -*- lexical-binding: t -*-

;; The backward for the 3:1 stack in `nl-llm-bonsai.el'.  Three things make it
;; different from the Qwen3 backward this project already has:
;;
;;   * every projection reads a rotated activation, so each one contributes an
;;     inverse rotation on the way back.  The transform is orthogonal, which is
;;     the only reason this is cheap -- and also the reason no norm-based check
;;     can tell a wrong rotation from a right one, so the rotations are checked
;;     against finite differences like everything else;
;;   * `ssm_alpha' and `ssm_beta' read the UNROTATED activation.  The forward
;;     got that wrong once and ran clean while poisoning the recurrence, so the
;;     backward keeps the two paths separate rather than inferring one;
;;   * the recurrence itself is `nl-llm-deltanet.el', whose vjps are already
;;     checked against a reference and finite differences.  This file is the
;;     wiring around them: norms, rotations, the residual and the projections.
;;
;; The base weights are frozen.  Only LoRA adapters take gradients, so a role
;; with no adapter costs one W^T.g and nothing is kept for it.

;;; Code:

(require 'nl-llm-bonsai)
(require 'nl-llm-deltanet)
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-backward)
(require 'nl-llm-weights-lora)
(require 'cl-lib)

;;; --- the context a block's forward and backward share ---------------------

(defun nl-llm-bonsai-bw-loras (&rest specs)
  "Build an adapter table from SPECS, each (LAYER ROLE ADAPTER).

Keyed by layer AND role, never by role alone.  One plist shared across blocks
gives every layer the same adapter, so a single optimiser step moves it once
per block it appears in -- at 28 blocks that is a 28x larger step than the
learning rate says, which is how a run collapses while every number in the log
still looks like progress."
  (let ((tbl (make-hash-table :test 'equal)))
    (dolist (s specs)
      (puthash (cons (nth 0 s) (nth 1 s)) (nth 2 s) tbl))
    tbl))

(defun nl-llm-bonsai-bw-make (sess &optional loras apply-fn wt-fn)
  "A backward context over SESS.
LORAS is a `nl-llm-bonsai-bw-loras' table, or nil for a frozen model.
APPLY-FN and WT-FN stand in for the CPU when a GPU path is in scope; they are
called with (LIN X BASE) and (LIN G)."
  (list :sess sess :cfg (plist-get sess :cfg) :loras loras
        :apply-fn apply-fn :wt-fn wt-fn
        :grads (make-hash-table :test 'equal)))

(defun nl-llm-bonsai-bw--lora (bc role)
  (let ((tbl (plist-get bc :loras)))
    (and tbl (gethash (cons (plist-get bc :layer) role) tbl))))

(defun nl-llm-bonsai-bw-grads (bc)
  "The accumulated adapter gradients, as an alist of (LAYER . ROLE) -> plist."
  (let (out)
    (maphash (lambda (key g) (push (cons key g) out)) (plist-get bc :grads))
    out))

(defun nl-llm-bonsai-bw-grad (bc layer role)
  "The accumulated (:da V :db V) for LAYER's ROLE, or nil."
  (gethash (cons layer role) (plist-get bc :grads)))

(defun nl-llm-bonsai-bw--rotate (sess v n back)
  "Rotate V (N long) in place, or pull a gradient BACK through the rotation.
Orthogonal, so the pullback is the inverse and costs the same as the forward."
  (let ((s (plist-get (plist-get sess :signs) n))
        (nl-llm-had-block (or (plist-get sess :block) nl-llm-had-block))
        (inv (plist-get sess :invert)))
    (unless s (error "nl-llm-bonsai-bw: no signs for width %d" n))
    (nl-llm-had-rotate v n s (if back (not inv) inv))))

(defun nl-llm-bonsai-bw--apply (bc lin x base)
  (if (plist-get bc :apply-fn)
      (funcall (plist-get bc :apply-fn) lin x base)
    (nl-llm-weights-apply lin x base)))

(defun nl-llm-bonsai-bw--transpose (bc lin g)
  (if (plist-get bc :wt-fn)
      (funcall (plist-get bc :wt-fn) lin g)
    (nl-llm-weights-apply-t lin g)))

(defun nl-llm-bonsai-bw--fwd (bc role lin x &optional base)
  "Apply LIN at ROLE to X at BASE; return (Y . SAVED).
SAVED is nil unless the role has an adapter, in which case it carries the
rank-long A.x and the input slice the backward needs."
  (let ((lora (nl-llm-bonsai-bw--lora bc role)))
    (if (null lora)
        (cons (nl-llm-bonsai-bw--apply bc lin x (or base 0)) nil)
      (let ((r (nl-llm-wlora-forward lin lora x (or base 0)
                                     (lambda (l xx bb)
                                       (nl-llm-bonsai-bw--apply bc l xx bb)))))
        (cons (nth 0 r) (list :u (nth 1 r) :xs (nth 2 r)))))))

(defun nl-llm-bonsai-bw--bwd (bc role lin saved g)
  "Pull G back through LIN at ROLE; return the input gradient.
An adapter's own gradients are accumulated on BC rather than returned, so a
caller that only wants to propagate does not have to thread them."
  (let ((lora (nl-llm-bonsai-bw--lora bc role)))
    (if (null lora)
        (nl-llm-bonsai-bw--transpose bc lin g)
      (let ((r (nl-llm-wlora-backward lin lora (plist-get saved :xs)
                                      (plist-get saved :u) g
                                      (lambda (l gg)
                                        (nl-llm-bonsai-bw--transpose bc l gg)))))
        (let* ((tbl (plist-get bc :grads))
               (key (cons (plist-get bc :layer) role))
               (old (gethash key tbl)))
          (if (null old)
              (puthash key (list :da (plist-get r :da) :db (plist-get r :db)) tbl)
            (let ((da (plist-get old :da)) (db (plist-get old :db))
                  (nda (plist-get r :da)) (ndb (plist-get r :db)))
              (dotimes (i (length da)) (aset da i (+ (aref da i) (aref nda i))))
              (dotimes (i (length db)) (aset db i (+ (aref db i) (aref ndb i)))))))
        (plist-get r :dx)))))

(defun nl-llm-bonsai-bw--add (dst src n &optional dbase sbase)
  (let ((db (or dbase 0)) (sb (or sbase 0)))
    (dotimes (i n) (aset dst (+ db i) (+ (aref dst (+ db i)) (aref src (+ sb i))))))
  dst)

;;; --- the feed-forward half, identical in both block types -----------------

(defun nl-llm-bonsai-bw--ffn-forward (bc mid seq)
  "Run the feed-forward half over MID in place; return its tape."
  (let* ((sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (dim (plist-get cfg :dim)) (ff (plist-get cfg :ff))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (wts (plist-get sess :wts))
         (layer (plist-get bc :layer))
         (ln2 (nl-llm-bonsai--gain sess :ln2g layer dim))
         (lins (nl-llm-bonsai-linears sess layer))
         (wg (plist-get lins :wg)) (wu (plist-get lins :wu))
         (wd (plist-get lins :wd))
         (mid0 (copy-sequence mid))
         (tape (make-vector seq nil)))
    (dotimes (tt seq)
      (let* ((b (nl-llm-wf--rmsnorm mid (* tt dim) dim ln2 eps))
             (brot (nl-llm-bonsai-bw--rotate sess (copy-sequence b) dim nil))
             (rg (nl-llm-bonsai-bw--fwd bc :wg wg brot))
             (ru (nl-llm-bonsai-bw--fwd bc :wu wu brot))
             (gg (car rg)) (uu (car ru))
             (hh (nl-llm-wf--silu-mul gg uu ff))
             (hrot (nl-llm-bonsai-bw--rotate sess (copy-sequence hh) ff nil))
             (rd (nl-llm-bonsai-bw--fwd bc :wd wd hrot)))
        (aset tape tt (list :gg gg :uu uu :sg (cdr rg) :su (cdr ru) :sd (cdr rd)))
        (dotimes (i dim)
          (aset mid (+ (* tt dim) i)
                (+ (aref mid (+ (* tt dim) i)) (aref (car rd) i))))))
    (list :mid0 mid0 :ln2 ln2 :per tape)))

(defun nl-llm-bonsai-bw--ffn-backward (bc ffn dout seq)
  "Pull DOUT back through the feed-forward half; return the gradient at MID0."
  (let* ((sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (dim (plist-get cfg :dim)) (ff (plist-get cfg :ff))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (layer (plist-get bc :layer))
         (lins (nl-llm-bonsai-linears sess layer))
         (wg (plist-get lins :wg)) (wu (plist-get lins :wu))
         (wd (plist-get lins :wd))
         (mid0 (plist-get ffn :mid0)) (ln2 (plist-get ffn :ln2))
         (per (plist-get ffn :per))
         (dmid (copy-sequence dout)))
    (dotimes (tt seq)
      (let* ((s (aref per tt))
             (dt (let ((v (make-vector dim 0.0)))
                   (dotimes (i dim) (aset v i (aref dout (+ (* tt dim) i)))) v))
             (dhrot (nl-llm-bonsai-bw--bwd bc :wd wd (plist-get s :sd) dt))
             (dhh (nl-llm-bonsai-bw--rotate sess dhrot ff t))
             (gu (nl-llm-wb-silu-mul-vjp (plist-get s :gg) (plist-get s :uu)
                                         dhh ff))
             (dbrot (nl-llm-bonsai-bw--bwd bc :wg wg (plist-get s :sg) (nth 0 gu)))
             (dbu (nl-llm-bonsai-bw--bwd bc :wu wu (plist-get s :su) (nth 1 gu))))
        (dotimes (i dim) (aset dbrot i (+ (aref dbrot i) (aref dbu i))))
        (let* ((db (nl-llm-bonsai-bw--rotate sess dbrot dim t))
               (dn (nl-llm-wb-rmsnorm-vjp mid0 (* tt dim) dim ln2 eps db)))
          (nl-llm-bonsai-bw--add dmid dn dim (* tt dim) 0))))
    dmid))

;;; --- the Gated DeltaNet block ---------------------------------------------

(defun nl-llm-bonsai-bw-deltanet-forward (bc layer x seq)
  "`nl-llm-bonsai-deltanet-block' with everything the backward needs kept.
Returns (OUT TAPE).  Checked against the plain forward for exact equality, so
the two cannot drift apart silently."
  (let* ((bc (plist-put bc :layer layer))
         (sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (wts (plist-get sess :wts))
         (dim (plist-get cfg :dim))
         (nk (plist-get cfg :ssm-groups)) (nv (plist-get cfg :ssm-heads))
         (hd (plist-get cfg :ssm-state)) (kern (plist-get cfg :ssm-conv-kernel))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd)) (grp (/ nv nk))
         (ln1 (nl-llm-bonsai--gain sess :ln1g layer dim))
         (snorm (nl-llm-bonsai--gain sess :ssm-norm layer hd))
         (alog (nl-llm-weights-row wts (nl-llm-weights-tensor wts :a-log layer) 0))
         (dtb (nl-llm-weights-row wts (nl-llm-weights-tensor wts :dt-bias layer) 0))
         (lins (nl-llm-bonsai-linears sess layer))
         (wqkv (plist-get lins :wqkv)) (wz (plist-get lins :wz))
         (wa (plist-get lins :walpha)) (wb (plist-get lins :wbeta))
         (wout (plist-get lins :wout))
         (convw (let* ((tn (nl-llm-weights-tensor wts :conv-w layer))
                       (o (make-vector (* cd kern) 0.0)))
                  (dotimes (c cd)
                    (let ((row (nl-llm-weights-row wts tn c)))
                      (dotimes (r kern) (aset o (+ (* c kern) r) (aref row r)))))
                  o))
         (convb (make-vector cd 0.0))
         (mixed (make-vector (* seq cd) 0.0))
         (z (make-vector (* seq vd) 0.0))
         (ba (make-vector (* seq 2 nv) 0.0))
         (proj (make-vector seq nil))
         (out (make-vector (* seq dim) 0.0)))
    (dotimes (tt seq)
      (let* ((plain (nl-llm-wf--rmsnorm x (* tt dim) dim ln1 eps))
             (a (nl-llm-bonsai-bw--rotate sess (copy-sequence plain) dim nil))
             (rq (nl-llm-bonsai-bw--fwd bc :wqkv wqkv a))
             (rz (nl-llm-bonsai-bw--fwd bc :wz wz a))
             (ra (nl-llm-bonsai-bw--fwd bc :walpha wa plain))
             (rb (nl-llm-bonsai-bw--fwd bc :wbeta wb plain)))
        (aset proj tt (list :sq (cdr rq) :sz (cdr rz) :sa (cdr ra) :sb (cdr rb)))
        (dotimes (i cd) (aset mixed (+ (* tt cd) i) (aref (car rq) i)))
        (dotimes (i vd) (aset z (+ (* tt vd) i) (aref (car rz) i)))
        (dotimes (i nv)
          (aset ba (+ (* tt 2 nv) i) (aref (car ra) i))
          (aset ba (+ (* tt 2 nv) nv i) (aref (car rb) i)))))
    (let* ((cv (nl-llm-dn-conv mixed convw convb seq cd kern))
           (conv-out (nth 0 cv)) (pre (nth 1 cv))
           (ctx (make-vector (* seq vd) 0.0))
           (heads (make-vector nv nil)))
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
          (let* ((fw (nl-llm-dn-forward qh khv vh ah bh (aref alog h)
                                        (aref dtb h) seq hd hd))
                 (oh (nth 0 fw)))
            (aset heads h (list :q qh :k khv :v vh :a ah :b bh :tape (nth 1 fw)))
            (dotimes (tt seq)
              (dotimes (i hd)
                (aset ctx (+ (* tt vd) (* h hd) i) (aref oh (+ (* tt hd) i))))))))
      (let ((gated (make-vector seq nil)))
        (dotimes (tt seq)
          (let ((g (make-vector vd 0.0)) (nrms (make-vector nv nil)))
            (dotimes (h nv)
              (let ((xs (make-vector hd 0.0)) (gs (make-vector hd 0.0)))
                (dotimes (i hd)
                  (aset xs i (aref ctx (+ (* tt vd) (* h hd) i)))
                  (aset gs i (aref z (+ (* tt vd) (* h hd) i))))
                (let ((r (nl-llm-dn-norm-gated xs gs snorm hd eps)))
                  (aset nrms h (nth 1 r))
                  (dotimes (i hd) (aset g (+ (* h hd) i) (aref (nth 0 r) i))))))
            (let* ((grot (nl-llm-bonsai-bw--rotate sess (copy-sequence g) vd nil))
                   (ro (nl-llm-bonsai-bw--fwd bc :wout wout grot)))
              (aset gated tt (list :nrms nrms :so (cdr ro)))
              (dotimes (i dim)
                (aset out (+ (* tt dim) i)
                      (+ (aref x (+ (* tt dim) i)) (aref (car ro) i)))))))
        (let ((ffn (nl-llm-bonsai-bw--ffn-forward bc out seq)))
          (list out
                (list :kind :deltanet :layer layer :x x
                      :ln1 ln1 :snorm snorm :alog alog :dtb dtb
                      :convw convw :mixed mixed :pre pre :conv-out conv-out
                      :z z :ba ba :ctx ctx :proj proj :gated gated :heads heads
                      :ffn ffn)))))))

(defun nl-llm-bonsai-bw-deltanet-backward (bc tape dout seq)
  "Gradient of `nl-llm-bonsai-bw-deltanet-forward' at its input."
  (let* ((bc (plist-put bc :layer (plist-get tape :layer)))
         (sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (dim (plist-get cfg :dim))
         (nk (plist-get cfg :ssm-groups)) (nv (plist-get cfg :ssm-heads))
         (hd (plist-get cfg :ssm-state)) (kern (plist-get cfg :ssm-conv-kernel))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (kd (* nk hd)) (vd (* nv hd)) (cd (+ kd kd vd)) (grp (/ nv nk))
         (layer (plist-get tape :layer))
         (lins (nl-llm-bonsai-linears sess layer))
         (wqkv (plist-get lins :wqkv)) (wz (plist-get lins :wz))
         (wa (plist-get lins :walpha)) (wb (plist-get lins :wbeta))
         (wout (plist-get lins :wout))
         (x (plist-get tape :x)) (ln1 (plist-get tape :ln1))
         (snorm (plist-get tape :snorm))
         (alog (plist-get tape :alog)) (dtb (plist-get tape :dtb))
         (ctx (plist-get tape :ctx)) (z (plist-get tape :z))
         (dmid (nl-llm-bonsai-bw--ffn-backward bc (plist-get tape :ffn) dout seq))
         (dx (copy-sequence dmid))       ; the residual out = x + o
         (dctx (make-vector (* seq vd) 0.0))
         (dz (make-vector (* seq vd) 0.0))
         (dconv (make-vector (* seq cd) 0.0))
         (dba (make-vector (* seq 2 nv) 0.0)))
    ;; out_proj, the gated norm, and back to ctx and z
    (dotimes (tt seq)
      (let* ((s (aref (plist-get tape :gated) tt))
             (dt (let ((v (make-vector dim 0.0)))
                   (dotimes (i dim) (aset v i (aref dmid (+ (* tt dim) i)))) v))
             (dgrot (nl-llm-bonsai-bw--bwd bc :wout wout (plist-get s :so) dt))
             (dg (nl-llm-bonsai-bw--rotate sess dgrot vd t)))
        (dotimes (h nv)
          (let ((xs (make-vector hd 0.0)) (gs (make-vector hd 0.0))
                (dh (make-vector hd 0.0)))
            (dotimes (i hd)
              (aset xs i (aref ctx (+ (* tt vd) (* h hd) i)))
              (aset gs i (aref z (+ (* tt vd) (* h hd) i)))
              (aset dh i (aref dg (+ (* h hd) i))))
            (let ((r (nl-llm-dn-norm-gated-vjp xs gs snorm
                                               (aref (plist-get s :nrms) h)
                                               dh hd eps)))
              (dotimes (i hd)
                (aset dctx (+ (* tt vd) (* h hd) i)
                      (+ (aref dctx (+ (* tt vd) (* h hd) i))
                         (aref (plist-get r :dx) i)))
                (aset dz (+ (* tt vd) (* h hd) i)
                      (+ (aref dz (+ (* tt vd) (* h hd) i))
                         (aref (plist-get r :dgate) i)))))))))
    ;; the recurrence, head by head
    (dotimes (h nv)
      (let* ((hs (aref (plist-get tape :heads) h))
             (kh (/ h grp))
             (doh (make-vector (* seq hd) 0.0)))
        (dotimes (tt seq)
          (dotimes (i hd)
            (aset doh (+ (* tt hd) i) (aref dctx (+ (* tt vd) (* h hd) i)))))
        (let ((r (nl-llm-dn-backward (plist-get hs :q) (plist-get hs :k)
                                     (plist-get hs :v) (plist-get hs :a)
                                     (plist-get hs :b) (aref alog h) (aref dtb h)
                                     seq hd hd (plist-get hs :tape) doh)))
          (dotimes (tt seq)
            (dotimes (i hd)
              (aset dconv (+ (* tt cd) (* kh hd) i)
                    (+ (aref dconv (+ (* tt cd) (* kh hd) i))
                       (aref (plist-get r :dq) (+ (* tt hd) i))))
              (aset dconv (+ (* tt cd) kd (* kh hd) i)
                    (+ (aref dconv (+ (* tt cd) kd (* kh hd) i))
                       (aref (plist-get r :dk) (+ (* tt hd) i))))
              (aset dconv (+ (* tt cd) kd kd (* h hd) i)
                    (+ (aref dconv (+ (* tt cd) kd kd (* h hd) i))
                       (aref (plist-get r :dv) (+ (* tt hd) i)))))
            (aset dba (+ (* tt 2 nv) h)
                  (+ (aref dba (+ (* tt 2 nv) h)) (aref (plist-get r :da) tt)))
            (aset dba (+ (* tt 2 nv) nv h)
                  (+ (aref dba (+ (* tt 2 nv) nv h)) (aref (plist-get r :db) tt)))))))
    ;; the convolution, then the four projections
    (let* ((cvj (nl-llm-dn-conv-vjp (plist-get tape :mixed) (plist-get tape :convw)
                                    (plist-get tape :pre) dconv seq cd kern))
           (dmixed (plist-get cvj :dx)))
      (dotimes (tt seq)
        (let* ((s (aref (plist-get tape :proj) tt))
               (dq (let ((v (make-vector cd 0.0)))
                     (dotimes (i cd) (aset v i (aref dmixed (+ (* tt cd) i)))) v))
               (dzz (let ((v (make-vector vd 0.0)))
                      (dotimes (i vd) (aset v i (aref dz (+ (* tt vd) i)))) v))
               (dya (let ((v (make-vector nv 0.0)))
                      (dotimes (i nv) (aset v i (aref dba (+ (* tt 2 nv) i)))) v))
               (dyb (let ((v (make-vector nv 0.0)))
                      (dotimes (i nv)
                        (aset v i (aref dba (+ (* tt 2 nv) nv i)))) v))
               (darot (nl-llm-bonsai-bw--bwd bc :wqkv wqkv (plist-get s :sq) dq))
               (dzrot (nl-llm-bonsai-bw--bwd bc :wz wz (plist-get s :sz) dzz)))
          (dotimes (i dim) (aset darot i (+ (aref darot i) (aref dzrot i))))
          ;; the rotated half comes back through the rotation; alpha and beta
          ;; never saw it, so their gradients join afterwards
          (let ((dplain (nl-llm-bonsai-bw--rotate sess darot dim t))
                (dpa (nl-llm-bonsai-bw--bwd bc :walpha wa (plist-get s :sa) dya))
                (dpb (nl-llm-bonsai-bw--bwd bc :wbeta wb (plist-get s :sb) dyb)))
            (dotimes (i dim)
              (aset dplain i (+ (aref dplain i) (aref dpa i) (aref dpb i))))
            (let ((dn (nl-llm-wb-rmsnorm-vjp x (* tt dim) dim ln1 eps dplain)))
              (nl-llm-bonsai-bw--add dx dn dim (* tt dim) 0))))))
    dx))


;;; --- the gated full-attention block, every fourth one ---------------------

(defun nl-llm-bonsai-bw--rope-vjp (dy base hd rdims pos rbase)
  "Pull a gradient back through the partial rotary at BASE, in place.
The rotary is a plane rotation by POS * freq, so its transpose is the rotation
by -POS -- and the dimensions past RDIMS, which the forward left alone, must
stay untouched here too."
  (nl-llm-bonsai--rope-partial dy base hd rdims (- pos) rbase))

(defun nl-llm-bonsai-bw-attn-forward (bc layer x seq)
  "`nl-llm-bonsai-attn-block' with what the backward needs kept.
Returns (OUT TAPE)."
  (let* ((bc (plist-put bc :layer layer))
         (sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (wts (plist-get sess :wts))
         (dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads)) (kvh (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         ;; no partial rotary declared means the whole head
         (rdims (or (plist-get cfg :rope-dims) hd))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kvh hd))
         (ln1 (nl-llm-bonsai--gain sess :ln1g layer dim))
         (qn (nl-llm-weights-row wts (nl-llm-weights-tensor wts :q-norm layer) 0))
         (kn (nl-llm-weights-row wts (nl-llm-weights-tensor wts :k-norm layer) 0))
         (lins (nl-llm-bonsai-linears sess layer))
         (wq (plist-get lins :wq)) (wk (plist-get lins :wk))
         (wv (plist-get lins :wv)) (wo (plist-get lins :wo))
         (gated (nl-llm-bonsai--gated-q-p lins cfg))
         (q (make-vector (* seq qdim) 0.0)) (k (make-vector (* seq kvdim) 0.0))
         (v (make-vector (* seq kvdim) 0.0)) (gate (make-vector (* seq qdim) 0.0))
         (proj (make-vector seq nil))
         (out (make-vector (* seq dim) 0.0)))
    (dotimes (tt seq)
      (let* ((plain (nl-llm-wf--rmsnorm x (* tt dim) dim ln1 eps))
             (a (nl-llm-bonsai-bw--rotate sess (copy-sequence plain) dim nil))
             (rq (nl-llm-bonsai-bw--fwd bc :wq wq a))
             (rk (nl-llm-bonsai-bw--fwd bc :wk wk a))
             (rv (nl-llm-bonsai-bw--fwd bc :wv wv a)))
        (aset proj tt (list :sq (cdr rq) :sk (cdr rk) :sv (cdr rv)))
        (dotimes (i qdim)
          (aset q (+ (* tt qdim) i) (aref (car rq) i))
          (when gated
            (aset gate (+ (* tt qdim) i) (aref (car rq) (+ qdim i)))))
        (dotimes (i kvdim)
          (aset k (+ (* tt kvdim) i) (aref (car rk) i))
          (aset v (+ (* tt kvdim) i) (aref (car rv) i)))))
    (let ((qpre (copy-sequence q)) (kpre (copy-sequence k)))
      (dotimes (tt seq)
        (dotimes (h heads)
          (let ((b (+ (* tt qdim) (* h hd))))
            (nl-llm-dn--rmsnorm-into q b hd qn eps)
            (nl-llm-bonsai--rope-partial q b hd rdims tt rbase)))
        (dotimes (h kvh)
          (let ((b (+ (* tt kvdim) (* h hd))))
            (nl-llm-dn--rmsnorm-into k b hd kn eps)
            (nl-llm-bonsai--rope-partial k b hd rdims tt rbase))))
      (let ((ctx (nl-llm-wf--attend q k v seq heads kvh hd))
            (gated (make-vector seq nil)))
        (dotimes (tt seq)
          (let ((g (make-vector qdim 0.0)))
            (dotimes (i qdim)
              (aset g i (if gated
                            (* (aref ctx (+ (* tt qdim) i))
                               (nl-llm-bonsai--gate (aref gate (+ (* tt qdim) i))))
                          (aref ctx (+ (* tt qdim) i)))))
            (let* ((grot (nl-llm-bonsai-bw--rotate sess (copy-sequence g) qdim nil))
                   (ro (nl-llm-bonsai-bw--fwd bc :wo wo grot)))
              (aset gated tt (list :so (cdr ro)))
              (dotimes (i dim)
                (aset out (+ (* tt dim) i)
                      (+ (aref x (+ (* tt dim) i)) (aref (car ro) i)))))))
        (let ((ffn (nl-llm-bonsai-bw--ffn-forward bc out seq)))
          (list out
                (list :kind :attn :layer layer :x x :ln1 ln1 :qn qn :kn kn :gated gated
                      :q q :k k :v v :qpre qpre :kpre kpre :gate gate :ctx ctx
                      :proj proj :gated gated :ffn ffn)))))))

(defun nl-llm-bonsai-bw-attn-backward (bc tape dout seq)
  "Gradient of `nl-llm-bonsai-bw-attn-forward' at its input."
  (let* ((bc (plist-put bc :layer (plist-get tape :layer)))
         (sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads)) (kvh (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         ;; no partial rotary declared means the whole head
         (rdims (or (plist-get cfg :rope-dims) hd))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kvh hd))
         (layer (plist-get tape :layer))
         (lins (nl-llm-bonsai-linears sess layer))
         (wq (plist-get lins :wq)) (wk (plist-get lins :wk))
         (wv (plist-get lins :wv)) (wo (plist-get lins :wo))
         (x (plist-get tape :x)) (ln1 (plist-get tape :ln1))
         (ctx (plist-get tape :ctx)) (gate (plist-get tape :gate))
         (gated (plist-get tape :gated))
         (dmid (nl-llm-bonsai-bw--ffn-backward bc (plist-get tape :ffn) dout seq))
         (dx (copy-sequence dmid))
         (dctx (make-vector (* seq qdim) 0.0))
         (dgate (make-vector (* seq qdim) 0.0)))
    ;; out_proj, then the output gate
    (dotimes (tt seq)
      (let* ((s (aref (plist-get tape :gated) tt))
             (dt (let ((vv (make-vector dim 0.0)))
                   (dotimes (i dim) (aset vv i (aref dmid (+ (* tt dim) i)))) vv))
             (dgrot (nl-llm-bonsai-bw--bwd bc :wo wo (plist-get s :so) dt))
             (dg (nl-llm-bonsai-bw--rotate sess dgrot qdim t)))
        (dotimes (i qdim)
          (if (not gated)
              (aset dctx (+ (* tt qdim) i) (aref dg i))
            (let* ((gi (aref gate (+ (* tt qdim) i)))
                   (ci (aref ctx (+ (* tt qdim) i)))
                   (sg (nl-llm-bonsai--gate gi)))
              (aset dctx (+ (* tt qdim) i) (* (aref dg i) sg))
              (aset dgate (+ (* tt qdim) i)
                    (* (aref dg i) ci sg (- 1.0 sg))))))))
    ;; attention, then the rotary and the QK-norm, in the order they ran
    (let* ((qkv (nl-llm-wb-attend-vjp (plist-get tape :q) (plist-get tape :k)
                                      (plist-get tape :v) seq heads kvh hd dctx))
           (dq (nth 0 qkv)) (dk (nth 1 qkv)) (dv (nth 2 qkv)))
      (dotimes (tt seq)
        (dotimes (h heads)
          (let ((b (+ (* tt qdim) (* h hd))))
            (nl-llm-bonsai-bw--rope-vjp dq b hd rdims tt rbase)))
        (nl-llm-wb-rmsnorm-heads-vjp (plist-get tape :qpre) (* tt qdim)
                                     heads hd (plist-get tape :qn) dq eps)
        (dotimes (h kvh)
          (let ((b (+ (* tt kvdim) (* h hd))))
            (nl-llm-bonsai-bw--rope-vjp dk b hd rdims tt rbase)))
        (nl-llm-wb-rmsnorm-heads-vjp (plist-get tape :kpre) (* tt kvdim)
                                     kvh hd (plist-get tape :kn) dk eps))
      ;; the three projections; the query's gradient carries the gate's half
      (dotimes (tt seq)
        (let* ((s (aref (plist-get tape :proj) tt))
               (dyq (make-vector (if gated (* 2 qdim) qdim) 0.0))
               (dyk (make-vector kvdim 0.0))
               (dyv (make-vector kvdim 0.0)))
          (dotimes (i qdim)
            (aset dyq i (aref dq (+ (* tt qdim) i)))
            (when gated
              (aset dyq (+ qdim i) (aref dgate (+ (* tt qdim) i)))))
          (dotimes (i kvdim)
            (aset dyk i (aref dk (+ (* tt kvdim) i)))
            (aset dyv i (aref dv (+ (* tt kvdim) i))))
          (let ((da (nl-llm-bonsai-bw--bwd bc :wq wq (plist-get s :sq) dyq))
                (dbk (nl-llm-bonsai-bw--bwd bc :wk wk (plist-get s :sk) dyk))
                (dbv (nl-llm-bonsai-bw--bwd bc :wv wv (plist-get s :sv) dyv)))
            (dotimes (i dim)
              (aset da i (+ (aref da i) (aref dbk i) (aref dbv i))))
            (let* ((dplain (nl-llm-bonsai-bw--rotate sess da dim t))
                   (dn (nl-llm-wb-rmsnorm-vjp x (* tt dim) dim ln1 eps dplain)))
              (nl-llm-bonsai-bw--add dx dn dim (* tt dim) 0))))))
    dx))

;;; --- dispatch --------------------------------------------------------------

;;;###autoload
(defun nl-llm-bonsai-bw-block-forward (bc layer x seq)
  "Run LAYER's block with a tape; return (OUT TAPE)."
  (let ((iv (nl-llm-bonsai--interval (plist-get bc :cfg))))
    (if (= (mod layer iv) (1- iv))
        (nl-llm-bonsai-bw-attn-forward bc layer x seq)
      (nl-llm-bonsai-bw-deltanet-forward bc layer x seq))))

;;;###autoload
(defun nl-llm-bonsai-bw-block-backward (bc tape dout seq)
  "Pull DOUT back through the block TAPE came from."
  (if (eq (plist-get tape :kind) :attn)
      (nl-llm-bonsai-bw-attn-backward bc tape dout seq)
    (nl-llm-bonsai-bw-deltanet-backward bc tape dout seq)))

;;;###autoload
(defun nl-llm-bonsai-bw-logits-backward (bc x seq dlogits &optional pos)
  "Pull DLOGITS back through the final norm and the head; return dX.
The head is rotated like every other rotated projection, so the pullback goes
through the inverse rotation before the norm."
  (let* ((sess (plist-get bc :sess)) (cfg (plist-get bc :cfg))
         (dim (plist-get cfg :dim)) (wts (plist-get sess :wts))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (lnf (nl-llm-bonsai--gain sess :lnf nil dim))
         (at (or pos (1- seq)))
         (head (nl-llm-bonsai-head sess))
         (dx (make-vector (* seq dim) 0.0))
         (dh (nl-llm-bonsai-bw--transpose bc head dlogits)))
    (nl-llm-bonsai-bw--rotate sess dh dim t)
    (let ((dn (nl-llm-wb-rmsnorm-vjp x (* at dim) dim lnf eps dh)))
      (dotimes (i dim) (aset dx (+ (* at dim) i) (aref dn i))))
    dx))

(provide 'nl-llm-bonsai-backward)
;;; nl-llm-bonsai-backward.el ends here
