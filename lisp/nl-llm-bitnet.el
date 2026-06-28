;;; nl-llm-bitnet.el --- BitNet b1.58 packed ternary-weight inference (Phase B)  -*- lexical-binding: t; -*-

;; Phase B of BitNet b1.58 (docs/design/01): after Phase-A QAT trains a
;; full-precision latent weight whose ternary quantization is the deployed
;; weight, Phase B *stores* that ternary weight compactly and runs the forward
;; from the packed form.  The nelisp-gpu substrate only has f32 buffers, so
;; instead of a new int8 dtype we pack PK ternary codes (tern+1 in {0,1,2}) as
;; base-4 digits into each f32 -- exact for PK<=10 (4^10 < 2^23).  At PK=8 the
;; weight buffer is 8x smaller (16 of 32 bits used), an 8x VRAM/bandwidth cut;
;; the `bitlinear-packed' kernel reads each packed float once and peels the codes
;; with integer %4 // /4.  (A further int8/DP4A path on Pascal is future work.)

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)         ; puts the nelisp-gpu sibling dir on the load-path
(require 'nelisp-gpu-server)  ; nelisp-gpu-server-run
(require 'nl-llm-arch)    ; nl-llm-rmsnorm, nl-llm-silu
(require 'nl-llm-attn)    ; nl-llm--rope-heads
(require 'nl-llm-decode)  ; nl-llm-dcache accessors

(defconst nl-llm-bitnet-pk 8
  "Ternary codes packed per f32 (base-4).  Must match the kernel push value.")

(defun nl-llm-bitnet-pack (w &optional pk)
  "Pack ternary-quantized weight W (out x in) into base-4 codes, PK per f32.
Returns (PACKED BETA FCOUNT): PACKED is (out*FCOUNT) floats, FCOUNT =
ceil(in/PK), BETA = mean|W|.  Each weight -> ternary in {-1,0,1} -> code
\(tern+1) in {0,1,2}; row o's codes are little-endian base-4 across its floats."
  (let* ((pk (or pk nl-llm-bitnet-pk)) (sh (photon-tensor-shape w))
         (out (car sh)) (in (nth 1 sh)) (wd (photon-tensor-data w)) (n (* out in))
         (fcount (/ (+ in pk -1) pk)) (packed (make-vector (* out fcount) 0.0)) (acc 0.0))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let ((beta (/ acc (float n))))
      (dotimes (o out)
        (dotimes (f fcount)
          (let ((val 0.0) (mul 1.0))
            (dotimes (z pk)
              (let ((i (+ (* f pk) z)))
                (when (< i in)
                  (let* ((q (if (> beta 0.0) (/ (aref wd (+ (* o in) i)) beta) 0.0))
                         (tern (cond ((>= q 0.5) 1) ((<= q -0.5) -1) (t 0))))
                    (setq val (+ val (* mul (float (+ tern 1)))))))
                (setq mul (* mul 4.0))))
            (aset packed (+ (* o fcount) f) val))))
      (list packed beta fcount))))

;;;###autoload
(defun nl-llm-bitnet-linear (x w bias &optional pk)
  "X (seq x in) . Wq^T + BIAS on the GPU with PACKED ternary weight W (out x in),
Wq = beta*ternary(W).  Returns the flat (seq*out) result vector.  The GPU server
must be running (`nl-llm-gpu-enable')."
  (let* ((pk (or pk nl-llm-bitnet-pk)) (sh (photon-tensor-shape x)) (seq (car sh)) (in (nth 1 sh))
         (out (car (photon-tensor-shape w)))
         (pk3 (nl-llm-bitnet-pack w pk)) (packed (nth 0 pk3)) (beta (nth 1 pk3)) (fcount (nth 2 pk3)))
    (nth 4 (nelisp-gpu-server-run
            'bitlinear-packed
            (list (copy-sequence (photon-tensor-data x)) packed
                  (copy-sequence (photon-tensor-data bias)) (vector beta) (make-vector (* seq out) 0.0))
            (vector seq in out pk fcount) (/ (+ (* seq out) 63) 64)))))

;; --- whole-model packed forward (KV-cache decode with packed linears) -------
(defun nl-llm-bitnet-ternarize (w)
  "Return beta*ternary(W) as an f32 tensor (the dequantized ternary weight) --
the exact value the packed kernel reconstructs, for an f32 reference forward."
  (let* ((sh (photon-tensor-shape w)) (wd (photon-tensor-data w)) (n (length wd)) (acc 0.0))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let* ((beta (/ acc (float n))) (o (make-vector n 0.0)))
      (dotimes (i n) (let ((q (if (> beta 0.0) (/ (aref wd i) beta) 0.0)))
        (aset o i (* beta (cond ((>= q 0.5) 1.0) ((<= q -0.5) -1.0) (t 0.0))))))
      (photon-tensor sh o))))

(defun nl-llm-bitnet--map-block (blk fn)
  "Return a copy of block plist BLK with each linear weight replaced by (FN W)."
  (let ((o nil) (kv blk))
    (while kv
      (let ((key (car kv)) (val (cadr kv)))
        (push key o)
        (push (if (memq key '(:wq :wk :wv :wo :wg :wu :wd)) (funcall fn val) val) o))
      (setq kv (cddr kv)))
    (nreverse o)))

(defun nl-llm-bitnet-pack-block (blk)
  "Pack a block's seven linear weights; biases and norms pass through.  Each
weight key becomes (PACKED BETA FCOUNT IN OUT) for `nl-llm-bitnet--run1'."
  (nl-llm-bitnet--map-block blk
    (lambda (w) (let* ((sh (photon-tensor-shape w)) (out (car sh)) (in (nth 1 sh)) (p (nl-llm-bitnet-pack w)))
                  (list (nth 0 p) (nth 1 p) (nth 2 p) in out)))))

(defun nl-llm-bitnet-ternarize-block (blk)
  "Return a block with its linear weights replaced by beta*ternary(W) (f32)."
  (nl-llm-bitnet--map-block blk #'nl-llm-bitnet-ternarize))

(defun nl-llm-bitnet--run1 (xrow pspec bias)
  "Packed linear on a single row XROW (1 x in); PSPEC = (PACKED BETA FCOUNT IN OUT).
Returns the flat (out) result vector."
  (let ((packed (nth 0 pspec)) (beta (nth 1 pspec)) (fcount (nth 2 pspec))
        (in (nth 3 pspec)) (out (nth 4 pspec)) (pk nl-llm-bitnet-pk))
    (nth 4 (nelisp-gpu-server-run
            'bitlinear-packed
            (list (copy-sequence (photon-tensor-data xrow)) packed
                  (copy-sequence (photon-tensor-data bias)) (vector beta) (make-vector out 0.0))
            (vector 1 in out pk fcount) (/ (+ out 63) 64)))))

(defun nl-llm-bitnet--blk (xrow pblk cache linfn &optional rope-base)
  "One pre-norm block with every linear routed through LINFN (a fn of XROW, the
block's weight spec, and the bias row).  Shared by the packed (`nl-llm-bitnet--run1')
and DP4A (`nl-llm-bitnet--dp4a-run1') whole-model decoders."
  (let* ((dim (nl-llm-dcache-dim cache)) (heads (nl-llm-dcache-heads cache)) (kvh (nl-llm-dcache-kvh cache))
         (hd (/ dim heads)) (kvdim (nl-llm-dcache-kvdim cache)) (grp (/ heads kvh))
         (pos (nl-llm-dcache-len cache)) (base (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (a (nl-llm-rmsnorm xrow (plist-get pblk :ln1g)))
         (qr (funcall linfn a (plist-get pblk :wq) (plist-get pblk :bq)))
         (kr (funcall linfn a (plist-get pblk :wk) (plist-get pblk :bk)))
         (vr (funcall linfn a (plist-get pblk :wv) (plist-get pblk :bv)))
         (kc (nl-llm-dcache-k cache)) (vc (nl-llm-dcache-v cache)) (out (make-vector dim 0.0)))
    (nl-llm--rope-heads qr 0 heads hd pos base)
    (nl-llm--rope-heads kr 0 kvh hd pos base)
    (dotimes (t0 kvdim) (aset kc (+ (* pos kvdim) t0) (aref kr t0)) (aset vc (+ (* pos kvdim) t0) (aref vr t0)))
    (setf (nl-llm-dcache-len cache) (1+ pos))
    (dotimes (h heads)
      (let ((c0q (* h hd)) (c0k (* (/ h grp) hd)) (scores (make-vector (1+ pos) 0.0)) (mx -1.0e30))
        (dotimes (j (1+ pos))
          (let ((kb (+ (* j kvdim) c0k)) (acc 0.0) (t0 0))
            (while (< t0 hd) (setq acc (+ acc (* (aref qr (+ c0q t0)) (aref kc (+ kb t0))))) (setq t0 (1+ t0)))
            (let ((sc (* acc scale))) (aset scores j sc) (when (> sc mx) (setq mx sc)))))
        (let ((sm 0.0))
          (dotimes (j (1+ pos)) (let ((e (exp (- (aref scores j) mx)))) (aset scores j e) (setq sm (+ sm e))))
          (let ((t0 0)) (while (< t0 hd)
            (let ((acc 0.0) (j 0)) (while (<= j pos) (setq acc (+ acc (* (/ (aref scores j) sm) (aref vc (+ (* j kvdim) c0k t0))))) (setq j (1+ j)))
              (aset out (+ c0q t0) acc)) (setq t0 (1+ t0)))))))
    (let* ((attn (funcall linfn (photon-tensor (list 1 dim) out) (plist-get pblk :wo) (plist-get pblk :bo)))
           (x1 (photon-tensor-add xrow (photon-tensor (list 1 dim) attn)))
           (bnorm (nl-llm-rmsnorm x1 (plist-get pblk :ln2g)))
           (g (funcall linfn bnorm (plist-get pblk :wg) (plist-get pblk :bg)))
           (u (funcall linfn bnorm (plist-get pblk :wu) (plist-get pblk :bu)))
           (sd (photon-tensor-data (nl-llm-silu (photon-tensor (list 1 (length g)) g))))
           (hh (make-vector (length g) 0.0)))
      (dotimes (i (length g)) (aset hh i (* (aref sd i) (aref u i))))
      (photon-tensor-add x1 (photon-tensor (list 1 dim)
                                           (funcall linfn (photon-tensor (list 1 (length hh)) hh)
                                                    (plist-get pblk :wd) (plist-get pblk :bd)))))))

;;;###autoload
(defun nl-llm-bitnet-decode-block (xrow pblk cache &optional rope-base)
  "Pre-norm block with every linear via the base-4 packed ternary kernel.  PBLK is
a `nl-llm-bitnet-pack-block' result."
  (nl-llm-bitnet--blk xrow pblk cache #'nl-llm-bitnet--run1 rope-base))

;;;###autoload
(defun nl-llm-bitnet-decode-step (token pblocks caches wte lnfg bh dim &optional rope-base)
  "Decode one TOKEN through packed PBLOCKS with KV CACHES; tied head stays f32.
Returns the (vocab) logit vector (cf. `nl-llm-decode-step')."
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim) (let ((v (make-vector dim 0.0)))
                                          (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl pblocks) (cl caches))
    (while bl (setq x (nl-llm-bitnet-decode-block x (car bl) (car cl) rope-base)) (setq bl (cdr bl) cl (cdr cl)))
    (photon-tensor-data (photon-tensor-linear (nl-llm-rmsnorm x lnfg) wte bh))))

;; --- packing the (tied) embedding/head too: fully-ternary model -------------
(defun nl-llm-bitnet-pack-wte (wte)
  "Pack the tied embedding/head WTE (vocab x dim) base-4.  Returns a spec
\(PACKED BETA FCOUNT DIM VOCAB) usable both as the head's `nl-llm-bitnet--run1'
weight and, row by row, as the embedding via `nl-llm-bitnet-unpack-row'."
  (let* ((sh (photon-tensor-shape wte)) (vocab (car sh)) (dim (nth 1 sh)) (p (nl-llm-bitnet-pack wte)))
    (list (nth 0 p) (nth 1 p) (nth 2 p) dim vocab)))

(defun nl-llm-bitnet-unpack-row (packed beta fcount token dim &optional pk)
  "Unpack row TOKEN of a base-4 PACKED weight into its (dim) ternary*BETA values
\(the ternary embedding for TOKEN)."
  (let ((pk (or pk nl-llm-bitnet-pk)) (v (make-vector dim 0.0)) (base (* token fcount)))
    (dotimes (i dim)
      (let* ((f (/ i pk)) (z (% i pk)) (code (% (/ (round (aref packed (+ base f))) (expt 4 z)) 4)))
        (aset v i (* beta (float (- code 1))))))
    v))

;;;###autoload
(defun nl-llm-bitnet-decode-step-fullpacked (token pblocks caches wte-spec lnfg bh dim &optional rope-base)
  "Decode one TOKEN with EVERYTHING ternary: WTE-SPEC (`nl-llm-bitnet-pack-wte')
is the packed tied weight -- the embedding is its unpacked row and the head is a
packed linear over it -- so no f32 weight matrix remains (only biases / norms).
Returns the (vocab) logit vector."
  (let* ((packed (nth 0 wte-spec)) (beta (nth 1 wte-spec)) (fcount (nth 2 wte-spec))
         (x (photon-tensor (list 1 dim) (nl-llm-bitnet-unpack-row packed beta fcount token dim)))
         (bl pblocks) (cl caches))
    (while bl (setq x (nl-llm-bitnet-decode-block x (car bl) (car cl) rope-base)) (setq bl (cdr bl) cl (cdr cl)))
    (nl-llm-bitnet--run1 (nl-llm-rmsnorm x lnfg) wte-spec bh)))

(defun nl-llm-bitnet-block-bytes (blk &optional pk)
  "Return (F32-BYTES . PACKED-BYTES) for the seven linear weights of block BLK."
  (let ((pk (or pk nl-llm-bitnet-pk)) (f32 0) (pkb 0) (kv blk))
    (while kv
      (when (memq (car kv) '(:wq :wk :wv :wo :wg :wu :wd))
        (let* ((sh (photon-tensor-shape (cadr kv))) (out (car sh)) (in (nth 1 sh)))
          (setq f32 (+ f32 (* out in 4)))
          (setq pkb (+ pkb (* out (/ (+ in pk -1) pk) 4)))))
      (setq kv (cddr kv)))
    (cons f32 pkb)))

;; --- fused resident packed linear: fewer dispatches, weights uploaded once ---
;; Several projections that share an input dim (Q|K|V, gate|up) are packed into ONE
;; per-row-beta weight and uploaded resident, so they run in ONE GPU dispatch with
;; no per-token weight re-upload (the dispatch/upload cost the plain `--run1' path
;; pays every token).  Used by the fused integrated decode (nl-llm-integrated.el).

(defun nl-llm-bitnet-pack-fused-res (wlist biaslist)
  "Pack f32 weights WLIST (all with the same input dim) into one per-row-beta
packed weight, upload it + the per-row beta + the concatenated bias BIASLIST as
resident buffers, and return a resident fused-linear spec (a plist with :ph :betah
:biash :in :out :fcount :splits).  Free the handles with
`nl-llm-bitnet-free-fused-res'."
  (let* ((pk nl-llm-bitnet-pk) (in (nth 1 (photon-tensor-shape (car wlist))))
         (fcount (/ (+ in pk -1) pk)) (splits nil) (out 0) (packs nil))
    (dolist (w wlist)
      (let ((p (nl-llm-bitnet-pack w)) (oi (car (photon-tensor-shape w))))
        (push (cons p oi) packs) (push oi splits) (setq out (+ out oi))))
    (setq packs (nreverse packs) splits (nreverse splits))
    (let ((pv (make-vector (* out fcount) 0.0)) (bv (make-vector out 0.0)) (biasv (make-vector out 0.0))
          (poff 0) (off 0) (bl biaslist))
      (dolist (pe packs)
        (let* ((p (car pe)) (oi (cdr pe)) (pkw (nth 0 p)) (beta (nth 1 p)) (bias (photon-tensor-data (car bl))))
          (dotimes (k (* oi fcount)) (aset pv (+ poff k) (aref pkw k)))
          (dotimes (r oi) (aset bv (+ off r) beta) (aset biasv (+ off r) (aref bias r)))
          (setq poff (+ poff (* oi fcount)) off (+ off oi) bl (cdr bl))))
      (list :ph (nelisp-gpu-server-upload pv) :betah (nelisp-gpu-server-upload bv)
            :biash (nelisp-gpu-server-upload biasv) :in in :out out :fcount fcount :splits splits))))

(defun nl-llm-bitnet-free-fused-res (spec)
  "Free the resident handles of a fused-linear SPEC."
  (ignore-errors (nelisp-gpu-server-free (plist-get spec :ph)))
  (ignore-errors (nelisp-gpu-server-free (plist-get spec :betah)))
  (ignore-errors (nelisp-gpu-server-free (plist-get spec :biash))))

(defun nl-llm-bitnet--run-fused-res (xrow spec)
  "Run the resident fused packed linear SPEC on row XROW (1 x in); return the flat
\(out) result (caller splits it with SPEC's :splits).  ONE dispatch, no re-upload."
  (let ((in (plist-get spec :in)) (out (plist-get spec :out)) (fcount (plist-get spec :fcount)) (pk nl-llm-bitnet-pk))
    (car (nelisp-gpu-server-run2
          'bitlinear-packed-v
          (list (cons 'in (copy-sequence (photon-tensor-data xrow)))
                (list 'res (plist-get spec :ph) (* out fcount))
                (list 'res (plist-get spec :biash) out)
                (list 'res (plist-get spec :betah) out)
                (cons 'out out))
          (vector 1 in out pk fcount) (/ (+ out 63) 64)))))

(defun nl-llm-bitnet--run-fused-res-io (in-handle out-handle spec)
  "Run fused packed linear SPEC reading input from resident IN-HANDLE and writing
the result to resident OUT-HANDLE -- nothing crosses the PCIe bus (ONE dispatch)."
  (let ((in (plist-get spec :in)) (out (plist-get spec :out)) (fcount (plist-get spec :fcount)) (pk nl-llm-bitnet-pk))
    (nelisp-gpu-server-run2
     'bitlinear-packed-v
     (list (list 'res in-handle in)
           (list 'res (plist-get spec :ph) (* out fcount))
           (list 'res (plist-get spec :biash) out)
           (list 'res (plist-get spec :betah) out)
           (list 'res out-handle out))
     (vector 1 in out pk fcount) (/ (+ out 63) 64))))

(defun nl-llm-bitnet--run-fused-res-rin (in-handle spec)
  "Run fused packed linear SPEC reading input from resident IN-HANDLE, returning
the result as a CPU vector (no input upload)."
  (let ((in (plist-get spec :in)) (out (plist-get spec :out)) (fcount (plist-get spec :fcount)) (pk nl-llm-bitnet-pk))
    (car (nelisp-gpu-server-run2
          'bitlinear-packed-v
          (list (list 'res in-handle in)
                (list 'res (plist-get spec :ph) (* out fcount))
                (list 'res (plist-get spec :biash) out)
                (list 'res (plist-get spec :betah) out)
                (cons 'out out))
          (vector 1 in out pk fcount) (/ (+ out 63) 64)))))

;; --- DP4A int8 path (compute win via hardware OpSDot) -----------------------
;; Four int8 lanes/group are carried in two f32 halves (each a 0..65535 value,
;; exact in f32): lo = b0 + 256*b1, hi = b2 + 256*b3, unsigned bytes b = int8&255.
(defun nl-llm-bitnet-pack-i8-act (x)
  "Per-row int8-quantize X (seq x in) and pack 4 lanes/group into two f32 halves.
Returns (ALO AHI GAMMA NG); ALO/AHI are (seq x NG), GAMMA (seq), NG = ceil(in/4)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (in (nth 1 sh)) (xd (photon-tensor-data x))
         (ng (/ (+ in 3) 4)) (alo (make-vector (* seq ng) 0.0)) (ahi (make-vector (* seq ng) 0.0))
         (gamma (make-vector seq 0.0)))
    (dotimes (s seq)
      (let ((amax 0.0))
        (dotimes (i in) (let ((av (abs (aref xd (+ (* s in) i))))) (when (> av amax) (setq amax av))))
        (let ((g (/ amax 127.0)))
          (aset gamma s g)
          (dotimes (grp ng)
            (let ((bytes (make-vector 4 0)))
              (dotimes (l 4)
                (let ((i (+ (* grp 4) l)))
                  (when (and (< i in) (> g 0.0))
                    (aset bytes l (logand (max -127 (min 127 (round (/ (aref xd (+ (* s in) i)) g)))) 255)))))
              (aset alo (+ (* s ng) grp) (float (+ (aref bytes 0) (* 256 (aref bytes 1)))))
              (aset ahi (+ (* s ng) grp) (float (+ (aref bytes 2) (* 256 (aref bytes 3))))))))))
    (list alo ahi gamma ng)))

(defun nl-llm-bitnet-pack-i8-w (w)
  "Ternary-quantize weight W (out x in) and pack 4 lanes/group into two f32 halves.
Returns (WLO WHI BETA NG); WLO/WHI are (out x NG), NG = ceil(in/4), BETA = mean|W|."
  (let* ((sh (photon-tensor-shape w)) (out (car sh)) (in (nth 1 sh)) (wd (photon-tensor-data w))
         (n (* out in)) (ng (/ (+ in 3) 4)) (acc 0.0)
         (wlo (make-vector (* out ng) 0.0)) (whi (make-vector (* out ng) 0.0)))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let ((beta (/ acc (float n))))
      (dotimes (o out)
        (dotimes (grp ng)
          (let ((bytes (make-vector 4 0)))
            (dotimes (l 4)
              (let ((i (+ (* grp 4) l)))
                (when (< i in)
                  (let ((q (if (> beta 0.0) (/ (aref wd (+ (* o in) i)) beta) 0.0)))
                    (aset bytes l (logand (cond ((>= q 0.5) 1) ((<= q -0.5) -1) (t 0)) 255))))))
            (aset wlo (+ (* o ng) grp) (float (+ (aref bytes 0) (* 256 (aref bytes 1)))))
            (aset whi (+ (* o ng) grp) (float (+ (aref bytes 2) (* 256 (aref bytes 3))))))))
      (list wlo whi beta ng))))

;;;###autoload
(defun nl-llm-bitnet-dp4a-linear (x w bias)
  "X (seq x in) . Wq^T + BIAS using hardware DP4A (int8 activations, ternary
weights) via the bitlinear-dp4a kernel.  Returns the flat (seq*out) result."
  (let* ((seq (car (photon-tensor-shape x))) (out (car (photon-tensor-shape w)))
         (pa (nl-llm-bitnet-pack-i8-act x)) (pw (nl-llm-bitnet-pack-i8-w w)))
    (nth 7 (nelisp-gpu-server-run
            'bitlinear-dp4a
            (list (nth 0 pa) (nth 1 pa) (nth 0 pw) (nth 1 pw)
                  (copy-sequence (photon-tensor-data bias)) (vector (nth 2 pw)) (nth 2 pa)
                  (make-vector (* seq out) 0.0))
            (vector seq out (nth 3 pa)) (/ (+ (* seq out) 63) 64)))))

;; --- DP4A with one f32 per 4 lanes (bitcast packing): memory + compute --------
(defun nl-llm-bitnet--word (b0 b1 b2 b3)
  "Pack four unsigned bytes (int8 & 255) into one little-endian uint32."
  (logior b0 (ash b1 8) (ash b2 16) (ash b3 24)))

(defun nl-llm-bitnet-pack-i8-act-1f (x)
  "Per-row int8-quantize X (seq x in) and pack 4 lanes per uint32 (1 word/4 lanes).
Returns (AP-U32 GAMMA NG): AP-U32 is (seq*NG) uint32 words, GAMMA (seq), NG=ceil(in/4)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (in (nth 1 sh)) (xd (photon-tensor-data x))
         (ng (/ (+ in 3) 4)) (ap (make-vector (* seq ng) 0)) (gamma (make-vector seq 0.0)))
    (dotimes (s seq)
      (let ((amax 0.0))
        (dotimes (i in) (let ((av (abs (aref xd (+ (* s in) i))))) (when (> av amax) (setq amax av))))
        (let ((g (/ amax 127.0)))
          (aset gamma s g)
          (dotimes (grp ng)
            (let ((by (make-vector 4 0)))
              (dotimes (l 4) (let ((i (+ (* grp 4) l)))
                (when (and (< i in) (> g 0.0))
                  (aset by l (logand (max -127 (min 127 (round (/ (aref xd (+ (* s in) i)) g)))) 255)))))
              (aset ap (+ (* s ng) grp) (nl-llm-bitnet--word (aref by 0) (aref by 1) (aref by 2) (aref by 3))))))))
    (list ap gamma ng)))

(defun nl-llm-bitnet-pack-i8-w-1f (w)
  "Ternary-quantize weight W (out x in) and pack 4 lanes per uint32.
Returns (WP-U32 BETA NG): WP-U32 is (out*NG) uint32 words."
  (let* ((sh (photon-tensor-shape w)) (out (car sh)) (in (nth 1 sh)) (wd (photon-tensor-data w))
         (n (* out in)) (ng (/ (+ in 3) 4)) (acc 0.0) (wp (make-vector (* out ng) 0)))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let ((beta (/ acc (float n))))
      (dotimes (o out)
        (dotimes (grp ng)
          (let ((by (make-vector 4 0)))
            (dotimes (l 4) (let ((i (+ (* grp 4) l)))
              (when (< i in)
                (let ((q (if (> beta 0.0) (/ (aref wd (+ (* o in) i)) beta) 0.0)))
                  (aset by l (logand (cond ((>= q 0.5) 1) ((<= q -0.5) -1) (t 0)) 255))))))
            (aset wp (+ (* o ng) grp) (nl-llm-bitnet--word (aref by 0) (aref by 1) (aref by 2) (aref by 3))))))
      (list wp beta ng))))

;;;###autoload
(defun nl-llm-bitnet-dp4a-1f-linear (x w bias)
  "X (seq x in) . Wq^T + BIAS via hardware DP4A with ONE f32 per 4 int8 lanes
\(bitcast packing: 1 byte/weight = 4x less than f32, plus DP4A).  Returns the flat
\(seq*out) result.  The GPU server must be running."
  (let* ((seq (car (photon-tensor-shape x))) (out (car (photon-tensor-shape w)))
         (pa (nl-llm-bitnet-pack-i8-act-1f x)) (pw (nl-llm-bitnet-pack-i8-w-1f w))
         (hap (nelisp-gpu-server-upload-u32 (nth 0 pa))) (hwp (nelisp-gpu-server-upload-u32 (nth 0 pw)))
         (ng (nth 2 pa)))
    ;; run2 returns only the out/inout descs (here just Y) in binding order
    (prog1 (nth 0 (nelisp-gpu-server-run2 'bitlinear-dp4a-1f
                   (list (list 'res hap (* seq ng)) (list 'res hwp (* out ng))
                         (cons 'in (copy-sequence (photon-tensor-data bias))) (cons 'in (vector (nth 1 pw)))
                         (cons 'in (nth 1 pa)) (cons 'out (* seq out)))
                   (list seq out ng) (/ (+ (* seq out) 63) 64)))
      (nelisp-gpu-server-free hap) (nelisp-gpu-server-free hwp))))

;; --- whole-model DP4A decode (every block linear via bitlinear-dp4a-1f) -------
(defun nl-llm-bitnet-dp4a-pack-block (blk)
  "Pack a block's seven linear weights into the DP4A 1-f32 form and upload them
RESIDENT (once).  Each weight key becomes (WHANDLE BETA NG IN OUT) for
`nl-llm-bitnet--dp4a-run1'; biases and norms pass through.  Free with
`nl-llm-bitnet-dp4a-free-blocks'."
  (let ((o nil) (kv blk))
    (while kv
      (let ((key (car kv)) (val (cadr kv)))
        (if (memq key '(:wq :wk :wv :wo :wg :wu :wd))
            (let* ((sh (photon-tensor-shape val)) (out (car sh)) (in (nth 1 sh))
                   (pw (nl-llm-bitnet-pack-i8-w-1f val)) (h (nelisp-gpu-server-upload-u32 (nth 0 pw))))
              (push key o) (push (list h (nth 1 pw) (nth 2 pw) in out) o))
          (push key o) (push val o)))
      (setq kv (cddr kv)))
    (nreverse o)))

(defun nl-llm-bitnet-dp4a-free-blocks (pblocks)
  "Free the resident weight handles held by DP4A-packed PBLOCKS."
  (dolist (blk pblocks)
    (let ((kv blk))
      (while kv (when (memq (car kv) '(:wq :wk :wv :wo :wg :wu :wd))
                  (ignore-errors (nelisp-gpu-server-free (car (cadr kv))))) (setq kv (cddr kv))))))

(defun nl-llm-bitnet--dp4a-run1 (xrow wspec bias)
  "One-row DP4A linear: int8-quantize XROW (1 x in), DP4A against the RESIDENT
ternary weight in WSPEC = (WHANDLE BETA NG IN OUT).  Returns the flat (out)."
  (let* ((whandle (nth 0 wspec)) (beta (nth 1 wspec)) (ng (nth 2 wspec)) (out (nth 4 wspec))
         (pa (nl-llm-bitnet-pack-i8-act-1f xrow)) (hap (nelisp-gpu-server-upload-u32 (nth 0 pa))))
    (prog1 (nth 0 (nelisp-gpu-server-run2 'bitlinear-dp4a-1f
                   (list (list 'res hap ng) (list 'res whandle (* out ng))
                         (cons 'in (copy-sequence (photon-tensor-data bias))) (cons 'in (vector beta))
                         (cons 'in (nth 1 pa)) (cons 'out out))
                   (list 1 out ng) (/ (+ out 63) 64)))
      (nelisp-gpu-server-free hap))))

;;;###autoload
(defun nl-llm-bitnet-dp4a-decode-block (xrow pblk cache &optional rope-base)
  "Pre-norm block with every linear via hardware DP4A (1-f32 packed int8 weights +
int8 activations).  PBLK is a `nl-llm-bitnet-dp4a-pack-block' result."
  (nl-llm-bitnet--blk xrow pblk cache #'nl-llm-bitnet--dp4a-run1 rope-base))

;;;###autoload
(defun nl-llm-bitnet-dp4a-decode-step (token pblocks caches wte lnfg bh dim &optional rope-base)
  "Decode one TOKEN through DP4A-packed PBLOCKS with KV CACHES; tied head stays f32.
Returns the (vocab) logit vector."
  (let* ((wd (photon-tensor-data wte))
         (x (photon-tensor (list 1 dim) (let ((v (make-vector dim 0.0)))
                                          (dotimes (j dim) (aset v j (aref wd (+ (* token dim) j)))) v)))
         (bl pblocks) (cl caches))
    (while bl (setq x (nl-llm-bitnet-dp4a-decode-block x (car bl) (car cl) rope-base)) (setq bl (cdr bl) cl (cdr cl)))
    (photon-tensor-data (photon-tensor-linear (nl-llm-rmsnorm x lnfg) wte bh))))

(provide 'nl-llm-bitnet)
;;; nl-llm-bitnet.el ends here
