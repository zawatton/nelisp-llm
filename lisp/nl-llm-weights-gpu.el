;;; nl-llm-weights-gpu.el --- imported int8 weights on the GPU  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 2d.  Runs a linear from an
;; imported table on the GPU without the weight ever becoming a float on the
;; Elisp side: the table's payload is already little-endian uint32 words of four
;; int8 lanes, which is exactly what the DP4A kernel reads back with `bitcast-u',
;; so the bytes go from `insert-file-contents-literally' to the GPU buffer
;; untouched.
;;
;; Two things had to exist for that:
;;
;; - `nelisp-gpu-server-upload-bytes'.  The older `upload-u32' takes a vector of
;;   Elisp integers and lists it before encoding; for a model's worth of words
;;   (~149M here) that is a 1.2 GB vector plus a 2.4 GB list, to send bytes that
;;   were already correct on disk.
;; - `bitlinear-dp4a-rows'.  Imported weights are quantized per output row, and
;;   `bitlinear-dp4a-1f' reads BETA[0].  Re-indexing that kernel was tempting
;;   and wrong: nl-llm-bitnet.el passes it a one-element BETA, so it would have
;;   read past the end.  The per-row scales got their own kernel instead.
;;
;; The arithmetic is W8A8: the weight is int8 with a per-row scale, and the
;; activation is quantized int8 with a per-row scale of its own.  That is not
;; the f32-activation path the CPU reference runs, so
;; `nl-llm-weights-apply-w8a8' reproduces the GPU's exact arithmetic on the CPU.
;; Comparing GPU against that isolates the transfer and the kernel; comparing it
;; against the f32 path measures what activation quantization costs.  Those are
;; different questions and one comparison cannot answer both.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)
(require 'nl-llm-weights)
(require 'nl-llm-gpu)            ; puts the nelisp-gpu sibling dir on load-path
(require 'nelisp-gpu-server)

(defun nl-llm-wgpu--i8 (v)
  "Round V to an int8 lane, clipped, as an unsigned byte."
  (let ((q (round v)))
    (logand (max -127 (min 127 q)) 255)))

;;;###autoload
(defun nl-llm-wgpu-pack-act (x base n)
  "Quantize N elements of X from BASE to int8 and pack four lanes per word.
Returns (BYTES . GAMMA): BYTES is a unibyte string of little-endian uint32
words, GAMMA the scale such that lane * GAMMA approximates the original."
  (let ((amax 0.0) (i 0))
    (while (< i n)
      (let ((a (abs (aref x (+ base i))))) (when (> a amax) (setq amax a)))
      (setq i (1+ i)))
    (let* ((gamma (if (> amax 0.0) (/ amax 127.0) 1.0))
           (ng (/ (+ n 3) 4))
           (out (make-string (* 4 ng) 0))
           (k 0))
      (while (< k (* 4 ng))
        (aset out k (if (< k n)
                        (nl-llm-wgpu--i8 (/ (aref x (+ base k)) gamma))
                      0))
        (setq k (1+ k)))
      (cons out gamma))))

;;;###autoload
(defun nl-llm-weights-apply-w8a8 (lin x &optional base)
  "Return LIN applied to X at BASE with the activation ALSO quantized to int8.
This is the GPU kernel's arithmetic written out on the CPU: integer lane
products accumulated, then scaled once by the weight row's scale times the
activation's.  Use it to check the GPU; use `nl-llm-weights-apply' to find out
what the activation quantization costs."
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (wb (nl-llm-weights-lin-bytes lin))
         (scales (nl-llm-weights-lin-scales lin))
         (pack (nl-llm-wgpu-pack-act x (or base 0) cols))
         (ab (car pack)) (gamma (cdr pack))
         (y (make-vector rows 0.0))
         (o 0))
    (while (< o rows)
      (let ((p (* o 4 words)) (acc 0) (i 0))
        (while (< i cols)
          (let ((w (aref wb (+ p i))) (a (aref ab i)))
            (setq acc (+ acc (* (if (> w 127) (- w 256) w)
                                (if (> a 127) (- a 256) a)))))
          (setq i (1+ i)))
        (aset y o (* (aref scales o) gamma acc)))
      (setq o (1+ o)))
    y))

;;;###autoload
(defun nl-llm-wgpu-upload (lin)
  "Upload LIN's int8 payload to a resident GPU buffer; return the handle.
The bytes go up exactly as they came off disk -- no float is built, and nothing
is allocated per word."
  (nl-llm-wgpu--upload-payload lin))

(defun nl-llm-wgpu--upload-scales (lin)
  "Put LIN\='s scales on the GPU, from the file when the file has them.

A per-row int8 weight has one scale a row; a ternary one has a scale every
128 columns, which for the output head is 9.9 million floats.
`nelisp-gpu--floats-bytes\=' encodes those one at a time in Elisp, so the
scales -- 3% of the bytes -- took ten times longer to upload than the weights
until they went the same way the weights do."
  (if (nl-llm-weights-lin-scale-offset lin)
      (nelisp-gpu-server-upload-file (nl-llm-weights-lin-path lin)
                                     (nl-llm-weights-lin-scale-offset lin)
                                     (nl-llm-weights-lin-scale-nbytes lin))
    (nelisp-gpu-server-upload-bytes
     (nelisp-gpu--floats-bytes (list (nl-llm-weights-lin-scales lin))))))

(defun nl-llm-wgpu--upload-payload (lin)
  "Put LIN's packed lanes on the GPU without routing them through Emacs.
The server reads the tensor's region of the weight file itself.  Sending the
same bytes down the pipe measured 3.6 MB/s against 2568 MB/s here, which for a
27B model is the difference between forty minutes of marshalling and one."
  (if (nl-llm-weights-lin-path lin)
      (nelisp-gpu-server-upload-file (nl-llm-weights-lin-path lin)
                                     (nl-llm-weights-lin-offset lin)
                                     (nl-llm-weights-lin-nbytes lin))
    ;; A linear built in memory -- `nl-llm-weights-quantize' for a test -- has
    ;; no file behind it, so those bytes still go down the pipe.
    (nelisp-gpu-server-upload-bytes (nl-llm-weights-lin-bytes lin))))

;;;###autoload
(defun nl-llm-wgpu-upload-lin (lin)
  "Upload LIN's payload *and its constants*; return a plist of handles.
(:w H :s H :b H :rows N :words N).

The constants are the point.  A weight's per-row scales do not change, and
Qwen3's projections have no bias at all, yet the inline form of these calls
re-sent both on every invocation -- and `nelisp-gpu--floats-bytes' encodes
float32 one element at a time in Elisp, measured at 4 microseconds each.  For
a 2048-row projection that is 2048 scales plus 2048 zeros, 16.4ms of encoding
against a kernel that runs in single-digit milliseconds, paid 1177 times in a
step.  Uploading them once turns the dominant cost of both directions into
nothing."
  (let ((rows (nl-llm-weights-lin-rows lin)))
    (list :w (nl-llm-wgpu--upload-payload lin)
          :s (nl-llm-wgpu--upload-scales lin)
          :b (nelisp-gpu-server-upload-bytes
              (nelisp-gpu--floats-bytes (list (make-vector rows 0.0))))
          :rows rows :words (nl-llm-weights-lin-words lin))))

(defun nl-llm-wgpu--resident-p (handle)
  "Non-nil when HANDLE is a plist from `nl-llm-wgpu-upload-lin'."
  (and (consp handle) (plist-member handle :w)))

;;;###autoload
(defun nl-llm-wgpu--nblk (lin)
  "How many scale blocks a row of LIN has."
  (let ((cols (nl-llm-weights-lin-cols lin))
        (bsize (or (nl-llm-weights-lin-block lin)
                   (nl-llm-weights-lin-cols lin))))
    (/ (+ cols bsize -1) bsize)))

(defun nl-llm-wgpu--apply-ternary (lin handle x base bias)
  "LIN applied on the GPU through `ternary-rows\='.

The activation goes across as float rather than packed to int8.  A ternary
weight is an add or a subtract, so there is nothing for DP4A to accelerate,
and not packing removes the activation quantization -- which after the
weights stop being requantized is the only lossy step left in this path."
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (nblk (nl-llm-wgpu--nblk lin))
         (res (nl-llm-wgpu--resident-p handle))
         (slice (if (and (= (or base 0) 0) (= (length x) cols))
                    x
                  (let ((v (make-vector cols 0.0)))
                    (dotimes (i cols) (aset v i (aref x (+ (or base 0) i))))
                    v))))
    (nth 0 (nelisp-gpu-server-run2
            'ternary-rows
            (list (cons 'in slice)
                  (list 'res (if res (plist-get handle :w) handle)
                        (* rows words))
                  (if (and res (null bias))
                      (list 'res (plist-get handle :b) rows)
                    (cons 'in (or bias (make-vector rows 0.0))))
                  (if res
                      (list 'res (plist-get handle :s) (* rows nblk))
                    (cons 'in (nl-llm-weights-lin-scales lin)))
                  (cons 'out rows))
            (list 1 rows cols nblk words)
            (/ (+ rows 63) 64)))))

;;;###autoload
(defun nl-llm-wgpu-apply (lin handle x &optional base bias)
  "Run LIN (resident at HANDLE) on X at BASE on the GPU.
`ternary-rows\=' for a two-bit weight, `bitlinear-dp4a-rows\=' for an int8 one.
BIAS defaults to zeros, which is what these projections have.  Returns the
ROWS-long result as a float vector."
  (if (nl-llm-weights-lin-ternary lin)
      (nl-llm-wgpu--apply-ternary lin handle x base bias)
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (res (nl-llm-wgpu--resident-p handle))
         (pack (nl-llm-wgpu-pack-act x (or base 0) cols))
         (hact (nelisp-gpu-server-upload-bytes (car pack))))
    (unwind-protect
        (nth 0 (nelisp-gpu-server-run2
                'bitlinear-dp4a-rows
                (list (list 'res hact words)
                      (list 'res (if res (plist-get handle :w) handle)
                            (* rows words))
                      (if (and res (null bias))
                          (list 'res (plist-get handle :b) rows)
                        (cons 'in (or bias (make-vector rows 0.0))))
                      (if res
                          (list 'res (plist-get handle :s) rows)
                        (cons 'in (nl-llm-weights-lin-scales lin)))
                      (cons 'in (vector (cdr pack)))
                      (cons 'out rows))
                (list 1 rows words)
                (/ (+ rows 63) 64)))
      (nelisp-gpu-server-free hact)))))

;;;###autoload
(defun nl-llm-wgpu-apply-t (lin handle g)
  "Return W^T applied to G on the GPU, for LIN resident at HANDLE.
The same quantity `nl-llm-weights-apply-t' computes on the CPU, and the one a
frozen base has to supply for anything upstream of it to be trainable.

Accumulation is float rather than DP4A: the per-row scale multiplies each term,
so it cannot be factored out of an integer dot product.  That makes this
comparable to the CPU path up to f32 rounding, which is what the suite checks
-- a DP4A variant would need the gradient quantized too, and a second, looser
comparison to go with it."
  (let* ((rows (nl-llm-weights-lin-rows lin))
         (cols (nl-llm-weights-lin-cols lin))
         (words (nl-llm-weights-lin-words lin)))
    (unless (= (length g) rows)
      (error "nl-llm-wgpu-apply-t: G is %d long, weight has %d rows"
             (length g) rows))
    (let ((res (nl-llm-wgpu--resident-p handle))
          (nblk (nl-llm-wgpu--nblk lin)))
      (nth 0 (nelisp-gpu-server-run2
              (if (nl-llm-weights-lin-ternary lin) 'ternary-rows-t 'dp4a-rows-t)
              (list (list 'res (if res (plist-get handle :w) handle)
                          (* rows words))
                    (if res
                        (list 'res (plist-get handle :s) (* rows nblk))
                      (cons 'in (nl-llm-weights-lin-scales lin)))
                    (cons 'in g)
                    (cons 'out cols))
              (if (nl-llm-weights-lin-ternary lin)
                  (list rows cols nblk words 1)
                (list rows cols words 1))
              (/ (+ cols 63) 64))))))

;;;###autoload
(defun nl-llm-wgpu-pack-act-seq (x base seq stride cols)
  "Pack SEQ slices of X into one byte string; return (BYTES . GAMMAS).
Slice P starts at BASE + P*STRIDE and is COLS wide.  The slices are quantized
*independently*, each with its own scale, which is what the kernel expects --
GAMMA is indexed by position there."
  (let ((parts nil) (gammas (make-vector seq 0.0)))
    (dotimes (p seq)
      (let ((pk (nl-llm-wgpu-pack-act x (+ base (* p stride)) cols)))
        (aset gammas p (cdr pk))
        (push (car pk) parts)))
    (cons (apply #'concat (nreverse parts)) gammas)))

;;;###autoload
(defun nl-llm-wgpu--apply-seq-ternary (lin handle x base seq stride)
  "LIN applied to SEQ slices of X through `ternary-rows\=', in one dispatch.
The activation goes across as float, so unlike the int8 path there is no
packing step -- only a gather when the slices are strided."
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (nblk (nl-llm-wgpu--nblk lin))
         (res (nl-llm-wgpu--resident-p handle))
         (st (or stride cols))
         (flat (if (and (= (or base 0) 0) (= st cols) (= (length x) (* seq cols)))
                   x
                 (let ((v (make-vector (* seq cols) 0.0)))
                   (dotimes (p seq)
                     (dotimes (i cols)
                       (aset v (+ (* p cols) i)
                             (aref x (+ (or base 0) (* p st) i)))))
                   v))))
    (nth 0 (nelisp-gpu-server-run2
            'ternary-rows
            (list (cons 'in flat)
                  (list 'res (if res (plist-get handle :w) handle)
                        (* rows words))
                  (if res
                      (list 'res (plist-get handle :b) rows)
                    (cons 'in (make-vector rows 0.0)))
                  (if res
                      (list 'res (plist-get handle :s) (* rows nblk))
                    (cons 'in (nl-llm-weights-lin-scales lin)))
                  (cons 'out (* seq rows)))
            (list seq rows cols nblk words)
            (/ (+ (* seq rows) 63) 64)))))

;;;###autoload
(defun nl-llm-wgpu-apply-seq (lin handle x base seq &optional stride)
  "Apply LIN to SEQ slices of X in one dispatch; return a SEQ x ROWS vector.
STRIDE defaults to LIN's COLS, which is right when X is a packed sequence of
this linear's inputs and wrong for `:wo', whose input is strided by the query
width -- so the caller says.

Arithmetic per (position, row) is what `nl-llm-wgpu-apply' does, so this is
bit-identical to calling it SEQ times; what changes is that one round trip and
one activation upload carry all of them instead of SEQ of each."
  (if (nl-llm-weights-lin-ternary lin)
      (nl-llm-wgpu--apply-seq-ternary lin handle x base seq stride)
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (res (nl-llm-wgpu--resident-p handle))
         (pack (nl-llm-wgpu-pack-act-seq x (or base 0) seq (or stride cols) cols))
         (hact (nelisp-gpu-server-upload-bytes (car pack))))
    (unwind-protect
        (nth 0 (nelisp-gpu-server-run2
                'bitlinear-dp4a-rows
                (list (list 'res hact (* seq words))
                      (list 'res (if res (plist-get handle :w) handle)
                            (* rows words))
                      (if res
                          (list 'res (plist-get handle :b) rows)
                        (cons 'in (make-vector rows 0.0)))
                      (if res
                          (list 'res (plist-get handle :s) rows)
                        (cons 'in (nl-llm-weights-lin-scales lin)))
                      (cons 'in (cdr pack))
                      (cons 'out (* seq rows)))
                (list seq rows words)
                (/ (+ (* seq rows) 63) 64)))
      (nelisp-gpu-server-free hact)))))

;;;###autoload
(defun nl-llm-wgpu-apply-t-seq (lin handle g seq)
  "Apply LIN's transpose to SEQ gradients at once; return a SEQ x COLS vector.
G is SEQ x ROWS.  Bit-identical to SEQ calls of `nl-llm-wgpu-apply-t', and the
reason to prefer it is the same as for the forward: the caller pays a round
trip and an Elisp-side float encoding of G per call, and this makes it one."
  (let* ((rows (nl-llm-weights-lin-rows lin))
         (cols (nl-llm-weights-lin-cols lin))
         (words (nl-llm-weights-lin-words lin))
         (res (nl-llm-wgpu--resident-p handle)))
    (unless (= (length g) (* seq rows))
      (error "nl-llm-wgpu-apply-t-seq: G is %d long, want %d x %d"
             (length g) seq rows))
    (nth 0 (nelisp-gpu-server-run2
            'dp4a-rows-t
            (list (list 'res (if res (plist-get handle :w) handle)
                        (* rows words))
                  (if res
                      (list 'res (plist-get handle :s) rows)
                    (cons 'in (nl-llm-weights-lin-scales lin)))
                  (cons 'in g)
                  (cons 'out (* seq cols)))
            (list rows cols words seq)
            (/ (+ (* seq cols) 63) 64)))))

;;; --- a whole layer, linears on the GPU -----------------------------------
;;
;; The linears are where the work is: at one position a layer is about 12.6M
;; multiply-accumulates across its seven matrices, against a few thousand for
;; the norms, the rotation and the attention.  So this puts the seven on the GPU
;; and leaves the elementwise glue on the CPU, which captures nearly all of the
;; arithmetic while keeping the parts that were hard to get right -- the
;; decoupled head width, the half-split rotation, QK-norm -- in the code the CPU
;; oracle already agrees with.

(require 'nl-llm-weights-forward)
;; nl-llm-wf-layer-lin and nl-llm-wb-transpose-fn live here; the backward is
;; what this file routes, so requiring it is honest about the dependency.
(require 'nl-llm-weights-backward)

(cl-defstruct (nl-llm-wgpu-layer (:constructor nl-llm-wgpu-layer--make))
  lins       ; plist ROLE -> nl-llm-weights-lin
  handles    ; plist ROLE -> resident GPU handle
  ln1g ln2g q-norm k-norm)

(defconst nl-llm-wgpu-roles '(:wq :wk :wv :wo :wg :wu :wd)
  "The quantized matrices of one imported block, in no particular order.")

;;;###autoload
(defun nl-llm-wgpu-load-layer (wts layer)
  "Load LAYER of WTS with its seven matrices resident on the GPU.
Weights are uploaded once and referenced by handle afterwards, so a 28-layer
forward pays the transfer a single time."
  (let (lins handles)
    (dolist (role nl-llm-wgpu-roles)
      (let ((lin (nl-llm-weights-linear wts role layer)))
        (setq lins (plist-put lins role lin))
        (setq handles (plist-put handles role (nl-llm-wgpu-upload lin)))))
    (nl-llm-wgpu-layer--make
     :lins lins :handles handles
     :ln1g (nl-llm-weights-row wts (nl-llm-weights-tensor wts :ln1g layer) 0)
     :ln2g (nl-llm-weights-row wts (nl-llm-weights-tensor wts :ln2g layer) 0)
     :q-norm (nl-llm-weights-row wts (nl-llm-weights-tensor wts :q-norm layer) 0)
     :k-norm (nl-llm-weights-row wts (nl-llm-weights-tensor wts :k-norm layer) 0))))

;;;###autoload
(defun nl-llm-wgpu-free-layer (lay)
  "Free LAY's resident GPU buffers."
  (dolist (role nl-llm-wgpu-roles)
    (let ((h (plist-get (nl-llm-wgpu-layer-handles lay) role)))
      (when h (nelisp-gpu-server-free h)))))

(defun nl-llm-wgpu--lin (lay role x &optional base)
  "Apply LAY's ROLE matrix to X at BASE on the GPU."
  (nl-llm-wgpu-apply (plist-get (nl-llm-wgpu-layer-lins lay) role)
                     (plist-get (nl-llm-wgpu-layer-handles lay) role)
                     x base))

;;;###autoload
(defun nl-llm-wgpu-block (lay x seq cfg)
  "Run one imported block over X (flat SEQ x dim) with its linears on the GPU.
Elementwise arithmetic stays on the CPU and is the same code
`nl-llm-wf-block' uses, so the two differ only in how the matrices are
multiplied -- and therefore only by the activation quantization the GPU path
does and the f32 path does not."
  (let* ((dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (eps (or (plist-get cfg :rms-eps) 1.0e-6))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ff (nl-llm-weights-lin-rows
              (plist-get (nl-llm-wgpu-layer-lins lay) :wg)))
         (q (make-vector (* seq qdim) 0.0))
         (k (make-vector (* seq kvdim) 0.0))
         (v (make-vector (* seq kvdim) 0.0)))
    (dotimes (i seq)
      (let ((a (nl-llm-wf--rmsnorm x (* i dim) dim
                                   (nl-llm-wgpu-layer-ln1g lay) eps)))
        (let ((qi (nl-llm-wgpu--lin lay :wq a))
              (ki (nl-llm-wgpu--lin lay :wk a))
              (vi (nl-llm-wgpu--lin lay :wv a)))
          (dotimes (t0 qdim) (aset q (+ (* i qdim) t0) (aref qi t0)))
          (dotimes (t0 kvdim) (aset k (+ (* i kvdim) t0) (aref ki t0)))
          (dotimes (t0 kvdim) (aset v (+ (* i kvdim) t0) (aref vi t0))))))
    (dotimes (i seq)
      (nl-llm--rmsnorm-heads q (* i qdim) heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wgpu-layer-q-norm lay))
                             eps)
      (nl-llm--rmsnorm-heads k (* i kvdim) kv-heads hd
                             (photon-tensor (list hd)
                                            (nl-llm-wgpu-layer-k-norm lay))
                             eps)
      (nl-llm--rope-heads q (* i qdim) heads hd i rbase 'half)
      (nl-llm--rope-heads k (* i kvdim) kv-heads hd i rbase 'half))
    (let ((ctx (nl-llm-wf--attend q k v seq heads kv-heads hd))
          (x1 (make-vector (* seq dim) 0.0)))
      (dotimes (i seq)
        (let ((o (nl-llm-wgpu--lin lay :wo ctx (* i qdim))))
          (dotimes (t0 dim)
            (aset x1 (+ (* i dim) t0)
                  (+ (aref x (+ (* i dim) t0)) (aref o t0))))))
      (let ((out (make-vector (* seq dim) 0.0)))
        (dotimes (i seq)
          (let* ((b (nl-llm-wf--rmsnorm x1 (* i dim) dim
                                        (nl-llm-wgpu-layer-ln2g lay) eps))
                 (g (nl-llm-wgpu--lin lay :wg b))
                 (u (nl-llm-wgpu--lin lay :wu b))
                 (h (nl-llm-wf--silu-mul g u ff))
                 (d (nl-llm-wgpu--lin lay :wd h)))
            (dotimes (t0 dim)
              (aset out (+ (* i dim) t0)
                    (+ (aref x1 (+ (* i dim) t0)) (aref d t0))))))
        out))))

;;;###autoload
(defun nl-llm-wgpu-block-cpu (wts layer x seq cfg)
  "Run one imported block with the GPU's W8A8 arithmetic, on the CPU.
The reference for `nl-llm-wgpu-block': identical except that every matrix goes
through `nl-llm-weights-apply-w8a8' instead of the kernel, so a difference
between the two is the transfer or the kernel and nothing else.  Compare
against `nl-llm-wf-block' instead to see what the activation quantization
costs."
  (let* ((lay (nl-llm-wf-load-layer wts layer))
         (dim (plist-get cfg :dim))
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
    (dotimes (i seq)
      (let ((a (nl-llm-wf--rmsnorm x (* i dim) dim
                                   (nl-llm-wf-layer-ln1g lay) eps)))
        (let ((qi (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wq lay) a))
              (ki (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wk lay) a))
              (vi (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wv lay) a)))
          (dotimes (t0 qdim) (aset q (+ (* i qdim) t0) (aref qi t0)))
          (dotimes (t0 kvdim) (aset k (+ (* i kvdim) t0) (aref ki t0)))
          (dotimes (t0 kvdim) (aset v (+ (* i kvdim) t0) (aref vi t0))))))
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
        (let ((o (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wo lay)
                                            ctx (* i qdim))))
          (dotimes (t0 dim)
            (aset x1 (+ (* i dim) t0)
                  (+ (aref x (+ (* i dim) t0)) (aref o t0))))))
      (let ((out (make-vector (* seq dim) 0.0)))
        (dotimes (i seq)
          (let* ((b (nl-llm-wf--rmsnorm x1 (* i dim) dim
                                        (nl-llm-wf-layer-ln2g lay) eps))
                 (g (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wg lay) b))
                 (u (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wu lay) b))
                 (h (nl-llm-wf--silu-mul g u ff))
                 (d (nl-llm-weights-apply-w8a8 (nl-llm-wf-layer-wd lay) h)))
            (dotimes (t0 dim)
              (aset out (+ (* i dim) t0)
                    (+ (aref x1 (+ (* i dim) t0)) (aref d t0))))))
        out))))

;;; --- routing the backward's transposes to the GPU ------------------------

(defvar nl-llm-wgpu--transposes nil
  "An eq hash of `nl-llm-weights-lin' -> resident handle, or nil.
Consulted by `nl-llm-wgpu-transpose', which stands in for the CPU loop while
`nl-llm-wgpu-with-transposes' is in scope.")

;;;###autoload
(defun nl-llm-wgpu-upload-transposes (layers)
  "Upload LAYERS' seven matrices each and return an eq hash LIN -> handle.
LAYERS are `nl-llm-wf-layer' structs -- the CPU ones the backward actually
holds.  Uploading *those* objects rather than a separately loaded copy is what
makes the hash lookup work: two loads of the same tensor are equal in content
and not `eq', and a lookup that missed would silently fall back to the CPU and
look merely slow."
  (let ((tbl (make-hash-table :test 'eq)))
    (dolist (lay layers)
      (dolist (role nl-llm-wgpu-roles)
        (let ((lin (nl-llm-wf-layer-lin lay role)))
          (puthash lin (nl-llm-wgpu-upload-lin lin) tbl))))
    tbl))

;;;###autoload
(defun nl-llm-wgpu-free-transposes (tbl)
  "Free every resident handle in TBL, whichever form it takes."
  (maphash (lambda (_lin h)
             (if (nl-llm-wgpu--resident-p h)
                 (dolist (k '(:w :s :b)) (nelisp-gpu-server-free (plist-get h k)))
               (nelisp-gpu-server-free h)))
           tbl))

;;;###autoload
(defun nl-llm-wgpu-transpose (lin g)
  "Return W^T.G for LIN on the GPU when it is resident, else on the CPU."
  (let ((h (and nl-llm-wgpu--transposes (gethash lin nl-llm-wgpu--transposes))))
    (if h (nl-llm-wgpu-apply-t lin h g) (nl-llm-weights-apply-t lin g))))

;;;###autoload
(defun nl-llm-wgpu-transpose-seq (lin g seq)
  "W^T.G for SEQ gradients on the GPU when LIN is resident, else on the CPU."
  (let ((h (and nl-llm-wgpu--transposes (gethash lin nl-llm-wgpu--transposes))))
    (if h (nl-llm-wgpu-apply-t-seq lin h g seq)
      (let* ((rows (nl-llm-weights-lin-rows lin))
             (cols (nl-llm-weights-lin-cols lin))
             (out (make-vector (* seq cols) 0.0)))
        (dotimes (p seq)
          (let ((gp (make-vector rows 0.0)))
            (dotimes (o rows) (aset gp o (aref g (+ (* p rows) o))))
            (let ((dx (nl-llm-weights-apply-t lin gp)))
              (dotimes (i cols) (aset out (+ (* p cols) i) (aref dx i))))))
        out))))

;;;###autoload
(defun nl-llm-wgpu-apply-seq-resident (lin x seq stride)
  "Apply LIN to SEQ slices of X on the GPU, using the current table.
Falls back to SEQ separate CPU applications when LIN is not resident, so a
partially uploaded model runs and is merely slower."
  (let ((h (and nl-llm-wgpu--transposes (gethash lin nl-llm-wgpu--transposes))))
    (if h (nl-llm-wgpu-apply-seq lin h x 0 seq stride)
      (let* ((rows (nl-llm-weights-lin-rows lin))
             (out (make-vector (* seq rows) 0.0)))
        (dotimes (p seq)
          (let ((y (nl-llm-weights-apply lin x (* p stride))))
            (dotimes (o rows) (aset out (+ (* p rows) o) (aref y o)))))
        out))))

;;;###autoload
(defun nl-llm-wgpu-apply-resident (lin x base)
  "Apply LIN to X at BASE on the GPU, using the handle from the current table.
Falls back to the CPU when LIN is not resident, so a partially uploaded model
runs and is merely slower."
  (let ((h (and nl-llm-wgpu--transposes (gethash lin nl-llm-wgpu--transposes))))
    (if h (nl-llm-wgpu-apply lin h x base) (nl-llm-weights-apply lin x base))))

;;;###autoload
(defmacro nl-llm-wgpu-with-linears (table &rest body)
  "Run BODY with BOTH directions of every resident linear on the GPU.
Unlike `nl-llm-wgpu-with-transposes' this also moves the forward, which is a
*different computation* and not merely a faster one: the kernel quantizes the
activation to int8, so a result computed here and one computed on the f32 path
differ by about 7e-03 per block, not by rounding.  Use it when the comparison
you intend is against the same W8A8 arithmetic; use the transposes-only macro
when you want the verified f32 reference to still apply."
  (declare (indent 1))
  `(let ((nl-llm-wgpu--transposes ,table)
         (nl-llm-wb-transpose-fn #'nl-llm-wgpu-transpose)
         (nl-llm-wb-transpose-seq-fn #'nl-llm-wgpu-transpose-seq)
         (nl-llm-wb-forward-fn #'nl-llm-wgpu-apply-resident)
         (nl-llm-wb-forward-seq-fn #'nl-llm-wgpu-apply-seq-resident))
     ,@body))

;;;###autoload
(defun nl-llm-wgpu-open-model (wts &optional nlayers)
  "Load NLAYERS of WTS and upload every matrix once; return a session plist.
(:layers LIST :table HASH :cfg PLIST).  The point is the `once': a step that
uploads per block pays about five seconds a block, which for 28 layers is more
than the arithmetic it enables.  Qwen3-0.6B is 568 MiB of int8, so the whole
model is resident inside a 6 GB card with room for the rest."
  (let* ((cfg (nl-llm-weights-config wts))
         (n (min (or nlayers (plist-get cfg :layers)) (plist-get cfg :layers)))
         (layers nil))
    (dotimes (ly n) (push (nl-llm-wf-load-layer wts ly) layers))
    (setq layers (nreverse layers))
    (let ((tbl (nl-llm-wgpu-upload-transposes layers))
          ;; The tied head goes in the same table.  It is one more linear --
          ;; 151936 x 1024, 155 MiB -- and both directions of it are the same
          ;; two kernels, so scoring the vocabulary and pushing the gradient
          ;; back from it need no new machinery.  Leaving it on the CPU would
          ;; make it the whole cost: 70s against the blocks' 7.5s.
          (head (nl-llm-weights-linear wts :wte)))
      (puthash head (nl-llm-wgpu-upload-lin head) tbl)
      (list :layers layers :table tbl :cfg cfg
            :head (list :lin head
                        :lnf (nl-llm-weights-row
                              wts (nl-llm-weights-tensor wts :lnf) 0))))))

;;;###autoload
(defun nl-llm-wgpu-close-model (session)
  "Free SESSION's resident buffers."
  (nl-llm-wgpu-free-transposes (plist-get session :table)))

;;;###autoload
(defmacro nl-llm-wgpu-with-transposes (table &rest body)
  "Run BODY with every frozen base's W^T.g routed through TABLE to the GPU.
TABLE is from `nl-llm-wgpu-upload-transposes'; a linear absent from it falls
back to the CPU loop, so a partially uploaded model still works and is merely
slower.

This wires only the *backward*.  The forward stays on the f32 path on purpose:
`nl-llm-wgpu-block' quantizes activations to int8, which is a different
computation from the CPU reference, and folding that in here would mean a
gradient difference could be either the wiring or the quantization.  Separating
them keeps the comparison against the verified CPU backward direct."
  (declare (indent 1))
  `(let ((nl-llm-wgpu--transposes ,table)
         (nl-llm-wb-transpose-fn #'nl-llm-wgpu-transpose)
         (nl-llm-wb-transpose-seq-fn #'nl-llm-wgpu-transpose-seq))
     ,@body))

;;;###autoload
(defun nl-llm-wgpu-next-token (wts tokens &optional nlayers progress)
  "Greedy next token for TOKENS with every linear on the GPU.
Returns (ID . LOGIT).  Layers are loaded, used and freed one at a time, so
peak VRAM is one layer rather than the whole model; the head is scored on the
CPU because it is read once and used once."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (seq (length tokens))
         (n (min (or nlayers (plist-get cfg :layers))
                 (plist-get cfg :layers)))
         (x (make-vector (* seq dim) 0.0))
         (i 0))
    (dolist (tk tokens)
      (let ((row (nl-llm-weights-embed wts tk)))
        (dotimes (t0 dim) (aset x (+ (* i dim) t0) (aref row t0))))
      (setq i (1+ i)))
    (dotimes (ly n)
      (let ((lay (nl-llm-wgpu-load-layer wts ly)))
        (unwind-protect
            (setq x (nl-llm-wgpu-block lay x seq cfg))
          (nl-llm-wgpu-free-layer lay)))
      (when progress (funcall progress ly)))
    (let ((final (nl-llm-wf-final-norm wts x seq)))
      (nl-llm-wf-argmax (nl-llm-wf-logits-all wts final seq (1- seq))))))

(provide 'nl-llm-weights-gpu)
;;; nl-llm-weights-gpu.el ends here
