;;; nl-llm-weights-fused.el --- a block as one GPU batch  -*- lexical-binding: t; -*-

;; A block's linears already run on the GPU, and so does everything between
;; them if asked separately.  Asking separately is the problem: measured on a
;; step at seq 6, float marshalling across the Elisp boundary is about 8.3s of
;; 32.5s and the Elisp glue another 15.5s, and both exist only because
;; consecutive operations sit on opposite sides of that boundary.  A tensor
;; that is produced on the device and consumed on the device should not visit
;; Elisp in between.
;;
;; So this assembles a block as one `nelisp-gpu-server-batch': intermediates
;; live in `tmp' slots, which stay resident for the batch and are never
;; marshalled.  Nothing here is new arithmetic -- every kernel it dispatches is
;; checked against its CPU counterpart elsewhere in the suite.  What is new is
;; that the answer comes back once instead of twenty times.
;;
;; The layer's weights and gains are uploaded once by `nl-llm-wfuse-open-layer'
;; and referenced by handle.  The gains go up as floats through the encoder,
;; which costs a few thousand elements once per layer rather than per call.

;;; Code:

(require 'nl-llm-gpu)            ; puts the nelisp-gpu sibling dir on load-path
(require 'nelisp-gpu-server)
(require 'nl-llm-weights)
(require 'nl-llm-weights-gpu)
(require 'nl-llm-weights-forward)

(defconst nl-llm-wfuse-roles '(:wq :wk :wv :wo :wg :wu :wd))

;;;###autoload
(defun nl-llm-wfuse-open-layer (wts layer)
  "Upload LAYER of WTS -- seven linears and four gains -- and return a plist.
The gains are float vectors and go through the encoder; that is a few thousand
elements once, against the millions a per-call path would send."
  (let ((pl (list :cfg (nl-llm-weights-config wts))))
    (dolist (role nl-llm-wfuse-roles)
      (setq pl (plist-put pl role
                          (nl-llm-wgpu-upload-lin
                           (nl-llm-weights-linear wts role layer)))))
    (dolist (pair '((:ln1g . :ln1g) (:ln2g . :ln2g)
                    (:qnorm . :q-norm) (:knorm . :k-norm)))
      (let* ((tn (ignore-errors
                   (nl-llm-weights-tensor wts (cdr pair) layer)))
             (v (and tn (nl-llm-weights-row wts tn 0))))
        (setq pl (plist-put pl (car pair)
                            (and v (cons (nelisp-gpu-server-upload-bytes
                                          (nelisp-gpu--floats-bytes (list v)))
                                         (length v)))))))
    pl))

;;;###autoload
(defun nl-llm-wfuse-close-layer (lay)
  "Free every handle LAY holds."
  (dolist (role nl-llm-wfuse-roles)
    (let ((h (plist-get lay role)))
      (when h (dolist (k '(:w :s :b)) (nelisp-gpu-server-free (plist-get h k))))))
  (dolist (k '(:ln1g :ln2g :qnorm :knorm))
    (let ((h (plist-get lay k))) (when h (nelisp-gpu-server-free (car h))))))

(defun nl-llm-wfuse--lin (lay role) (plist-get lay role))

(defconst nl-llm-wfuse-tape-keys
  '(:x1 :istd1 :istd2 :qpre :kpre :q :k :v :p :g :u)
  "The forward intermediates a backward reads, in no particular order.
Everything else a block computes is consumed inside the batch and can stay in
a `tmp' slot; these have to outlive it, so they are resident buffers a kernel
writes into.  `:p' is why a training forward decomposes the attention:
`attn-causal-gqa' discards the probabilities and every attention vjp takes
them.")

;;;###autoload
(defun nl-llm-wfuse-alloc-tape (cfg seq ff)
  "Allocate the resident buffers a block's backward will read; return a plist.
Allocation is a zeroed unibyte string through `upload-bytes', so it costs a
memcpy rather than a pass through the float encoder.  Reuse one across steps:
the sizes depend only on SEQ."
  (let* ((dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (sizes (list :x1 (* seq dim) :istd1 seq :istd2 seq
                      :qpre (* seq qdim) :kpre (* seq kvdim)
                      :q (* seq qdim) :k (* seq kvdim) :v (* seq kvdim)
                      :p (* heads seq seq) :g (* seq ff) :u (* seq ff)))
         (out nil))
    (dolist (k nl-llm-wfuse-tape-keys)
      (let ((n (plist-get sizes k)))
        (setq out (plist-put out k
                             (cons (nelisp-gpu-server-upload-bytes
                                    (make-string (* 4 n) 0))
                                   n)))))
    out))

;;;###autoload
(defun nl-llm-wfuse-free-tape (tape)
  "Free every buffer in TAPE."
  (dolist (k nl-llm-wfuse-tape-keys)
    (let ((h (plist-get tape k))) (when h (nelisp-gpu-server-free (car h))))))

(defun nl-llm-wfuse--slot (tape key n)
  "A slot for KEY: resident when TAPE carries it, a `tmp' of N otherwise."
  (let ((h (and tape (plist-get tape key))))
    (if h (list 'res (car h) (cdr h)) (cons 'tmp n))))

;;;###autoload
(defun nl-llm-wfuse-block (lay x seq &optional tape)
  "Run a whole block over X (SEQ x dim) as one batch; return its output.
Every intermediate -- the normalised activations, their packed forms, q, k, v,
the rotated forms, the context, the SwiGLU halves -- stays on the device.  The
answer crosses the boundary once.

Equal to `nl-llm-wb-block-forward' under `nl-llm-wgpu-with-linears' to about
4e-07, which is the same W8A8 arithmetic with the glue in f32 rather than f64.
It is not equal to the f32 CPU block and does not claim to be: the activation
quantization those two differ by is measured separately.

With TAPE from `nl-llm-wfuse-alloc-tape', the intermediates a backward reads
are written into its resident buffers instead of into `tmp' slots, and the
attention is run as scores/softmax/context so the probabilities survive --
`attn-causal-gqa' discards them and every attention vjp needs them.  Without
TAPE the block is the inference form and nothing outlives the batch."
  (let* ((cfg (plist-get lay :cfg))
         (dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (ng (/ dim 4)) (qng (/ qdim 4))
         (wq (nl-llm-wfuse--lin lay :wq)) (wk (nl-llm-wfuse--lin lay :wk))
         (wv (nl-llm-wfuse--lin lay :wv)) (wo (nl-llm-wfuse--lin lay :wo))
         (wg (nl-llm-wfuse--lin lay :wg)) (wu (nl-llm-wfuse--lin lay :wu))
         (wd (nl-llm-wfuse--lin lay :wd))
         (ff (plist-get wg :rows)) (fng (/ ff 4))
         (g64 (lambda (n) (/ (+ n 63) 64)))
         ;; slots, in the order the batch declares them
         (slots (list (cons 'in x)                       ; 0  X
                      (nl-llm-wfuse--slot tape :istd1 seq)          ; 1  istd
                      (cons 'tmp (* seq dim))            ; 2  A
                      (cons 'tmp (* seq ng))             ; 3  AQ
                      (cons 'tmp seq)                    ; 4  AG
                      (nl-llm-wfuse--slot tape :qpre (* seq qdim))  ; 5  Q
                      (nl-llm-wfuse--slot tape :kpre (* seq kvdim)) ; 6  K
                      (nl-llm-wfuse--slot tape :v (* seq kvdim))    ; 7  V
                      (cons 'tmp (* seq qdim))           ; 8  Qn
                      (cons 'tmp (* seq kvdim))          ; 9  Kn
                      (nl-llm-wfuse--slot tape :q (* seq qdim))     ; 10 Qr
                      (nl-llm-wfuse--slot tape :k (* seq kvdim))    ; 11 Kr
                      (cons 'tmp (* seq qdim))           ; 12 CTX
                      (cons 'tmp (* seq qng))            ; 13 CQ
                      (cons 'tmp seq)                    ; 14 CG
                      (cons 'tmp (* seq dim))            ; 15 O
                      (nl-llm-wfuse--slot tape :x1 (* seq dim))     ; 16 X1
                      (list 'res (car (plist-get lay :ln1g)) dim)   ; 17
                      (list 'res (car (plist-get lay :qnorm)) hd)   ; 18
                      (list 'res (car (plist-get lay :knorm)) hd)   ; 19
                      (list 'res (plist-get wq :w) (* (plist-get wq :rows) ng))
                      (list 'res (plist-get wq :b) (plist-get wq :rows))
                      (list 'res (plist-get wq :s) (plist-get wq :rows))
                      (list 'res (plist-get wk :w) (* (plist-get wk :rows) ng))
                      (list 'res (plist-get wk :b) (plist-get wk :rows))
                      (list 'res (plist-get wk :s) (plist-get wk :rows))
                      (list 'res (plist-get wv :w) (* (plist-get wv :rows) ng))
                      (list 'res (plist-get wv :b) (plist-get wv :rows))
                      (list 'res (plist-get wv :s) (plist-get wv :rows))
                      (list 'res (plist-get wo :w) (* (plist-get wo :rows) qng))
                      (list 'res (plist-get wo :b) (plist-get wo :rows))   ; 30
                      (list 'res (plist-get wo :s) (plist-get wo :rows))   ; 31
                      ;; --- the feed-forward half ---
                      (nl-llm-wfuse--slot tape :istd2 seq)          ; 32 istd2
                      (cons 'tmp (* seq dim))            ; 33 B
                      (cons 'tmp (* seq ng))             ; 34 BQ
                      (cons 'tmp seq)                    ; 35 BG
                      (nl-llm-wfuse--slot tape :g (* seq ff))       ; 36 G
                      (nl-llm-wfuse--slot tape :u (* seq ff))       ; 37 U
                      (cons 'tmp (* seq ff))             ; 38 H
                      (cons 'tmp (* seq fng))            ; 39 HQ
                      (cons 'tmp seq)                    ; 40 HG
                      (cons 'tmp (* seq dim))            ; 41 D
                      (cons 'out (* seq dim))            ; 42 OUT
                      (list 'res (car (plist-get lay :ln2g)) dim)   ; 43
                      (list 'res (plist-get wg :w) (* (plist-get wg :rows) ng))
                      (list 'res (plist-get wg :b) (plist-get wg :rows))
                      (list 'res (plist-get wg :s) (plist-get wg :rows))
                      (list 'res (plist-get wu :w) (* (plist-get wu :rows) ng))
                      (list 'res (plist-get wu :b) (plist-get wu :rows))
                      (list 'res (plist-get wu :s) (plist-get wu :rows))
                      (list 'res (plist-get wd :w) (* (plist-get wd :rows) fng))
                      (list 'res (plist-get wd :b) (plist-get wd :rows))
                      (list 'res (plist-get wd :s) (plist-get wd :rows))
                      ;; --- only used when a tape is given ---
                      (cons 'tmp (* heads seq seq))                     ; 53 S
                      (nl-llm-wfuse--slot tape :p (* heads seq seq))))  ; 54 P
         ;; With a tape the probabilities have to survive, so the attention is
         ;; decomposed into scores, softmax and context; without one the fused
         ;; kernel does it in a single dispatch.  Everything else is the same
         ;; list, which is the point of splicing rather than writing it twice.
         (attn
          (if (null tape)
              (list (list 'attn-causal-gqa '(10 11 7 12)
                          (list seq heads kv-heads hd)
                          (funcall g64 (* heads seq))))
            (list (list 'attn-scores '(10 11 53) (list seq qdim heads kv-heads)
                        (funcall g64 (* heads seq seq)))
                  (list 'softmax '(53 54) (list (* heads seq) seq)
                        (funcall g64 (* heads seq)))
                  (list 'attn-context '(54 7 12) (list seq qdim heads kv-heads)
                        (funcall g64 (* seq qdim))))))
         (disps
          (append
          (list
           ;; a = RMSNorm(x) * ln1g
           (list 'rmsnorm-istd '(0 1) (list seq dim) (funcall g64 seq))
           (list 'rmsnorm-fwd '(0 1 17 2) (list seq dim) (funcall g64 (* seq dim)))
           ;; quantize it once; q, k and v all read the same packed activation
           (list 'pack-act-rows '(2 3 4) (list seq dim ng) (funcall g64 seq))
           ;; q = Wq.a, k = Wk.a, v = Wv.a
           (list 'bitlinear-dp4a-rows '(3 20 21 22 4 5)
                 (list seq (plist-get wq :rows) ng)
                 (funcall g64 (* seq (plist-get wq :rows))))
           (list 'bitlinear-dp4a-rows '(3 23 24 25 4 6)
                 (list seq (plist-get wk :rows) ng)
                 (funcall g64 (* seq (plist-get wk :rows))))
           (list 'bitlinear-dp4a-rows '(3 26 27 28 4 7)
                 (list seq (plist-get wv :rows) ng)
                 (funcall g64 (* seq (plist-get wv :rows))))
           ;; QK-norm, then the half-split rotation
           (list 'rmsnorm-heads '(5 18 8) (list (* seq heads) 1 hd)
                 (funcall g64 (* seq heads)))
           (list 'rmsnorm-heads '(6 19 9) (list (* seq kv-heads) 1 hd)
                 (funcall g64 (* seq kv-heads)))
           (list 'rope-half '(8 10) (list seq heads hd
                                          (nelisp-gpu--f32-bits (float rbase))
                                          (nelisp-gpu--f32-bits 1.0))
                 (funcall g64 (* seq heads (/ hd 2))))
           (list 'rope-half '(9 11) (list seq kv-heads hd
                                          (nelisp-gpu--f32-bits (float rbase))
                                          (nelisp-gpu--f32-bits 1.0))
                 (funcall g64 (* seq kv-heads (/ hd 2)))))
          attn
          (list
           ;; causal GQA, then the output projection and the residual
           (list 'pack-act-rows '(12 13 14) (list seq qdim qng) (funcall g64 seq))
           (list 'bitlinear-dp4a-rows '(13 29 30 31 14 15)
                 (list seq (plist-get wo :rows) qng)
                 (funcall g64 (* seq (plist-get wo :rows))))
           (list 'add2 '(0 15 16) (list (* seq dim))
                 (funcall g64 (* seq dim)))
           ;; b = RMSNorm(x1) * ln2g
           (list 'rmsnorm-istd '(16 32) (list seq dim) (funcall g64 seq))
           (list 'rmsnorm-fwd '(16 32 43 33) (list seq dim)
                 (funcall g64 (* seq dim)))
           (list 'pack-act-rows '(33 34 35) (list seq dim ng) (funcall g64 seq))
           ;; the SwiGLU: gate and up from the same packed activation
           (list 'bitlinear-dp4a-rows '(34 44 45 46 35 36)
                 (list seq ff ng) (funcall g64 (* seq ff)))
           (list 'bitlinear-dp4a-rows '(34 47 48 49 35 37)
                 (list seq ff ng) (funcall g64 (* seq ff)))
           (list 'silu-mul '(36 37 38) (list (* seq ff)) (funcall g64 (* seq ff)))
           ;; down, and the second residual
           (list 'pack-act-rows '(38 39 40) (list seq ff fng) (funcall g64 seq))
           (list 'bitlinear-dp4a-rows '(39 50 51 52 40 41)
                 (list seq (plist-get wd :rows) fng)
                 (funcall g64 (* seq (plist-get wd :rows))))
           (list 'add2 '(16 41 42) (list (* seq dim))
                 (funcall g64 (* seq dim)))))))
    (car (nelisp-gpu-server-batch slots disps))))


;;;###autoload
(defun nl-llm-wfuse-open-model (wts &optional nlayers)
  "Upload every layer once and return a list of layer plists.
A block's arithmetic is a fraction of what uploading its weights costs, so
opening and closing a layer per block hides the fusion entirely -- measured
end to end that way, fused and unfused blocks came out 1.1x apart while one
block alone is 5.4x.  The upload has to happen once, like everything else in
this file."
  (let* ((cfg (nl-llm-weights-config wts))
         (n (min (or nlayers (plist-get cfg :layers)) (plist-get cfg :layers)))
         (out nil))
    (dotimes (ly n) (push (nl-llm-wfuse-open-layer wts ly) out))
    (nreverse out)))

;;;###autoload
(defun nl-llm-wfuse-close-model (layers)
  "Free every layer in LAYERS."
  (dolist (lay layers) (nl-llm-wfuse-close-layer lay)))

;;;###autoload
(defun nl-llm-wfuse-run (layers wts tokens)
  "Run TOKENS through resident LAYERS; return the post-final-norm hidden state."
  (let* ((cfg (nl-llm-weights-config wts))
         (dim (plist-get cfg :dim))
         (seq (length tokens))
         (x (make-vector (* seq dim) 0.0))
         (i 0))
    (dolist (tk tokens)
      (let ((row (nl-llm-weights-embed wts tk)))
        (dotimes (t0 dim) (aset x (+ (* i dim) t0) (aref row t0))))
      (setq i (1+ i)))
    (dolist (lay layers) (setq x (nl-llm-wfuse-block lay x seq)))
    (nl-llm-wf-final-norm wts x seq)))

;;;###autoload
(defun nl-llm-wfuse-next-token (wts tokens &optional nlayers progress)
  "Greedy next token for TOKENS with each block run as one batch.
Returns (ID . LOGIT).  Layers are opened, used and closed one at a time, so
peak residency is one layer rather than the whole model -- the same shape as
`nl-llm-wgpu-next-token', which this is meant to be compared against.

The head is scored on the CPU here, exactly as that function does, so the
difference between the two is the blocks and nothing else."
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
      (let ((lay (nl-llm-wfuse-open-layer wts ly)))
        (unwind-protect
            (setq x (nl-llm-wfuse-block lay x seq))
          (nl-llm-wfuse-close-layer lay)))
      (when progress (funcall progress ly)))
    (let ((final (nl-llm-wf-final-norm wts x seq)))
      (nl-llm-wf-argmax (nl-llm-wf-logits-all wts final seq (1- seq))))))


;;;###autoload
(defun nl-llm-wfuse-block-backward (lay tape x dout seq)
  "Gradient of one block for output gradient DOUT, as one batch; return DX.
TAPE is what `nl-llm-wfuse-block' wrote, X the block's input.  Base only: the
adapter's own gradients are not computed here, so this is the frozen model's
dL/dx and is what `nl-llm-wb-block-backward' returns as its car with no LoRAs
attached.

Staged the way the CPU version is, and for the same reason: within a position
the feed-forward's gradient runs wd, then the SwiGLU's vjp, then wg and wu, so
the batch for wd has to complete before wg has an input.  A single batch can
express that -- a memory barrier sits between consecutive dispatches -- which
is why it is one call and not four."
  (let* ((cfg (plist-get lay :cfg))
         (dim (plist-get cfg :dim))
         (heads (plist-get cfg :heads))
         (kv-heads (plist-get cfg :kv-heads))
         (hd (plist-get cfg :head-dim))
         (rbase (plist-get cfg :rope-base))
         (qdim (* heads hd)) (kvdim (* kv-heads hd))
         (wq (nl-llm-wfuse--lin lay :wq)) (wk (nl-llm-wfuse--lin lay :wk))
         (wv (nl-llm-wfuse--lin lay :wv)) (wo (nl-llm-wfuse--lin lay :wo))
         (wg (nl-llm-wfuse--lin lay :wg)) (wu (nl-llm-wfuse--lin lay :wu))
         (wd (nl-llm-wfuse--lin lay :wd))
         (ff (plist-get wg :rows))
         (ng (/ dim 4)) (qng (/ qdim 4)) (fng (/ ff 4))
         (ns (* heads seq seq))
         (g64 (lambda (n) (/ (+ n 63) 64)))
         (tp (lambda (k) (let ((h (plist-get tape k))) (list 'res (car h) (cdr h)))))
         (rb (nelisp-gpu--f32-bits (float rbase)))
         (slots
          (list (cons 'in dout)                          ; 0  DOUT
                (cons 'in x)                             ; 1  X
                (funcall tp :x1)                         ; 2  X1
                (funcall tp :istd1)                      ; 3  istd1
                (funcall tp :istd2)                      ; 4  istd2
                (funcall tp :qpre)                       ; 5  Qpre
                (funcall tp :kpre)                       ; 6  Kpre
                (funcall tp :q)                          ; 7  Q
                (funcall tp :k)                          ; 8  K
                (funcall tp :v)                          ; 9  V
                (funcall tp :p)                          ; 10 P
                (funcall tp :g)                          ; 11 G
                (funcall tp :u)                          ; 12 U
                (list 'res (car (plist-get lay :ln1g)) dim)   ; 13
                (list 'res (car (plist-get lay :ln2g)) dim)   ; 14
                (list 'res (car (plist-get lay :qnorm)) hd)   ; 15
                (list 'res (car (plist-get lay :knorm)) hd)   ; 16
                (cons 'tmp (* seq ff))                   ; 17 DH
                (cons 'tmp (* seq ff))                   ; 18 DG
                (cons 'tmp (* seq ff))                   ; 19 DU
                (cons 'tmp (* seq dim))                  ; 20 DB
                (cons 'tmp (* seq dim))                  ; 21 DBu
                (cons 'tmp (* seq dim))                  ; 22 DX1
                (cons 'tmp (* seq qdim))                 ; 23 DCTX
                (cons 'tmp ns)                           ; 24 DP
                (cons 'tmp ns)                           ; 25 DS
                (cons 'tmp (* seq qdim))                 ; 26 DQ
                (cons 'tmp (* seq kvdim))                ; 27 DK
                (cons 'tmp (* seq kvdim))                ; 28 DV
                (cons 'tmp (* seq qdim))                 ; 29 DQr
                (cons 'tmp (* seq kvdim))                ; 30 DKr
                (cons 'tmp (* seq qdim))                 ; 31 DQn
                (cons 'tmp (* seq kvdim))                ; 32 DKn
                (cons 'tmp (* seq dim))                  ; 33 DA
                (cons 'tmp (* seq dim))                  ; 34 DAq
                (cons 'tmp (* seq dim))                  ; 35 DAv
                (cons 'tmp (* seq dim))                  ; 36 DXa
                (cons 'out (* seq dim))                  ; 37 DX
                (list 'res (plist-get wd :w) (* (plist-get wd :rows) fng)) ; 38
                (list 'res (plist-get wd :s) (plist-get wd :rows))         ; 39
                (list 'res (plist-get wg :w) (* ff ng))                    ; 40
                (list 'res (plist-get wg :s) ff)                           ; 41
                (list 'res (plist-get wu :w) (* ff ng))                    ; 42
                (list 'res (plist-get wu :s) ff)                           ; 43
                (list 'res (plist-get wo :w) (* (plist-get wo :rows) qng)) ; 44
                (list 'res (plist-get wo :s) (plist-get wo :rows))         ; 45
                (list 'res (plist-get wq :w) (* (plist-get wq :rows) ng))  ; 46
                (list 'res (plist-get wq :s) (plist-get wq :rows))         ; 47
                (list 'res (plist-get wk :w) (* (plist-get wk :rows) ng))  ; 48
                (list 'res (plist-get wk :s) (plist-get wk :rows))         ; 49
                (list 'res (plist-get wv :w) (* (plist-get wv :rows) ng))  ; 50
                (list 'res (plist-get wv :s) (plist-get wv :rows))         ; 51
                ;; The per-head inverse standard deviations QK-norm's vjp
                ;; needs.  They are NOT the tape's :istd1 and :istd2, which
                ;; are per *row* and one sixteenth the length -- writing these
                ;; there overruns the buffer, which is what the first version
                ;; of this did.  It returned numbers.
                (cons 'tmp (* seq heads))                                  ; 52
                (cons 'tmp (* seq kv-heads))))                             ; 53
         (disps
          (list
           ;; the feed-forward half
           (list 'dp4a-rows-t '(38 39 0 17) (list (plist-get wd :rows) ff fng seq)
                 (funcall g64 (* seq ff)))
           (list 'silu-mul-bwd '(17 11 12 18 19) (list (* seq ff))
                 (funcall g64 (* seq ff)))
           (list 'dp4a-rows-t '(40 41 18 20) (list ff dim ng seq)
                 (funcall g64 (* seq dim)))
           (list 'dp4a-rows-t '(42 43 19 21) (list ff dim ng seq)
                 (funcall g64 (* seq dim)))
           (list 'add2 '(20 21 33) (list (* seq dim)) (funcall g64 (* seq dim)))
           ;; through the second norm, and the residual into x1
           (list 'rmsnorm-dx '(33 2 4 14 22) (list seq dim) (funcall g64 seq))
           (list 'add2 '(22 0 36) (list (* seq dim)) (funcall g64 (* seq dim)))
           ;; the attention half: wo, then the vjps, then the rotation and
           ;; QK-norm in reverse
           (list 'dp4a-rows-t '(44 45 36 23) (list (plist-get wo :rows) qdim qng seq)
                 (funcall g64 (* seq qdim)))
           (list 'attn-ctx-dp '(23 9 24) (list seq qdim heads kv-heads)
                 (funcall g64 ns))
           (list 'attn-ctx-dv '(23 10 28) (list seq qdim heads kv-heads)
                 (funcall g64 (* seq kvdim)))
           (list 'softmax-bwd '(10 24 25) (list (* heads seq) seq)
                 (funcall g64 (* heads seq)))
           (list 'attn-sc-dq '(25 8 26) (list seq qdim heads kv-heads)
                 (funcall g64 (* seq qdim)))
           (list 'attn-sc-dk '(25 7 27) (list seq qdim heads kv-heads)
                 (funcall g64 (* seq kvdim)))
           (list 'rope-half '(26 29) (list seq heads hd rb
                                           (nelisp-gpu--f32-bits -1.0))
                 (funcall g64 (* seq heads (/ hd 2))))
           (list 'rope-half '(27 30) (list seq kv-heads hd rb
                                           (nelisp-gpu--f32-bits -1.0))
                 (funcall g64 (* seq kv-heads (/ hd 2))))
           (list 'rmsnorm-istd '(5 52) (list (* seq heads) hd)
                 (funcall g64 (* seq heads)))
           (list 'rmsnorm-dx '(29 5 52 15 31) (list (* seq heads) hd)
                 (funcall g64 (* seq heads)))
           (list 'rmsnorm-istd '(6 53) (list (* seq kv-heads) hd)
                 (funcall g64 (* seq kv-heads)))
           (list 'rmsnorm-dx '(30 6 53 16 32) (list (* seq kv-heads) hd)
                 (funcall g64 (* seq kv-heads)))
           ;; back through the three projections into the normalised input
           (list 'dp4a-rows-t '(46 47 31 34) (list (plist-get wq :rows) dim ng seq)
                 (funcall g64 (* seq dim)))
           (list 'dp4a-rows-t '(48 49 32 35) (list (plist-get wk :rows) dim ng seq)
                 (funcall g64 (* seq dim)))
           (list 'add2 '(34 35 33) (list (* seq dim)) (funcall g64 (* seq dim)))
           (list 'dp4a-rows-t '(50 51 28 34) (list (plist-get wv :rows) dim ng seq)
                 (funcall g64 (* seq dim)))
           (list 'add2 '(33 34 35) (list (* seq dim)) (funcall g64 (* seq dim)))
           ;; the first norm, and the residual into x.  istd1 comes from the
           ;; tape rather than being recomputed, which it now can because
           ;; nothing has overwritten it.
           (list 'rmsnorm-dx '(35 1 3 13 33) (list seq dim) (funcall g64 seq))
           (list 'add2 '(33 36 37) (list (* seq dim))
                 (funcall g64 (* seq dim))))))
    (car (nelisp-gpu-server-batch slots disps))))

(provide 'nl-llm-weights-fused)
;;; nl-llm-weights-fused.el ends here
