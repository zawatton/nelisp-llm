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

;;;###autoload
(defun nl-llm-wfuse-block (lay x seq)
  "Run a whole block over X (SEQ x dim) as one batch; return its output.
Every intermediate -- the normalised activations, their packed forms, q, k, v,
the rotated forms, the context, the SwiGLU halves -- stays on the device.  The
answer crosses the boundary once.

Equal to `nl-llm-wb-block-forward' under `nl-llm-wgpu-with-linears' to about
4e-07, which is the same W8A8 arithmetic with the glue in f32 rather than f64.
It is not equal to the f32 CPU block and does not claim to be: the activation
quantization those two differ by is measured separately."
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
                      (cons 'tmp seq)                    ; 1  istd
                      (cons 'tmp (* seq dim))            ; 2  A
                      (cons 'tmp (* seq ng))             ; 3  AQ
                      (cons 'tmp seq)                    ; 4  AG
                      (cons 'tmp (* seq qdim))           ; 5  Q
                      (cons 'tmp (* seq kvdim))          ; 6  K
                      (cons 'tmp (* seq kvdim))          ; 7  V
                      (cons 'tmp (* seq qdim))           ; 8  Qn
                      (cons 'tmp (* seq kvdim))          ; 9  Kn
                      (cons 'tmp (* seq qdim))           ; 10 Qr
                      (cons 'tmp (* seq kvdim))          ; 11 Kr
                      (cons 'tmp (* seq qdim))           ; 12 CTX
                      (cons 'tmp (* seq qng))            ; 13 CQ
                      (cons 'tmp seq)                    ; 14 CG
                      (cons 'tmp (* seq dim))            ; 15 O
                      (cons 'tmp (* seq dim))            ; 16 X1
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
                      (cons 'tmp seq)                    ; 32 istd2
                      (cons 'tmp (* seq dim))            ; 33 B
                      (cons 'tmp (* seq ng))             ; 34 BQ
                      (cons 'tmp seq)                    ; 35 BG
                      (cons 'tmp (* seq ff))             ; 36 G
                      (cons 'tmp (* seq ff))             ; 37 U
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
                      (list 'res (plist-get wd :s) (plist-get wd :rows))))
         (disps
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
                                          (nelisp-gpu--f32-bits (float rbase)))
                 (funcall g64 (* seq heads (/ hd 2))))
           (list 'rope-half '(9 11) (list seq kv-heads hd
                                          (nelisp-gpu--f32-bits (float rbase)))
                 (funcall g64 (* seq kv-heads (/ hd 2))))
           ;; causal GQA, then the output projection and the residual
           (list 'attn-causal-gqa '(10 11 7 12) (list seq heads kv-heads hd)
                 (funcall g64 (* heads seq)))
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
                 (funcall g64 (* seq dim))))))
    (car (nelisp-gpu-server-batch slots disps))))

(provide 'nl-llm-weights-fused)
;;; nl-llm-weights-fused.el ends here
