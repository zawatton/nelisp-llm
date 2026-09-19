;;; rope-style-test.el --- the donor's rotation convention and QK-norm  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/rope-style-test.el
;;
;; Doc 08 Phase 2c.  Two more conventions where the donor differs from what this
;; repo implemented, both silent:
;;
;; 1. RoPE pairing.  nl-llm--rope-block rotates ADJACENT pairs (2m, 2m+1), the
;;    GPT-J convention.  Qwen3 and Llama rotate HALF-SPLIT pairs (i, i + hd/2),
;;    the GPT-NeoX convention that HuggingFace calls rotate_half.  Same angles,
;;    different pairing; measured on an 8-wide head at position 3 the two differ
;;    by 5.82.  Nothing raises -- the vector is the right length either way.
;;
;; 2. QK-norm.  Qwen3 RMSNorms each head's q and k with a learned per-head-width
;;    gain BEFORE RoPE.  Omit it and the attention scores are simply scaled
;;    wrong, again with no error.
;;
;; As in test/head-dim-test.el the suite carries its own reference rather than
;; comparing two library paths, because a shared assumption is exactly what is
;; under test.  Defaults are unchanged: no :rope-style means the interleaved
;; pairing this repo has always used, and no :q-norm/:k-norm means no QK-norm.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-attn)
(require 'nl-llm-block)

(defvar rs--fail 0)
(defun rs--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq rs--fail (1+ rs--fail)) "FAIL"))
                 (or extra ""))))

(defun rs--mk (rows cols seed)
  (let ((v (make-vector (* rows cols) 0.0)) (i 0))
    (while (< i (* rows cols))
      (aset v i (* 0.11 (- (mod (+ (* (1+ i) 5) seed) 13) 6)))
      (setq i (1+ i)))
    (photon-tensor (list rows cols) v)))

(defun rs--gain (n seed)
  (let ((v (make-vector n 0.0)))
    (dotimes (i n) (aset v i (+ 0.7 (* 0.05 (mod (+ i seed) 5)))))
    (photon-tensor (list n) v)))

(defun rs--maxdiff (a b)
  (let ((ad (photon-tensor-data a)) (bd (photon-tensor-data b)) (m 0.0))
    (if (/= (length ad) (length bd)) 1.0e30
      (dotimes (i (length ad))
        (let ((d (abs (- (aref ad i) (aref bd i))))) (when (> d m) (setq m d))))
      m)))

;;; --- independent reference ------------------------------------------------

(defun rs--rope-ref (vec rowbase nheads hd pos rbase style)
  "Rotate NHEADS blocks of width HD at ROWBASE by RoPE at POS, using STYLE.
`half' pairs (i, i+hd/2) as Qwen3 and Llama do; `interleaved' pairs (2m, 2m+1)."
  (dotimes (h nheads)
    (let* ((base (+ rowbase (* h hd))) (half (/ hd 2))
           (orig (make-vector hd 0.0)))
      (dotimes (i hd) (aset orig i (aref vec (+ base i))))
      (dotimes (i half)
        (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 i) (float hd)))))
               (c (cos theta)) (s (sin theta)))
          (if (eq style 'half)
              (let ((a (aref orig i)) (b (aref orig (+ i half))))
                (aset vec (+ base i) (- (* a c) (* b s)))
                (aset vec (+ base i half) (+ (* b c) (* a s))))
            (let ((a (aref orig (* 2 i))) (b (aref orig (1+ (* 2 i)))))
              (aset vec (+ base (* 2 i)) (- (* a c) (* b s)))
              (aset vec (+ base (1+ (* 2 i))) (+ (* a s) (* b c))))))))))

(defun rs--qknorm-ref (vec rowbase nheads hd gain eps)
  "RMSNorm each of NHEADS blocks of width HD at ROWBASE with GAIN."
  (when gain
    (let ((g (photon-tensor-data gain)))
      (dotimes (h nheads)
        (let ((base (+ rowbase (* h hd))) (ss 0.0))
          (dotimes (i hd)
            (let ((v (aref vec (+ base i)))) (setq ss (+ ss (* v v)))))
          (let ((inv (/ 1.0 (sqrt (+ (/ ss (float hd)) eps)))))
            (dotimes (i hd)
              (aset vec (+ base i)
                    (* (aref vec (+ base i)) inv (aref g i))))))))))

(defun rs--ref-gqa (x layer heads kv-heads hd style &optional rope-base)
  "Reference GQA with an explicit rotation STYLE and optional QK-norm."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (qdim (* heads hd)) (kvdim (* kv-heads hd)) (grp (/ heads kv-heads))
         (rbase (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (eps 1.0e-6)
         (q (copy-sequence
             (photon-tensor-data (photon-tensor-linear x (plist-get layer :wq)))))
         (k (copy-sequence
             (photon-tensor-data (photon-tensor-linear x (plist-get layer :wk)))))
         (v (photon-tensor-data (photon-tensor-linear x (plist-get layer :wv))))
         (ctx (make-vector (* seq qdim) 0.0)))
    (dotimes (i seq)
      ;; QK-norm before the rotation, as Qwen3 applies it.
      (rs--qknorm-ref q (* i qdim) heads hd (plist-get layer :q-norm) eps)
      (rs--qknorm-ref k (* i kvdim) kv-heads hd (plist-get layer :k-norm) eps)
      (rs--rope-ref q (* i qdim) heads hd i rbase style)
      (rs--rope-ref k (* i kvdim) kv-heads hd i rbase style))
    (dotimes (h heads)
      (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((scores (make-vector (1+ i) 0.0)) (mx -1.0e30))
            (dotimes (j (1+ i))
              (let ((acc 0.0))
                (dotimes (t0 hd)
                  (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                      (aref k (+ (* j kvdim) kc t0))))))
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
    (photon-tensor-linear (photon-tensor (list seq qdim) ctx)
                          (plist-get layer :wo))))

(defun rs--layer (dim heads kv-heads hd &rest extra)
  (let ((qdim (* heads hd)) (kvdim (* kv-heads hd)))
    (append extra
            (list :head-dim hd
                  :wq (rs--mk qdim dim 1) :wk (rs--mk kvdim dim 2)
                  :wv (rs--mk kvdim dim 3) :wo (rs--mk dim qdim 4)))))

;;; --- checks ---------------------------------------------------------------

(let* ((dim 8) (seq 3) (heads 2) (kv-heads 1) (hd 6)
       (x (rs--mk seq dim 9))
       (qn (rs--gain hd 1)) (kn (rs--gain hd 2)))

  ;; 1. Regression guard: the default must stay the interleaved pairing.
  (let* ((layer (rs--layer dim heads kv-heads hd))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (rs--ref-gqa x layer heads kv-heads hd 'interleaved)))
    (rs--ck "default rope style is interleaved" (< (rs--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff got want))))

  ;; 2. Asking for interleaved explicitly changes nothing.
  (let* ((a (nl-llm-gqa x (rs--layer dim heads kv-heads hd) heads kv-heads))
         (b (nl-llm-gqa x (rs--layer dim heads kv-heads hd
                                     :rope-style 'interleaved)
                        heads kv-heads)))
    (rs--ck "explicit :rope-style interleaved is a no-op"
            (< (rs--maxdiff a b) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff a b))))

  ;; 3. The donor's pairing.  Red before Phase 2c: the key was ignored.
  (let* ((layer (rs--layer dim heads kv-heads hd :rope-style 'half))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (rs--ref-gqa x layer heads kv-heads hd 'half)))
    (rs--ck "half-split rope == reference" (< (rs--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff got want))))

  ;; 4. The two conventions really do disagree, so check 3 is not vacuous.
  (let* ((li (rs--layer dim heads kv-heads hd :rope-style 'interleaved))
         (lh (rs--layer dim heads kv-heads hd :rope-style 'half))
         (a (nl-llm-gqa x li heads kv-heads))
         (b (nl-llm-gqa x lh heads kv-heads)))
    (rs--ck "the two rope styles differ" (> (rs--maxdiff a b) 1.0e-6)
            (format "maxdiff %.3f" (rs--maxdiff a b))))

  ;; 5. QK-norm, on its own.
  (let* ((layer (rs--layer dim heads kv-heads hd :q-norm qn :k-norm kn))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (rs--ref-gqa x layer heads kv-heads hd 'interleaved)))
    (rs--ck "QK-norm == reference" (< (rs--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff got want))))

  ;; 6. QK-norm must actually do something.
  (let* ((plain (rs--layer dim heads kv-heads hd))
         (normed (rs--layer dim heads kv-heads hd :q-norm qn :k-norm kn))
         (a (nl-llm-gqa x plain heads kv-heads))
         (b (nl-llm-gqa x normed heads kv-heads)))
    (rs--ck "QK-norm changes the result" (> (rs--maxdiff a b) 1.0e-6)
            (format "maxdiff %.3f" (rs--maxdiff a b))))

  ;; 7. Both together: the full Qwen3 attention shape.
  (let* ((layer (rs--layer dim heads kv-heads hd :rope-style 'half
                           :q-norm qn :k-norm kn))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (rs--ref-gqa x layer heads kv-heads hd 'half)))
    (rs--ck "half-split + QK-norm == reference"
            (< (rs--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff got want))))

  ;; 8. The cached decode path must agree with the full forward under both.
  (let* ((layer (rs--layer dim heads kv-heads hd :rope-style 'half
                           :q-norm qn :k-norm kn))
         (full (nl-llm-gqa x layer heads kv-heads))
         (cached (nl-llm-gqa-cached x layer heads kv-heads)))
    (rs--ck "cached == full under half-split + QK-norm"
            (< (rs--maxdiff full cached) 1.0e-12)
            (format "maxdiff %.2e" (rs--maxdiff full cached))))

  ;; 9. An unknown style is a typo, not a silent fallback to the default.
  (rs--ck "an unknown :rope-style signals"
          (condition-case err
              (progn (nl-llm-gqa x (rs--layer dim heads kv-heads hd
                                              :rope-style 'neox)
                                 heads kv-heads)
                     nil)
            (error (and (string-match-p "rope-style" (error-message-string err))
                        t))))

  ;; 10. The paths that do not implement these keys must refuse them, the same
  ;; contract :head-dim established.
  (require 'nl-llm-stream)
  (rs--ck "nl-llm-stream-block refuses :rope-style half"
          (condition-case err
              (progn (nl-llm-stream-block
                      (rs--mk 1 dim 11)
                      (rs--layer dim heads kv-heads (/ dim heads)
                                 :rope-style 'half)
                      (nl-llm-scache-new 1 2 dim heads kv-heads))
                     nil)
            (error (and (string-match-p "rope-style\\|head-dim"
                                        (error-message-string err))
                        t)))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop rs--fail) "rope-style OK" "rope-style") rs--fail))
(when (> rs--fail 0) (kill-emacs 1))
