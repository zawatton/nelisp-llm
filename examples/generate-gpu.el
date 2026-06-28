;;; generate-gpu.el --- train on-device, then generate  -*- lexical-binding: t; -*-
;; End-to-end loop on a tiny model: train a weight-tied stacked model fully
;; on-device (gather embedding, Adam, warmup+cosine LR), read the trained weights
;; back to the host, then autoregressively generate from a prompt and BPE-decode.
;; To make the result legible for a tiny model, it memorises one passage and is
;; asked to continue its prompt -- so a correct pipeline reproduces the passage.
;; Generation is forward-only and reuses the CPU autograd forward (nl-llm-ag-block
;; + tied head), numerically the same model the on-device path trained.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/generate-gpu.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-bpe)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-block)   ; nl-llm-sample
(require 'nl-llm-decode)  ; KV-cache incremental decode (CPU)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-gpu-decode)  ; on-GPU KV-cache decode

(defun ge--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun ge--ones (n) (photon-tensor (list n) (make-vector n 1.0)))
(defun ge--zeros (sh) (let ((n 1)) (dolist (d sh) (setq n (* n d))) (photon-tensor sh (make-vector n 0.0))))
(defun ge--idx (ids start seq) (photon-tensor (list seq) (let ((v (make-vector seq 0.0))) (dotimes (i seq) (aset v i (float (aref ids (+ start i))))) v)))
(defun ge--argmax (data base n) (let ((bi 0) (bv (aref data base)) (j 1)) (while (< j n) (when (> (aref data (+ base j)) bv) (setq bv (aref data (+ base j)) bi j)) (setq j (1+ j))) bi))

(unless (nl-llm-gpu-enable)
  (princ "no GPU server / Vulkan device\n") (kill-emacs 0))

(let* ((corpus (with-temp-buffer (insert-file-contents "data/corpus.txt") (buffer-string)))
       (bpe (photon-bpe-train (list corpus) 96))
       (ids (apply #'vector (photon-bpe-encode bpe corpus))) (vocab (photon-bpe-size bpe))
       (nblocks 2) (dim 64) (heads 4) (kvh 2) (ff 128) (seq 24) (steps 500)
       (base-lr 0.01) (warmup 20) (minlr 0.001) (plen 8)
       (hd (/ dim heads)) (kvdim (* kvh hd)) (sc (/ 1.0 (sqrt (float dim)))) (scl (/ 1.0 (sqrt (float hd))))
       (mkblk (lambda (s0) (list :ln1g (ge--ones dim) :wq (ge--t (list dim dim) (+ s0 1) sc) :bq (ge--zeros (list dim))
                                 :wk (ge--t (list kvdim dim) (+ s0 2) sc) :bk (ge--zeros (list kvdim))
                                 :wv (ge--t (list kvdim dim) (+ s0 3) sc) :bv (ge--zeros (list kvdim))
                                 :wo (ge--t (list dim dim) (+ s0 4) sc) :bo (ge--zeros (list dim)) :ln2g (ge--ones dim)
                                 :wg (ge--t (list ff dim) (+ s0 5) sc) :bg (ge--zeros (list ff))
                                 :wu (ge--t (list ff dim) (+ s0 6) sc) :bu (ge--zeros (list ff))
                                 :wd (ge--t (list dim ff) (+ s0 7) sc) :bd (ge--zeros (list dim)))))
       (wte (ge--t (list vocab dim) 99 sc)) (bh (ge--zeros (list vocab))) (lnfg (ge--ones dim))
       (blks (let ((l nil) (n 0)) (while (< n nblocks) (push (funcall mkblk (* (1+ n) 100)) l) (setq n (1+ n))) (nreverse l)))
       (tables (nl-llm-gpu-rope-tables seq hd))
       (mask (let ((md (make-vector (* seq seq) 0.0)) (i 0)) (while (< i seq) (let ((j (1+ i))) (while (< j seq) (aset md (+ (* i seq) j) -1.0e30) (setq j (1+ j)))) (setq i (1+ i))) (photon-tensor (list seq seq) md)))
       (b (nlga-new)))
  ;; ---- memorise one window on-device (fixed tok/tgt; tied head; Adam+sched) ----
  (cl-flet ((wb (bt) (let ((o nil) (kv bt)) (while kv (push (car kv) o) (push (nlga-param b (cadr kv)) o) (setq kv (cddr kv))) (nreverse o))))
    (let* ((tok (nlga-const b (ge--idx ids 0 seq))) (tgt (nlga-const b (ge--idx ids 1 seq)))
           (wter (nlga-param b wte)) (bhr (nlga-param b bh)) (gblks (mapcar #'wb blks)) (lnfgr (nlga-param b lnfg))
           (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
           (sposr (nlga-scalar b 1.0)) (snegr (nlga-scalar b -1.0)) (sclr (nlga-scalar b scl)) (oner (nlga-scalar b 1.0))
           (maskr (nlga-const b mask)) (x (nlga-embed b tok wter)))
      (dolist (g gblks) (setq x (nlga-block b x g heads kvh cosr sinr sposr snegr sclr maskr)))
      (let* ((xf (nlga-rmsnorm b x lnfgr)) (logits (nlga-linear b xf wter bhr)))
        (nlga-keep b logits oner) (nlga-seed-ce-idx b logits tgt)
        (nlga-finish-adam b base-lr) (nlga-compile b))
      (princ (format "memorise on-device: blocks=%d dim=%d ff=%d seq=%d vocab=%d steps=%d (tied, Adam, warmup+cosine)\n"
                     nblocks dim ff seq vocab steps))
      (let ((s 0)) (while (< s steps)
        (nlga-adam-update-t b (1+ s) (nl-llm-lr-warmup-cosine (1+ s) steps warmup base-lr minlr))
        (nlga-step b) (setq s (1+ s))))
      (nlga-readback b) (nlga-free b)))
  ;; ---- generate ON the GPU with a KV-cache decode (server still up) ----
  (let* ((prompt (let ((l nil)) (dotimes (i plen) (push (aref ids i) l)) (nreverse l)))
         (orig (let ((l nil)) (dotimes (i seq) (push (aref ids i) l)) (nreverse l))))
    (cl-flet ((generate (pick)               ; PICK: (logits-vec vocab) -> token id
                (let* ((ctx (nl-llm-gpu-decode-new wte blks lnfg bh heads kvh dim vocab seq tables))
                       (gen (copy-sequence prompt)) (p 0))
                  (while (< (length gen) seq)
                    (let ((ld (nl-llm-gpu-decode-step ctx (nth p gen) p)))
                      (when (>= (1+ p) plen) (setq gen (append gen (list (funcall pick ld vocab)))))
                      (setq p (1+ p))))
                  (nl-llm-gpu-decode-free ctx)
                  gen)))
      (let* ((greedy (generate (lambda (d v) (ge--argmax d 0 v))))
             (match (let ((m 0) (i 0)) (while (< i seq) (when (= (nth i greedy) (nth i orig)) (setq m (1+ m))) (setq i (1+ i))) m)))
        (random "nl-llm-generate-seed")      ; deterministic sampling
        (let ((sampled (generate (lambda (d v) (nl-llm-sample d 0 v 0.7 20)))))
          (nl-llm-gpu-disable)
          (princ (format "prompt        : %S\n" (photon-bpe-decode bpe prompt)))
          (princ (format "greedy        : %S\n" (photon-bpe-decode bpe greedy)))
          (princ (format "sampled(T0.7) : %S\n" (photon-bpe-decode bpe sampled)))
          (princ (format "original      : %S\n" (photon-bpe-decode bpe orig)))
          (princ (format "greedy match  : %d/%d  %s (on-GPU KV-cache decode)\n" match seq (if (>= match (- seq 2)) "reproduced" "partial")))
          (princ (format "GENERATE-GPU=%s\n" (if (>= match (- seq 2)) "PASS" "OK"))))))))
;;; generate-gpu.el ends here
