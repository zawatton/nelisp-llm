;;; gpu-test.el --- GPU backend correctness for the modern block  -*- lexical-binding: t; -*-
;; Verifies the GPU wiring is *correct*, not just fast:
;;   A. resident-weight invalidation (the training-safety fix)
;;   B. modern model forward matches the CPU result (inference)
;;   C. the modern block (GQA + MoE) actually trains on the GPU (loss falls)
;; Skips cleanly (exit 0) when no Vulkan device / server is available.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-arch)
(require 'nl-llm-attn)
(require 'nl-llm-moe)
(require 'nl-llm-block)
(require 'nl-llm-autograd)
(require 'nl-llm-gpu)

(defvar g--fail 0)
(defun g--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name
                 (if ok "PASS" (progn (setq g--fail (1+ g--fail)) "FAIL")) (or extra ""))))
(defun g--t (shape seed sc)
  (let ((n 1)) (dolist (d shape) (setq n (* n d)))
    (photon-tensor-from-list
     shape
     (mapcar (lambda (i) (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536))
                                         65536.0) 0.5)))
             (number-sequence 0 (1- n))))))
(defun g--ones (n) (photon-tensor-create (list n) 1.0))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU SKIP (no GPU server / Vulkan device)\n")
  (kill-emacs 0))

;; --- A. resident-weight invalidation -------------------------------------
;; y = x * w with a 1x1 weight.  Mutate w in place: without invalidation the
;; GPU keeps the first upload (stale); invalidation refreshes it.
(let* ((w (photon-tensor (list 1 1) (vector 2.0)))
       (x (photon-tensor (list 1 1) (vector 3.0)))
       (y0 (aref (photon-tensor-data (photon-tensor-linear x w)) 0)))   ; 3*2 = 6
  (aset (photon-tensor-data w) 0 5.0)                                   ; mutate in place
  (let ((y-stale (aref (photon-tensor-data (photon-tensor-linear x w)) 0)))
    (nelisp-gpu-server-invalidate (photon-tensor-data w))
    (let ((y-fresh (aref (photon-tensor-data (photon-tensor-linear x w)) 0)))  ; 3*5 = 15
      (g--ck "gpu resident: initial upload"        (< (abs (- y0 6.0)) 1e-3)  (format "y=%.3f" y0))
      (g--ck "gpu resident: stale before refresh"  (< (abs (- y-stale 6.0)) 1e-3) (format "y=%.3f (cached)" y-stale))
      (g--ck "gpu resident: refreshed by invalidate" (< (abs (- y-fresh 15.0)) 1e-3) (format "y=%.3f" y-fresh)))))

;; --- B. forward (inference) equivalence: GPU == CPU ----------------------
;; SwiGLU FFN block (no MoE routing) so the result is selection-deterministic.
(let* ((vocab 12) (dim 16) (ff 32) (heads 4) (kvh 2)
       (hd (/ dim heads)) (kvdim (* kvh hd)) (sc (/ 1.0 (sqrt (float dim))))
       (tokens '(1 2 3 4 5 6 7 8))
       (block (list :ln1g (g--ones dim) :wq (g--t (list dim dim) 2 sc)
                    :wk (g--t (list kvdim dim) 3 sc) :wv (g--t (list kvdim dim) 4 sc)
                    :wo (g--t (list dim dim) 5 sc) :ln2g (g--ones dim)
                    :wg (g--t (list ff dim) 6 sc) :wu (g--t (list ff dim) 7 sc)
                    :wd (g--t (list dim ff) 8 sc)))
       (model (list :wte (g--t (list vocab dim) 1 sc) :blocks (list block) :lnf (g--ones dim)
                    :head (g--t (list vocab dim) 9 sc) :dim dim :heads heads :kv-heads kvh)))
  (let ((gpu (photon-tensor-data (nl-llm-model-forward model tokens))))
    (photon-tensor-use-cpu-backend)
    (let* ((cpu (photon-tensor-data (nl-llm-model-forward model tokens)))
           (n (length cpu)) (maxrel 0.0) (i 0))
      (while (< i n)
        (setq maxrel (max maxrel (/ (abs (- (aref gpu i) (aref cpu i)))
                                    (max 1e-4 (abs (aref cpu i))))))
        (setq i (1+ i)))
      (g--ck "modern model forward: GPU == CPU" (< maxrel 5e-3) (format "maxrel=%.2e" maxrel)))
    (photon-tensor-use-gpu-backend)))

;; --- C. the full modern block (GQA + MoE) trains on the GPU --------------
(cl-flet ((p (shape seed sc) (photon-autograd-const (g--t shape seed sc)))
          (ones (n) (photon-autograd-const (g--ones n)))
          (zeros (n) (photon-autograd-const (photon-tensor-create (list n) 0.0))))
  (let* ((vocab 12) (dim 16) (ff 32) (heads 4) (kvh 2) (ne 3) (topk 2)
         (hd (/ dim heads)) (kvdim (* kvh hd)) (sc (/ 1.0 (sqrt (float dim))))
         (wte (p (list vocab dim) 1 sc)) (ln1g (ones dim))
         (Wq (p (list dim dim) 2 sc)) (bq (zeros dim))
         (Wk (p (list kvdim dim) 3 sc)) (bk (zeros kvdim))
         (Wv (p (list kvdim dim) 4 sc)) (bv (zeros kvdim))
         (Wo (p (list dim dim) 5 sc)) (bo (zeros dim)) (ln2g (ones dim))
         (router (p (list ne dim) 6 sc)) (brouter (zeros ne))
         (experts (let ((ex nil) (e 0))
                    (while (< e ne)
                      (push (list :wg (p (list ff dim) (+ 30 e) sc) :bg (zeros ff)
                                  :wu (p (list ff dim) (+ 40 e) sc) :bu (zeros ff)
                                  :wd (p (list dim ff) (+ 50 e) sc) :bd (zeros dim)) ex)
                      (setq e (1+ e)))
                    (nreverse ex)))
         (lnfg (ones dim)) (Wh (p (list vocab dim) 9 sc)) (bh (zeros vocab))
         (block (list :ln1g ln1g :wq Wq :bq bq :wk Wk :bk bk :wv Wv :bv bv :wo Wo :bo bo
                      :ln2g ln2g :router router :brouter brouter :experts experts :top-k topk))
         (epavs (apply #'append
                       (mapcar (lambda (q) (list (plist-get q :wg) (plist-get q :bg)
                                                 (plist-get q :wu) (plist-get q :bu)
                                                 (plist-get q :wd) (plist-get q :bd)))
                               experts)))
         (params (append (list wte ln1g Wq bq Wk bk Wv bv Wo bo ln2g router brouter)
                         epavs (list lnfg Wh bh)))
         (toks '(1 2 3 4 5 6 7 8)) (tgts (vector 2 3 4 5 6 7 8 9))
         (l0 nil) (ln nil))
    (cl-flet ((fwd ()
                (photon-autograd-reset-tape)
                (let* ((x (photon-autograd-embedding wte toks dim))
                       (h (nl-llm-ag-block x block heads kvh))
                       (xf (nl-llm-ag-rmsnorm h lnfg)))
                  (photon-autograd-softmax-ce (photon-autograd-linear xf Wh bh) tgts))))
      (setq l0 (aref (photon-tensor-data (pav-value (fwd))) 0))
      (let ((step 0))
        (while (< step 10)
          (let ((l (fwd)))
            (photon-autograd-zero-grad params)
            (photon-autograd-backward l)
            (photon-autograd-sgd params 0.3)
            (nl-llm-gpu-invalidate params))   ; refresh GPU weight copies
          (setq step (1+ step))))
      (setq ln (aref (photon-tensor-data (pav-value (fwd))) 0))
      (g--ck "modern block (GQA+MoE) trains on GPU" (< ln (- l0 0.05))
             (format "loss %.4f -> %.4f" l0 ln)))))

(nl-llm-gpu-disable)
(princ (format "NL-LLM-GPU %s (%d failures)\n"
               (if (= g--fail 0) "ALL-PASS" "HAS-FAILURES") g--fail))
(kill-emacs (if (= g--fail 0) 0 1))
;;; gpu-test.el ends here
