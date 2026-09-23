;;; bench-native-inference.el --- source vs byte-compiled native-agent decode  -*- lexical-binding: t; -*-

;; Reproduce the CPU cost of feeding a complete rendered agent history through
;; the pure-Elisp KV decoder.  This deliberately does not use the GPU/native
;; backends: "compiled" below means selected numeric Elisp functions compiled
;; in memory with `byte-compile', with no .elc written to the checkout.
;;
;;   emacs -Q --batch -l examples/bench-native-inference.el

;;; Code:

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'bytecomp)

(defconst bninf--examples-dir
  (file-name-directory (or load-file-name buffer-file-name)))
(defconst bninf--repo-dir
  (file-name-as-directory (expand-file-name ".." bninf--examples-dir)))
(add-to-list 'load-path (expand-file-name "lisp" bninf--repo-dir))

;; Load the measured implementation from source even if a developer happens to
;; have stale .elc files elsewhere on `load-path'.
(load (expand-file-name "../nelisp-photon/lisp/photon-tensor.el"
                        bninf--repo-dir)
      nil nil t)
(load (expand-file-name "lisp/nl-llm-arch.el" bninf--repo-dir) nil nil t)
(load (expand-file-name "lisp/nl-llm-attn.el" bninf--repo-dir) nil nil t)
(load (expand-file-name "lisp/nl-llm-decode.el" bninf--repo-dir) nil nil t)
(load (expand-file-name "lisp/nl-llm-agent-model.el" bninf--repo-dir)
      nil nil t)

(defvar nl-llm-agent-char-vocab)
(defvar nl-llm-agent-model-inference-mode)
(defvar nl-llm-inference-runtime--dependencies)
(declare-function photon-tensor "photon-tensor" (shape data))
(declare-function nl-llm-agent--char->id "nl-llm-agent-model" (character))
(declare-function nl-llm-agent--render "nl-llm-agent-model" (messages))
(declare-function nl-llm-agent-grammar-message "nl-llm-agent-model"
                  (n &optional allow))
(declare-function nl-llm-agent-model-policy "nl-llm-agent-model"
                  (model grammar &optional maxseq))
(declare-function nl-llm-agent-model-step-fn "nl-llm-agent-model"
                  (model caches))
(declare-function nl-llm-dcache-new "nl-llm-decode"
                  (max-seq dim heads kvh))
(declare-function nl-llm-inference-runtime-prepare "nl-llm-inference-runtime"
                  (&optional mode))

(defconst bninf--kernel-functions
  '(photon-tensor-linear photon-tensor-add photon-tensor-hadamard
    nl-llm-rmsnorm nl-llm-silu)
  "Dense projection, normalization, and SwiGLU element kernels.")

(defconst bninf--decode-functions
  '(nl-llm--rope-block nl-llm--rope-heads nl-llm--swiglu-b
    nl-llm-decode-block nl-llm-decode-step)
  "RoPE and the history-dependent incremental attention path.")

(defconst bninf--all-hot-functions
  (append bninf--kernel-functions bninf--decode-functions))

(defun bninf--tensor (shape seed scale)
  "Return a deterministic raw Photon tensor with SHAPE, SEED, and SCALE."
  (let ((size 1))
    (dolist (dimension shape)
      (setq size (* size dimension)))
    (photon-tensor
     shape
     (let ((data (make-vector size 0.0))
           (i 0))
       (while (< i size)
         (aset data i
               (* scale 2.0
                  (- (/ (float
                         (mod (+ (* (1+ i) 2654435761)
                                 (* (1+ seed) 40503))
                              65536))
                        65536.0)
                     0.5)))
         (setq i (1+ i)))
       data))))

(defun bninf--constant (n value)
  "Return a raw length-N Photon tensor filled with VALUE."
  (photon-tensor (list n) (make-vector n value)))

(defun bninf--raw-p5-model (dim ff blocks heads)
  "Build a deterministic artifact-shaped raw P5 inference model.
The embedding and output head are independent tensors, as in promoted P5
artifacts.  DIM must be divisible by HEADS and have an even head dimension."
  (let* ((vocab nl-llm-agent-char-vocab)
         (head-dim (/ dim heads))
         (scale (/ 1.0 (sqrt (float dim))))
         (make-block
          (lambda (seed)
            (list
             :ln1g (bninf--constant dim 1.0)
             :wq (bninf--tensor (list dim dim) (+ seed 1) scale)
             :bq (bninf--constant dim 0.0)
             :wk (bninf--tensor (list dim dim) (+ seed 2) scale)
             :bk (bninf--constant dim 0.0)
             :wv (bninf--tensor (list dim dim) (+ seed 3) scale)
             :bv (bninf--constant dim 0.0)
             :wo (bninf--tensor (list dim dim) (+ seed 4) scale)
             :bo (bninf--constant dim 0.0)
             :ln2g (bninf--constant dim 1.0)
             :wg (bninf--tensor (list ff dim) (+ seed 5) scale)
             :bg (bninf--constant ff 0.0)
             :wu (bninf--tensor (list ff dim) (+ seed 6) scale)
             :bu (bninf--constant ff 0.0)
             :wd (bninf--tensor (list dim ff) (+ seed 7) scale)
             :bd (bninf--constant dim 0.0)))))
    (unless (and (= (% dim heads) 0) (= (% head-dim 2) 0))
      (error "DIM/HEADS must produce an even head dimension"))
    (list :wte (bninf--tensor (list vocab dim) 1 scale)
          :blocks (cl-loop for i below blocks
                           collect (funcall make-block (* 20 (1+ i))))
          :lnfg (bninf--constant dim 1.0)
          :wh (bninf--tensor (list vocab dim) 9 scale)
          :bh (bninf--constant vocab 0.0)
          :dim dim :ff ff :vocab vocab :heads heads :kvh heads
          :nblocks blocks)))

(defun bninf--messages (rendered-length)
  "Return messages whose complete agent rendering is RENDERED-LENGTH chars."
  (let* ((skeleton (list (cons 'system "native inference benchmark")
                         (cons 'user "")))
         (fixed (length (nl-llm-agent--render skeleton)))
         (body-length (- rendered-length fixed)))
    (when (< body-length 0)
      (error "Rendered length %d is shorter than fixed framing %d"
             rendered-length fixed))
    (let ((messages
           (list (cons 'system "native inference benchmark")
                 (cons 'user (make-string body-length ?x)))))
      (unless (= (length (nl-llm-agent--render messages)) rendered-length)
        (error "Prompt construction lost input"))
      messages)))

(defun bninf--tokens (rendered-length)
  "Return every token from an exactly RENDERED-LENGTH rendered prompt."
  (mapcar #'nl-llm-agent--char->id
          (string-to-list
           (nl-llm-agent--render (bninf--messages rendered-length)))))

(defun bninf--logits-by-position (model tokens)
  "Decode all TOKENS through MODEL and retain every position's logits."
  (let* ((dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (kvh (plist-get model :kvh))
         (blocks (plist-get model :blocks))
         (caches (mapcar (lambda (_block)
                           (nl-llm-dcache-new (length tokens)
                                                dim heads kvh))
                         blocks))
         (step (nl-llm-agent-model-step-fn model caches))
         (result nil))
    (dolist (token tokens)
      (push (copy-sequence (funcall step token)) result))
    (nreverse result)))

(defun bninf--max-logit-difference (left right)
  "Return maximum absolute difference across every logit in LEFT and RIGHT."
  (unless (= (length left) (length right))
    (error "Parity result position counts differ"))
  (let ((maximum 0.0))
    (while left
      (let ((a (car left)) (b (car right)) (i 0))
        (unless (= (length a) (length b))
          (error "Parity result vocabulary sizes differ"))
        (while (< i (length a))
          (setq maximum (max maximum (abs (- (aref a i) (aref b i)))))
          (setq i (1+ i))))
      (setq left (cdr left) right (cdr right)))
    maximum))

(defun bninf--elapsed (thunk &optional repetitions)
  "Return minimum wall seconds for THUNK over REPETITIONS fresh runs."
  (let ((runs (or repetitions 1)) (best nil))
    (dotimes (_ runs)
      (garbage-collect)
      (let ((start (float-time)))
        (funcall thunk)
        (let ((elapsed (- (float-time) start)))
          (setq best (if best (min best elapsed) elapsed)))))
    best))

(defun bninf--compile-functions (symbols)
  "Install in-memory byte-compiled definitions for SYMBOLS and return seconds."
  (let ((start (float-time)))
    (dolist (symbol symbols)
      (fset symbol (byte-compile (symbol-function symbol))))
    (- (float-time) start)))

(defun bninf--snapshot-symbols (symbols)
  "Snapshot function definitions and properties for SYMBOLS."
  (mapcar (lambda (symbol)
            (list symbol (symbol-function symbol)
                  (copy-tree (symbol-plist symbol))))
          symbols))

(defun bninf--restore-symbols (snapshots)
  "Restore function definitions and properties from SNAPSHOTS."
  (dolist (snapshot snapshots)
    (fset (nth 0 snapshot) (nth 1 snapshot))
    (setplist (nth 0 snapshot) (nth 2 snapshot))))

(defun bninf--timed-prefill (label model length repetitions)
  "Time complete LENGTH-position decode of MODEL and print a LABEL row."
  (let* ((tokens (bninf--tokens length))
         (elapsed (bninf--elapsed
                   (lambda () (bninf--logits-by-position model tokens))
                   repetitions)))
    (princ (format "  %-18s len=%4d  %8.3fs  %8.1f pos/s\n"
                   label length elapsed (/ length elapsed)))
    elapsed))

(defun bninf--policy-run (model messages grammar)
  "Run the actual native agent policy without altering or truncating MESSAGES."
  (let* ((rendered (nl-llm-agent--render messages))
         (maxseq (+ (length rendered) 64))
         (policy (nl-llm-agent-model-policy model grammar maxseq)))
    (cons (length rendered) (funcall policy messages))))

(defun bninf--main ()
  "Run the isolated source-versus-byte-compiled inference benchmark."
  (let* ((nl-llm-agent-model-inference-mode 'source)
         (saved-functions
          (mapcar (lambda (symbol) (cons symbol (symbol-function symbol)))
                  bninf--all-hot-functions))
         (saved-dependencies
          (bninf--snapshot-symbols
           nl-llm-inference-runtime--dependencies))
         (saved-gc-cons-threshold gc-cons-threshold)
         (saved-gc-cons-percentage gc-cons-percentage)
         (tiny (bninf--raw-p5-model 2 2 1 1))
         (larger (bninf--raw-p5-model 8 8 1 1))
         (parity-tokens (bninf--tokens 128))
         (grammar (nl-llm-agent-grammar-message 4 "DONE"))
         (long-messages (bninf--messages 2800))
         source-parity source-output compiled-parity compiled-output
         source-128 source-512 source-dim8 kernel-512
         compiled-128 compiled-512 compiled-dim8 source-wall compiled-wall
         runtime-output runtime-wall)
    (unwind-protect
        (progn
          (dolist (symbol bninf--all-hot-functions)
            (when (byte-code-function-p (symbol-function symbol))
              (error "%s did not load from source" symbol)))
          ;; Keep collection policy stable during each timed region.  It is
          ;; restored even when compilation, parity, or inference fails.
          (setq gc-cons-threshold (* 128 1024 1024)
                gc-cons-percentage 0.5)
          (princ "=== native agent CPU inference: source vs in-memory byte-code ===\n")
          (princ "model: artifact-shaped raw P5, independent :wh, vocab=96\n")
          (princ (format "hot functions: %S\n\n" bninf--all-hot-functions))

          (setq source-parity
                (bninf--logits-by-position tiny parity-tokens))
          (princ "-- source reference preflight --\n")
          (setq source-128 (bninf--timed-prefill "source tiny" tiny 128 3)
                source-512 (bninf--timed-prefill "source tiny" tiny 512 2)
                source-dim8 (bninf--timed-prefill
                             "source dim8" larger 128 2))

          (let ((kernel-compile
                 (bninf--compile-functions bninf--kernel-functions)))
            (princ (format "\ncompile kernels in memory: %.6fs\n" kernel-compile))
            (setq kernel-512
                  (bninf--timed-prefill "kernel-compiled" tiny 512 2)))
          (let ((decode-compile
                 (bninf--compile-functions bninf--decode-functions)))
            (princ (format "compile decode in memory : %.6fs\n\n"
                           decode-compile)))

          (setq compiled-parity
                (bninf--logits-by-position tiny parity-tokens))
          (let ((difference
                 (bninf--max-logit-difference source-parity compiled-parity)))
            (princ (format "all-position parity (128 x 96 logits): maxdiff=%.3e %s\n"
                           difference (if (= difference 0.0) "PASS" "FAIL")))
            (unless (= difference 0.0)
              (error "Source/compiled logit parity failed")))

          (princ "\n-- fully compiled numeric path preflight --\n")
          (setq compiled-128
                (bninf--timed-prefill "compiled tiny" tiny 128 3)
                compiled-512
                (bninf--timed-prefill "compiled tiny" tiny 512 2)
                compiled-dim8
                (bninf--timed-prefill "compiled dim8" larger 128 2))
          (princ (format
                  "speedup: tiny-128 %.2fx, tiny-512 %.2fx, dim8-128 %.2fx\n"
                  (/ source-128 compiled-128)
                  (/ source-512 compiled-512)
                  (/ source-dim8 compiled-dim8)))
          (princ (format "kernel-only share: tiny-512 %.2fx; decode compilation supplies the remaining gain\n"
                         (/ source-512 kernel-512)))

          ;; Restore source definitions for the first actual policy run, then
          ;; reinstall the already-produced byte-code objects for its peer.
          (dolist (entry saved-functions)
            (fset (car entry) (cdr entry)))
          (bninf--restore-symbols saved-dependencies)
          (let ((elapsed
                 (bninf--elapsed
                  (lambda ()
                    (setq source-output
                          (bninf--policy-run tiny long-messages grammar))))))
            (setq source-wall elapsed)
            (princ (format "\nsource policy   : prompt=%d output=%S wall=%.3fs\n"
                           (car source-output) (cdr source-output) elapsed)))
          (let ((start (float-time)))
            (dolist (entry saved-functions)
              (fset (car entry) (byte-compile (cdr entry))))
            (princ (format "policy-path compile overhead: %.6fs\n"
                           (- (float-time) start))))
          (let ((elapsed
                 (bninf--elapsed
                  (lambda ()
                    (setq compiled-output
                          (bninf--policy-run tiny long-messages grammar))))))
            (setq compiled-wall elapsed)
            (princ (format "compiled policy : prompt=%d output=%S wall=%.3fs\n"
                           (car compiled-output) (cdr compiled-output) elapsed)))
          (unless (and (= (car source-output) 2800)
                       (= (car compiled-output) 2800)
                       (equal source-output compiled-output))
            (error "Long policy output/input parity failed"))
          (princ (format
                  "long policy input/output parity: PASS (all 2800 chars fed), speedup %.2fx\n"
                  (/ source-wall compiled-wall)))

          ;; Measure the production adapter separately.  Start from the saved
          ;; source bindings so manual compilation above cannot contaminate it.
          (dolist (entry saved-functions)
            (fset (car entry) (cdr entry)))
          (bninf--restore-symbols saved-dependencies)
          (nl-llm-inference-runtime-prepare 'source)
          (let ((nl-llm-agent-model-inference-mode 'auto))
            (setq runtime-wall
                  (bninf--elapsed
                   (lambda ()
                     (setq runtime-output
                           (bninf--policy-run tiny long-messages grammar))))))
          (princ (format
                  "production auto : mode=%S prompt=%d output=%S wall=%.3fs speedup=%.2fx\n"
                  (if (byte-code-function-p
                       (symbol-function 'nl-llm-decode-block))
                      'byte-code 'source)
                  (car runtime-output) (cdr runtime-output) runtime-wall
                  (/ source-wall runtime-wall)))
          (unless (equal source-output runtime-output)
            (error "Production runtime output/input parity failed"))
          (nl-llm-inference-runtime-prepare 'source))
      (nl-llm-inference-runtime-prepare 'source)
      (dolist (entry saved-functions)
        (fset (car entry) (cdr entry)))
      (bninf--restore-symbols saved-dependencies)
      (setq gc-cons-threshold saved-gc-cons-threshold
            gc-cons-percentage saved-gc-cons-percentage))))

(bninf--main)

(provide 'bench-native-inference)
;;; bench-native-inference.el ends here
