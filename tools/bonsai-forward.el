;;; bonsai-forward.el --- run every block of the imported model -*- lexical-binding: t -*-

;; The decisive instrument for the one thing the metadata does not say: which
;; way round the folded incoherence rotation goes.  Both candidates are
;; orthogonal, so norms, amax and residual growth are blind to the difference
;; by construction -- and the stored weights are ternary, so their distribution
;; carries no memory of the rotation either.  What is not blind is whether 64
;; blocks in sequence produce a token that follows from the prompt.
;;
;; One layer is resident at a time: 6 GB of device memory against 26.92 GB of
;; weights.  Per-layer residual magnitudes are printed so a blow-up localises
;; to a block rather than to the stack.
;;
;; Usage: emacs -Q --batch -L lisp -L <photon> -l tools/bonsai-forward.el
;; Env: NL_BONSAI_WTS (weight table), NL_BONSAI_INVERT (non-empty = the other
;; rotation), NL_BONSAI_LAYERS (stop early), NL_BONSAI_TOKENS (prompt ids).

(require 'nl-llm-bonsai)
(require 'nl-llm-weights-gpu)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defun bf--rms (v &optional base n)
  (let ((base (or base 0)) (n (or n (length v))) (s 0.0))
    (dotimes (i n) (setq s (+ s (* (aref v (+ base i)) (aref v (+ base i))))))
    (sqrt (/ s (float n)))))

(defun bf--vocab (path)
  "Id -> printable token from the TSV `tools/bonsai-tokenizer.py --dump' writes.
A predicted id says nothing on its own; the point of the whole run is whether
what comes out reads as a continuation of what went in."
  (when (and path (file-readable-p path))
    (let ((v (make-vector 300000 nil)))
      (with-temp-buffer
        (let ((coding-system-for-read 'utf-8))
          (insert-file-contents path))
        (goto-char (point-min))
        (while (not (eobp))
          (let* ((eol (line-end-position))
                 (tab (save-excursion (search-forward "\t" eol t))))
            (when tab
              (let ((id (string-to-number
                         (buffer-substring-no-properties (point) (1- tab)))))
                (when (< id (length v))
                  (aset v id (buffer-substring-no-properties tab eol))))))
          (forward-line 1)))
      v)))

(defun bf--show (vocab id)
  (let ((s (and vocab (< id (length vocab)) (aref vocab id))))
    (if s (format "%d %S" id (string-replace "\u0120" " " s)) (format "%d" id))))

(defun bf--getenv (name default)
  (let ((v (getenv name))) (if (and v (> (length v) 0)) v default)))

(defun bf-run ()
  (let* ((path (bf--getenv "NL_BONSAI_WTS" "build/bonsai/wts-full.bin"))
         (invert (and (getenv "NL_BONSAI_INVERT")
                      (> (length (getenv "NL_BONSAI_INVERT")) 0)))
         (sess (nl-llm-bonsai-open path invert))
         (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (nlayers (plist-get cfg :layers))
         (cap (string-to-number (bf--getenv "NL_BONSAI_LAYERS" "0")))
         (ids (mapcar #'string-to-number
                      (split-string (bf--getenv "NL_BONSAI_TOKENS"
                                                "9707 11 847 525 498 30")
                                    "[ ,]+" t)))
         (seq (length ids))
         (wts (plist-get sess :wts))
         (vocab (bf--vocab (getenv "NL_BONSAI_VOCAB")))
         (x (make-vector (* seq dim) 0.0))
         (i 0) (t-start (float-time)) (t-up 0.0) (t-run 0.0))
    (when (and (> cap 0) (< cap nlayers)) (setq nlayers cap))
    (dolist (tk ids)
      (let ((e (nl-llm-bonsai-embed sess tk)))
        (dotimes (j dim) (aset x (+ (* i dim) j) (aref e j))))
      (setq i (1+ i)))
    (message "  prompt         %s"
             (mapconcat (lambda (id) (bf--show vocab id)) ids " "))
    (message "%s  dim %d  layers %d  seq %d  rotation %s"
             (file-name-nondirectory path) dim nlayers seq
             (if invert "inverse" "forward"))
    (message "  embedding      rms %.5f" (bf--rms x))
    (nelisp-gpu-server-start)
    (unwind-protect
        (progn
          (progn
            (dotimes (ly nlayers)
              (let* ((lins (nl-llm-bonsai-linears sess ly))
                     (tbl (make-hash-table :test 'eq))
                     (t0 (float-time)))
                (cl-loop for (k v) on lins by #'cddr
                         do (puthash v (nl-llm-wgpu-upload-lin v) tbl))
                (let ((t1 (float-time)))
                  (setq t-up (+ t-up (- t1 t0)))
                  (unwind-protect
                      (let ((nl-llm-wgpu--transposes tbl)
                            (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident))
                        (setq x (nl-llm-bonsai-block sess ly x seq)))
                    (nl-llm-wgpu-free-transposes tbl)
                    (nl-llm-bonsai-forget-layer sess ly))
                  (setq t-run (+ t-run (- (float-time) t1)))
                  (message "  block %2d       rms %10.4f   last-pos rms %10.4f  \
(up %.2fs run %.1fs)"
                           ly (bf--rms x) (bf--rms x (* (1- seq) dim) dim)
                           (- t1 t0) (- (float-time) t1)))))

            (let ((dump (getenv "NL_BONSAI_DUMP")))
              (when (and dump (> (length dump) 0))
                (let ((coding-system-for-write 'binary))
                  (write-region (nelisp-gpu--floats-bytes (list x)) nil dump
                                nil 'silent))
                (message "  dumped %s (%d floats)" dump (length x))))
            ;; The head, rotated like every other rotated projection.
            (let* ((t0 (float-time))
                   (head (nl-llm-bonsai-head sess))
                   (tbl (make-hash-table :test 'eq)))
              (puthash head (nl-llm-wgpu-upload-lin head) tbl)
              (let ((t1 (float-time)))
                (unwind-protect
                    (let ((nl-llm-wgpu--transposes tbl)
                          (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident))
                      (let* ((lg (nl-llm-bonsai-logits sess x seq))
                             (best (nl-llm-bonsai-argmax lg))
                             (sorted (sort (cl-loop for k from 0 below (length lg)
                                                    collect (cons k (aref lg k)))
                                           (lambda (a b) (> (cdr a) (cdr b))))))
                        (message "  head           up %.2fs  run %.1fs"
                                 (- t1 t0) (- (float-time) t1))
                        (message "  logits         rms %.4f  max %.4f  min %.4f"
                                 (bf--rms lg) (cdr best)
                                 (apply #'min (append lg nil)))
                        (message "  PREDICTED      %s" (bf--show vocab (car best)))
                        (dolist (c (cl-subseq sorted 0 10))
                          (message "    %8.3f  %s" (cdr c) (bf--show vocab (car c))))))
                  (nl-llm-wgpu-free-transposes tbl))))))
      (nelisp-gpu-server-stop))
    (message "TOTAL %.1fs  (upload %.1fs, compute %.1fs)"
             (- (float-time) t-start) t-up t-run)
    (message "DONE")))

(bf-run)
