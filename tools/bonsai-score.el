;;; bonsai-score.el --- cross entropy through the hybrid driver -*- lexical-binding: t -*-

;; The same measurement the numpy reference and the donor's reference make,
;; run through `nl-llm-bonsai.el' itself.  On a model whose forward is already
;; checked layer by layer -- the Qwen3-0.6B donor, which scores 3.12 nats --
;; this says whether the driver reproduces a known answer.  No amount of
;; testing against Ternary Bonsai can say that while that model's own fidelity
;; is open, which is exactly how two defects survived a dozen sweeps.
;;
;; Env: NL_BONSAI_WTS, NL_BONSAI_TOKENS, NL_BONSAI_LAYERS.

(require 'nl-llm-bonsai)
(require 'nl-llm-weights-gpu)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defun bs--env (n d) (let ((v (getenv n))) (if (and v (> (length v) 0)) v d)))

(defun bs--rms (v)
  (let ((s 0.0))
    (dotimes (i (length v)) (setq s (+ s (* (aref v i) (aref v i)))))
    (sqrt (/ s (float (length v))))))

(defun bs--score (sess x seq ids vocab)
  "Cross entropy and ranks of IDS under the final hidden states X."
  (let ((total 0.0) (n 0) (ranks nil))
    (dotimes (at (1- seq))
      (let* ((lg (nl-llm-bonsai-logits sess x seq at))
             (tgt (nth (1+ at) ids))
             (mx (aref lg 0)) (sum 0.0) (rank 1))
        (dotimes (j (length lg)) (when (> (aref lg j) mx) (setq mx (aref lg j))))
        (dotimes (j (length lg)) (setq sum (+ sum (exp (- (aref lg j) mx)))))
        (dotimes (j (length lg))
          (when (> (aref lg j) (aref lg tgt)) (setq rank (1+ rank))))
        (push rank ranks)
        (setq total (+ total (- (+ mx (log sum)) (aref lg tgt))))
        (setq n (1+ n))))
    (setq ranks (sort ranks #'<))
    (message "  cross entropy %.4f nats over %d targets (chance %.2f)"
             (/ total n) n (log vocab))
    (let ((cap (string-to-number (bs--env "NL_BONSAI_CE_MAX" "0"))))
      (when (and (> cap 0.0) (> (/ total n) cap))
        (message "FAIL: %.4f nats is above the %.2f this model should reach"
                 (/ total n) cap)
        (kill-emacs 1)))
    (message "  median rank %d of %d, top-1 %.1f%%, top-10 %.1f%%"
             (nth (/ n 2) ranks) vocab
             (* 100.0 (/ (float (cl-count 1 ranks)) n))
             (* 100.0 (/ (float (cl-count-if (lambda (r) (<= r 10)) ranks)) n)))))

(defun bs--forward (sess nlayers x seq)
  "Run NLAYERS blocks with every weight resident; return the final X."
  (let ((tbl (make-hash-table :test 'eq)))
    (dotimes (ly nlayers)
      (cl-loop for (_k v) on (nl-llm-bonsai-linears sess ly) by #'cddr
               do (puthash v (nl-llm-wgpu-upload-lin v) tbl)))
    (let ((head (nl-llm-bonsai-head sess)))
      (puthash head (nl-llm-wgpu-upload-lin head) tbl))
    (message "  resident %d buffers" (hash-table-count tbl))
    (let ((nl-llm-wgpu--transposes tbl)
          (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident))
      (dotimes (ly nlayers)
        (setq x (nl-llm-bonsai-block sess ly x seq)))
      ;; the table stays live: the head is in it and the scoring needs it
      (list x tbl))))

(defun bs-run ()
  (let* ((path (bs--env "NL_BONSAI_WTS" "build/donor/qwen3-0.6b/weights.bin"))
         (sess (nl-llm-bonsai-open path))
         (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (nlayers (plist-get cfg :layers))
         (cap (string-to-number (bs--env "NL_BONSAI_LAYERS" "0")))
         (ids (mapcar #'string-to-number
                      (split-string (bs--env "NL_BONSAI_TOKENS" "9707 1879") "[ ,]+" t)))
         (seq (length ids))
         (x (make-vector (* seq dim) 0.0))
         (i 0) (t0 (float-time)))
    (when (and (> cap 0) (< cap nlayers)) (setq nlayers cap))
    (message "%s  dim %d  layers %d  seq %d  rotation %s  gains %s  interval %d"
             (file-name-nondirectory path) dim nlayers seq
             (if (plist-get sess :rotate) "yes" "none")
             (if (plist-get sess :folded) "folded" "applied")
             (nl-llm-bonsai--interval cfg))
    (dolist (tk ids)
      (let ((e (nl-llm-bonsai-embed sess tk)))
        (dotimes (j dim) (aset x (+ (* i dim) j) (aref e j))))
      (setq i (1+ i)))
    (nelisp-gpu-server-start)
    (unwind-protect
        (let* ((r (bs--forward sess nlayers x seq))
               (tbl (nth 1 r)))
          (setq x (nth 0 r))
          (message "  forward done (%.1fs), residual rms %.4f"
                   (- (float-time) t0) (bs--rms x))
          (unwind-protect
              (let ((nl-llm-wgpu--transposes tbl)
                    (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident))
                (bs--score sess x seq ids (plist-get cfg :vocab)))
            (nl-llm-wgpu-free-transposes tbl)))
      (nelisp-gpu-server-stop))
    (message "TOTAL %.1fs" (- (float-time) t0))
    (message "DONE")))

(bs-run)
