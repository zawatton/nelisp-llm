;;; weights-gpu-test.el --- imported int8 weights through the DP4A kernel  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/weights-gpu-test.el
;;
;; Doc 08 Phase 2d.  A linear from the imported table, uploaded as bytes and run
;; through `bitlinear-dp4a-rows', checked two ways that answer two different
;; questions:
;;
;;   GPU vs CPU-W8A8   did the transfer and the kernel do the right arithmetic
;;   f32 act vs W8A8   what does quantizing the activation cost
;;
;; One comparison cannot answer both.  `nl-llm-weights-apply-w8a8' reproduces the
;; kernel's exact arithmetic on the CPU -- integer lane products, scaled once by
;; the weight row's scale times the activation's -- so the first check has a
;; tight threshold, while the second is a quality number to report rather than
;; to assert away.
;;
;; The activation matters more than it looks.  A raw embedding row is
;; lane * scale, i.e. already exactly int8-representable, so re-quantizing it is
;; lossless and "f32 act vs W8A8" comes out at 0.000e+00 -- which reads as
;; "activation quantization is free" and is really "this input could not
;; measure it".  The suite therefore uses a post-RMSNorm activation, the kind a
;; linear actually sees, AND asserts the input is not int8-exact, so the check
;; cannot quietly become trivial again.
;;
;; Needs the GPU server and the donor table; skips cleanly without either.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-weights)
(require 'nl-llm-lora)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-lora)
(require 'nl-llm-weights-backward)

(defvar wg--fail 0)
(defvar wg--table (expand-file-name "build/donor/qwen3-0.6b/weights.bin"))

(defun wg--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name
                 (if ok "PASS" (progn (setq wg--fail (1+ wg--fail)) "FAIL"))
                 (or extra ""))))

(defun wg--rel (a b scale)
  (let ((m 0.0))
    (dotimes (i (length a))
      (let ((d (abs (- (aref a i) (aref b i))))) (when (> d m) (setq m d))))
    (/ m (if (> scale 0.0) scale 1.0))))

(defun wg--amax (v)
  (let ((m 0.0)) (dotimes (i (length v)) (setq m (max m (abs (aref v i))))) m))

(if (not (file-readable-p wg--table))
    (princ (format "SKIP: weight table missing\n  %s\n\
  regenerate with:  make qwen-weights-table\n" wg--table))

  (if (not (require 'nl-llm-weights-gpu nil t))
      (princ "SKIP: nelisp-gpu is not loadable\n")

    (let ((started (ignore-errors (nelisp-gpu-server-start)
                                  (nelisp-gpu-server-up-p))))
      (if (not started)
          (princ "SKIP: the GPU server would not start (no Vulkan device?)\n")

        (unwind-protect
            (let* ((wts (nl-llm-weights-open wg--table))
                   (cfg (nl-llm-weights-config wts))
                   (dim (plist-get cfg :dim))
                   (lin (nl-llm-weights-linear wts :wq 0))
                   (emb (nl-llm-weights-embed wts 785))
                   (ln1g (nl-llm-weights-row
                          wts (nl-llm-weights-tensor wts :ln1g 0) 0))
                   (act (nl-llm-wf--rmsnorm emb 0 dim ln1g 1.0e-6)))

              ;; Guard the trap described above: if this ever passes, the
              ;; activation-quantization number below is measuring nothing.
              (let* ((pack (nl-llm-wgpu-pack-act act 0 dim))
                     (gamma (cdr pack))
                     (exact t))
                (dotimes (i dim)
                  (let ((q (/ (aref act i) gamma)))
                    (when (> (abs (- q (round q))) 1.0e-9) (setq exact nil))))
                (wg--ck "the test activation is NOT already int8-exact"
                        (not exact)
                        "a raw embedding row would be, and would measure nothing"))

              (let* ((t0 (float-time))
                     (handle (nl-llm-wgpu-upload lin))
                     (up (- (float-time) t0)))
                (wg--ck "weight payload uploads as bytes"
                        (integerp handle)
                        (format "%d bytes in %.3fs, handle %S"
                                (length (nl-llm-weights-lin-bytes lin))
                                up handle))

                (unwind-protect
                    (let* ((t1 (float-time))
                           (gpu (nl-llm-wgpu-apply lin handle act))
                           (gsec (- (float-time) t1))
                           (t2 (float-time))
                           (cpu (nl-llm-weights-apply-w8a8 lin act))
                           (csec (- (float-time) t2))
                           (f32 (nl-llm-weights-apply lin act))
                           (scale (wg--amax f32)))

                      (wg--ck "output has one entry per output row"
                              (= (length gpu) (nl-llm-weights-lin-rows lin))
                              (format "%d rows" (length gpu)))

                      ;; The question the kernel has to answer.
                      (let ((rel (wg--rel gpu cpu scale)))
                        (wg--ck "GPU == the same W8A8 arithmetic on the CPU"
                                (< rel 1.0e-6)
                                (format "rel %.3e (f32 accumulate vs f64)" rel)))

                      ;; The separate question: what the activation costs.
                      (let ((rel (wg--rel f32 cpu scale)))
                        (wg--ck "activation quantization cost is reported" t
                                (format "rel %.3e" rel)))

                      (wg--ck "the GPU is faster than the Elisp loop" (< gsec csec)
                              (format "%.3fs vs %.3fs (%.0fx)"
                                      gsec csec (/ csec (max gsec 1.0e-6))))

                      ;; Negative control: a wrong scale vector must be caught,
                      ;; so a green run above cannot mean the comparison is inert.
                      (let* ((scales (nl-llm-weights-lin-scales lin))
                             (saved (aref scales 0)))
                        (aset scales 0 (* saved 2.0))
                        (let* ((bad (nl-llm-wgpu-apply lin handle act))
                               (rel (wg--rel bad cpu scale)))
                          (aset scales 0 saved)
                          (wg--ck "control: a doubled row scale is detected"
                                  (> rel 1.0e-6)
                                  (format "rel %.3e" rel))))

                      ;; --- the transpose, which training needs ------------
                      ;;
                      ;; Two checks with different strengths.  Against the CPU
                      ;; it can only be as tight as f32 allows -- this kernel
                      ;; accumulates in float across every row, because the
                      ;; per-row scale multiplies each term and cannot be
                      ;; factored out of an integer dot product the way the
                      ;; forward's can.  So the inner-product identity carries
                      ;; the structural weight: it needs no reference at all and
                      ;; no index error survives it.
                      (let* ((gvec (make-vector (nl-llm-weights-lin-rows lin) 0.0)))
                        (dotimes (i (length gvec))
                          (aset gvec i (* 0.01 (- (mod (* (1+ i) 7919) 211) 105))))
                        (let* ((t4 (float-time))
                               (gt (nl-llm-wgpu-apply-t lin handle gvec))
                               (gsec (- (float-time) t4))
                               (t5 (float-time))
                               (ct (nl-llm-weights-apply-t lin gvec))
                               (csec (- (float-time) t5))
                               (sc (wg--amax ct)))
                          (wg--ck "GPU transpose == the CPU transpose"
                                  (< (wg--rel gt ct sc) 1.0e-4)
                                  (format "rel %.3e (f32 accumulate over %d rows)"
                                          (wg--rel gt ct sc)
                                          (nl-llm-weights-lin-rows lin)))
                          (wg--ck "the GPU transpose is faster than the Elisp loop"
                                  (< gsec csec)
                                  (format "%.4fs vs %.4fs (%.0fx)"
                                          gsec csec (/ csec (max gsec 1.0e-6))))
                          ;; <W.x, g> = <x, W^T.g>, with W^T from the GPU.
                          (let* ((wx (nl-llm-weights-apply lin act))
                                 (lhs (let ((s 0.0))
                                        (dotimes (i (length wx))
                                          (setq s (+ s (* (aref wx i) (aref gvec i)))))
                                        s))
                                 (rhs (let ((s 0.0))
                                        (dotimes (i (length gt))
                                          (setq s (+ s (* (aref act i) (aref gt i)))))
                                        s))
                                 (rel (/ (abs (- lhs rhs))
                                         (max (abs lhs) (abs rhs) 1.0e-30))))
                            (wg--ck "GPU transpose satisfies <W.x, g> = <x, W^T.g>"
                                    (< rel 1.0e-4)
                                    (format "%.6f vs %.6f (rel %.2e)" lhs rhs rel)))))

                      ;; --- the block backward, routed to the GPU ----------
                      ;;
                      ;; Only the transposes move, so every gradient that
                      ;; depends on one may differ by f32 accumulation.  That is
                      ;; all of them: an adapter's dA and dB do not touch its
                      ;; own W^T, but they are built from the gradient arriving
                      ;; at that linear, and for a LoRA on :wv that has already
                      ;; come back through :wo's transpose and the attention.
                      ;;
                      ;; Worth writing down because the first version of this
                      ;; check asserted dA and dB were bit-identical, on exactly
                      ;; that "they never touch W^T" reasoning, and a probe
                      ;; seemed to agree -- but only because the probe left B at
                      ;; zero, which makes dA identically zero on both sides.
                      ;; The suite failed the moment B was perturbed.
                      (let* ((lay (nl-llm-wf-load-layer wts 0))
                             (blin (nl-llm-wf-layer-lin lay :wv))
                             (lora (nl-llm-lora-make
                                    (nl-llm-weights-lin-rows blin)
                                    (nl-llm-weights-lin-cols blin) 8 16 0))
                             (loras (list :wv lora))
                             (bx (nl-llm-weights-embed wts 785))
                             (dout (make-vector dim 0.0)))
                        (dotimes (i (length (photon-tensor-data
                                             (plist-get lora :b))))
                          (aset (photon-tensor-data (plist-get lora :b)) i
                                (* 0.001 (- (mod (* (1+ i) 31) 17) 8))))
                        (dotimes (i dim)
                          (aset dout i (* 0.01 (- (mod (* (1+ i) 7919) 211) 105))))
                        (let* ((fw (nl-llm-wb-block-forward lay bx 1 cfg loras))
                               (tape (nth 1 fw))
                               (t6 (float-time))
                               (cpu (nl-llm-wb-block-backward
                                     lay tape dout 1 cfg loras))
                               (csec (- (float-time) t6))
                               (tbl (nl-llm-wgpu-upload-transposes (list lay)))
                               (t7 (float-time))
                               (gpu (nl-llm-wgpu-with-transposes tbl
                                      (nl-llm-wb-block-backward
                                       lay tape dout 1 cfg loras)))
                               (gsec (- (float-time) t7)))
                          (unwind-protect
                              (progn
                                (wg--ck "block backward on the GPU == the CPU (dx)"
                                        (< (wg--rel (car gpu) (car cpu)
                                                    (wg--amax (car cpu)))
                                           1.0e-4)
                                        (format "rel %.3e"
                                                (wg--rel (car gpu) (car cpu)
                                                         (wg--amax (car cpu)))))
                                (let* ((ca (plist-get (plist-get (cdr cpu) :wv) :da))
                                       (ga (plist-get (plist-get (cdr gpu) :wv) :da))
                                       (cb (plist-get (plist-get (cdr cpu) :wv) :db))
                                       (gb (plist-get (plist-get (cdr gpu) :wv) :db))
                                       (ra (wg--rel ga ca (wg--amax ca)))
                                       (rb (wg--rel gb cb (wg--amax cb))))
                                  (wg--ck "the adapter's gradients agree to f32"
                                          (and (< ra 1.0e-4) (< rb 1.0e-4)
                                               (> (wg--amax ca) 0.0))
                                          (format "dA rel %.2e, dB rel %.2e (nonzero: %s)"
                                                  ra rb (> (wg--amax ca) 0.0))))
                                (wg--ck "the GPU block backward is faster"
                                        (< gsec csec)
                                        (format "%.3fs vs %.2fs (%.0fx)"
                                                gsec csec (/ csec (max gsec 1.0e-6))))
                                ;; A linear absent from the table must fall back
                                ;; rather than fail, which is what makes a
                                ;; partially uploaded model merely slower.
                                (let* ((empty (make-hash-table :test 'eq))
                                       (fb (nl-llm-wgpu-with-transposes empty
                                             (nl-llm-wb-block-backward
                                              lay tape dout 1 cfg loras))))
                                  (wg--ck "an empty table falls back to the CPU"
                                          (equal (append (car fb) nil)
                                                 (append (car cpu) nil))
                                          "identical to the CPU backward")))
                            (nl-llm-wgpu-free-transposes tbl))))

                      ;; The acceptance criterion for this phase, behind an env
                      ;; var because it is a four-minute run: does the whole
                      ;; W8A8 GPU stack still predict what the CPU oracle
                      ;; predicted?  Activation quantization costs about 0.7%
                      ;; per layer, so surviving 28 of them is a real question
                      ;; rather than a formality.
                      (if (not (getenv "NL_LLM_GPU_E2E"))
                          (princ (format "%-50s %s  %s\n"
                                         "end-to-end greedy token" "----"
                                         "set NL_LLM_GPU_E2E=1 to run (~4 min)"))
                        (let* ((t3 (float-time))
                               (best (nl-llm-wgpu-next-token
                                      wts (list 785 6722 315 9625 374)))
                               (secs (- (float-time) t3)))
                          (wg--ck "end-to-end greedy token == CPU oracle"
                                  (= (car best) 12095)
                                  (format "got %d want 12095, logit %.6f vs \
17.189348 (oracle), %.0fs" (car best) (cdr best) secs)))))
                  (nelisp-gpu-server-free handle))))
          (nelisp-gpu-server-stop))

        (princ (format "\n%s: %d failure(s)\n"
                       (if (zerop wg--fail) "weights-gpu OK" "weights-gpu")
                       wg--fail))
        (when (> wg--fail 0) (kill-emacs 1))))))
