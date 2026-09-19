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
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)

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
                                  (format "rel %.3e" rel)))))
                  (nelisp-gpu-server-free handle))))
          (nelisp-gpu-server-stop))

        (princ (format "\n%s: %d failure(s)\n"
                       (if (zerop wg--fail) "weights-gpu OK" "weights-gpu")
                       wg--fail))
        (when (> wg--fail 0) (kill-emacs 1))))))
