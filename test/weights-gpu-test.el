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
                          ;; What the GPU does NOT reproduce, stated as a
                          ;; measurement rather than left to be discovered.
                          ;; The device flushes subnormal inputs to zero, so a
                          ;; gradient whose every entry is subnormal comes back
                          ;; as zeros while the CPU returns something of order
                          ;; 1e-41.  That is not a defect and not a rounding
                          ;; difference -- it is a different answer -- and it
                          ;; is harmless only because no real gradient looks
                          ;; like that.  Pinning it here means the day one does,
                          ;; this check says so instead of the training loop
                          ;; quietly learning nothing.
                          (let* ((tiny (make-vector (nl-llm-weights-lin-rows lin) 0.0)))
                            (dotimes (i (length tiny))
                              (aset tiny i (* 1.0e-42 (1+ (mod i 7)))))
                            (let* ((tg (nl-llm-wgpu-apply-t lin handle tiny))
                                   (tc (nl-llm-weights-apply-t lin tiny))
                                   (gmax (wg--amax tg)) (cmax (wg--amax tc)))
                              (wg--ck "subnormal inputs flush to zero on the GPU"
                                      (and (= gmax 0.0) (> cmax 0.0))
                                      (format "GPU amax %.3e, CPU amax %.3e"
                                              gmax cmax))))
                          ;; And the mixed case, which is what actually occurs:
                          ;; a softmax tail is mostly subnormal but its mass
                          ;; sits in the entries that are not, so the agreement
                          ;; survives.  Before the encoder in nelisp-gpu was
                          ;; fixed this disagreed by 7.7e+30, because every
                          ;; subnormal was encoded as a number around 1e+33.
                          (let* ((soft (make-vector (nl-llm-weights-lin-rows lin) 0.0))
                                 (sub 0))
                            (dotimes (i (length soft))
                              (aset soft i (exp (- (* 0.05 i)))))
                            (dotimes (i (length soft))
                              (when (and (> (aref soft i) 0.0)
                                         (< (aref soft i) 1.1754943508222875e-38))
                                (setq sub (1+ sub))))
                            (let* ((sg (nl-llm-wgpu-apply-t lin handle soft))
                                   (sc2 (nl-llm-weights-apply-t lin soft))
                                   (sm (wg--amax sc2)))
                              (wg--ck "a softmax-shaped gradient still agrees"
                                      (and (> sub 0) (< (wg--rel sg sc2 sm) 1.0e-4))
                                      (format "rel %.3e, %d of %d entries subnormal"
                                              (wg--rel sg sc2 sm) sub (length soft)))))
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

                      ;; --- a tape that lives on the device -----------------
                      ;;
                      ;; A `tmp' slot lives for one batch.  A tape does not:
                      ;; the forward writes it and the backward reads it in a
                      ;; later call.  So the fused design rests on a kernel
                      ;; being able to write into a *resident* buffer and have
                      ;; it survive, which is cheap to establish and expensive
                      ;; to assume.
                      ;;
                      ;; The control is the load-bearing half.  If the write
                      ;; had not survived, the second dispatch would have
                      ;; rotated a buffer of zeros and returned zeros -- which
                      ;; disagrees with the CPU, so the first check would fail,
                      ;; but for a reason it does not name.  Asserting the
                      ;; output is not zero says which of the two happened.
                      (let* ((tseq 6) (tnh (plist-get cfg :heads))
                             (thd (plist-get cfg :head-dim))
                             (trb (plist-get cfg :rope-base))
                             (tn (* tseq tnh thd))
                             (tx (make-vector tn 0.0))
                             (tg (make-vector thd 0.0)))
                        (dotimes (i tn)
                          (aset tx i (* 0.05 (- (mod (* (1+ i) 7919) 101) 50))))
                        (dotimes (i thd) (aset tg i (+ 0.8 (* 0.004 (mod i 51)))))
                        (let ((cpu (copy-sequence tx)))
                          (dotimes (p tseq)
                            (nl-llm--rmsnorm-heads cpu (* p tnh thd) tnh thd
                                                   (photon-tensor (list thd) tg)
                                                   1.0e-6))
                          (dotimes (p tseq)
                            (nl-llm--rope-heads cpu (* p tnh thd) tnh thd p trb 'half))
                          (let ((tape (nelisp-gpu-server-upload-bytes
                                       (make-string (* 4 tn) 0))))
                            (unwind-protect
                                (progn
                                  (nelisp-gpu-server-run2
                                   'rmsnorm-heads
                                   (list (cons 'in tx) (cons 'in tg)
                                         (list 'res tape tn))
                                   (list tseq tnh thd)
                                   (/ (+ (* tseq tnh) 63) 64))
                                  (let* ((out (car (nelisp-gpu-server-run2
                                                    'rope-half
                                                    (list (list 'res tape tn)
                                                          (cons 'out tn))
                                                    (list tseq tnh thd
                                                          (nelisp-gpu--f32-bits
                                                           (float trb)))
                                                    (/ (+ (* tseq tnh (/ thd 2)) 63)
                                                       64))))
                                         (sc (wg--amax cpu)))
                                    (wg--ck "a kernel's write to a resident buffer survives"
                                            (< (wg--rel out cpu sc) 1.0e-5)
                                            (format "rel %.3e across two dispatches"
                                                    (wg--rel out cpu sc)))
                                    (wg--ck "control: the second pass is not reading zeros"
                                            (> (wg--amax out) 1.0e-3)
                                            (format "amax %.4f" (wg--amax out)))))
                              (nelisp-gpu-server-free tape)))))

                      ;; --- the two glue kernels a fused block still needed --
                      ;;
                      ;; QK-norm and the rotation were the pieces missing from
                      ;; the kernel set: `rmsnorm-fwd' normalises rows, not
                      ;; heads, and `rope-apply' pairs adjacent elements where
                      ;; Qwen3 pairs i with i + hd/2.
                      ;;
                      ;; The rotation carries a control, because this is the
                      ;; mistake the project has already paid for once: the two
                      ;; conventions take the same shapes and produce different
                      ;; numbers, so a kernel quietly using the wrong one is
                      ;; invisible to anything but a comparison against the
                      ;; right one.  The control asserts the other convention
                      ;; really is different here, so "matches the CPU" is not
                      ;; passing because both happen to agree.
                      (let* ((gseq 6) (gnh (plist-get cfg :heads))
                             (ghd (plist-get cfg :head-dim))
                             (grb (plist-get cfg :rope-base))
                             (gn (* gseq gnh ghd))
                             (gx (make-vector gn 0.0))
                             (ggain (make-vector ghd 0.0)))
                        (dotimes (i gn)
                          (aset gx i (* 0.05 (- (mod (* (1+ i) 7919) 101) 50))))
                        (dotimes (i ghd) (aset ggain i (+ 0.8 (* 0.004 (mod i 51)))))
                        (let ((cpu (copy-sequence gx)))
                          (dotimes (p gseq)
                            (nl-llm--rmsnorm-heads cpu (* p gnh ghd) gnh ghd
                                                   (photon-tensor (list ghd) ggain)
                                                   1.0e-6))
                          (let* ((gpu (car (nelisp-gpu-server-run2
                                            'rmsnorm-heads
                                            (list (cons 'in gx) (cons 'in ggain)
                                                  (cons 'out gn))
                                            (list gseq gnh ghd)
                                            (/ (+ (* gseq gnh) 63) 64))))
                                 (sc (wg--amax cpu)))
                            (wg--ck "per-head RMSNorm on the GPU (QK-norm)"
                                    (< (wg--rel gpu cpu sc) 1.0e-5)
                                    (format "rel %.3e" (wg--rel gpu cpu sc)))))
                        (let ((cpu (copy-sequence gx))
                              (other (copy-sequence gx)))
                          (dotimes (p gseq)
                            (nl-llm--rope-heads cpu (* p gnh ghd) gnh ghd p grb 'half)
                            (nl-llm--rope-heads other (* p gnh ghd) gnh ghd p grb
                                                'interleaved))
                          (let* ((gpu (car (nelisp-gpu-server-run2
                                            'rope-half
                                            (list (cons 'in gx) (cons 'out gn))
                                            (list gseq gnh ghd
                                                  (nelisp-gpu--f32-bits (float grb)))
                                            (/ (+ (* gseq gnh (/ ghd 2)) 63) 64))))
                                 (sc (wg--amax cpu)))
                            (wg--ck "half-split RoPE on the GPU"
                                    (< (wg--rel gpu cpu sc) 1.0e-5)
                                    (format "rel %.3e" (wg--rel gpu cpu sc)))
                            (wg--ck "control: the interleaved convention differs"
                                    (> (wg--rel gpu other (wg--amax other)) 0.1)
                                    (format "rel %.3e against it"
                                            (wg--rel gpu other (wg--amax other)))))))

                      ;; --- quantizing on the device, and what it costs -----
                      ;;
                      ;; `pack-act-rows' is the input side of the int8 linear
                      ;; done on the GPU, so a tensor already there can feed one
                      ;; without coming back.  Checked as a fused batch --
                      ;; pack into a tmp slot, apply from it -- because the
                      ;; packed words must NOT come back: they are arbitrary
                      ;; bit patterns and the return path decodes floats, so a
                      ;; word landing in the NaN range does not survive it.
                      ;; An earlier version of this check read them back and
                      ;; reported a broken kernel; the kernel was right and the
                      ;; harness was destroying its output.
                      ;;
                      ;; The result is *not* bit-identical to CPU packing and
                      ;; cannot be: the GPU divides by gamma in f32 and Elisp in
                      ;; f64, so a lane sitting on a rounding boundary can go
                      ;; either way.  What that is worth is the number below.
                      (let* ((ng (/ dim 4))
                             (fseq 6)
                             (fx (make-vector (* fseq dim) 0.0))
                             (flin (nl-llm-weights-linear wts :wq 0))
                             (frows (nl-llm-weights-lin-rows flin)))
                        (dotimes (p fseq)
                          (dotimes (i dim)
                            (aset fx (+ (* p dim) i) (* (aref act i) (+ 1.0 (* 0.1 p))))))
                        (let ((fh (nl-llm-wgpu-upload-lin flin)))
                          (unwind-protect
                              (let* ((t0 (float-time))
                                     (cpu (nl-llm-wgpu-apply-seq flin fh fx 0 fseq))
                                     (t1 (float-time))
                                     (fused (car (nelisp-gpu-server-batch
                                                  (list (cons 'in fx)
                                                        (cons 'tmp (* fseq ng))
                                                        (cons 'tmp fseq)
                                                        (list 'res (plist-get fh :w) (* frows ng))
                                                        (list 'res (plist-get fh :b) frows)
                                                        (list 'res (plist-get fh :s) frows)
                                                        (cons 'out (* fseq frows)))
                                                  (list (list 'pack-act-rows '(0 1 2)
                                                              (list fseq dim ng)
                                                              (/ (+ fseq 63) 64))
                                                        (list 'bitlinear-dp4a-rows '(1 3 4 5 2 6)
                                                              (list fseq frows ng)
                                                              (/ (+ (* fseq frows) 63) 64))))))
                                     (t2 (float-time))
                                     (sc (wg--amax cpu))
                                     (rel (wg--rel fused cpu sc)))
                                (wg--ck "packing on the device agrees with packing here"
                                        (< rel 1.0e-5)
                                        (format "rel %.3e (f32 vs f64 division by gamma)" rel))
                                ;; And the part that decides the design: fusing
                                ;; only pays when the input is already on the
                                ;; device.  Here it is not, so the fused form
                                ;; sends f32 where the other sends packed bytes
                                ;; -- four times the data through the encoder.
                                ;;
                                ;; The claim is that it does not *win*, not that
                                ;; it loses.  An earlier version asserted
                                ;; "slower" from a single sample of 0.042
                                ;; against 0.030; the two have since measured
                                ;; 0.0407 against 0.0411, the same within noise.
                                ;; A check that keeps passing while the
                                ;; observation it names changes sign is not
                                ;; checking anything.
                                (wg--ck "fusing from a host tensor buys nothing"
                                        (> (- t2 t1) (* 0.7 (- t1 t0)))
                                        (format "fused %.4fs vs packed-here %.4fs (f32 in vs int8 bytes in)" (- t2 t1) (- t1 t0))))
                            (dolist (k '(:w :s :b))
                              (nelisp-gpu-server-free (plist-get fh k))))))

                      ;; --- causal attention on the GPU, and why it is not
                      ;; --- wired in -------------------------------------
                      ;;
                      ;; Attention is the one part of a block that is
                      ;; quadratic in the sequence, and in Elisp that is the
                      ;; wall: measured on these shapes it is 6.6s a step at
                      ;; seq 6 across 28 layers and 311s at seq 48.  So
                      ;; `attn-causal-gqa' exists and agrees with the CPU.
                      ;;
                      ;; It is still not used, and the reason is worth a check
                      ;; rather than a comment.  Calling it means marshalling
                      ;; q, k and v across the Elisp boundary, and that costs
                      ;; about 3 microseconds a float out and 1.2 back --
                      ;; 0.71s of the 1.06s a seq-48 call takes, whatever the
                      ;; kernel does.  The fix is not a faster kernel but
                      ;; keeping q, k and v on the device between the
                      ;; projections and the attention, which is a different
                      ;; shape of change.
                      (let* ((heads (plist-get cfg :heads))
                             (kv-heads (plist-get cfg :kv-heads))
                             (hd (plist-get cfg :head-dim))
                             (aseq 24)
                             (qd (* heads hd)) (kvd (* kv-heads hd))
                             (qq (make-vector (* aseq qd) 0.0))
                             (kk (make-vector (* aseq kvd) 0.0))
                             (vv (make-vector (* aseq kvd) 0.0)))
                        (dotimes (i (* aseq qd))
                          (aset qq i (* 0.05 (- (mod (* (1+ i) 7919) 101) 50))))
                        (dotimes (i (* aseq kvd))
                          (aset kk i (* 0.05 (- (mod (* (1+ i) 5387) 101) 50)))
                          (aset vv i (* 0.05 (- (mod (* (1+ i) 3319) 101) 50))))
                        (let* ((t0 (float-time))
                               (ccpu (nl-llm-wf--attend qq kk vv aseq heads kv-heads hd))
                               (t1 (float-time))
                               (cgpu (car (nelisp-gpu-server-run2
                                           'attn-causal-gqa
                                           (list (cons 'in qq) (cons 'in kk)
                                                 (cons 'in vv)
                                                 (cons 'out (* aseq qd)))
                                           (list aseq heads kv-heads hd)
                                           (/ (+ (* heads aseq) 63) 64))))
                               (t2 (float-time))
                               (t3 (float-time))
                               (_ (nelisp-gpu--floats-bytes (list qq kk vv)))
                               (enc (- (float-time) t3))
                               (sc (wg--amax ccpu))
                               (rel (wg--rel cgpu ccpu sc)))
                          (wg--ck "causal GQA attention matches the CPU"
                                  (< rel 1.0e-5)
                                  (format "rel %.3e, seq %d" rel aseq))
                          ;; The claim that the boundary dominates, as a
                          ;; number: if this ever stops holding, the kernel
                          ;; becomes worth wiring in.
                          (wg--ck "and its cost is the boundary, not the kernel"
                                  (> enc (* 0.3 (- t2 t1)))
                                  (format "encoding q,k,v %.3fs of a %.3fs call (CPU %.3fs)" enc (- t2 t1) (- t1 t0)))))

                      ;; --- the block backward, batched and not -------------
                      ;;
                      ;; The same check as for the forward, and it matters more
                      ;; here.  The forward has an absolute reference -- its
                      ;; output is compared bit for bit against
                      ;; `nl-llm-wf-block' -- while the backward's own suite
                      ;; compares gradients against finite differences, with a
                      ;; tolerance.  A staging error that perturbed the order
                      ;; of accumulation would stay inside that tolerance and
                      ;; pass.  Running the block both ways and demanding
                      ;; identical gradients is what actually pins the
                      ;; restructure: float addition is not associative, so
                      ;; "identical" is a claim about order as well as sum.
                      (let* ((lay (nl-llm-wf-load-layer wts 0))
                             (seq 4)
                             (blin (nl-llm-wf-layer-lin lay :wv))
                             (lora (nl-llm-lora-make
                                    (nl-llm-weights-lin-rows blin)
                                    (nl-llm-weights-lin-cols blin) 4 8.0 3))
                             (xs (make-vector (* seq dim) 0.0))
                             (dout (make-vector (* seq dim) 0.0)))
                        ;; B nonzero, or the adapter's dA is identically zero
                        ;; and the check cannot see a difference in it.
                        (let ((b (plist-get lora :b)))
                          (dotimes (i (length (photon-tensor-data b)))
                            (aset (photon-tensor-data b) i
                                  (* 0.01 (- (mod (* (1+ i) 31) 7) 3)))))
                        (dotimes (p seq)
                          (let ((row (nl-llm-weights-embed wts (+ 785 p))))
                            (dotimes (i dim)
                              (aset xs (+ (* p dim) i) (aref row i)))))
                        (dotimes (i (* seq dim))
                          (aset dout i (* 0.01 (- (mod (* (1+ i) 7919) 211) 105.0))))
                        (let ((tbl (nl-llm-wgpu-upload-transposes (list lay)))
                              (loras (list :wv lora)))
                          (unwind-protect
                              (let* ((tape (nth 1 (nl-llm-wb-block-forward
                                                   lay xs seq cfg loras)))
                                     (batched (nl-llm-wgpu-with-transposes tbl
                                                (nl-llm-wb-block-backward
                                                 lay tape dout seq cfg loras)))
                                     (per-pos (nl-llm-wgpu-with-transposes tbl
                                                (let ((nl-llm-wb-transpose-seq-fn nil))
                                                  (nl-llm-wb-block-backward
                                                   lay tape dout seq cfg loras))))
                                     (a (car batched)) (b (car per-pos))
                                     (bad 0) (badg 0))
                                (dotimes (i (length a))
                                  (unless (= (aref a i) (aref b i))
                                    (setq bad (1+ bad))))
                                (let ((ga (plist-get (cdr batched) :wv))
                                      (gb (plist-get (cdr per-pos) :wv)))
                                  (dolist (key '(:da :db))
                                    (let ((va (plist-get ga key))
                                          (vb (plist-get gb key)))
                                      (dotimes (i (length va))
                                        (unless (= (aref va i) (aref vb i))
                                          (setq badg (1+ badg)))))))
                                (wg--ck "block backward: batched == per-position"
                                        (and (zerop bad) (zerop badg))
                                        (format "dx %d of %d differ, adapter %d"
                                                bad (length a) badg)))
                            (nl-llm-wgpu-free-transposes tbl))))

                      ;; --- the block forward, batched and not --------------
                      ;;
                      ;; `nl-llm-wb-block-forward' now applies the linears a
                      ;; role at a time rather than a position at a time.  The
                      ;; direct check on that restructure is to run it both
                      ;; ways on the same GPU table and demand the outputs be
                      ;; identical: unbinding the batched hook inside
                      ;; `with-linears' puts the position loop back, and
                      ;; everything else is the same code.  A difference here
                      ;; is a layout error in the batching, which is the one
                      ;; mistake that still produces plausible numbers.
                      (let* ((lay (nl-llm-wf-load-layer wts 0))
                             (seq 4)
                             (xs (make-vector (* seq dim) 0.0)))
                        (dotimes (p seq)
                          (let ((row (nl-llm-weights-embed wts (+ 785 p))))
                            (dotimes (i dim)
                              (aset xs (+ (* p dim) i) (aref row i)))))
                        (let ((tbl (nl-llm-wgpu-upload-transposes (list lay))))
                          (unwind-protect
                              (let* ((batched (nl-llm-wgpu-with-linears tbl
                                                (nl-llm-wb-block-forward
                                                 lay xs seq cfg)))
                                     (per-pos (nl-llm-wgpu-with-linears tbl
                                                (let ((nl-llm-wb-forward-seq-fn nil))
                                                  (nl-llm-wb-block-forward
                                                   lay xs seq cfg))))
                                     (a (nth 0 batched)) (b (nth 0 per-pos))
                                     (bad 0))
                                (dotimes (i (length a))
                                  (unless (= (aref a i) (aref b i))
                                    (setq bad (1+ bad))))
                                (wg--ck "block forward: batched == per-position"
                                        (zerop bad)
                                        (format "%d of %d differ, seq %d"
                                                bad (length a) seq)))
                            (nl-llm-wgpu-free-transposes tbl))))

                      ;; --- a batch of positions in one dispatch ------------
                      ;;
                      ;; Equality here is exact, not approximate, and that is
                      ;; the claim: batching changes how many dispatches carry
                      ;; the arithmetic, not the arithmetic.  Each position is
                      ;; still quantized with its own scale and each (position,
                      ;; row) still accumulates the same way, so anything other
                      ;; than a bit-for-bit match means a layout error -- which
                      ;; a tolerance would hide, since a wrong position offset
                      ;; still produces plausible numbers.
                      ;;
                      ;; Both handle forms are checked because both are live:
                      ;; a bare integer sends the weight's constants inline,
                      ;; a plist has them resident.
                      (let* ((seq 6)
                             (rows (nl-llm-weights-lin-rows lin))
                             (cols (nl-llm-weights-lin-cols lin))
                             (xs (make-vector (* seq cols) 0.0))
                             (gs (make-vector (* seq rows) 0.0)))
                        (dotimes (p seq)
                          (dotimes (i cols)
                            (aset xs (+ (* p cols) i)
                                  (* (aref act i) (+ 1.0 (* 0.1 p))))))
                        (dotimes (i (* seq rows))
                          (aset gs i (* 0.001 (- (mod (* (1+ i) 7919) 211) 105.0))))
                        (dolist (form '(bare resident))
                          (let ((h (if (eq form 'bare)
                                       (nl-llm-wgpu-upload lin)
                                     (nl-llm-wgpu-upload-lin lin))))
                            (unwind-protect
                                (let* ((t0 (float-time))
                                       (one (let (acc)
                                              (dotimes (p seq)
                                                (push (nl-llm-wgpu-apply lin h xs (* p cols))
                                                      acc))
                                              (nreverse acc)))
                                       (t1 (float-time))
                                       (many (nl-llm-wgpu-apply-seq lin h xs 0 seq))
                                       (t2 (float-time))
                                       (bad 0))
                                  (dotimes (p seq)
                                    (dotimes (o rows)
                                      (unless (= (aref (nth p one) o)
                                                 (aref many (+ (* p rows) o)))
                                        (setq bad (1+ bad)))))
                                  (wg--ck (format "%S handle: batched forward is exact" form)
                                          (zerop bad)
                                          (format "%d of %d differ, %.3fs -> %.3fs (%.1fx)"
                                                  bad (* seq rows) (- t1 t0) (- t2 t1)
                                                  (/ (- t1 t0) (max (- t2 t1) 1.0e-6))))
                                  (let* ((t3 (float-time))
                                         (ot (let (acc)
                                               (dotimes (p seq)
                                                 (let ((gp (make-vector rows 0.0)))
                                                   (dotimes (o rows)
                                                     (aset gp o (aref gs (+ (* p rows) o))))
                                                   (push (nl-llm-wgpu-apply-t lin h gp) acc)))
                                               (nreverse acc)))
                                         (t4 (float-time))
                                         (mt (nl-llm-wgpu-apply-t-seq lin h gs seq))
                                         (t5 (float-time))
                                         (badt 0))
                                    (dotimes (p seq)
                                      (dotimes (i cols)
                                        (unless (= (aref (nth p ot) i)
                                                   (aref mt (+ (* p cols) i)))
                                          (setq badt (1+ badt)))))
                                    (wg--ck (format "%S handle: batched transpose is exact" form)
                                            (zerop badt)
                                            (format "%d of %d differ, %.3fs -> %.3fs (%.1fx)"
                                                    badt (* seq cols) (- t4 t3) (- t5 t4)
                                                    (/ (- t4 t3) (max (- t5 t4) 1.0e-6))))))
                              (if (eq form 'bare)
                                  (nelisp-gpu-server-free h)
                                (dolist (k '(:w :s :b))
                                  (nelisp-gpu-server-free (plist-get h k))))))))

                      ;; --- the tied head, the other half of a step --------
                      ;;
                      ;; 151936 x 1024, the largest single matrix in the
                      ;; model.  It is the same two kernels as a block's
                      ;; linears, so there is no new arithmetic here; what is
                      ;; new is that leaving it on the CPU makes it the whole
                      ;; cost of a step, since every training position has to
                      ;; score the vocabulary and push a gradient back.
                      ;;
                      ;; Two traps this section exists to pin down, both of
                      ;; which produced confident wrong numbers first:
                      ;;
                      ;;   * the head's input is a *normalised* hidden state.
                      ;;     An embedding row is lane*scale and therefore
                      ;;     already int8-exact, so measuring activation
                      ;;     quantization on one measures nothing -- it
                      ;;     reported 3.9e-08 and meant it.
                      ;;   * the gradient the head receives is dense.  A probe
                      ;;     with 64 nonzeros out of 151936 made the CPU
                      ;;     transpose look ten times *faster* than the GPU,
                      ;;     because `nl-llm-weights-apply-t' skips rows whose
                      ;;     scale times gradient is zero and that gradient
                      ;;     gave it almost nothing to do.
                      (let* ((hlin (nl-llm-weights-linear wts :wte))
                             (lnf (nl-llm-weights-row
                                   wts (nl-llm-weights-tensor wts :lnf) 0))
                             (vocab (nl-llm-weights-lin-rows hlin))
                             (hid (nl-llm-wf--rmsnorm emb 0 dim lnf 1.0e-6))
                             (t6 (float-time))
                             (hh (nl-llm-wgpu-upload hlin))
                             (hup (- (float-time) t6)))
                        (unwind-protect
                            (progn
                              (wg--ck "the tied head uploads as one buffer"
                                      (integerp hh)
                                      (format "%d x %d, %.0f MiB in %.1fs"
                                              vocab dim
                                              (/ (length (nl-llm-weights-lin-bytes hlin))
                                                 1048576.0)
                                              hup))
                              (let* ((pack (nl-llm-wgpu-pack-act hid 0 dim))
                                     (gam (cdr pack)) (ex t))
                                (dotimes (i dim)
                                  (let ((q (/ (aref hid i) gam)))
                                    (when (> (abs (- q (round q))) 1.0e-9)
                                      (setq ex nil))))
                                (wg--ck "the head's input is NOT int8-exact"
                                        (not ex)
                                        "normalised hidden state, not an embedding row"))
                              (let* ((t7 (float-time))
                                     (gl (nl-llm-wgpu-apply hlin hh hid 0))
                                     (fsec (- (float-time) t7)))
                                (wg--ck "the head scores the vocabulary on the GPU"
                                        (= (length gl) vocab)
                                        (format "%d logits in %.2fs" vocab fsec))
                                ;; The gradient of a completion-only
                                ;; cross-entropy is softmax minus one-hot.
                                ;; Count it rather than assume it: the sparse
                                ;; probe above is only wrong because this is
                                ;; true.
                                (let* ((sm (copy-sequence gl)) (mx (aref sm 0))
                                       (sum 0.0) (nz 0))
                                  (dotimes (i vocab) (setq mx (max mx (aref sm i))))
                                  (dotimes (i vocab)
                                    (aset sm i (exp (- (aref sm i) mx)))
                                    (setq sum (+ sum (aref sm i))))
                                  (dotimes (i vocab)
                                    (aset sm i (/ (aref sm i) sum))
                                    (when (/= (aref sm i) 0.0) (setq nz (1+ nz))))
                                  (wg--ck "the gradient a head receives is dense"
                                          (> nz (/ vocab 2))
                                          (format "%d of %d nonzero after softmax"
                                                  nz vocab)))
                                ;; The gradient for the identity below is the
                                ;; head's own output, which makes the left side
                                ;; <W.x, W.x> -- a sum of squares.  A signed
                                ;; pseudo-random gradient was tried first and
                                ;; is the wrong instrument here: over 151936
                                ;; terms it cancels down to about 4.6 out of
                                ;; individual terms near 1.8, so the forward's
                                ;; 0.2% activation-quantization error lands on
                                ;; a small difference of large numbers and the
                                ;; identity reads 8e-02.  That is arithmetic,
                                ;; not a defect -- the same transpose satisfies
                                ;; the identity to 4.8e-08 against an f32
                                ;; forward -- but a check whose tolerance has
                                ;; to be loosened to 0.2 is not checking much.
                                (let* ((gv (copy-sequence gl)))
                                  (let* ((t8 (float-time))
                                         (gt (nl-llm-wgpu-apply-t hlin hh gv))
                                         (tsec (- (float-time) t8)))
                                    (wg--ck "the head's transpose runs on the GPU"
                                            (= (length gt) dim)
                                            (format "%d x %d in %.2fs" dim vocab tsec))
                                    ;; Structural, and cheap: no reference
                                    ;; implementation, and no index error
                                    ;; survives it.  Loose because the two
                                    ;; sides are not the same arithmetic --
                                    ;; the forward quantizes the activation,
                                    ;; the transpose accumulates in f32.
                                    (let* ((lhs 0.0) (rhs 0.0))
                                      (dotimes (i vocab)
                                        (setq lhs (+ lhs (* (aref gl i) (aref gv i)))))
                                      (dotimes (i dim)
                                        (setq rhs (+ rhs (* (aref hid i) (aref gt i)))))
                                      (let ((rel (/ (abs (- lhs rhs))
                                                    (max (abs lhs) (abs rhs) 1.0e-30))))
                                        (wg--ck "the head satisfies <W.x, g> = <x, W^T.g>"
                                                (< rel 1.0e-2)
                                                (format "%.1f vs %.1f (rel %.2e, W8A8 vs f32)"
                                                        lhs rhs rel))))
                                    ;; The comparison against the CPU is
                                    ;; behind an env var only because it is
                                    ;; two minutes of Elisp over 151936 rows.
                                    (if (not (getenv "NL_LLM_GPU_HEAD"))
                                        (princ (format "%-50s %s  %s\n"
                                                       "head vs the CPU reference" "----"
                                                       "set NL_LLM_GPU_HEAD=1 to run (~2 min)"))
                                      (let* ((t9 (float-time))
                                             (w8 (nl-llm-weights-apply-w8a8 hlin hid 0))
                                             (f32 (nl-llm-weights-apply hlin hid 0))
                                             (csec (- (float-time) t9))
                                             (sc (wg--amax f32))
                                             (ta (float-time))
                                             (ct (nl-llm-weights-apply-t hlin gv))
                                             (tcsec (- (float-time) ta))
                                             (tsc (wg--amax ct)))
                                        (wg--ck "head forward == the same W8A8 on the CPU"
                                                (< (wg--rel gl w8 sc) 1.0e-4)
                                                (format "rel %.3e" (wg--rel gl w8 sc)))
                                        (wg--ck "head forward picks the f32 argmax"
                                                (= (car (nl-llm-wf-argmax gl))
                                                   (car (nl-llm-wf-argmax f32)))
                                                (format "%d, activation quantization rel %.3e, \
CPU %.0fs" (car (nl-llm-wf-argmax gl)) (wg--rel gl f32 sc) csec))
                                        (wg--ck "head transpose == the CPU transpose"
                                                (< (wg--rel gt ct tsc) 1.0e-4)
                                                (format "rel %.3e, CPU %.0fs (dense gradient)"
                                                        (wg--rel gt ct tsc) tcsec))
                                        ;; The identity again, now with the f32
                                        ;; forward on the left, so the only
                                        ;; approximation in it is the
                                        ;; transpose's own f32 accumulation.
                                        ;; This is the tight form; the cheap
                                        ;; one above pays for avoiding a
                                        ;; two-minute CPU pass.
                                        (let ((lhs 0.0) (rhs 0.0))
                                          (dotimes (i vocab)
                                            (setq lhs (+ lhs (* (aref f32 i) (aref gv i)))))
                                          (dotimes (i dim)
                                            (setq rhs (+ rhs (* (aref hid i) (aref gt i)))))
                                          (let ((rel (/ (abs (- lhs rhs))
                                                        (max (abs lhs) (abs rhs) 1.0e-30))))
                                            (wg--ck "the identity against an f32 forward"
                                                    (< rel 1.0e-4)
                                                    (format "%.1f vs %.1f (rel %.2e)"
                                                            lhs rhs rel))))))))))
                          (nelisp-gpu-server-free hh)))

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
