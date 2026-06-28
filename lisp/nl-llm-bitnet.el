;;; nl-llm-bitnet.el --- BitNet b1.58 packed ternary-weight inference (Phase B)  -*- lexical-binding: t; -*-

;; Phase B of BitNet b1.58 (docs/design/01): after Phase-A QAT trains a
;; full-precision latent weight whose ternary quantization is the deployed
;; weight, Phase B *stores* that ternary weight compactly and runs the forward
;; from the packed form.  The nelisp-gpu substrate only has f32 buffers, so
;; instead of a new int8 dtype we pack PK ternary codes (tern+1 in {0,1,2}) as
;; base-4 digits into each f32 -- exact for PK<=10 (4^10 < 2^23).  At PK=8 the
;; weight buffer is 8x smaller (16 of 32 bits used), an 8x VRAM/bandwidth cut;
;; the `bitlinear-packed' kernel reads each packed float once and peels the codes
;; with integer %4 // /4.  (A further int8/DP4A path on Pascal is future work.)

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nelisp-gpu-server)

(defconst nl-llm-bitnet-pk 8
  "Ternary codes packed per f32 (base-4).  Must match the kernel push value.")

(defun nl-llm-bitnet-pack (w &optional pk)
  "Pack ternary-quantized weight W (out x in) into base-4 codes, PK per f32.
Returns (PACKED BETA FCOUNT): PACKED is (out*FCOUNT) floats, FCOUNT =
ceil(in/PK), BETA = mean|W|.  Each weight -> ternary in {-1,0,1} -> code
\(tern+1) in {0,1,2}; row o's codes are little-endian base-4 across its floats."
  (let* ((pk (or pk nl-llm-bitnet-pk)) (sh (photon-tensor-shape w))
         (out (car sh)) (in (nth 1 sh)) (wd (photon-tensor-data w)) (n (* out in))
         (fcount (/ (+ in pk -1) pk)) (packed (make-vector (* out fcount) 0.0)) (acc 0.0))
    (dotimes (i n) (setq acc (+ acc (abs (aref wd i)))))
    (let ((beta (/ acc (float n))))
      (dotimes (o out)
        (dotimes (f fcount)
          (let ((val 0.0) (mul 1.0))
            (dotimes (z pk)
              (let ((i (+ (* f pk) z)))
                (when (< i in)
                  (let* ((q (if (> beta 0.0) (/ (aref wd (+ (* o in) i)) beta) 0.0))
                         (tern (cond ((>= q 0.5) 1) ((<= q -0.5) -1) (t 0))))
                    (setq val (+ val (* mul (float (+ tern 1)))))))
                (setq mul (* mul 4.0))))
            (aset packed (+ (* o fcount) f) val))))
      (list packed beta fcount))))

;;;###autoload
(defun nl-llm-bitnet-linear (x w bias &optional pk)
  "X (seq x in) . Wq^T + BIAS on the GPU with PACKED ternary weight W (out x in),
Wq = beta*ternary(W).  Returns the flat (seq*out) result vector.  The GPU server
must be running (`nl-llm-gpu-enable')."
  (let* ((pk (or pk nl-llm-bitnet-pk)) (sh (photon-tensor-shape x)) (seq (car sh)) (in (nth 1 sh))
         (out (car (photon-tensor-shape w)))
         (pk3 (nl-llm-bitnet-pack w pk)) (packed (nth 0 pk3)) (beta (nth 1 pk3)) (fcount (nth 2 pk3)))
    (nth 4 (nelisp-gpu-server-run
            'bitlinear-packed
            (list (copy-sequence (photon-tensor-data x)) packed
                  (copy-sequence (photon-tensor-data bias)) (vector beta) (make-vector (* seq out) 0.0))
            (vector seq in out pk fcount) (/ (+ (* seq out) 63) 64)))))

(provide 'nl-llm-bitnet)
;;; nl-llm-bitnet.el ends here
