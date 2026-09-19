;;; nl-llm-weights-gpu.el --- imported int8 weights on the GPU  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 2d.  Runs a linear from an
;; imported table on the GPU without the weight ever becoming a float on the
;; Elisp side: the table's payload is already little-endian uint32 words of four
;; int8 lanes, which is exactly what the DP4A kernel reads back with `bitcast-u',
;; so the bytes go from `insert-file-contents-literally' to the GPU buffer
;; untouched.
;;
;; Two things had to exist for that:
;;
;; - `nelisp-gpu-server-upload-bytes'.  The older `upload-u32' takes a vector of
;;   Elisp integers and lists it before encoding; for a model's worth of words
;;   (~149M here) that is a 1.2 GB vector plus a 2.4 GB list, to send bytes that
;;   were already correct on disk.
;; - `bitlinear-dp4a-rows'.  Imported weights are quantized per output row, and
;;   `bitlinear-dp4a-1f' reads BETA[0].  Re-indexing that kernel was tempting
;;   and wrong: nl-llm-bitnet.el passes it a one-element BETA, so it would have
;;   read past the end.  The per-row scales got their own kernel instead.
;;
;; The arithmetic is W8A8: the weight is int8 with a per-row scale, and the
;; activation is quantized int8 with a per-row scale of its own.  That is not
;; the f32-activation path the CPU reference runs, so
;; `nl-llm-weights-apply-w8a8' reproduces the GPU's exact arithmetic on the CPU.
;; Comparing GPU against that isolates the transfer and the kernel; comparing it
;; against the f32 path measures what activation quantization costs.  Those are
;; different questions and one comparison cannot answer both.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)
(require 'nl-llm-weights)
(require 'nl-llm-gpu)            ; puts the nelisp-gpu sibling dir on load-path
(require 'nelisp-gpu-server)

(defun nl-llm-wgpu--i8 (v)
  "Round V to an int8 lane, clipped, as an unsigned byte."
  (let ((q (round v)))
    (logand (max -127 (min 127 q)) 255)))

;;;###autoload
(defun nl-llm-wgpu-pack-act (x base n)
  "Quantize N elements of X from BASE to int8 and pack four lanes per word.
Returns (BYTES . GAMMA): BYTES is a unibyte string of little-endian uint32
words, GAMMA the scale such that lane * GAMMA approximates the original."
  (let ((amax 0.0) (i 0))
    (while (< i n)
      (let ((a (abs (aref x (+ base i))))) (when (> a amax) (setq amax a)))
      (setq i (1+ i)))
    (let* ((gamma (if (> amax 0.0) (/ amax 127.0) 1.0))
           (ng (/ (+ n 3) 4))
           (out (make-string (* 4 ng) 0))
           (k 0))
      (while (< k (* 4 ng))
        (aset out k (if (< k n)
                        (nl-llm-wgpu--i8 (/ (aref x (+ base k)) gamma))
                      0))
        (setq k (1+ k)))
      (cons out gamma))))

;;;###autoload
(defun nl-llm-weights-apply-w8a8 (lin x &optional base)
  "Return LIN applied to X at BASE with the activation ALSO quantized to int8.
This is the GPU kernel's arithmetic written out on the CPU: integer lane
products accumulated, then scaled once by the weight row's scale times the
activation's.  Use it to check the GPU; use `nl-llm-weights-apply' to find out
what the activation quantization costs."
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (wb (nl-llm-weights-lin-bytes lin))
         (scales (nl-llm-weights-lin-scales lin))
         (pack (nl-llm-wgpu-pack-act x (or base 0) cols))
         (ab (car pack)) (gamma (cdr pack))
         (y (make-vector rows 0.0))
         (o 0))
    (while (< o rows)
      (let ((p (* o 4 words)) (acc 0) (i 0))
        (while (< i cols)
          (let ((w (aref wb (+ p i))) (a (aref ab i)))
            (setq acc (+ acc (* (if (> w 127) (- w 256) w)
                                (if (> a 127) (- a 256) a)))))
          (setq i (1+ i)))
        (aset y o (* (aref scales o) gamma acc)))
      (setq o (1+ o)))
    y))

;;;###autoload
(defun nl-llm-wgpu-upload (lin)
  "Upload LIN's int8 payload to a resident GPU buffer; return the handle.
The bytes go up exactly as they came off disk -- no float is built, and nothing
is allocated per word."
  (nelisp-gpu-server-upload-bytes (nl-llm-weights-lin-bytes lin)))

;;;###autoload
(defun nl-llm-wgpu-apply (lin handle x &optional base bias)
  "Run LIN (resident at HANDLE) on X at BASE through `bitlinear-dp4a-rows'.
BIAS defaults to zeros, which is what Qwen3's projections have.  Returns the
ROWS-long result as a float vector."
  (let* ((cols (nl-llm-weights-lin-cols lin))
         (rows (nl-llm-weights-lin-rows lin))
         (words (nl-llm-weights-lin-words lin))
         (pack (nl-llm-wgpu-pack-act x (or base 0) cols))
         (hact (nelisp-gpu-server-upload-bytes (car pack))))
    (unwind-protect
        (nth 0 (nelisp-gpu-server-run2
                'bitlinear-dp4a-rows
                (list (list 'res hact words)
                      (list 'res handle (* rows words))
                      (cons 'in (or bias (make-vector rows 0.0)))
                      (cons 'in (nl-llm-weights-lin-scales lin))
                      (cons 'in (vector (cdr pack)))
                      (cons 'out rows))
                (list 1 rows words)
                (/ (+ rows 63) 64)))
      (nelisp-gpu-server-free hact))))

(provide 'nl-llm-weights-gpu)
;;; nl-llm-weights-gpu.el ends here
