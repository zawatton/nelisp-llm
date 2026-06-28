;;; gpu-paged-spike-test.el --- PagedAttention dynamic-index gather/scatter spike  -*- lexical-binding: t; -*-
;; De-risks the one load-bearing unknown for paged KV (docs/design/03): can the
;; kernel compiler emit a DATA-DEPENDENT dynamic index -- read a physical block id
;; from a TABLE buffer and use it to index a POOL buffer?  Tests both directions
;; (gather = attention read, scatter = cache append) against a CPU computation.
;; Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/gpu-paged-spike-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'nl-llm-gpu)        ; starts the server with the default kernel list
(require 'nelisp-gpu-server) ; nelisp-gpu-server-run

(defvar ps--fail 0)
(defun ps--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name (if ok "PASS" (progn (setq ps--fail (1+ ps--fail)) "FAIL")) (or extra ""))))

(unless (nl-llm-gpu-enable)
  (princ "NL-LLM-GPU-PAGED-SPIKE SKIP (no GPU server / Vulkan device)\n") (kill-emacs 0))

(let* ((n 4) (bs 3) (nphys 6)
       ;; logical block -> physical block id (a permutation, not identity)
       (table (vector 2.0 0.0 5.0 3.0))
       ;; pool of NPHYS physical blocks x BS, each element = phys*100 + off (distinct)
       (pool (let ((v (make-vector (* nphys bs) 0.0)))
               (dotimes (p nphys) (dotimes (o bs) (aset v (+ (* p bs) o) (float (+ (* p 100) o))))) v))
       (out (make-vector (* n bs) -1.0))
       (src (let ((v (make-vector (* n bs) 0.0)))
              (dotimes (blk n) (dotimes (o bs) (aset v (+ (* blk bs) o) (float (+ (* blk 7) o 1))))) v)))
  ;; --- gather: OUT[blk*bs+off] = POOL[table[blk]*bs+off] ---
  (let* ((res (nelisp-gpu-server-run 'gather-spike (list (copy-sequence table) (copy-sequence pool) out)
                                     (vector n bs) (/ (+ (* n bs) 63) 64)))
         (g (nth 2 res)) (ok t) (mx 0.0))
    (dotimes (blk n) (dotimes (o bs)
      (let ((want (aref pool (+ (* (truncate (aref table blk)) bs) o)))
            (got (aref g (+ (* blk bs) o))))
        (setq mx (max mx (abs (- want got)))) (unless (= want got) (setq ok nil)))))
    (ps--ck "gather: OUT == POOL[TABLE[blk]*bs+off]" ok (format "maxdiff=%.1f" mx)))
  ;; --- scatter: POOL2[table[blk]*bs+off] = SRC[blk*bs+off] ---
  (let* ((pool2 (make-vector (* nphys bs) 0.0))
         (res (nelisp-gpu-server-run 'scatter-spike (list (copy-sequence table) (copy-sequence src) pool2)
                                     (vector n bs) (/ (+ (* n bs) 63) 64)))
         (p2 (nth 2 res)) (ok t) (mx 0.0))
    (dotimes (blk n) (dotimes (o bs)
      (let ((want (aref src (+ (* blk bs) o)))
            (got (aref p2 (+ (* (truncate (aref table blk)) bs) o))))
        (setq mx (max mx (abs (- want got)))) (unless (= want got) (setq ok nil)))))
    (ps--ck "scatter: POOL[TABLE[blk]*bs+off] == SRC" ok (format "maxdiff=%.1f" mx)))
  (nl-llm-gpu-disable)
  (princ (format "NL-LLM-GPU-PAGED-SPIKE %s (%d failures)\n" (if (= ps--fail 0) "ALL-PASS" "HAS-FAILURES") ps--fail))
  (kill-emacs (if (= ps--fail 0) 0 1)))
;;; gpu-paged-spike-test.el ends here
