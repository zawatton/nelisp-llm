;;; ternary-test.el --- the model's own ternary, read back -*- lexical-binding: t -*-

;; The int8 export requantizes: 0.0039 relative on the worst tensor, and every
;; weight moves a little.  The ternary export does not, because the model's
;; weights already are ternary inside each 128-element block, so the block's
;; absmax is its scale and the values divide into it exactly.  These checks
;; hold that to EXACT equality rather than a tolerance -- a tolerance would
;; pass an off-by-one in the packing, which produces numbers that are still
;; plausible.
;;
;; The fixture comes from the donor GGUF, not from the exporter, so a failure
;; here against a pass in tools/ternary-verify.py separates "the reader
;; unpacks it wrong" from "the exporter wrote it wrong".

(require 'nl-llm-weights)
(require 'cl-lib)

(defvar tt-pass 0)
(defvar tt-fail 0)
(defvar tt-seed 12345)
(defvar tt-table (or (getenv "NL_TERNARY_TABLE") "build/bonsai/wts-t2.bin"))
(defvar tt-fixture (or (getenv "NL_TERNARY_FIXTURE") "build/ternary-fixture.eld"))

(defun tt-check (name ok fmt &rest args)
  (if ok (setq tt-pass (1+ tt-pass)) (setq tt-fail (1+ tt-fail)))
  (message "%-52s %s  %s" name (if ok "PASS" "FAIL") (apply #'format fmt args)))

(defun tt-rnd ()
  (setq tt-seed (mod (+ (* tt-seed 1103515245) 12345) 2147483648))
  (/ (float tt-seed) 2147483648.0))

(defun tt-pack (trits rows cols words)
  "Pack TRITS into two-bit fields: sixteen per uint32, four per byte."
  (let ((bytes (make-string (* rows words 4) 0)))
    (dotimes (o rows)
      (dotimes (i cols)
        (let* ((v (aref trits (+ (* o cols) i)))
               (code (cond ((= v 1) 1) ((= v -1) 3) (t 0)))
               (idx (+ (* o words 4) (ash i -2))))
          (aset bytes idx (logior (aref bytes idx)
                                  (ash code (* 2 (logand i 3))))))))
    bytes))

(defun tt-lin (bytes sc rows cols words bsize name)
  (nl-llm-weights-lin--make
   :payload bytes :scales sc :rows rows :cols cols :words words
   :block bsize :ternary t :name name))

(defun tt-fixture-rows ()
  "Every fixture row must come back bit for bit."
  (let ((wts (nl-llm-weights-open tt-table))
        (rows (with-temp-buffer
                (insert-file-contents tt-fixture)
                (goto-char (point-min))
                (read (current-buffer)))))
    (dolist (r rows)
      (let* ((role (plist-get r :role))
             (layer (plist-get r :layer))
             (want (plist-get r :values))
             (tn (nl-llm-weights-tensor wts role (and (>= layer 0) layer)))
             (got (nl-llm-weights-row wts tn (plist-get r :row)))
             (worst 0.0) (i 0))
        (tt-check (format "%s@%d is ternary2" role layer)
                  (equal (plist-get tn :kind) "ternary2")
                  "kind %s" (plist-get tn :kind))
        (dolist (v want)
          (setq worst (max worst (abs (- v (aref got i)))))
          (setq i (1+ i)))
        (tt-check (format "%s@%d row %d is exact" role layer (plist-get r :row))
                  (= worst 0.0) "worst |difference| %.3e over %d values"
                  worst (length want))))))

(defun tt-arithmetic ()
  "Unpacking and per-block scaling, on a matrix small enough to write out.
The real tensors are tens of millions of multiply-accumulates in Elisp, and
what these check is the decode and the scaling, which a small one exercises
exactly as well."
  (let* ((rows 5) (cols 256) (bsize 128) (nb (/ cols bsize))
         (words (/ (+ cols 15) 16))
         (trits (make-vector (* rows cols) 0))
         (sc (make-vector (* rows nb) 0.0)))
    (dotimes (o rows)
      (dotimes (k nb) (aset sc (+ (* o nb) k) (+ 0.01 (* 0.04 (tt-rnd)))))
      (dotimes (i cols)
        (aset trits (+ (* o cols) i) (- (truncate (* 3.0 (tt-rnd))) 1))))
    (let* ((bytes (tt-pack trits rows cols words))
           (lin (tt-lin bytes sc rows cols words bsize "hand-packed"))
           (x (make-vector cols 0.0))
           (g (make-vector rows 0.0)))
      (dotimes (i cols) (aset x i (- (* 2.0 (tt-rnd)) 1.0)))
      (dotimes (o rows) (aset g o (- (* 2.0 (tt-rnd)) 1.0)))
      (tt-definition lin trits sc x rows cols bsize nb)
      (tt-transpose lin x g rows cols)
      (tt-controls bytes sc trits rows cols words bsize))))

(defun tt-definition (lin trits sc x rows cols bsize nb)
  (let ((y (nl-llm-weights-apply lin x)) (worst 0.0))
    (dotimes (o rows)
      (let ((acc 0.0))
        (dotimes (i cols)
          (setq acc (+ acc (* (aref trits (+ (* o cols) i))
                              (aref sc (+ (* o nb) (/ i bsize)))
                              (aref x i)))))
        (setq worst (max worst (/ (abs (- acc (aref y o)))
                                  (max 1.0e-8 (abs acc)))))))
    (tt-check "apply matches the definition, per-block scales"
              (< worst 1.0e-12) "worst rel %.3e" worst)))

(defun tt-transpose (lin x g rows cols)
  (let* ((y (nl-llm-weights-apply lin x))
         (xt (nl-llm-weights-apply-t lin g))
         (l 0.0) (r 0.0))
    (dotimes (o rows) (setq l (+ l (* (aref y o) (aref g o)))))
    (dotimes (i cols) (setq r (+ r (* (aref x i) (aref xt i)))))
    (tt-check "transpose: <W.x, g> = <x, W'.g>"
              (< (/ (abs (- l r)) (max 1.0e-8 (abs l))) 1.0e-10)
              "%.9f against %.9f" l r)))

(defun tt-controls (bytes sc trits rows cols words bsize)
  "The packing and the per-block scales must both be load-bearing."
  (let ((lin (tt-lin bytes sc rows cols words bsize "base"))
        (probe (make-vector cols 0.0))
        (flipped (copy-sequence bytes)))
    (aset probe 0 1.0)
    (let ((y0 (aref (nl-llm-weights-apply lin probe) 0)))
      (aset flipped 0 (logxor (aref flipped 0) 3))
      (let ((y1 (aref (nl-llm-weights-apply
                       (tt-lin flipped sc rows cols words bsize "flip") probe)
                      0)))
        (tt-check "control: flipping one two-bit field changes the answer"
                  (/= y0 y1) "%.6f -> %.6f" y0 y1))))
  (let ((lin (tt-lin bytes sc rows cols words bsize "base"))
        (sc2 (copy-sequence sc))
        (probe (make-vector cols 0.0))
        ;; a column in the SECOND block whose trit is not zero, or the probe
        ;; would read a zero and the control would pass on nothing
        (col (let ((i bsize))
               (while (and (< i cols) (= (aref trits i) 0)) (setq i (1+ i)))
               i)))
    (when (>= col cols) (error "no non-zero trit in the second block"))
    (aset probe col 1.0)
    (let ((y0 (aref (nl-llm-weights-apply lin probe) 0)))
      (aset sc2 1 (* 2.0 (aref sc2 1)))
      (let ((y1 (aref (nl-llm-weights-apply
                       (tt-lin bytes sc2 rows cols words bsize "rescaled") probe)
                      0)))
        (tt-check "control: the second block uses the second scale"
                  (and (/= y0 0.0) (< (abs (- y1 (* 2.0 y0))) 1.0e-12))
                  "column %d: %.9f -> %.9f" col y0 y1)))))

(defun tt-run ()
  (unless (and (file-readable-p tt-table) (file-readable-p tt-fixture))
    (message "SKIP: no %s / %s -- run `make ternary-export'" tt-table tt-fixture)
    (kill-emacs 0))
  (tt-fixture-rows)
  (tt-arithmetic)
  (message "ternary: %d passed, %d failed" tt-pass tt-fail)
  (when (> tt-fail 0) (kill-emacs 1)))

(tt-run)
