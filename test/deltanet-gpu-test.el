;;; deltanet-gpu-test.el --- the recurrence on the device -*- lexical-binding: t -*-

;; `gdn-step' against `nl-llm-dn-forward', head by head, on the same inputs.
;; The CPU side is already checked against an independent transcription of the
;; reference and against finite differences, so what this adds is that the
;; device computes the same thing -- including the part a kernel gets wrong
;; quietly, which is the ordering: the decay applies before the memory is read
;; and the write happens before the output is taken.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-deltanet)
(require 'nl-llm-deltanet-gpu)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defvar dg-pass 0)
(defvar dg-fail 0)
(defvar dg-seed 4711)

(defun dg-check (name ok fmt &rest args)
  (if ok (setq dg-pass (1+ dg-pass)) (setq dg-fail (1+ dg-fail)))
  (message "%-52s %s  %s" name (if ok "PASS" "FAIL") (apply #'format fmt args)))

(defun dg-rnd ()
  (setq dg-seed (mod (+ (* dg-seed 1103515245) 12345) 2147483648))
  (- (/ (float dg-seed) 1073741824.0) 1.0))

(defun dg-err (got want)
  "Worst difference against the reference's own scale."
  (let ((scale 0.0) (worst 0.0))
    (dotimes (i (length want)) (setq scale (max scale (abs (aref want i)))))
    (dotimes (i (length want))
      (setq worst (max worst (abs (- (aref got i) (aref want i))))))
    (/ worst (max 1.0e-12 scale))))

(defun dg-run ()
  (let* ((seq 6) (nv 5) (dk 128) (dv 128)
         (q (make-vector (* seq nv dk) 0.0))
         (k (make-vector (* seq nv dk) 0.0))
         (v (make-vector (* seq nv dv) 0.0))
         (a (make-vector (* seq nv) 0.0))
         (b (make-vector (* seq nv) 0.0))
         (alog (make-vector nv 0.0))
         (dtb (make-vector nv 0.0)))
    (dotimes (i (* seq nv dk)) (aset q i (dg-rnd)) (aset k i (dg-rnd)))
    (dotimes (i (* seq nv dv)) (aset v i (* 0.3 (dg-rnd))))
    (dotimes (i (* seq nv)) (aset a i (dg-rnd)) (aset b i (dg-rnd)))
    (dotimes (h nv)
      (aset alog h (* 0.5 (dg-rnd)))
      (aset dtb h (* 0.5 (dg-rnd))))
    ;; the CPU answer, one head at a time, from the raw inputs
    (let ((want (make-vector (* seq nv dv) 0.0)))
      (dotimes (h nv)
        (let ((qh (make-vector (* seq dk) 0.0)) (kh (make-vector (* seq dk) 0.0))
              (vh (make-vector (* seq dv) 0.0))
              (ah (make-vector seq 0.0)) (bh (make-vector seq 0.0)))
          (dotimes (tt seq)
            (dotimes (i dk)
              (aset qh (+ (* tt dk) i) (aref q (+ (* (+ (* tt nv) h) dk) i)))
              (aset kh (+ (* tt dk) i) (aref k (+ (* (+ (* tt nv) h) dk) i))))
            (dotimes (j dv)
              (aset vh (+ (* tt dv) j) (aref v (+ (* (+ (* tt nv) h) dv) j))))
            (aset ah tt (aref a (+ (* tt nv) h)))
            (aset bh tt (aref b (+ (* tt nv) h))))
          (let ((oh (nth 0 (nl-llm-dn-forward qh kh vh ah bh (aref alog h)
                                              (aref dtb h) seq dk dv))))
            (dotimes (tt seq)
              (dotimes (j dv)
                (aset want (+ (* (+ (* tt nv) h) dv) j)
                      (aref oh (+ (* tt dv) j))))))))
      (nelisp-gpu-server-start)
      (unwind-protect
          (let* ((prep (nl-llm-dngpu-prepare q k a b alog dtb seq nv dk))
                 (got (nl-llm-dngpu-scan (nth 0 prep) (nth 1 prep) v
                                         (nth 2 prep) (nth 3 prep)
                                         seq nv dk dv)))
            (dg-check "gdn-step matches nl-llm-dn-forward"
                      (< (dg-err got want) 1.0e-4)
                      "worst %.3e of the reference's scale over %d values"
                      (dg-err got want) (length want))
            (let ((nonzero (cl-some (lambda (x) (/= x 0.0)) (append got nil))))
              (dg-check "control: the output is not all zero" nonzero
                        (if nonzero "it is not" "every value came back zero")))
            ;; the recurrence has to be a recurrence.  Perturb head 0's value
            ;; at position 0 and a LATER position of the same head must move:
            ;; a kernel that mapped each position independently would not.
            ;; Later positions of other heads must not, and neither must
            ;; position 0 of any other head -- the state is per head.
            (let ((v2 (copy-sequence v)))
              (dotimes (j dv) (aset v2 j (+ (aref v2 j) 1.0)))
              (let* ((g2 (nl-llm-dngpu-scan (nth 0 prep) (nth 1 prep) v2
                                            (nth 2 prep) (nth 3 prep)
                                            seq nv dk dv))
                     (h0-next (* nv dv))          ; position 1, head 0
                     (h1-next (+ (* nv dv) dv))   ; position 1, head 1
                     (moved 0) (spilled 0))
                (dotimes (j dv)
                  (unless (= (aref got (+ h0-next j)) (aref g2 (+ h0-next j)))
                    (setq moved (1+ moved)))
                  (unless (= (aref got (+ h1-next j)) (aref g2 (+ h1-next j)))
                    (setq spilled (1+ spilled))))
                (dg-check "control: a later position carries the state"
                          (> moved (/ dv 2)) "%d of %d outputs moved" moved dv)
                (dg-check "control: and the state does not cross heads"
                          (= spilled 0) "%d outputs of another head moved"
                          spilled))))
        (nelisp-gpu-server-stop))))
  (message "deltanet-gpu: %d passed, %d failed" dg-pass dg-fail)
  (when (> dg-fail 0) (kill-emacs 1)))

(dg-run)
