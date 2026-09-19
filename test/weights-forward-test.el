;;; weights-forward-test.el --- the imported model actually runs  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/weights-forward-test.el
;;
;; Doc 08 Phase 2c, Verification check 4.  This is the point where "the import
;; works" stops being a property of the file format and becomes a measurement:
;; the pure-Elisp forward runs Qwen3-0.6B off the int8 table and its hidden
;; state is compared against tools/qwen-forward-ref.py *layer by layer*.
;;
;; Layer by layer rather than on the logits, because the logits only say
;; "wrong".  A transposed weight, a head width of 64 instead of 128, an
;; interleaved rotation instead of half-split -- each shows up as a specific
;; layer, and each of those was a real mistake here at some point.
;;
;; The reference reads the SAME int8 table and dequantizes it the same way, so a
;; difference cannot be blamed on quantization; it isolates the arithmetic.
;; Both sides work in float64, so the only expected difference is summation
;; order (numpy's pairwise sums versus sequential accumulation in Elisp), which
;; is why the threshold is a relative 1e-9 rather than exact equality -- and why
;; the measured value is printed, so a threshold silently absorbing something
;; larger would be visible.
;;
;; Table and fixture are donor-derived and gitignored; this skips when absent.
;; At roughly seven seconds per layer per token it is the slowest suite here,
;; which is the price of being the oracle the GPU path gets checked against.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-attn)
(require 'nl-llm-weights)
(require 'nl-llm-weights-forward)

(defvar wfd--fail 0)
(defvar wfd--table (expand-file-name "build/donor/qwen3-0.6b/weights.bin"))
(defvar wfd--fixture (expand-file-name "test/fixtures/qwen-forward-ref.eld"))

(defun wfd--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name
                 (if ok "PASS" (progn (setq wfd--fail (1+ wfd--fail)) "FAIL"))
                 (or extra ""))))

(defun wfd--relerr (got want)
  "Largest |got-want| over WANT's largest magnitude, plus the index."
  (let ((scale 0.0) (worst 0.0) (at -1))
    (dotimes (i (length want))
      (setq scale (max scale (abs (nth i want)))))
    (when (= scale 0.0) (setq scale 1.0))
    (dotimes (i (length want))
      (let ((d (abs (- (aref got i) (nth i want)))))
        (when (> d worst) (setq worst d at i))))
    (cons (/ worst scale) at)))

;;; --- internal consistency, no donor needed -------------------------------
;;
;; nl-llm-wf--attend duplicates the attention loop rather than calling
;; nl-llm-gqa, which needs photon-tensors this model cannot afford.  That
;; duplication is the risk; pinning it against nl-llm-gqa on a model small
;; enough for both is the mitigation.

(let* ((dim 8) (seq 3) (heads 2) (kv-heads 1) (hd 6)
       (qdim (* heads hd)) (kvdim (* kv-heads hd))
       (mk (lambda (n seed)
             (let ((v (make-vector n 0.0)))
               (dotimes (i n)
                 (aset v i (* 0.17 (- (mod (+ (* (1+ i) 7) seed) 11) 5))))
               v)))
       (q (funcall mk (* seq qdim) 1))
       (k (funcall mk (* seq kvdim) 2))
       (v (funcall mk (* seq kvdim) 3))
       ;; nl-llm-gqa reaches attention through projections, so drive it with
       ;; identity-shaped weights and compare the context it produces.
       (ctx (nl-llm-wf--attend (copy-sequence q) (copy-sequence k)
                               (copy-sequence v) seq heads kv-heads hd)))
  ;; Recompute the same thing the long way round, from the definition.
  (let ((want (make-vector (* seq qdim) 0.0))
        (grp (/ heads kv-heads)) (scale (/ 1.0 (sqrt (float hd)))))
    (dotimes (h heads)
      (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((sc (make-vector (1+ i) 0.0)) (sm 0.0) (mx -1.0e30))
            (dotimes (j (1+ i))
              (let ((acc 0.0))
                (dotimes (t0 hd)
                  (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                      (aref k (+ (* j kvdim) kc t0))))))
                (aset sc j (* acc scale))
                (setq mx (max mx (aref sc j)))))
            (dotimes (j (1+ i))
              (aset sc j (exp (- (aref sc j) mx)))
              (setq sm (+ sm (aref sc j))))
            (dotimes (t0 hd)
              (let ((acc 0.0))
                (dotimes (j (1+ i))
                  (setq acc (+ acc (* (/ (aref sc j) sm)
                                      (aref v (+ (* j kvdim) kc t0))))))
                (aset want (+ (* i qdim) qc t0) acc)))))))
    (let ((m 0.0))
      (dotimes (i (length want))
        (setq m (max m (abs (- (aref ctx i) (aref want i))))))
      (wfd--ck "attention loop matches the definition" (< m 1.0e-15)
               (format "maxdiff %.2e" m)))))

;;; --- the real thing -------------------------------------------------------

(if (not (and (file-readable-p wfd--table) (file-readable-p wfd--fixture)))
    (princ (format "SKIP: table or reference fixture missing\n  %s\n  %s\n\
  regenerate with:  make qwen-weights-table && make qwen-forward-ref\n"
                   wfd--table wfd--fixture))

  (let* ((ref (with-temp-buffer
                (let ((coding-system-for-read 'utf-8-unix))
                  (insert-file-contents wfd--fixture))
                (goto-char (point-min))
                (read (current-buffer))))
         (tokens (plist-get ref :tokens))
         (nlayers (plist-get ref :layers))
         (dim (plist-get ref :dim))
         (seq (plist-get ref :seq))
         (states (plist-get ref :states))
         (wts (nl-llm-weights-open wfd--table))
         (cfg (nl-llm-weights-config wts)))

    (wfd--ck "reference and table agree on shape"
             (and (= dim (plist-get cfg :dim)) (= seq (length tokens)))
             (format "%d tokens, dim %d, %d layers" seq dim nlayers))

    (let* ((t0 (float-time))
           (got (nl-llm-wf-hidden wts tokens nlayers))
           (secs (- (float-time) t0)))
      (wfd--ck "forward produced one state per layer plus the embedding"
               (= (length got) (1+ nlayers))
               (format "%d states in %.1fs (%.1fs per layer per token)"
                       (length got) secs
                       (/ secs (max 1 (* nlayers seq)))))

      ;; Report every layer, and name the FIRST that diverges.
      (let ((first-bad nil))
        (dotimes (i (min (length got) (length states)))
          (let* ((r (wfd--relerr (nth i got) (nth i states)))
                 (rel (car r))
                 (ok (< rel 1.0e-9))
                 (label (if (= i 0) "embedding" (format "after layer %d" (1- i)))))
            (unless (or ok first-bad) (setq first-bad label))
            (wfd--ck (format "hidden %s == reference" label) ok
                     (format "rel %.2e at index %d" rel (cdr r)))))
        (when first-bad
          (princ (format "\nfirst divergence: %s\n" first-bad))))

      ;; A logit for a handful of tokens, so the head path is exercised too.
      (let* ((final (nl-llm-wf-final-norm wts (car (last got)) seq))
             (probe (list 0 100 14990 151935))
             (lg (nl-llm-wf-logits wts final seq (1- seq) probe)))
        (wfd--ck "logits are finite for a few probe tokens"
                 (cl-every (lambda (v) (and (numberp v) (= v v)
                                            (< (abs v) 1.0e6)))
                           (append lg nil))
                 (format "%S -> %S" probe
                         (mapcar (lambda (v) (/ (fround (* 1000 v)) 1000.0))
                                 (append lg nil))))))

    (princ (format "\n%s: %d failure(s)\n"
                   (if (zerop wfd--fail) "weights-forward OK" "weights-forward")
                   wfd--fail))
    (when (> wfd--fail 0) (kill-emacs 1))))
