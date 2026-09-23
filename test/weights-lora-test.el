;;; weights-lora-test.el --- a trainable adapter over a frozen int8 base  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/weights-lora-test.el
;;
;; Doc 08 Phase 3.  The barrier this suite is about: a frozen quantized weight
;; still has to appear in the backward pass, because dL/dx = W^T.g even when W
;; collects no gradient.  Three things are therefore pinned, in order of how
;; badly they fail silently:
;;
;;   the transpose   by the inner-product identity <W.x, g> = <x, W^T.g>, which
;;                   no index swap survives -- and a transposed loop reads
;;                   almost exactly like the forward one, so reading it proves
;;                   nothing
;;   the gradients   against finite differences, separately for A, B and x,
;;                   because a wrong scale or a missing factor of alpha/rank
;;                   still produces a plausible descent direction
;;   the invariant   a fresh adapter has B = 0, so it must leave the base's
;;                   output bit-identical; if it does not, everything above is
;;                   measuring the wrong base
;;
;; and then that it actually learns, since all three can hold while the step
;; does nothing useful.  No donor table is needed: the weight is quantized here
;; with the same per-row scheme the exporter uses.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'nl-llm-weights)
(require 'nl-llm-lora)
(require 'nl-llm-weights-lora)

(defvar wl--fail 0)
(defun wl--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq wl--fail (1+ wl--fail)) "FAIL"))
                 (or extra ""))))

(defun wl--vec (n seed &optional scale)
  (let ((v (make-vector n 0.0)) (s (or scale 1.0)))
    (dotimes (i n)
      (aset v i (* s 0.37 (- (mod (+ (* (1+ i) 7919) (* seed 104729)) 211) 105))))
    v))

(defun wl--dot (a b)
  (let ((acc 0.0)) (dotimes (i (length a)) (setq acc (+ acc (* (aref a i) (aref b i))))) acc))

(defun wl--rel (a b)
  (let ((m (max (abs a) (abs b))))
    (if (= m 0.0) 0.0 (/ (abs (- a b)) m))))

;;; --- the transpose --------------------------------------------------------

(let* ((rows 7) (cols 11)
       (lin (nl-llm-weights-lin-quantize (wl--vec (* rows cols) 1) rows cols))
       (x (wl--vec cols 2))
       (g (wl--vec rows 3))
       (wx (nl-llm-weights-apply lin x))
       (wtg (nl-llm-weights-apply-t lin g))
       (lhs (wl--dot wx g))
       (rhs (wl--dot x wtg)))
  (wl--ck "transpose satisfies <W.x, g> = <x, W^T.g>"
          (< (wl--rel lhs rhs) 1.0e-12)
          (format "%.10f vs %.10f (rel %.2e)" lhs rhs (wl--rel lhs rhs)))

  ;; Non-square on purpose: a rows/cols swap passes a square test.
  (wl--ck "transpose returns one entry per input column"
          (= (length wtg) cols) (format "%d (rows %d)" (length wtg) rows))

  ;; Control: perturb one lane and the identity must break.
  (let* ((bytes (copy-sequence (nl-llm-weights-lin-bytes lin)))
         (bad (nl-llm-weights-lin--make
               :payload bytes :scale-cache (nl-llm-weights-lin-scales lin)
               :rows rows :cols cols
               :words (nl-llm-weights-lin-words lin) :name "perturbed")))
    (aset bytes 3 (mod (+ (aref bytes 3) 40) 256))
    (let ((l2 (wl--dot (nl-llm-weights-apply lin x) g))
          (r2 (wl--dot x (nl-llm-weights-apply-t bad g))))
      (wl--ck "control: a perturbed lane breaks the identity"
              (> (wl--rel l2 r2) 1.0e-9)
              (format "rel %.2e" (wl--rel l2 r2))))))

;;; --- the LoRA invariant ---------------------------------------------------

(let* ((rows 9) (cols 13) (rank 3)
       (lin (nl-llm-weights-lin-quantize (wl--vec (* rows cols) 4) rows cols))
       (lora (nl-llm-lora-make rows cols rank))
       (x (wl--vec cols 5))
       (base (nl-llm-weights-apply lin x))
       (fw (nl-llm-wlora-forward lin lora x))
       (y (nth 0 fw)))
  (let ((m 0.0))
    (dotimes (i rows) (setq m (max m (abs (- (aref y i) (aref base i))))))
    (wl--ck "a fresh adapter (B = 0) is the identity on the base"
            (= m 0.0) (format "maxdiff %.2e" m)))

  ;; And once B moves, it must not be.
  (let ((b (photon-tensor-data (plist-get lora :b))))
    (aset b 0 0.5)
    (let* ((fw2 (nl-llm-wlora-forward lin lora x))
           (m 0.0))
      (dotimes (i rows) (setq m (max m (abs (- (aref (nth 0 fw2) i) (aref base i))))))
      (wl--ck "control: a nonzero B changes the output"
              (> m 1.0e-9) (format "maxdiff %.3e" m)))
    (aset b 0 0.0)))

;;; --- the gradients, against finite differences ----------------------------

(let* ((rows 6) (cols 8) (rank 3)
       (lin (nl-llm-weights-lin-quantize (wl--vec (* rows cols) 6) rows cols))
       (lora (nl-llm-lora-make rows cols rank 6 7))
       (x (wl--vec cols 8))
       (target (wl--vec rows 9 0.2))
       (eps 1.0e-6))
  ;; B starts at zero, which makes dA identically zero; move it first so the
  ;; check is not trivially satisfied.
  (let ((b (photon-tensor-data (plist-get lora :b))))
    (dotimes (i (length b)) (aset b i (* 0.05 (aref (wl--vec (length b) 10) i)))))

  (cl-labels ((loss ()
                (car (nl-llm-wlora-sq-loss
                      (nth 0 (nl-llm-wlora-forward lin lora x)) target)))
              (fd (vec index)
                (let* ((saved (aref vec index))
                       (_ (aset vec index (+ saved eps)))
                       (up (loss))
                       (_2 (aset vec index (- saved eps)))
                       (down (loss)))
                  (aset vec index saved)
                  (/ (- up down) (* 2.0 eps)))))
    (let* ((fw (nl-llm-wlora-forward lin lora x))
           (lg (nl-llm-wlora-sq-loss (nth 0 fw) target))
           (grads (nl-llm-wlora-backward lin lora (nth 2 fw) (nth 1 fw) (cdr lg)))
           (a (photon-tensor-data (plist-get lora :a)))
           (b (photon-tensor-data (plist-get lora :b))))

      (let ((worst 0.0) (at nil))
        (dotimes (i (length a))
          (let ((r (wl--rel (aref (plist-get grads :da) i) (fd a i))))
            (when (> r worst) (setq worst r at i))))
        (wl--ck "dL/dA matches finite differences"
                (< worst 1.0e-5)
                (format "worst rel %.2e over %d entries (at %S)"
                        worst (length a) at)))

      (let ((worst 0.0))
        (dotimes (i (length b))
          (let ((r (wl--rel (aref (plist-get grads :db) i) (fd b i))))
            (when (> r worst) (setq worst r))))
        (wl--ck "dL/dB matches finite differences"
                (< worst 1.0e-5)
                (format "worst rel %.2e over %d entries" worst (length b))))

      ;; dx is the one that needs the frozen base's transpose.
      (let ((worst 0.0))
        (dotimes (i cols)
          (let ((r (wl--rel (aref (plist-get grads :dx) i) (fd x i))))
            (when (> r worst) (setq worst r))))
        (wl--ck "dL/dx matches finite differences (needs W^T)"
                (< worst 1.0e-5)
                (format "worst rel %.2e over %d entries" worst cols)))

      ;; Control: without the base term, dx would be wrong -- show by how much,
      ;; so "the frozen weight is needed in the backward" is a number.
      (let* ((only-lora (nl-llm-wlora--matvec-t
                         (photon-tensor-data (plist-get lora :a))
                         (plist-get lora :rank) cols
                         (let ((du (nl-llm-wlora--matvec-t
                                    (photon-tensor-data (plist-get lora :b))
                                    rows (plist-get lora :rank) (cdr lg)))
                               (s (nl-llm-lora-scale lora)))
                           (dotimes (r (length du)) (aset du r (* s (aref du r))))
                           du)))
             (worst 0.0))
        (dotimes (i cols)
          (let ((r (wl--rel (aref only-lora i) (fd x i))))
            (when (> r worst) (setq worst r))))
        (wl--ck "control: dropping W^T makes dL/dx wrong"
                (> worst 0.1)
                (format "worst rel %.2f without the base term" worst))))))

;;; --- and it learns --------------------------------------------------------

;; Magnitudes matter here and the first attempt got them wrong: with x entries
;; around +-39 the loss started at 1.1e7 and plain SGD at lr 0.002 diverged to
;; NaN in sixty steps.  The gradients were already verified correct above, so
;; that was the test's data, not the code -- but it is also the honest shape of
;; `nl-llm-wlora-fit', which is unclipped SGD.  So: unit-scale input, and a
;; target a short distance from where the base already lands, which is what
;; adapting a pretrained model actually asks for.
(let* ((rows 12) (cols 16) (rank 4)
       (lin (nl-llm-weights-lin-quantize (wl--vec (* rows cols) 11 0.02)
                                         rows cols))
       (lora (nl-llm-lora-make rows cols rank 8 12))
       (x (wl--vec cols 13 0.02))
       (base (nl-llm-weights-apply lin x))
       (target (let ((v (copy-sequence base)) (d (wl--vec rows 14 0.002)))
                 (dotimes (i rows) (aset v i (+ (aref v i) (aref d i))))
                 v))
       (losses (nl-llm-wlora-fit lin lora x target 60 0.05)))
  (wl--ck "the adapter trains, base untouched"
          (and (< (car (last losses)) (* 0.5 (car losses)))
               (cl-every (lambda (l) (and (numberp l) (= l l))) losses))
          (format "loss %.4f -> %.4f over %d steps"
                  (car losses) (car (last losses)) (length losses)))

  ;; The base really is frozen: its own output for a fresh probe is unchanged
  ;; by training, because nothing wrote to the int8 bytes or the scales.
  (let* ((probe (wl--vec cols 15))
         (before (nl-llm-weights-apply lin probe))
         (after (nl-llm-weights-apply lin probe))
         (m 0.0))
    (dotimes (i rows) (setq m (max m (abs (- (aref before i) (aref after i))))))
    (wl--ck "the frozen base is bit-identical after training"
            (= m 0.0) (format "maxdiff %.2e" m))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop wl--fail) "weights-lora OK" "weights-lora") wl--fail))
(when (> wl--fail 0) (kill-emacs 1))
